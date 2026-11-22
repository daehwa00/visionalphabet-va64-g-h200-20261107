"""Tensor Core path projection for interleaved vertical scan features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from torch import Tensor

# A mode shares weights across this many adjacent output cells.
_CELL_TILE = 64


# Triton pointer and scalar argument types are inferred at specialization.
@triton.jit
def _project(  # noqa: ANN202
    feature_ptr, wi_ptr, bi_ptr, wo_ptr, bo_ptr,  # noqa: ANN001
    real_ptr, imag_ptr, dp_ptr, partial_ptr,  # noqa: ANN001
    cells, modes: tl.constexpr, hidden: tl.constexpr,  # noqa: ANN001
    real_stride: tl.constexpr, imag_stride: tl.constexpr,
    tile_hidden: tl.constexpr, tile_cells: tl.constexpr, backward: tl.constexpr,
):
    tile = tl.program_id(0)
    mode = tl.program_id(1)
    cell = tile * tile_cells + tl.arange(0, tile_cells)
    col = tl.arange(0, 16)
    unit = tl.arange(0, tile_hidden)
    feature_dtype = feature_ptr.dtype.element_ty
    input_offset = (cell[:, None] * modes + mode) * 8 + col[None, :]
    p = tl.load(feature_ptr + input_offset, (cell[:, None] < cells) & (col[None, :] < 8), 0)
    wi = tl.load(
        wi_ptr + mode * hidden * 8 + unit[None, :] * 8 + col[:, None],
        (col[:, None] < 8) & (unit[None, :] < hidden),
        0,
    )
    bi = tl.load(bi_ptr + mode * hidden + unit, unit < hidden, 0).to(tl.float32)
    pre = tl.dot(p, wi).to(feature_dtype).to(tl.float32)
    pre = (pre + bi[None, :]).to(feature_dtype).to(tl.float32)
    sigmoid = tl.sigmoid(pre)
    act = (pre * sigmoid).to(feature_dtype)
    wo = tl.load(
        wo_ptr + mode * 2 * hidden + col[:, None] * hidden + unit[None, :],
        (col[:, None] < 2) & (unit[None, :] < hidden),
        0,
    )

    if backward:
        real = tl.load(real_ptr + cell * modes * real_stride + mode, cell < cells, 0)
        imag = tl.load(imag_ptr + cell * modes * imag_stride + mode, cell < cells, 0)
        dy = tl.where(col[None, :] == 0, real[:, None], 0)
        dy += tl.where(col[None, :] == 1, imag[:, None], 0)
        dy = dy.to(feature_dtype)
        da = tl.dot(dy, wo).to(feature_dtype).to(tl.float32)
        dz = (da * (sigmoid * (1 + pre * (1 - sigmoid)))).to(feature_dtype)
        dp = tl.dot(dz, tl.trans(wi))
        tl.store(dp_ptr + input_offset, dp, (cell[:, None] < cells) & (col[None, :] < 8))

        row = partial_ptr + (tile * modes + mode) * (11 * hidden + 2)
        dwi = tl.dot(tl.trans(dz), p)
        dwo = tl.dot(tl.trans(dy), act)
        tl.store(
            row + unit[:, None] * 8 + col[None, :], dwi,
            (unit[:, None] < hidden) & (col[None, :] < 8),
        )
        tl.store(row + 8 * hidden + unit, tl.sum(dz.to(tl.float32), 0), unit < hidden)
        tl.store(
            row + 9 * hidden + col[:, None] * hidden + unit[None, :], dwo,
            (col[:, None] < 2) & (unit[None, :] < hidden),
        )
        tl.store(row + 11 * hidden + col, tl.sum(dy.to(tl.float32), 0), col < 2)
    else:
        out = tl.dot(act, tl.trans(wo)).to(feature_dtype).to(tl.float32)
        bo = tl.load(bo_ptr + mode * 2 + col, col < 2, 0).to(tl.float32)
        out = out + bo[None, :]
        real = tl.sum(tl.where(col[None, :] == 0, out, 0), 1)
        imag = tl.sum(tl.where(col[None, :] == 1, out, 0), 1)
        tl.store(real_ptr + cell * modes * real_stride + mode, real, cell < cells)
        tl.store(imag_ptr + cell * modes * imag_stride + mode, imag, cell < cells)


def path_projection(
    features: Tensor,
    path: tuple[Tensor, Tensor, Tensor, Tensor],
    re: Tensor,
    im: Tensor,
    *,
    backward: bool,
) -> tuple[Tensor, tuple[Tensor, ...]] | None:
    wi, bi, wo, bo = path
    modes, hidden, _ = wi.shape
    cells = re.numel() // modes
    tiles = triton.cdiv(cells, _CELL_TILE)
    if backward:
        dp = torch.empty_like(features)
        partial = features.new_empty((tiles, modes, 11 * hidden + 2), dtype=torch.float32)
    else:
        dp = partial = features
    _project[(tiles, modes)](
        features, wi, bi, wo, bo, re, im, dp, partial,
        cells, modes, hidden, 1 if re.is_contiguous() else 2, 1 if im.is_contiguous() else 2,
        max(16, triton.next_power_of_2(hidden)), _CELL_TILE, backward, num_warps=4,
    )
    if not backward:
        return None
    fields = partial.sum(0).split([8 * hidden, hidden, 2 * hidden, 2], dim=-1)
    return dp, tuple(field.reshape_as(target) for field, target in zip(fields, path, strict=True))
