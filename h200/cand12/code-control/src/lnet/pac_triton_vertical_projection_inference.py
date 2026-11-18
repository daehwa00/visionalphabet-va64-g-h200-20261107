"""Opt-in vertical scan/path collapse/complex projection inference candidate.

A CTA owns one spatial column and one projection output tile. It scans and
collapses successive pole tiles on chip, accumulating the four complex GEMMs.
There is no materialized path-collapse tensor. Scans are repeated across output
tiles, so this is a measured candidate rather than an assumed improvement.
"""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# pyright: reportPrivateUsage=false, reportGeneralTypeIssues=false
import math
from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional

from .pac_product_scan_contracts import DEFAULT_EPSILON, ProductGainNormalization, gain_kind
from .pac_triton_product_scan_kernels import _compose_complex_pair
from .pac_triton_scan2d_inference import _collapse_full_paths

Pole = tuple[Tensor, Tensor, Tensor, Tensor]
Path = tuple[Tensor, Tensor, Tensor, Tensor]
Auxiliary = tuple[Tensor, Tensor, Tensor]


def vertical_scan_projection_reference(
    pole_y: Pole,
    horizontal: Tensor,
    auxiliary: Auxiliary,
    path: Path,
    weight_real: Tensor,
    weight_imag: Tensor,
    *,
    gain_normalization: ProductGainNormalization = "global",
    epsilon: float = DEFAULT_EPSILON,
    path_swiglu: bool = False,
) -> Tensor:
    """Mirror CUDA's fused-path BF16 boundaries using ordinary PyTorch.

    Vertical recurrence and gain multiplication remain FP32 until the packed
    path input cast. Both path linear+bias pairs round once, as in the existing
    fused scan/path kernel. Projection coordinate GEMMs round separately.
    """
    batch, height, width, modes = horizontal.shape[1:]
    ar, ai, gr, gi = (value.float().reshape(1, 1, modes) for value in pole_y)
    vx, vy, global_gain = auxiliary
    if gain_normalization == "global" and not global_gain.numel():
        global_gain = torch.rsqrt((vx[0].mean(0) * vy[0].mean(0)).clamp_min(epsilon))
    paths_real, paths_imag = [], []
    for reverse in (False, True):
        decay_imag = -ai if reverse else ai
        gamma_imag = -gi if reverse else gi
        positions = range(height - 1, -1, -1) if reverse else range(height)
        for direction in (0, 1):
            state_real = horizontal.new_zeros((batch, width, modes), dtype=torch.float32)
            state_imag = torch.zeros_like(state_real)
            rows_real, rows_imag = [], []
            for y in positions:
                source_real = horizontal[2 * direction, :, y].float()
                source_imag = horizontal[2 * direction + 1, :, y].float()
                next_real = ar * state_real - decay_imag * state_imag
                next_imag = ar * state_imag + decay_imag * state_real
                state_real = next_real + gr * source_real - gamma_imag * source_imag
                state_imag = next_imag + gr * source_imag + gamma_imag * source_real
                rows_real.append(state_real)
                rows_imag.append(state_imag)
            real = torch.stack(rows_real, dim=1)
            imag = torch.stack(rows_imag, dim=1)
            if reverse:
                real, imag = real.flip(1), imag.flip(1)
            if gain_normalization == "global":
                inverse = global_gain.reshape(1, 1, 1, modes)
            else:
                variance = vx[direction].reshape(1, 1, width, modes) * vy[int(reverse)].reshape(
                    1, height, 1, modes,
                )
                inverse = torch.rsqrt(variance.clamp_min(epsilon))
            paths_real.append(real * inverse)
            paths_imag.append(imag * inverse)
    packed_path = torch.stack((*paths_real, *paths_imag), dim=-1).bfloat16()
    w1, b1, w2, b2 = (value.bfloat16() for value in path)
    joint = (
        torch.einsum("...pi,phi->...ph", packed_path.float(), w1.float()) + b1.float()
    ).bfloat16()
    if path_swiglu:
        hidden = joint.shape[-1] // 3
        gate = functional.silu(joint[..., 2 * hidden :]).bfloat16()
        activation = joint[..., : 2 * hidden] * torch.cat((gate, gate), dim=-1)
    else:
        activation = functional.silu(joint).bfloat16()
    collapsed = (
        torch.einsum("...ph,poh->...po", activation.float(), w2.float()) + b2.float()
    ).bfloat16()
    real, imag = collapsed[..., 0], collapsed[..., 1]
    wr, wi = weight_real.bfloat16(), weight_imag.bfloat16()
    return torch.cat(
        (functional.linear(real, wr) - functional.linear(imag, wi),
         functional.linear(real, wi) + functional.linear(imag, wr)), dim=-1,
    ).contiguous()


@triton.jit
def _vertical_path_projection_kernel(
    HORIZONTAL, AYR, AYI, GYR, GYI, VX, VY, GLOBAL_GAIN,
    W1, B1, W2, B2, WR, WI, OUTPUT,
    N: tl.constexpr, H: tl.constexpr, W: tl.constexpr, P: tl.constexpr, M: tl.constexpr,
    KIND: tl.constexpr, EPSILON: tl.constexpr, PATH_SWIGLU: tl.constexpr,
    HIDDEN: tl.constexpr, VALUE: tl.constexpr, BH: tl.constexpr,
    BP: tl.constexpr, BM: tl.constexpr, BLOCK_HIDDEN: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
):
    column = tl.program_id(0)
    batch, x = column // W, column % W
    output_mode = tl.program_id(1) * BM + tl.arange(0, BM)
    y = tl.arange(0, BH)[:, None, None]
    pole_lane = tl.arange(0, BP)
    rr = tl.full((BH, BM), 0.0, tl.float32)
    ri = tl.full((BH, BM), 0.0, tl.float32)
    ir = tl.full((BH, BM), 0.0, tl.float32)
    ii = tl.full((BH, BM), 0.0, tl.float32)
    plane_size: tl.constexpr = N * H * W * P
    for start in range(tl.cdiv(P, BP)):
        poles = start * BP + pole_lane
        mode = poles[None, None, :]
        active = (y < H) & (mode < P)
        offset = ((batch * H + y) * W + x) * P + mode
        sar = tl.load(HORIZONTAL + offset, active, 0).to(tl.float32)
        sai = tl.load(HORIZONTAL + plane_size + offset, active, 0).to(tl.float32)
        sbr = tl.load(HORIZONTAL + 2 * plane_size + offset, active, 0).to(tl.float32)
        sbi = tl.load(HORIZONTAL + 3 * plane_size + offset, active, 0).to(tl.float32)
        ar = tl.load(AYR + mode, mode < P, 0).to(tl.float32)
        ai = tl.load(AYI + mode, mode < P, 0).to(tl.float32)
        gr = tl.load(GYR + mode, mode < P, 0).to(tl.float32)
        gi = tl.load(GYI + mode, mode < P, 0).to(tl.float32)
        scan_ar, scan_ai = tl.where(active, ar, 1.0), tl.where(active, ai, 0.0)
        positive = tl.associative_scan(
            (scan_ar, scan_ai, gr * sar - gi * sai, gr * sai + gi * sar,
             gr * sbr - gi * sbi, gr * sbi + gi * sbr),
            axis=0, combine_fn=_compose_complex_pair,
        )
        negative = tl.associative_scan(
            (scan_ar, -scan_ai, gr * sar + gi * sai, gr * sai - gi * sar,
             gr * sbr + gi * sbi, gr * sbi - gi * sbr),
            axis=0, combine_fn=_compose_complex_pair, reverse=True,
        )
        if KIND == 1:
            inv0 = tl.load(GLOBAL_GAIN + mode, mode < P, 0)
            inv1, inv2, inv3 = inv0, inv0, inv0
        else:
            vx0 = tl.load(VX + x * P + mode, mode < P, 0)
            vx1 = tl.load(VX + W * P + x * P + mode, mode < P, 0)
            vy0 = tl.load(VY + y * P + mode, active, 0)
            vy1 = tl.load(VY + H * P + y * P + mode, active, 0)
            inv0 = tl.rsqrt(tl.maximum(vx0 * vy0, EPSILON))
            inv1 = tl.rsqrt(tl.maximum(vx1 * vy0, EPSILON))
            inv2 = tl.rsqrt(tl.maximum(vx0 * vy1, EPSILON))
            inv3 = tl.rsqrt(tl.maximum(vx1 * vy1, EPSILON))
        r0 = tl.where(active, positive[2] * inv0, 0.0)
        i0 = tl.where(active, positive[3] * inv0, 0.0)
        r1 = tl.where(active, positive[4] * inv1, 0.0)
        i1 = tl.where(active, positive[5] * inv1, 0.0)
        r2 = tl.where(active, negative[2] * inv2, 0.0)
        i2 = tl.where(active, negative[3] * inv2, 0.0)
        r3 = tl.where(active, negative[4] * inv3, 0.0)
        i3 = tl.where(active, negative[5] * inv3, 0.0)
        collapsed_real, collapsed_imag = _collapse_full_paths(
            r0, i0, r1, i1, r2, i2, r3, i3, W1, B1, W2, B2,
            poles, P, HIDDEN, BH, 1, BP, BLOCK_HIDDEN, BLOCK_VALUE, VALUE, PATH_SWIGLU,
        )
        # This cast replaces the BF16 global-memory path-collapse store.
        xr = tl.reshape(collapsed_real, (BH, BP)).to(tl.bfloat16)
        xi = tl.reshape(collapsed_imag, (BH, BP)).to(tl.bfloat16)
        weight_offset = output_mode[None, :] * P + poles[:, None]
        weight_active = (output_mode[None, :] < M) & (poles[:, None] < P)
        wr = tl.load(WR + weight_offset, weight_active, 0).to(tl.bfloat16)
        wi = tl.load(WI + weight_offset, weight_active, 0).to(tl.bfloat16)
        rr = tl.dot(xr, wr, rr)
        ri = tl.dot(xi, wr, ri)
        ir = tl.dot(xr, wi, ir)
        ii = tl.dot(xi, wi, ii)
    # Four coordinate GEMMs each store BF16 before the complex add/subtract.
    pr = (rr.to(tl.bfloat16).to(tl.float32) - ii.to(tl.bfloat16).to(tl.float32)).to(
        tl.bfloat16,
    )
    pi = (ir.to(tl.bfloat16).to(tl.float32) + ri.to(tl.bfloat16).to(tl.float32)).to(
        tl.bfloat16,
    )
    rows = tl.arange(0, BH)[:, None]
    output_offset = ((batch * H + rows) * W + x) * (2 * M) + output_mode[None, :]
    valid_output = (rows < H) & (output_mode[None, :] < M)
    tl.store(OUTPUT + output_offset, pr, valid_output)
    tl.store(OUTPUT + output_offset + M, pi, valid_output)


@triton_op("lnet::vertical_path_projection_inference", mutates_args={})
def _vertical_path_projection_op(
    horizontal: Tensor, poles: list[Tensor], auxiliary: list[Tensor], path: list[Tensor],
    weight_real: Tensor, weight_imag: Tensor,
    kind: int, epsilon: float, path_swiglu: bool,
    pole_tile: int, output_tile: int, num_warps: int,
) -> Tensor:
    if not horizontal.is_cuda:
        return vertical_scan_projection_reference(
            cast("Pole", tuple(poles)), horizontal, cast("Auxiliary", tuple(auxiliary)),
            cast("Path", tuple(path)), weight_real, weight_imag,
            gain_normalization="global" if kind == 1 else "pointwise",
            epsilon=epsilon, path_swiglu=path_swiglu,
        )
    batch, height, width, modes = horizontal.shape[1:]
    outputs = weight_real.shape[0]
    hidden = path[0].shape[1]
    value = hidden // 3 if path_swiglu else 1
    output = horizontal.new_empty((batch, height, width, 2 * outputs))
    wrap_triton(_vertical_path_projection_kernel)[
        (batch * width, triton.cdiv(outputs, output_tile))
    ](
        horizontal, *poles, *auxiliary, *path, weight_real, weight_imag, output,
        batch, height, width, modes, outputs, kind, epsilon, path_swiglu,
        hidden, value, max(16, triton.next_power_of_2(height)), pole_tile, output_tile,
        max(16, triton.next_power_of_2(hidden)),
        max(16, triton.next_power_of_2(2 * value)), num_warps=num_warps, num_stages=1,
    )
    return output


def run_vertical_scan_projection_inference(  # noqa: C901, PLR0912
    pole_y: Pole,
    horizontal: Tensor,
    auxiliary: Auxiliary,
    path: Path,
    weight_real: Tensor,
    weight_imag: Tensor,
    *,
    gain_normalization: ProductGainNormalization = "global",
    epsilon: float = DEFAULT_EPSILON,
    path_swiglu: bool = False,
    pole_tile: int = 32,
    output_tile: int = 32,
    num_warps: int = 8,
    experimental_large_height: bool = False,
) -> Tensor:
    """Fuse vertical scan, D4 path collapse and four complex projection GEMMs.

    ``horizontal`` contains real+/imag+/real-/imag- planes ``[4,N,H,W,P]``.
    ``auxiliary`` is returned by ``static_product_scan_auxiliary`` for both
    original poles and ``horizontal[0]``. Output is packed ``[N,H,W,2M]``.
    Frozen inference only; the default height limit is 16. Explicit experiments
    may use heights through 56, with no resource or performance guarantee.
    """
    if horizontal.ndim != 5 or horizontal.shape[0] != 4 or min(horizontal.shape) < 1:
        raise ValueError("vertical projection requires positive [4,N,H,W,P] horizontal planes")
    if horizontal.dtype != torch.bfloat16:
        raise TypeError("vertical projection requires BF16 horizontal planes")
    _, _, height, width, modes = horizontal.shape
    max_height = 56 if experimental_large_height else 16
    if height > max_height:
        raise ValueError(f"vertical projection requires height <={max_height}")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if len(pole_y) != 4 or any(value.shape != (1, 1, 1, modes) for value in pole_y):
        raise ValueError("vertical projection requires compact 111P poles")
    if len(auxiliary) != 3:
        raise ValueError("vertical projection requires VX, VY, global gain")
    vx, vy, global_gain = auxiliary
    kind = gain_kind(gain_normalization)
    if kind == 1:
        if global_gain.numel() != modes and (horizontal.is_cuda or global_gain.numel()):
            raise ValueError("global gain must contain one value per pole")
        if not global_gain.numel() and (
            vx.shape != (2, width, modes) or vy.shape != (2, height, modes)
        ):
            raise ValueError("CPU global normalization requires valid variance tables")
    elif vx.shape != (2, width, modes) or vy.shape != (2, height, modes):
        raise ValueError("pointwise variance tables must match spatial dimensions")
    if weight_real.ndim != 2 or weight_real.shape != weight_imag.shape:
        raise ValueError("projection weights must have matching matrix shapes")
    if weight_real.shape[0] < 1 or weight_real.shape[1] != modes:
        raise ValueError("projection weights must match pole count")
    if len(path) != 4:
        raise ValueError("path requires four weight/bias tensors")
    w1, b1, w2, b2 = path
    if w1.ndim != 3 or w1.shape[0] != modes or w1.shape[2] != 8:
        raise ValueError("path input weights must have [P,hidden,8] shape")
    hidden = w1.shape[1]
    if hidden < 1 or (path_swiglu and hidden % 3):
        raise ValueError("path hidden width is invalid")
    output_hidden = 2 * (hidden // 3) if path_swiglu else hidden
    if b1.shape != (modes, hidden) or w2.shape != (modes, 2, output_hidden):
        raise ValueError("path hidden parameter shapes are inconsistent")
    if b2.shape != (modes, 2):
        raise ValueError("path output bias must have [P,2] shape")
    tensors = (horizontal, *pole_y, *auxiliary, *path, weight_real, weight_imag)
    if any(value.device != horizontal.device or not value.is_contiguous() for value in tensors):
        raise ValueError("all inputs must be contiguous and share a device")
    if any(value.dtype not in (torch.float32, torch.bfloat16) for value in tensors):
        raise TypeError("all inputs require FP32 or BF16 dtype")
    if any(value.dtype != torch.float32 for value in auxiliary):
        raise TypeError("variance tables and global gain require FP32 dtype")
    if torch.is_grad_enabled() and any(value.requires_grad for value in tensors):
        raise RuntimeError("vertical projection is inference only")
    if pole_tile not in (16, 32, 64) or output_tile not in (32, 64, 128):
        raise ValueError("unsupported vertical projection tile sizes")
    if num_warps not in (4, 8):
        raise ValueError("vertical projection num_warps must be 4 or 8")
    return _vertical_path_projection_op(
        horizontal, list(pole_y), list(auxiliary), list(path), weight_real, weight_imag,
        kind, epsilon, path_swiglu, pole_tile, output_tile, num_warps,
    )
