"""Terminal correlations from updated vertical states and their writes."""

from __future__ import annotations

# Triton infers pointer types from the launch arguments.
# ruff: noqa: ANN001
# pyright: reportArgumentType=false, reportCallIssue=false
import os
from typing import cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .general_cuda_scans import (
    general_scans_enabled,
    horizontal_forward,
    try_general_horizontal_forward,
)
from .model_w28c import model_w28c_enabled
from .product_scan_reference import _causal_weights, bidirectional_product_scan_reference
from .triton_bidirectional_product_scan import (
    _compose_complex,
    triton_bidirectional_product_scan,
)

PoleTensors = tuple[Tensor, Tensor, Tensor, Tensor]
SourcePair = tuple[Tensor, Tensor]
StateQuad = tuple[Tensor, Tensor, Tensor, Tensor]


@triton.jit
def _terminal_vertical_scan_kernel(
    x_r,
    x_i,
    a_r,
    a_i,
    y_r,
    y_i,
    height: tl.constexpr,
    width: tl.constexpr,
    modes: tl.constexpr,
    real_strides: tl.constexpr,
    imag_strides: tl.constexpr,
    reverse: tl.constexpr,
    block_height: tl.constexpr,
    block_columns: tl.constexpr,
) -> None:
    batch = tl.program_id(1).to(tl.int64)
    row = tl.arange(0, block_height)[:, None]
    column = tl.program_id(0) * block_columns + tl.arange(0, block_columns)[None, :]
    x = column // modes
    mode = column % modes
    valid = (row < height) & (x < width)
    r = tl.load(
        x_r
        + batch * real_strides[0]
        + row * real_strides[1]
        + x * real_strides[2]
        + mode * real_strides[3],
        valid,
        other=0,
    )
    i = tl.load(
        x_i
        + batch * imag_strides[0]
        + row * imag_strides[1]
        + x * imag_strides[2]
        + mode * imag_strides[3],
        valid,
        other=0,
    )
    decay_r = tl.where(valid, tl.load(a_r + mode), 1.0)
    decay_i = tl.where(valid, tl.load(a_i + mode), 0.0)
    _, _, r, i = tl.associative_scan(
        (decay_r, decay_i, r, i),
        0,
        _compose_complex,
        reverse=reverse,
    )
    offset = (batch * height + row) * (width * modes) + column
    tl.store(y_r + offset, r, valid)
    tl.store(y_i + offset, i, valid)


@triton_op("lnet::terminal_vertical_scan", mutates_args=())
def _vertical_scan(
    real: Tensor,
    imag: Tensor,
    decay_r: Tensor,
    decay_i: Tensor,
    reverse: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor]:
    batch, height, width, modes = real.shape
    out_r, out_i = real.new_empty(real.shape), imag.new_empty(imag.shape)
    if batch and height and width and modes:
        block_height = triton.next_power_of_2(height)
        block_columns = max(4, min(32, 1024 // block_height))
        grid = (triton.cdiv(width * modes, block_columns), batch)
        wrap_triton(_terminal_vertical_scan_kernel)[grid](
            real,
            imag,
            decay_r,
            decay_i,
            out_r,
            out_i,
            height,
            width,
            modes,
            real.stride(),
            imag.stride(),
            reverse,
            block_height,
            block_columns,
        )
    return out_r, out_i


def _save_vertical_scan(ctx, inputs, output) -> None:
    _, _, decay_r, decay_i, ctx.reverse = inputs
    ctx.save_for_backward(decay_r, decay_i, *output)


def _differentiate_vertical_scan(ctx, grad_r, grad_i) -> tuple:
    decay_r, decay_i, out_r, out_i = ctx.saved_tensors
    # The adjoint recurrence traverses the line in the opposite direction
    # with conjugated decay. Each decay gradient uses the previous state.
    dx_r, dx_i = _vertical_scan(grad_r, grad_i, decay_r, -decay_i, not ctx.reverse)
    incoming, previous = (
        (slice(None, -1), slice(1, None)) if ctx.reverse else (slice(1, None), slice(None, -1))
    )
    ir, ii = dx_r[:, incoming], dx_i[:, incoming]
    pr, pi = out_r[:, previous], out_i[:, previous]
    return (
        dx_r,
        dx_i,
        (ir * pr + ii * pi).sum((0, 1, 2)),
        (ii * pr - ir * pi).sum((0, 1, 2)),
        None,
    )


_vertical_scan.register_autograd(_differentiate_vertical_scan, setup_context=_save_vertical_scan)


def _scan(
    pole: PoleTensors, source: SourcePair, *, transposed: bool = False,
) -> StateQuad:
    if source[0].is_cuda and general_scans_enabled(forward=True):
        if torch.compiler.is_compiling():
            # Let the custom op plan its launch with real CUDA tensors.
            states = horizontal_forward(list(pole), *source)
        else:
            states = try_general_horizontal_forward(pole, source)
        if states is not None:
            return cast("StateQuad", states.unbind(0))
    if transposed and source[0].is_cuda:
        source = source[0].contiguous(), source[1].contiguous()
    return (
        triton_bidirectional_product_scan(pole, source)
        if source[0].is_cuda
        else bidirectional_product_scan_reference(pole, source)
    )


def _vertical_correlations(
    pole: PoleTensors,
    source: SourcePair,
    *,
    normalize: bool = False,
    forward_vertical_only: bool = False,
) -> list[tuple[Tensor, Tensor]]:
    real, imag = (v.float() for v in source)
    gr, gi = (v.float().reshape(-1) for v in pole[2:])
    if forward_vertical_only:
        ar, ai = (v.float().reshape(-1) for v in pole[:2])
        ur, ui = gr * real - gi * imag, gr * imag + gi * real
        if real.is_cuda:
            sr, si = _vertical_scan(ur, ui, ar.contiguous(), ai.contiguous(), False)  # noqa: FBT003
        else:
            source_complex = torch.complex(
                real.transpose(1, 2).contiguous(), imag.transpose(1, 2).contiguous()
            )
            states = torch.einsum(
                "tjm,bhjm->bhtm",
                _causal_weights(torch.complex(ar, ai), real.shape[1]),
                torch.complex(gr, gi) * source_complex,
            )
            sr, si = states.real.clone().transpose(1, 2), states.imag.clone().transpose(1, 2)
        return [_correlate(sr, si, ur, ui, normalize=normalize)]

    states = _scan(pole, (real.transpose(1, 2).contiguous(), imag.transpose(1, 2).contiguous()))
    pairs: list[tuple[Tensor, Tensor]] = []
    for reverse in (False, True):
        gain_imag = -gi if reverse else gi
        ur, ui = gr * real - gain_imag * imag, gr * imag + gain_imag * real
        sr, si = (v.transpose(1, 2) for v in states[2 * int(reverse) : 2 * int(reverse) + 2])
        pairs.append(_correlate(sr, si, ur, ui, normalize=normalize))
    return pairs


def _correlate(
    sr: Tensor,
    si: Tensor,
    ur: Tensor,
    ui: Tensor,
    *,
    normalize: bool,
) -> tuple[Tensor, Tensor]:
    power_r, power_i = ur.square(), ui.square()
    re = (2 * (sr * ur + si * ui) - power_r - power_i).mean((1, 2))
    im = (2 * (si * ur - sr * ui)).mean((1, 2))
    if normalize:
        denominator = (power_r + power_i).mean((1, 2)).clamp_min(1e-6).detach()
        re, im = re / denominator, im / denominator
    return re, im


@triton.jit
def _nr4_correlation_kernel(
    forward_r, forward_i, reverse_r, reverse_i,
    decay_r, decay_i, gain_r, gain_i,
    partials, upstream, grad_source, grad_pole,
    height: tl.constexpr, width: tl.constexpr, modes: tl.constexpr,
    strides: tl.constexpr, block_rows: tl.constexpr, block_cols: tl.constexpr,
    normalize: tl.constexpr, backward: tl.constexpr,
) -> None:
    batch = tl.program_id(1).to(tl.int64)
    direction = tl.program_id(2)
    row = tl.arange(0, block_rows)[:, None]
    col = tl.program_id(0) * block_cols + tl.arange(0, block_cols)
    mode = col % modes
    columns: tl.constexpr = width * modes
    valid = (row < height) & (col < columns)
    offset = (
        batch * strides[0] + row * strides[1] + (col // modes) * strides[2]
        + mode * strides[3]
    )
    x_r = tl.load(tl.where(direction == 0, forward_r, reverse_r) + offset, valid, 0).to(tl.float32)
    x_i = tl.load(tl.where(direction == 0, forward_i, reverse_i) + offset, valid, 0).to(tl.float32)
    ar = tl.where(row < height, tl.load(decay_r + mode).to(tl.float32), 1.0)
    ai = tl.where(row < height, tl.load(decay_i + mode).to(tl.float32), 0.0)
    gr = tl.load(gain_r + mode).to(tl.float32)
    gi = tl.load(gain_i + mode).to(tl.float32)
    ur, ui = gr * x_r - gi * x_i, gr * x_i + gi * x_r
    _, _, sr, si = tl.associative_scan((ar, ai, ur, ui), 0, _compose_complex)
    groups = tl.num_programs(1)
    plane = (direction * groups + batch) * columns + col
    step = 2 * groups * columns

    if backward:
        re = tl.load(upstream + (direction * groups + batch) * modes + mode)
        im = tl.load(upstream + ((direction + 2) * groups + batch) * modes + mode)
        state_r = 2 * (re * ur - im * ui)
        state_i = 2 * (re * ui + im * ur)
        _, _, dr, di = tl.associative_scan(
            (ar, -ai, state_r, state_i), 0, _compose_complex, reverse=True,
        )
        wr = dr + 2 * (re * (sr - ur) + im * si)
        wi = di + 2 * (re * (si - ui) - im * sr)
        grad_offset = ((2 * direction * groups + batch) * height + row) * columns + col
        tl.store(grad_source + grad_offset, gr * wr + gi * wi, valid)
        tl.store(grad_source + grad_offset + groups * height * columns, gr * wi - gi * wr, valid)

        prev = tl.broadcast_to(tl.maximum(row - 1, 0), (block_rows, block_cols))
        pr, pi = tl.gather(sr, prev, 0), tl.gather(si, prev, 0)
        recurrence = valid & (row > 0)
        decay_grad_r = tl.sum(tl.where(recurrence, dr * pr + di * pi, 0.0), 0)
        decay_grad_i = tl.sum(tl.where(recurrence, di * pr - dr * pi, 0.0), 0)
        gain_grad_r = tl.sum(tl.where(valid, wr * x_r + wi * x_i, 0.0), 0)
        gain_grad_i = tl.sum(tl.where(valid, wi * x_r - wr * x_i, 0.0), 0)
        tl.store(grad_pole + plane, decay_grad_r, col < columns)
        tl.store(grad_pole + plane + step, decay_grad_i, col < columns)
        tl.store(grad_pole + plane + 2 * step, gain_grad_r, col < columns)
        tl.store(grad_pole + plane + 3 * step, gain_grad_i, col < columns)
    else:
        power_r, power_i = ur * ur, ui * ui
        re = 2 * (sr * ur + si * ui) - power_r - power_i
        im = 2 * (si * ur - sr * ui)
        tl.store(partials + plane, tl.sum(tl.where(valid, re, 0.0), 0), col < columns)
        tl.store(partials + plane + step, tl.sum(tl.where(valid, im, 0.0), 0), col < columns)
        if normalize:
            tl.store(
                partials + plane + 2 * step, tl.sum(tl.where(valid, power_r + power_i, 0.0), 0),
                col < columns,
            )


def _nr4_launch(
    states: StateQuad, pole: PoleTensors, partials: Tensor, upstream: Tensor,
    grad_source: Tensor, grad_pole: Tensor, *, normalize: bool, backward: bool, warp_tile: bool,
) -> None:
    batch, height, width, modes = states[0].shape
    block_rows = triton.next_power_of_2(height)
    block_cols = max(4, min(32, 512 // block_rows))
    # For a 32-column tile, one warp can scan complete columns without
    # exchanging intermediate states across warps.
    warps = 1 if warp_tile and block_rows <= 16 and modes % 32 == 0 else 4
    grid = (triton.cdiv(width * modes, block_cols), batch, 2)
    wrap_triton(_nr4_correlation_kernel)[grid](
        *states, *(p.contiguous() for p in pole),
        partials, upstream, grad_source, grad_pole,
        height, width, modes, states[0].stride(), block_rows, block_cols,
        normalize, backward, num_warps=warps, enable_fp_fusion=False,
    )


@triton_op("lnet::nr4_correlation", mutates_args=())
def _nr4_correlation(
    states: list[Tensor], pole: list[Tensor], normalize: bool, warp_tile: bool,  # noqa: FBT001
) -> Tensor:
    batch, height, width, modes = states[0].shape
    partials = states[0].new_empty(
        (3 if normalize else 2, 2, batch, width, modes), dtype=torch.float32,
    )
    _nr4_launch(
        cast("StateQuad", tuple(states)), cast("PoleTensors", tuple(pole)),
        partials, partials, partials, partials,
        normalize=normalize, backward=False, warp_tile=warp_tile,
    )
    return partials.sum(dim=-2) / (height * width)


@triton_op("lnet::nr4_correlation_backward", mutates_args=())
def _nr4_correlation_backward(
    states: list[Tensor], pole: list[Tensor], upstream: Tensor, warp_tile: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor]:
    batch, height, width, modes = states[0].shape
    grad_source = states[0].new_empty((4, batch, height, width, modes))
    grad_pole = upstream.new_empty((4, 2, batch, width, modes))
    upstream = (upstream / (height * width)).contiguous()
    _nr4_launch(
        cast("StateQuad", tuple(states)), cast("PoleTensors", tuple(pole)),
        grad_pole, upstream, grad_source, grad_pole,
        normalize=False, backward=True, warp_tile=warp_tile,
    )
    return grad_source, grad_pole.sum((1, 2, 3))


def _save_nr4_correlation(ctx, inputs, output) -> None:  # noqa: ARG001
    states, pole, _, ctx.warp_tile = inputs
    ctx.save_for_backward(*states, *pole)


def _differentiate_nr4_correlation(ctx, grad) -> tuple:
    states, pole = ctx.saved_tensors[:4], ctx.saved_tensors[4:]
    grad_source, adjoints = _nr4_correlation_backward(
        list(states), list(pole), grad, ctx.warp_tile,
    )
    grad_pole = [
        value.reshape_as(p).to(p.dtype) for value, p in zip(adjoints.unbind(0), pole, strict=True)
    ]
    return list(grad_source.unbind(0)), grad_pole, None, None


_nr4_correlation.register_autograd(
    _differentiate_nr4_correlation, setup_context=_save_nr4_correlation,
)


def _nr4_axis(
    pole: PoleTensors, states: StateQuad, *, normalize: bool, fused: bool, warp_tile: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if fused:
        stats = _nr4_correlation(list(states), list(pole), normalize, warp_tile)
        correlation = stats[:2]
        if normalize:
            correlation = correlation / stats[2].clamp_min(1e-6).detach()
        (fwd_re, rev_re), (fwd_im, rev_im) = correlation.unbind(0)
    else:
        (fwd_re, fwd_im), (rev_re, rev_im) = (
            _vertical_correlations(
                pole, (states[index], states[index + 1]), normalize=normalize,
                forward_vertical_only=True,
            )[0]
            for index in (0, 2)
        )
    return fwd_re, rev_re, fwd_im, rev_im


def complex_correlation_descriptor(
    pole_x: PoleTensors,
    pole_y: PoleTensors,
    drive: Tensor,
    *,
    gain_normalization: str | None = None,
    normalize: bool = False,
    forward_vertical_only: bool = False,
    non_redundant_four_directions: bool = False,
) -> Tensor:
    """Return the selected directions as real parts followed by imaginary parts."""
    del gain_normalization
    real_drive, imag_drive = drive.chunk(2, dim=-1)
    horizontal = _scan(pole_x, (real_drive, imag_drive))
    if non_redundant_four_directions:
        fused = (
            drive.is_cuda and drive.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and min(drive.shape) > 0 and max(drive.shape[1:3]) <= 128
            and drive.shape[0] <= 65_535
            and os.environ.get("LNET_NR4_FUSED") != "0"
        )
        warp_tile = (
            fused and drive.dtype in (torch.float16, torch.bfloat16)
            and model_w28c_enabled(drive.device, "nr4")
        )
        h_fwd_re, h_rev_re, h_fwd_im, h_rev_im = _nr4_axis(
            pole_y, horizontal, normalize=normalize, fused=fused, warp_tile=warp_tile,
        )
        transposed_drive = drive.transpose(1, 2)
        if not fused:
            transposed_drive = transposed_drive.contiguous()
        vertical = _scan(pole_y, transposed_drive.chunk(2, dim=-1), transposed=fused)
        v_fwd_re, v_rev_re, v_fwd_im, v_rev_im = _nr4_axis(
            pole_x, vertical, normalize=normalize, fused=fused, warp_tile=warp_tile,
        )
        return torch.cat(
            (h_fwd_re, h_rev_re, v_fwd_re, v_rev_re, h_fwd_im, h_rev_im, v_fwd_im, v_rev_im),
            dim=-1,
        )
    pairs = [
        _vertical_correlations(
            pole_y,
            (horizontal[h], horizontal[h + 1]),
            normalize=normalize,
            forward_vertical_only=forward_vertical_only,
        )
        for h in (0, 2)
    ]
    return torch.cat(
        [
            pairs[h][rev][component]
            for component in (0, 1)
            for rev in range(len(pairs[0]))
            for h in (0, 1)
        ],
        dim=-1,
    )
