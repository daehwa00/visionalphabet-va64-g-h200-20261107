"""Shape-autotuned real Reader forward and gradients.

No weight-derived value is cached across optimizer steps.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .pac_triton_real_spatial import _input_backward, _validate, _weight_backward


@triton.autotune(
    configs=[
        triton.Config({"PIXELS": px, "POLES": pl}, num_warps=w)
        for px, pl, w in (
            (4, 16, 4),
            (8, 16, 4),
            (4, 32, 4),
            (8, 32, 4),
            (8, 32, 8),
            (16, 32, 4),
            (16, 32, 8),
            (8, 64, 8),
        )
    ],
    key=["N", "H", "W", "P", "R", "K"],
    cache_results=True,
)
@triton.jit
def _tuned_forward(
    X,
    WGT,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    PIXELS: tl.constexpr,
    POLES: tl.constexpr,
):
    pixel = tl.program_id(0) * PIXELS + tl.arange(0, PIXELS)
    pole = tl.program_id(1) * POLES + tl.arange(0, POLES)
    batch = pixel // (H * W)
    row, col = pixel // W % H, pixel % W
    valid_pixel = pixel < N * H * W
    valid_pole = pole < P
    out0 = tl.full((PIXELS, POLES), 0, tl.float32)
    out1 = tl.full((PIXELS, POLES), 0, tl.float32)
    for rank in tl.static_range(R):
        for ky in tl.static_range(K):
            for kx in tl.static_range(K):
                ih, iw = row + ky - K // 2, col + kx - K // 2
                active_px = valid_pixel & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_base = ((batch * H + ih) * W + iw)[:, None] * (P * R) + pole[None, :] * R + rank
                x = tl.load(X + x_base, active_px[:, None] & valid_pole[None, :], 0).to(tl.float32)
                w_off = ((pole * 2 * R + rank) * K + ky) * K + kx
                w0 = tl.load(WGT + w_off, valid_pole, 0).to(tl.float32)
                w1 = tl.load(WGT + w_off + R * K * K, valid_pole, 0).to(tl.float32)
                out0 += x * w0[None, :]
                out1 += x * w1[None, :]
    y_off = pixel[:, None] * (2 * P) + pole[None, :]
    mask = valid_pixel[:, None] & valid_pole[None, :]
    tl.store(Y + y_off, out0, mask)
    tl.store(Y + y_off + P, out1, mask)


_tuned_input = triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in (64, 128, 256, 512) for w in (4, 8)],
    key=["N", "H", "W", "P", "R", "K"],
    cache_results=True,
)(_input_backward)
_tuned_weight = triton.autotune(
    configs=[
        triton.Config({"ROWS": rows, "POLES": p}, num_warps=4)
        for rows in (4, 8, 16)
        for p in (8, 16)
    ],
    key=["N", "H", "W", "P", "R", "K", "SPLITS"],
    cache_results=True,
)(_weight_backward)


@triton.autotune(
    configs=[
        triton.Config({"ROWS": rows, "POLES": p, "TAPS": t}, num_warps=w)
        for rows, p, t, w in [
            (8, 8, 4, 4),
            (16, 8, 4, 4),
            (16, 8, 4, 8),
            (16, 16, 4, 4),
            (16, 16, 4, 8),
            (32, 8, 4, 4),
            (16, 8, 8, 4),
            (16, 8, 8, 8),
            (16, 16, 8, 8),
            (32, 16, 8, 8),
        ]
    ],
    key=["N", "H", "W", "P", "R", "K", "SPLITS"],
    cache_results=True,
)
@triton.jit
def _weight_backward_vector(
    X,
    G,
    PARTIAL,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    ROWS: tl.constexpr,
    POLES: tl.constexpr,
    TAPS: tl.constexpr,
):
    poles = tl.program_id(0) * POLES + tl.arange(0, POLES)
    taps = tl.program_id(1) * TAPS + tl.arange(0, TAPS)
    split = tl.program_id(2)
    # Tile neighboring rank coordinates of the same spatial tap together.
    # NHW(PR) stores rank contiguously; the original rank-major tap ordering
    # groups strided source coordinates when a tile spans only spatial taps.
    spatial_tap = taps // R
    rank = taps % R
    ky = spatial_tap // K
    kx = spatial_tap % K
    accum0_3d = tl.full((ROWS, TAPS, POLES), 0, tl.float32)
    accum1_3d = tl.full((ROWS, TAPS, POLES), 0, tl.float32)
    for start in range(split * ROWS, N * H * W, SPLITS * ROWS):
        pixels = start + tl.arange(0, ROWS)
        batch = pixels // (H * W)
        row = pixels // W % H
        col = pixels % W
        ih = row[:, None] + ky[None, :] - K // 2
        iw = col[:, None] + kx[None, :] - K // 2
        source_offset = (
            ((batch[:, None] * H + ih) * W + iw)[:, :, None] * (P * R)
            + poles[None, None, :] * R
            + rank[None, :, None]
        )
        mask = (
            (pixels[:, None, None] < N * H * W)
            & (taps[None, :, None] < R * K * K)
            & (poles[None, None, :] < P)
            & (ih[:, :, None] >= 0)
            & (ih[:, :, None] < H)
            & (iw[:, :, None] >= 0)
            & (iw[:, :, None] < W)
        )
        source = tl.load(X + source_offset, mask, 0).to(tl.float32)
        grad_offset = pixels[:, None] * (2 * P) + poles[None, :]
        grad_mask = (pixels[:, None] < N * H * W) & (poles[None, :] < P)
        g0 = tl.load(G + grad_offset, grad_mask, 0).to(tl.float32)
        g1 = tl.load(G + grad_offset + P, grad_mask, 0).to(tl.float32)
        accum0_3d += source * g0[:, None, :]
        accum1_3d += source * g1[:, None, :]
    accum0 = tl.sum(accum0_3d, axis=0)
    accum1 = tl.sum(accum1_3d, axis=0)
    # Keep the public partial/weight layout P(2RKK) unchanged.
    weight_tap = rank * (K * K) + spatial_tap
    offset = split * (P * 2 * R * K * K) + poles[None, :] * (2 * R * K * K) + weight_tap[:, None]
    active = (poles[None, :] < P) & (taps[:, None] < R * K * K)
    tl.store(PARTIAL + offset, accum0, active)
    tl.store(PARTIAL + offset + R * K * K, accum1, active)


@triton_op("lnet::real_training_spatial", mutates_args={})
def real_training_spatial(x: Tensor, weight: Tensor, vector: bool) -> Tensor:
    del vector  # Used by the autograd setup, not the forward arithmetic.
    _validate(x, weight)
    n, h, w, _ = x.shape
    p, _, r, k, _ = weight.shape
    output = x.new_empty((n, h, w, 2 * p))
    wrap_triton(_tuned_forward)[
        lambda cfg: (triton.cdiv(n * h * w, cfg["PIXELS"]), triton.cdiv(p, cfg["POLES"]))
    ](x, weight, output, n, h, w, p, r, k)
    return output


@triton_op("lnet::real_training_spatial_backward", mutates_args={})
def _spatial_backward(
    x: Tensor, weight: Tensor, grad: Tensor, vector: bool
) -> tuple[Tensor, Tensor]:
    grad = grad.contiguous()
    n, h, w, _ = x.shape
    p, _, r, k, _ = weight.shape
    dx = torch.empty_like(x)
    wrap_triton(_tuned_input)[lambda cfg: (triton.cdiv(n * h * w * p * r, cfg["BLOCK"]),)](
        grad, weight, dx, n, h, w, p, r, k
    )
    splits = min(128, (n * h * w + 127) // 128)
    partials = torch.empty((splits, *weight.shape), device=x.device, dtype=torch.float32)
    if vector:
        wrap_triton(_weight_backward_vector)[
            lambda cfg: (triton.cdiv(p, cfg["POLES"]), triton.cdiv(r * k * k, cfg["TAPS"]), splits)
        ](x, grad, partials, n, h, w, p, r, k, splits)
    else:
        wrap_triton(_tuned_weight)[lambda cfg: (triton.cdiv(p, cfg["POLES"]), splits)](
            x,
            grad,
            partials,
            n,
            h,
            w,
            p,
            r,
            k,
            SPLITS=splits,
            TAPS=triton.next_power_of_2(r * k * k),
        )
    return dx, partials.sum(0).to(weight.dtype)


def _spatial_setup(ctx, inputs, output):
    del output
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.vector = inputs[2]


def _spatial_grad(ctx, grad):
    x, weight = ctx.saved_tensors
    dx, dw = _spatial_backward(x, weight, grad, ctx.vector)
    return dx, dw, None


torch.library.register_autograd(
    "lnet::real_training_spatial", _spatial_grad, setup_context=_spatial_setup
)
