"""Shape-autotuned real Reader forward and gradients.

No weight-derived value is cached across optimizer steps.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .pac_triton_real_spatial import _forward, _input_backward, _validate, _weight_backward

_tuned_forward = triton.autotune(
    configs=[triton.Config({"BLOCK": b}, num_warps=w) for b in (64, 128, 256, 512) for w in (4, 8)],
    key=["N", "H", "W", "P", "R", "K"],
    cache_results=True,
)(_forward)
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
        triton.Config({"ROWS": rows, "POLES": p, "TAPS": t}, num_warps=4)
        for rows, p, t in [
            (8, 8, 4),
            (16, 8, 4),
            (16, 16, 4),
            (32, 8, 4),
            (32, 16, 4),
            (16, 8, 8),
            (32, 8, 8),
            (32, 16, 8),
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
    accum0 = tl.full((TAPS, POLES), 0, tl.float32)
    accum1 = tl.full((TAPS, POLES), 0, tl.float32)
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
        accum0 += tl.sum(source * g0[:, None, :], axis=0)
        accum1 += tl.sum(source * g1[:, None, :], axis=0)
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
    wrap_triton(_tuned_forward)[lambda cfg: (triton.cdiv(n * h * w * p, cfg["BLOCK"]),)](
        x, weight, output, n, h, w, p, r, k
    )
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
