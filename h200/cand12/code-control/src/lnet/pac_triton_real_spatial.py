"""Unconstrained real per-pole spatial filtering into two memory coordinates."""

from __future__ import annotations

# Triton pointer arguments and static ranges are DSL values, not Python iterators.
# pyright: reportMissingParameterType=false, reportGeneralTypeIssues=false
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


class _Context(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _forward(
    X,
    WGT,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pixel, pole = index // P, index % P
    batch = pixel // (H * W)
    row, col = pixel // W % H, pixel % W
    out0 = tl.full((BLOCK,), 0, tl.float32)
    out1 = tl.full((BLOCK,), 0, tl.float32)
    for rank in tl.static_range(R):
        for ky in tl.static_range(K):
            for kx in tl.static_range(K):
                ih, iw = row + ky - K // 2, col + kx - K // 2
                active = (pixel < N * H * W) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x = tl.load(
                    X + ((batch * H + ih) * W + iw) * (P * R) + pole * R + rank, active, 0
                ).to(tl.float32)
                offset = ((pole * 2 * R + rank) * K + ky) * K + kx
                w0 = tl.load(WGT + offset, pixel < N * H * W, 0).to(tl.float32)
                w1 = tl.load(WGT + offset + R * K * K, pixel < N * H * W, 0).to(tl.float32)
                out0 += x * w0
                out1 += x * w1
    tl.store(Y + pixel * (2 * P) + pole, out0, pixel < N * H * W)
    tl.store(Y + pixel * (2 * P) + P + pole, out1, pixel < N * H * W)


@triton.jit
def _input_backward(
    G,
    WGT,
    DX,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pixel, channel = index // (P * R), index % (P * R)
    pole, rank = channel // R, channel % R
    batch = pixel // (H * W)
    row, col = pixel // W % H, pixel % W
    grad = tl.full((BLOCK,), 0, tl.float32)
    for ky in tl.static_range(K):
        for kx in tl.static_range(K):
            oh, ow = row - ky + K // 2, col - kx + K // 2
            active = (pixel < N * H * W) & (oh >= 0) & (oh < H) & (ow >= 0) & (ow < W)
            offset = ((batch * H + oh) * W + ow) * (2 * P) + pole
            g0 = tl.load(G + offset, active, 0).to(tl.float32)
            g1 = tl.load(G + offset + P, active, 0).to(tl.float32)
            weight = ((pole * 2 * R + rank) * K + ky) * K + kx
            w0 = tl.load(WGT + weight, pixel < N * H * W, 0).to(tl.float32)
            w1 = tl.load(WGT + weight + R * K * K, pixel < N * H * W, 0).to(tl.float32)
            grad += g0 * w0 + g1 * w1
    tl.store(DX + index, grad, pixel < N * H * W)


@triton.jit
def _weight_backward(
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
    pole = tl.program_id(0) * POLES + tl.arange(0, POLES)
    split = tl.program_id(1)
    tap = tl.arange(0, TAPS)
    grad0 = tl.full((TAPS, POLES), 0, tl.float32)
    grad1 = tl.full((TAPS, POLES), 0, tl.float32)
    for start in range(split * ROWS, N * H * W, SPLITS * ROWS):
        pixel = start + tl.arange(0, ROWS)
        batch = pixel // (H * W)
        row, col = pixel // W % H, pixel % W
        mask = (pixel[:, None] < N * H * W) & (pole[None, :] < P)
        offset = pixel[:, None] * (2 * P) + pole[None, :]
        g0 = tl.load(G + offset, mask, 0).to(tl.float32)
        g1 = tl.load(G + offset + P, mask, 0).to(tl.float32)
        for rank in tl.static_range(R):
            for ky in tl.static_range(K):
                for kx in tl.static_range(K):
                    ih, iw = row + ky - K // 2, col + kx - K // 2
                    active = mask & ((ih >= 0) & (ih < H) & (iw >= 0) & (iw < W))[:, None]
                    source = (
                        ((batch * H + ih) * W + iw)[:, None] * (P * R) + pole[None, :] * R + rank
                    )
                    x = tl.load(X + source, active, 0).to(tl.float32)
                    selected = tap[:, None] == (rank * K + ky) * K + kx
                    grad0 += tl.where(selected, tl.sum(x * g0, axis=0)[None, :], 0.0)
                    grad1 += tl.where(selected, tl.sum(x * g1, axis=0)[None, :], 0.0)
    offset = split * (P * 2 * R * K * K) + pole[None, :] * (2 * R * K * K) + tap[:, None]
    mask = (pole[None, :] < P) & (tap[:, None] < R * K * K)
    tl.store(PARTIAL + offset, grad0, mask)
    tl.store(PARTIAL + offset + R * K * K, grad1, mask)


def _validate(x: Tensor, weight: Tensor) -> None:
    if not x.is_cuda or x.device != weight.device or x.ndim != 4 or weight.ndim != 5:
        raise ValueError("real spatial filter requires CUDA NHWC and P2RKK tensors")
    p, outputs, rank, k, k2 = weight.shape
    if outputs != 2 or k != k2 or k % 2 != 1 or x.shape[-1] != p * rank:
        raise ValueError("real spatial filter dimensions do not match")
    if min(*x.shape, *weight.shape) <= 0 or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("real spatial filter requires positive contiguous tensors")
    if x.dtype not in (torch.float32, torch.bfloat16) or x.dtype != weight.dtype:
        raise TypeError("real spatial filter requires matching FP32/BF16 tensors")


@triton_op("lnet::real_factorized_spatial", mutates_args={})
def real_factorized_spatial(x: Tensor, weight: Tensor) -> Tensor:
    _validate(x, weight)
    n, h, w, _ = x.shape
    p, _, r, k, _ = weight.shape
    output = x.new_empty((n, h, w, 2 * p))
    wrap_triton(_forward)[(triton.cdiv(n * h * w * p, 256),)](
        x, weight, output, n, h, w, p, r, k, BLOCK=256
    )
    return output


@triton_op("lnet::real_factorized_spatial_backward", mutates_args={})
def _backward_op(x: Tensor, weight: Tensor, grad: Tensor) -> tuple[Tensor, Tensor]:
    grad = grad.contiguous()
    n, h, w, _ = x.shape
    p, _, r, k, _ = weight.shape
    dx = torch.empty_like(x)
    wrap_triton(_input_backward)[(triton.cdiv(n * h * w * p * r, 256),)](
        grad, weight, dx, n, h, w, p, r, k, BLOCK=256
    )
    splits = min(128, (n * h * w + 127) // 128)
    partials = torch.empty((splits, *weight.shape), device=x.device, dtype=torch.float32)
    wrap_triton(_weight_backward)[(triton.cdiv(p, 16), splits)](
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
        ROWS=4,
        POLES=16,
        TAPS=triton.next_power_of_2(r * k * k),
    )
    return dx, partials.sum(0).to(weight.dtype)


def _setup_context(ctx: _Context, inputs: tuple[Tensor, Tensor], output: Tensor) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _backward(ctx: _Context, grad: Tensor | None) -> tuple[Tensor, Tensor]:
    x, weight = ctx.saved_tensors
    if grad is None:
        grad = x.new_zeros((*x.shape[:-1], 2 * weight.shape[0]))
    return _backward_op(x, weight, grad)


torch.library.register_autograd(
    "lnet::real_factorized_spatial", _backward, setup_context=_setup_context
)
