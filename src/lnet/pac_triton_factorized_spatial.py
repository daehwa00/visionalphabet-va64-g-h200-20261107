"""Complex spatial filtering and rank reduction without grouped-conv intermediates.

These kernels consume component, pole, rank projection channel ordering
directly and write either separate fields or one packed output buffer.
Forward, input gradients and spatial-weight gradients all execute on the GPU.
"""

from __future__ import annotations

from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@triton.jit
def _forward_kernel(
    X,
    WR,
    WI,
    YR,
    YI,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED_OUTPUT: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pixel, pole = index // P, index % P
    batch = pixel // (H * W)
    row, col = (pixel // W) % H, pixel % W
    real = tl.full((BLOCK,), 0, tl.float32)
    imag = tl.full((BLOCK,), 0, tl.float32)
    for rank in tl.static_range(R):
        for ky in tl.static_range(K):
            for kx in tl.static_range(K):
                ih, iw = row + ky - K // 2, col + kx - K // 2
                valid = (pixel < N * H * W) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                offset = batch * S0 + (pole * R + rank) * S1 + ih * S2 + iw * S3
                xr = tl.load(X + offset, valid, 0).to(tl.float32)
                xi = tl.load(X + offset + P * R * S1, valid, 0).to(tl.float32)
                weight = ((pole * R + rank) * K + ky) * K + kx
                wr = tl.load(WR + weight, pixel < N * H * W, 0).to(tl.float32)
                wi = tl.load(WI + weight, pixel < N * H * W, 0).to(tl.float32)
                real += xr * wr - xi * wi
                imag += xr * wi + xi * wr
    if PACKED_OUTPUT:
        output = pixel * (2 * P) + pole
        tl.store(YR + output, real, pixel < N * H * W)
        tl.store(YR + output + P, imag, pixel < N * H * W)
    else:
        output = pixel * P + pole
        tl.store(YR + output, real, pixel < N * H * W)
        tl.store(YI + output, imag, pixel < N * H * W)


@triton.jit
def _input_backward_kernel(
    G,
    GI,
    WR,
    WI,
    DX,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED_OUTPUT: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pixel, channel = index // (P * R), index % (P * R)
    pole = channel // R
    batch = pixel // (H * W)
    row, col = (pixel // W) % H, pixel % W
    real = tl.full((BLOCK,), 0, tl.float32)
    imag = tl.full((BLOCK,), 0, tl.float32)
    for ky in tl.static_range(K):
        for kx in tl.static_range(K):
            oh, ow = row - ky + K // 2, col - kx + K // 2
            valid = (pixel < N * H * W) & (oh >= 0) & (oh < H) & (ow >= 0) & (ow < W)
            offset = batch * G0 + pole * G1 + oh * G2 + ow * G3
            gr = tl.load(G + offset, valid, 0).to(tl.float32)
            if PACKED_OUTPUT:
                gi = tl.load(G + offset + P * G1, valid, 0).to(tl.float32)
            else:
                gi = tl.load(GI + offset, valid, 0).to(tl.float32)
            weight = (channel * K + ky) * K + kx
            wr = tl.load(WR + weight, pixel < N * H * W, 0).to(tl.float32)
            wi = tl.load(WI + weight, pixel < N * H * W, 0).to(tl.float32)
            real += gr * wr + gi * wi
            imag += gi * wr - gr * wi
    output = batch * S0 + channel * S1 + row * S2 + col * S3
    tl.store(DX + output, real, pixel < N * H * W)
    tl.store(DX + output + P * R * S1, imag, pixel < N * H * W)


@triton.jit
def _weight_backward_kernel(
    X,
    G,
    GI,
    PARTIAL,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    G2: tl.constexpr,
    G3: tl.constexpr,
    SPLITS: tl.constexpr,
    ROWS: tl.constexpr,
    POLES: tl.constexpr,
    TAPS: tl.constexpr,
    PACKED_OUTPUT: tl.constexpr,
):
    pole = tl.program_id(0) * POLES + tl.arange(0, POLES)
    split = tl.program_id(1)
    tap = tl.arange(0, TAPS)
    real = tl.full((TAPS, POLES), 0, tl.float32)
    imag = tl.full((TAPS, POLES), 0, tl.float32)
    for start in range(split * ROWS, N * H * W, SPLITS * ROWS):
        pixel = start + tl.arange(0, ROWS)
        batch = pixel // (H * W)
        row, col = (pixel // W) % H, pixel % W
        goffset = (batch * G0 + row * G2 + col * G3)[:, None] + pole[None, :] * G1
        gmask = (pixel[:, None] < N * H * W) & (pole[None, :] < P)
        gr = tl.load(G + goffset, gmask, 0).to(tl.float32)
        if PACKED_OUTPUT:
            gi = tl.load(G + goffset + P * G1, gmask, 0).to(tl.float32)
        else:
            gi = tl.load(GI + goffset, gmask, 0).to(tl.float32)
        for rank in tl.static_range(R):
            for ky in tl.static_range(K):
                for kx in tl.static_range(K):
                    ih, iw = row + ky - K // 2, col + kx - K // 2
                    valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                    mask = valid[:, None] & gmask
                    offset = (batch * S0 + ih * S2 + iw * S3)[:, None] + (pole * R + rank)[
                        None, :
                    ] * S1
                    xr = tl.load(X + offset, mask, 0).to(tl.float32)
                    xi = tl.load(X + offset + P * R * S1, mask, 0).to(tl.float32)
                    vr = tl.sum(xr * gr + xi * gi, axis=0)
                    vi = tl.sum(xr * gi - xi * gr, axis=0)
                    active = tap[:, None] == (rank * K + ky) * K + kx
                    real += tl.where(active, vr[None, :], 0.0)
                    imag += tl.where(active, vi[None, :], 0.0)
    weights = P * R * K * K
    offset = pole[None, :] * (R * K * K) + tap[:, None]
    mask = (pole[None, :] < P) & (tap[:, None] < R * K * K)
    tl.store(PARTIAL + split * 2 * weights + offset, real, mask)
    tl.store(PARTIAL + (split * 2 + 1) * weights + offset, imag, mask)


def _validate(projected: Tensor, real: Tensor, imag: Tensor) -> None:
    if not projected.is_cuda or real.device != projected.device or imag.device != projected.device:
        raise ValueError("factorized spatial tensors must share a CUDA device")
    if projected.ndim != 4 or real.ndim != 4 or real.shape != imag.shape:
        raise ValueError("factorized spatial tensor dimensions are incompatible")
    poles, rank, height, width = real.shape
    if min(projected.shape) <= 0 or min(real.shape) <= 0:
        raise ValueError("factorized spatial dimensions must be positive")
    if height != width or height % 2 != 1 or projected.shape[1] != 2 * poles * rank:
        raise ValueError("factorized spatial channels or kernel support are incompatible")
    if projected.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("factorized spatial projection must use FP32 or BF16")
    if real.dtype != projected.dtype or imag.dtype != projected.dtype:
        raise TypeError("factorized spatial tensors must share a dtype")
    if not (
        projected.is_contiguous() or projected.is_contiguous(memory_format=torch.channels_last)
    ):
        raise ValueError("factorized spatial projection must have a dense NCHW or NHWC layout")
    if not real.is_contiguous() or not imag.is_contiguous():
        raise ValueError("factorized spatial factors must be contiguous")


@triton_op("lnet::factorized_spatial_v2", mutates_args={})
def factorized_spatial(projected: Tensor, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
    """Return separate contiguous NHWM fields, without a packed-output copy."""
    _validate(projected, real, imag)
    batch, _, height, width = projected.shape
    poles, rank, kernel, _ = real.shape
    output_real = torch.empty(
        (batch, height, width, poles),
        device=projected.device,
        dtype=projected.dtype,
    )
    output_imag = torch.empty_like(output_real)
    wrap_triton(_forward_kernel)[(triton.cdiv(batch * height * width * poles, 256),)](
        projected,
        real,
        imag,
        output_real,
        output_imag,
        batch,
        height,
        width,
        poles,
        rank,
        kernel,
        *projected.stride(),
        BLOCK=256,
        PACKED_OUTPUT=False,
    )
    return output_real, output_imag


def _spatial_backward(
    projected: Tensor,
    real: Tensor,
    imag: Tensor,
    grad_real: Tensor,
    grad_imag: Tensor,
    *,
    packed_output: bool = False,
) -> tuple[Tensor, Tensor]:
    grad_real = grad_real.contiguous()
    grad_imag = grad_real if packed_output else grad_imag.contiguous()
    grad_strides = (
        grad_real.stride(0),
        grad_real.stride(3),
        grad_real.stride(1),
        grad_real.stride(2),
    )
    batch, _, height, width = projected.shape
    poles, rank, kernel, _ = real.shape
    grad_input = torch.empty_like(projected)
    wrap_triton(_input_backward_kernel)[(triton.cdiv(batch * height * width * poles * rank, 256),)](
        grad_real,
        grad_imag,
        real,
        imag,
        grad_input,
        batch,
        height,
        width,
        poles,
        rank,
        kernel,
        *grad_input.stride(),
        *grad_strides,
        BLOCK=256,
        PACKED_OUTPUT=packed_output,
    )
    splits = min(128, triton.cdiv(batch * height * width, 128))
    partials = torch.empty(
        (splits, 2, poles, rank, kernel, kernel),
        device=projected.device,
        dtype=torch.float32,
    )
    wrap_triton(_weight_backward_kernel)[(triton.cdiv(poles, 32), splits)](
        projected,
        grad_real,
        grad_imag,
        partials,
        batch,
        height,
        width,
        poles,
        rank,
        kernel,
        *projected.stride(),
        *grad_strides,
        SPLITS=splits,
        ROWS=4,
        POLES=32,
        TAPS=triton.next_power_of_2(rank * kernel * kernel),
        PACKED_OUTPUT=packed_output,
    )
    grad_weight = partials.sum(dim=0).to(real.dtype)
    return grad_input, grad_weight


@triton_op("lnet::factorized_spatial_v2_backward", mutates_args={})
def _backward_op(
    projected: Tensor,
    real: Tensor,
    imag: Tensor,
    grad_real: Tensor,
    grad_imag: Tensor,
) -> tuple[Tensor, Tensor]:
    return _spatial_backward(projected, real, imag, grad_real, grad_imag)


@triton_op("lnet::factorized_spatial_packed_v1", mutates_args={})
def factorized_spatial_packed(projected: Tensor, real: Tensor, imag: Tensor) -> Tensor:
    """Write directly into one contiguous NHW[real | imag] tensor."""
    _validate(projected, real, imag)
    batch, _, height, width = projected.shape
    poles, rank, kernel, _ = real.shape
    output = torch.empty(
        (batch, height, width, 2 * poles), device=projected.device, dtype=projected.dtype,
    )
    wrap_triton(_forward_kernel)[(triton.cdiv(batch * height * width * poles, 256),)](
        projected,
        real,
        imag,
        output,
        output,
        batch,
        height,
        width,
        poles,
        rank,
        kernel,
        *projected.stride(),
        BLOCK=256,
        PACKED_OUTPUT=True,
    )
    return output


@triton_op("lnet::factorized_spatial_packed_v1_backward", mutates_args={})
def _packed_backward_op(
    projected: Tensor,
    real: Tensor,
    imag: Tensor,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor]:
    return _spatial_backward(
        projected, real, imag, grad_output, grad_output, packed_output=True,
    )


def _setup_context(
    ctx: _AutogradContext, inputs: tuple[Tensor, ...], output: object,
) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _backward(
    ctx: _AutogradContext, grad_real: Tensor | None, grad_imag: Tensor | None
) -> tuple[Tensor, Tensor, Tensor]:
    projected, real, imag = ctx.saved_tensors
    shape = (projected.shape[0], projected.shape[2], projected.shape[3], real.shape[0])
    if grad_real is None:
        grad_real = torch.zeros(shape, dtype=projected.dtype, device=projected.device)
    if grad_imag is None:
        grad_imag = torch.zeros(shape, dtype=projected.dtype, device=projected.device)
    grad_input, grad_weight = _backward_op(projected, real, imag, grad_real, grad_imag)
    return grad_input, grad_weight[0], grad_weight[1]


torch.library.register_autograd(
    "lnet::factorized_spatial_v2",
    _backward,
    setup_context=_setup_context,
)


def _packed_backward(ctx: _AutogradContext, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    projected, real, imag = ctx.saved_tensors
    grad_input, grad_weight = _packed_backward_op(projected, real, imag, grad_output)
    return grad_input, grad_weight[0], grad_weight[1]


torch.library.register_autograd(
    "lnet::factorized_spatial_packed_v1",
    _packed_backward,
    setup_context=_setup_context,
)
