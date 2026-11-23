"""Experimental direct NHWC stride-two convolution for small input channels."""

# Custom op and driver launch schemas use positional flags.
# ruff: noqa: FBT001, FBT003
from __future__ import annotations

import ctypes as ct
import os
from functools import cache

import torch
from torch import Tensor
from torch.nn import functional

from .cuda_nvrtc import _check_nvrtc, _CudaModule, _ensure_context, _include_dirs, _nvrtc

_MODE = "LNET_DIRECT_CONV_STEM"
_THREADS = 128
_TYPES = {torch.float16: "__half", torch.bfloat16: "__nv_bfloat16"}

_CUDA = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
using T = SCALAR;
constexpr int taps = 9 * CI;
constexpr int rows = taps + BIAS;
constexpr int column_groups = (CO + 31) / 32;

struct Patch {
    int base;
    unsigned valid;
};

__device__ Patch locate_patch(int p, int h, int w, int oh, int ow, int pixels) {
    if (p >= pixels) return {0, 0};
    int row = p / ow;
    int batch = row / oh;
    int y = 2 * (row - batch * oh) - 1;
    int x = 2 * (p - row * ow) - 1;
    unsigned horizontal = unsigned(x >= 0) | (unsigned(x + 1 < w) << 1)
                          | (unsigned(x + 2 < w) << 2);
    unsigned valid = ((y >= 0) ? horizontal : 0)
                     | ((y + 1 < h) ? horizontal << 3 : 0)
                     | ((y + 2 < h) ? horizontal << 6 : 0);
    return {((batch * h + y) * w + x) * CI, valid};
}

__device__ T load_patch(const T* x, Patch patch, int k, int w, T missing) {
    int tap = k / CI;
    if (k >= taps || !(patch.valid & (1u << tap))) return missing;
    return x[patch.base + (tap / 3 * w + tap % 3) * CI + k % CI];
}

__device__ int weight_index(int c, int k, int nhwc) {
    return nhwc ? c * taps + k : (c * CI + k % CI) * 9 + k / CI;
}

extern "C" __global__ void direct_conv_fprop(
    const T* x, const T* weight, const T* bias, T* y,
    int nhwc, int h, int w, int oh, int ow, int pixels
) {
    constexpr int depth = (taps + 15) / 16;
    __shared__ __align__(32) T weights[depth][256];
    __shared__ Patch patches[64];
    __shared__ __align__(32) T input[4][256];
    __shared__ __align__(32) float result[4][256];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int p0 = (blockIdx.x * 4 + warp) * 16;
    int c0 = blockIdx.y * 16;
    if (threadIdx.x < 64)
        patches[threadIdx.x] = locate_patch(blockIdx.x * 64 + threadIdx.x, h, w, oh, ow, pixels);
    for (int i = threadIdx.x; i < depth * 256; i += blockDim.x) {
        int c = c0 + i / 16 % 16;
        int k = i / 256 * 16 + i % 16;
        weights[i / 256][i % 256] =
            k < taps && c < CO ? weight[weight_index(c, k, nhwc)] : T(0.f);
    }
    __syncthreads();
    if (p0 >= pixels) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> lhs;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> rhs;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum;
    wmma::fill_fragment(sum, 0.f);
    for (int tile = 0; tile < depth; ++tile) {
        for (int i = lane; i < 256; i += 32) {
            int k = tile * 16 + i % 16;
            input[warp][i] = load_patch(x, patches[warp * 16 + i / 16], k, w, T(0.f));
        }
        __syncwarp();
        wmma::load_matrix_sync(lhs, input[warp], 16);
        wmma::load_matrix_sync(rhs, weights[tile], 16);
        wmma::mma_sync(sum, lhs, rhs, sum);
        __syncwarp();
    }
    wmma::store_matrix_sync(result[warp], sum, 16, wmma::mem_row_major);
    __syncwarp();
    if constexpr (CO % 8 == 0) {
        union Pack { uint4 bits; T values[8]; __device__ Pack() {} };
        int p = p0 + lane / 2;
        int c = c0 + lane % 2 * 8;
        if (p < pixels && c < CO) {
            Pack values;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                values.values[i] =
                    T(result[warp][lane * 8 + i] + (BIAS ? float(bias[c + i]) : 0.f));
            *reinterpret_cast<uint4*>(y + p * CO + c) = values.bits;
        }
    } else {
        for (int i = lane; i < 256; i += 32) {
            int p = p0 + i / 16;
            int c = c0 + i % 16;
            if (p < pixels && c < CO)
                y[p * CO + c] = T(result[warp][i] + (BIAS ? float(bias[c]) : 0.f));
        }
    }
}

extern "C" __global__ void direct_conv_dgrad(
    const T* grad, const T* weight, T* dx,
    int nhwc, int h, int w, int oh, int ow, int count
) {
    constexpr int max_depth = (4 * CO + 15) / 16;
    __shared__ __align__(32) T weights[max_depth][256];
    __shared__ __align__(32) T input[4][256];
    __shared__ __align__(32) float result[4][256];
    int py = blockIdx.y / 2;
    int px = blockIdx.y % 2;
    int kw = 1 + px;
    int inner = (1 + py) * kw * CO;
    int depth = (inner + 15) / 16;
    int c0 = blockIdx.z * 16;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int p0 = (blockIdx.x * 4 + warp) * 16;
    for (int i = threadIdx.x; i < depth * 256; i += blockDim.x) {
        int c = c0 + i / 16 % 16;
        int k = i / 256 * 16 + i % 16;
        int ky = 1 - py + 2 * (k / CO / kw);
        int kx = 1 - px + 2 * (k / CO % kw);
        int index = (ky * 3 + kx) * CI + c;
        weights[i / 256][i % 256] =
            k < inner && c < CI ? weight[weight_index(k % CO, index, nhwc)] : T(0.f);
    }
    __syncthreads();
    if (p0 >= count) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> lhs;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> rhs;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum;
    wmma::fill_fragment(sum, 0.f);
    int p = p0 + lane / 2;
    int b = p / (ow * oh);
    int oy = p / ow % oh;
    int ox = p % ow;
    bool vector = CO % 16 == 0 && reinterpret_cast<unsigned long long>(grad) % 16 == 0;
    for (int tile = 0; tile < depth; ++tile) {
        if (vector) {
            int row = oy + py * (1 - (tile * 16 / CO / kw));
            int col = ox + px * (1 - (tile * 16 / CO % kw));
            int channel = tile * 16 % CO + lane % 2 * 8;
            *reinterpret_cast<uint4*>(input[warp] + lane * 8) =
                p < count && row < oh && col < ow
                ? *reinterpret_cast<const uint4*>(grad + ((b * oh + row) * ow + col) * CO + channel)
                : make_uint4(0, 0, 0, 0);
        } else {
            for (int i = lane; i < 256; i += 32) {
                int q = p0 + i / 16;
                int k = tile * 16 + i % 16;
                int row = q / ow % oh + py * (1 - (k / CO / kw));
                int col = q % ow + px * (1 - (k / CO % kw));
                int image = q / (oh * ow);
                input[warp][i] = q < count && k < inner && row < oh && col < ow
                    ? grad[((image * oh + row) * ow + col) * CO + k % CO] : T(0.f);
            }
        }
        __syncwarp();
        wmma::load_matrix_sync(lhs, input[warp], 16);
        wmma::load_matrix_sync(rhs, weights[tile], 16);
        wmma::mma_sync(sum, lhs, rhs, sum);
        __syncwarp();
    }
    wmma::store_matrix_sync(result[warp], sum, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        int q = p0 + i / 16;
        int c = c0 + i % 16;
        int row = 2 * (q / ow % oh) + py;
        int col = 2 * (q % ow) + px;
        int image = q / (oh * ow);
        if (q < count && row < h && col < w && c < CI)
            dx[((image * h + row) * w + col) * CI + c] = T(result[warp][i]);
    }
}

// Compute [tap/channel, output channel] from contiguous spatial partitions.
extern "C" __global__ void direct_conv_wgrad(
    const T* x, const T* grad, float* partial, int h, int w, int oh, int ow, int pixels
) {
    __shared__ __align__(32) T operands[4][256];
    __shared__ __align__(32) float result[4][256];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row0 = blockIdx.y / column_groups * 32;
    int col0 = blockIdx.y % column_groups * 32;
    int row = row0 + warp / 2 * 16;
    int col = col0 + warp % 2 * 16;
    int chunks = (pixels - 1) / 16 + 1;
    int segment = (chunks - 1) / gridDim.x + 1;
    int begin = blockIdx.x * segment;
    int end = min(chunks, begin + segment);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> lhs;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> rhs;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sum;
    wmma::fill_fragment(sum, 0.f);
    for (int chunk = begin; chunk < end; ++chunk) {
        if (warp < 2) {
            Patch patch = locate_patch(chunk * 16 + lane % 16, h, w, oh, ow, pixels);
            int k = row0 + warp * 16 + lane % 16;
            #pragma unroll
            for (int i = lane; i < 256; i += 32) {
                int p = i / 16;
                Patch row_patch = {__shfl_sync(0xffffffff, patch.base, p),
                                   __shfl_sync(0xffffffff, patch.valid, p)};
                T missing = T(float(BIAS && k == taps && chunk * 16 + p < pixels));
                operands[warp][i] = load_patch(x, row_patch, k, w, missing);
            }
        } else if constexpr (CO % 16 == 0) {
            int p = chunk * 16 + lane / 2;
            int c = col0 + (warp - 2) * 16 + lane % 2 * 8;
            *reinterpret_cast<uint4*>(operands[warp] + lane * 8) =
                p < pixels && c < CO ? *reinterpret_cast<const uint4*>(grad + p * CO + c)
                                     : make_uint4(0, 0, 0, 0);
        } else {
            for (int i = lane; i < 256; i += 32) {
                int p = chunk * 16 + i / 16;
                int c = col0 + (warp - 2) * 16 + i % 16;
                operands[warp][i] = p < pixels && c < CO ? grad[p * CO + c] : T(0.f);
            }
        }
        __syncthreads();
        wmma::load_matrix_sync(lhs, operands[warp / 2], 16);
        wmma::load_matrix_sync(rhs, operands[2 + warp % 2], 16);
        wmma::mma_sync(sum, lhs, rhs, sum);
        __syncthreads();
    }
    wmma::store_matrix_sync(result[warp], sum, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        int k = row + i / 16;
        int c = col + i % 16;
        if (k < rows && c < CO)
            partial[(blockIdx.x * rows + k) * CO + c] = result[warp][i];
    }
}

extern "C" __global__ void direct_conv_reduce(
    const float* partial, T* dw, T* db, int nhwc, int parts
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * CO) return;
    float sum = 0;
    for (int p = 0; p < parts; ++p) sum += partial[p * rows * CO + i];
    int c = i % CO;
    int k = i / CO;
    if (k < taps) {
        if (dw) dw[weight_index(c, k, nhwc)] = T(sum);
    } else if (db) {
        db[c] = T(sum);
    }
}
"""


def _ceildiv(n: int, divisor: int) -> int:
    return (n + divisor - 1) // divisor


@cache
def _compile_ptx(
    capability: tuple[int, int],
    ci: int,
    co: int,
    dtype: torch.dtype,
    bias: bool,
) -> ct.Array:
    source = (
        f"#define CI {ci}\n#define CO {co}\n#define SCALAR {_TYPES[dtype]}\n"
        f"#define BIAS {int(bias)}\n{_CUDA}"
    )
    nvrtc = _nvrtc()
    program = ct.c_void_p()
    _check_nvrtc(
        nvrtc,
        nvrtc.nvrtcCreateProgram(
            ct.byref(program), source.encode(), b"direct_conv.cu", 0, None, None
        ),
        program,
    )
    options = [
        f"--gpu-architecture=compute_{capability[0]}{capability[1]}".encode(),
        b"--std=c++17",
        *(f"--include-path={path}".encode() for path in _include_dirs()),
    ]
    try:
        flags = (ct.c_char_p * len(options))(*options)
        _check_nvrtc(nvrtc, nvrtc.nvrtcCompileProgram(program, len(flags), flags), program)
        size = ct.c_size_t()
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), program)
        ptx = ct.create_string_buffer(size.value)
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTX(program, ptx), program)
    finally:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
    return ptx


@cache
def _module(device: int, ci: int, co: int, dtype: torch.dtype, bias: bool) -> _CudaModule:
    ptx = _compile_ptx(torch.cuda.get_device_capability(device), ci, co, dtype, bias)
    module = _CudaModule(ptx, device)
    kernels = ("direct_conv_fprop", "direct_conv_dgrad", "direct_conv_wgrad", "direct_conv_reduce")
    for name in kernels:
        module.load_function(name, 0)
    return module


def _launch(
    x: Tensor,
    w: Tensor,
    bias: bool,
    name: str,
    grid: tuple[int, int, int],
    tensors: tuple[Tensor | None, ...],
    *sizes: int,
) -> None:
    with torch.cuda.device(x.device):
        device = x.get_device()
        module = _module(device, x.size(1), w.size(0), x.dtype, bias)
        _ensure_context(module.driver, device)
        module.launch(
            name,
            grid,
            (_THREADS, 1, 1),
            0,
            torch.cuda.current_stream().cuda_stream,
            [
                *(ct.c_void_p(t.data_ptr() if t is not None else 0) for t in tensors),
                *(ct.c_int(n) for n in sizes),
            ],
        )


@torch.library.custom_op("va_direct_conv::forward", mutates_args=(), device_types="cuda")
def _forward(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    y = _forward_fake(x, weight, bias)
    oh, ow = y.shape[2:]
    pixels = x.size(0) * oh * ow
    _launch(
        x,
        weight,
        bias is not None,
        "direct_conv_fprop",
        (_ceildiv(pixels, 64), _ceildiv(weight.size(0), 16), 1),
        (x, weight, bias, y),
        int(not weight.is_contiguous()),
        *x.shape[2:],
        oh,
        ow,
        pixels,
    )
    return y


@_forward.register_fake
def _forward_fake(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:  # noqa: ARG001
    b, _, h, w = x.shape
    return x.new_empty((b, _ceildiv(h, 2), _ceildiv(w, 2), weight.size(0))).permute(0, 3, 1, 2)


@torch.library.custom_op(
    "va_direct_conv::backward",
    mutates_args=(),
    device_types="cuda",
    schema="(Tensor grad, Tensor x, Tensor weight, bool need_x, bool need_w, bool need_b)"
    " -> (Tensor?, Tensor?, Tensor?)",
)
def _backward(
    grad: Tensor,
    x: Tensor,
    weight: Tensor,
    need_x: bool,
    need_w: bool,
    need_b: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    grad = grad.contiguous(memory_format=torch.channels_last)
    dx, dw, db = _backward_fake(grad, x, weight, need_x, need_w, need_b)
    oh, ow = grad.shape[2:]
    dims = (*x.shape[2:], oh, ow)
    pixels = x.size(0) * oh * ow
    if dx is not None:
        _launch(
            x,
            weight,
            False,
            "direct_conv_dgrad",
            (_ceildiv(pixels, 64), 4, _ceildiv(x.size(1), 16)),
            (grad, weight, dx),
            int(not weight.is_contiguous()),
            *dims,
            pixels,
        )
    if need_w or need_b:
        # Sliced gradients may have a channels-last stride with a misaligned base.
        if weight.size(0) % 16 == 0 and grad.data_ptr() % 16:
            grad = grad.clone(memory_format=torch.channels_last)
        parts = min(512, _ceildiv(pixels, 128))
        rows = 9 * x.size(1) + int(need_b)
        co = weight.size(0)
        partial = x.new_empty((parts, rows, co), dtype=torch.float32)
        tile_groups = _ceildiv(rows, 32) * _ceildiv(co, 32)
        _launch(
            x,
            weight,
            need_b,
            "direct_conv_wgrad",
            (parts, tile_groups, 1),
            (x, grad, partial),
            *dims,
            pixels,
        )
        _launch(
            x,
            weight,
            need_b,
            "direct_conv_reduce",
            (_ceildiv(rows * co, _THREADS), 1, 1),
            (partial, dw, db),
            int(not weight.is_contiguous()),
            parts,
        )
    return dx, dw, db


@_backward.register_fake
def _backward_fake(
    grad: Tensor,
    x: Tensor,
    weight: Tensor,
    need_x: bool,
    need_w: bool,
    need_b: bool,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    return (
        torch.empty_like(x) if need_x else None,
        torch.empty_like(weight) if need_w else None,
        grad.new_empty((weight.size(0),)) if need_b else None,
    )


def _setup_context(ctx, inputs, output) -> None:  # noqa: ANN001, ARG001
    x, weight, _ = inputs
    ctx.save_for_backward(x, weight)


def _backward_autograd(ctx, grad: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:  # noqa: ANN001
    return _backward(grad, *ctx.saved_tensors, *ctx.needs_input_grad)


_forward.register_autograd(_backward_autograd, setup_context=_setup_context)


def try_direct_conv3x3s2(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor | None:
    """Return a direct convolution when the experimental path accepts the inputs."""
    mode = os.environ.get(_MODE, "auto")
    autocast = torch.is_autocast_enabled("cuda")
    dtype = torch.get_autocast_dtype("cuda") if autocast else x.dtype
    operands = (x, weight) if bias is None else (x, weight, bias)
    if (
        mode == "on"
        and x.is_cuda
        and x.ndim == 4
        and x.is_contiguous(memory_format=torch.channels_last)
        and weight.ndim == 4
        and (weight.is_contiguous() or weight.is_contiguous(memory_format=torch.channels_last))
        and weight.shape[1:] == (x.size(1), 3, 3)
        and 1 <= x.size(1) <= 8
        and 1 <= weight.size(0) <= 64
        and (bias is None or (bias.shape == (weight.size(0),) and bias.is_contiguous()))
        and all(t.device == x.device for t in operands)
        and all(t.dtype == dtype or (autocast and t.dtype == torch.float32) for t in operands)
        and dtype in _TYPES
        and 0 < x.numel() <= 2**30
        and x.size(0) * _ceildiv(x.size(2), 2) * _ceildiv(x.size(3), 2) * weight.size(0) <= 2**30
        and torch.cuda.get_device_capability(x.device)[0] >= 8
    ):
        return _forward(x.to(dtype), weight.to(dtype), bias.to(dtype) if bias is not None else None)
    return None


def conv3x3s2(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Apply a channels-last convolution with the configured CUDA backend."""
    result = try_direct_conv3x3s2(x, weight, bias)
    if result is not None:
        return result
    return functional.conv2d(x, weight, bias, stride=2, padding=1)
