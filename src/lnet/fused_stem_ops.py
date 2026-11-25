"""Fused, exact-GELU LayerNorm over physically contiguous pixel channels."""

from __future__ import annotations

# Custom op schemas use positional flags.
# ruff: noqa: ANN001, FBT001, N803
import ctypes as ct
import os
from functools import cache

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional
from triton.language.extra.cuda import libdevice

from .cuda_nvrtc import _check_cu, _check_nvrtc, _CudaModule, _ensure_context, _include_dirs, _nvrtc

_TYPES = {torch.float32: "float", torch.float16: "__half", torch.bfloat16: "__nv_bfloat16"}
_THREADS = 128
_VECTOR_BYTES = 16
_MAX_VECTOR_CHANNELS = 1024
_WARP_SIZE = 32
_MAX_FORWARD_BLOCKS = 65535
_REDUCE_CHANNELS = 8


# CUDA kernels
_CUDA = r"""
#include <cuda_fp16.h>
#if BF16
#include <cuda_bf16.h>
#endif

using X = INPUT;
using W = WEIGHT;
using B = BIAS;
using Y = OUTPUT;
using CB = CONV_BIAS;
constexpr int threads = BLOCK_THREADS;
constexpr int elements = VEC ? VECTOR_BYTES / sizeof(X) : 1;
constexpr int vectors = C / elements;
constexpr int group = vectors <= 1 ? 1 : vectors <= 2 ? 2 : vectors <= 4 ? 4 :
                      vectors <= 8 ? 8 : vectors <= 16 ? 16 : WARP_SIZE;
constexpr int rows = threads / group;
constexpr int tiles = (vectors + group - 1) / group;

template<typename T>
union Pack {
    static constexpr int words = elements * sizeof(T) / VECTOR_BYTES;
    uint4 bits[words ? words : 1];
    T value[elements];
    __device__ Pack() : bits{} {}

    __device__ void load(const T* ptr, long long vector) {
        #pragma unroll
        for (int i = 0; i < words; i++)
            bits[i] = reinterpret_cast<const uint4*>(ptr)[vector * words + i];
        if constexpr (!VEC) value[0] = ptr[vector];
    }

    __device__ void store(T* ptr, long long vector) {
        #pragma unroll
        for (int i = 0; i < words; i++)
            reinterpret_cast<uint4*>(ptr)[vector * words + i] = bits[i];
        if constexpr (!VEC) ptr[vector] = value[0];
    }
};

__device__ float sum_group(float v) {
    for (int i = group / 2; i > 0; i /= 2)
        v += __shfl_xor_sync(0xffffffff, v, i, group);
    return v;
}

__device__ float affine(float norm, float w, float b) {
    float a = fmaf(norm, w, b);
    if constexpr (ROUND_AFFINE) return float(Y(a));
    return a;
}

__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}

__device__ float gelu_grad(float x, float dy) {
    float derivative = 0.5f * (1.0f + erff(x * 0.7071067811865475244f))
        + x * 0.39894228040143267794f * expf(-0.5f * x * x);
    float g = dy * derivative;
    if constexpr (ROUND_AFFINE) return float(Y(g));
    return g;
}

template<bool backward, typename Out>
__device__ void pixel_rows(
    const X* x, const W* w, const B* b, const CB* cb, const Y* dy,
    Out* out, float* partial, long long n, float eps
) {
    int lane = threadIdx.x % group, local_row = threadIdx.x / group;
    float scale[tiles][elements], bias[tiles][elements], conv_offset[tiles][elements];
    float accum_cb[tiles][elements] = {};
    float accum_w[tiles][elements] = {}, accum_b[tiles][elements] = {};
    #pragma unroll
    for (int t = 0; t < tiles; t++) {
        int v = t * group + lane;
        #pragma unroll
        for (int e = 0; e < elements; e++) {
            scale[t][e] = v < vectors ? float(w[v * elements + e]) : 0.f;
            bias[t][e] = v < vectors ? float(b[v * elements + e]) : 0.f;
            if constexpr (HAS_CONV_BIAS)
                conv_offset[t][e] = v < vectors ? float(cb[v * elements + e]) : 0.f;
        }
    }
    // All lanes execute shuffles even when the last tile contains inactive rows.
    for (long long base = blockIdx.x * rows; base < n; base += gridDim.x * rows) {
        long long row = base + local_row;
        bool valid = row < n;
        float values[tiles][elements];
        float total = 0.f, squares = 0.f;
        #pragma unroll
        for (int t = 0; t < tiles; t++) {
            int v = t * group + lane;
            Pack<X> input;
            if (valid && v < vectors)
                input.load(x, row * vectors + v);
            #pragma unroll
            for (int e = 0; e < elements; e++) {
                values[t][e] = float(input.value[e]);
                if constexpr (HAS_CONV_BIAS) values[t][e] += conv_offset[t][e];
                total += values[t][e];
            }
        }
        float mu = sum_group(total) / C;
        #pragma unroll
        for (int t = 0; t < tiles; t++) {
            #pragma unroll
            for (int e = 0; e < elements; e++) {
                float d = values[t][e] - mu;
                if (t * group + lane < vectors) squares = fmaf(d, d, squares);
            }
        }
        float inv = rsqrtf(sum_group(squares) / C + eps);
        float projected[tiles][elements];
        total = squares = 0.f;
        #pragma unroll
        for (int t = 0; t < tiles; t++) {
            int v = t * group + lane;
            Pack<Y> grad, result;
            if constexpr (backward)
                if (valid && v < vectors)
                    grad.load(dy, row * vectors + v);
            #pragma unroll
            for (int e = 0; e < elements; e++) {
                float norm = (values[t][e] - mu) * inv;
                float a = affine(norm, scale[t][e], bias[t][e]);
                if constexpr (backward) {
                    float g = gelu_grad(a, float(grad.value[e]));
                    if (valid && v < vectors) {
                        accum_w[t][e] = fmaf(g, norm, accum_w[t][e]);
                        accum_b[t][e] += g;
                        total += g * scale[t][e];
                        squares = fmaf(g * scale[t][e], norm, squares);
                    }
                    projected[t][e] = g * scale[t][e];
                    values[t][e] = norm;
                } else {
                    result.value[e] = Y(gelu(a));
                }
            }
            if constexpr (!backward)
                if (valid && v < vectors)
                    result.store(out, row * vectors + v);
        }
        if constexpr (backward) {
            total = sum_group(total) / C;
            squares = sum_group(squares) / C;
            #pragma unroll
            for (int t = 0; t < tiles; t++) {
                int v = t * group + lane;
                Pack<X> result;
                #pragma unroll
                for (int e = 0; e < elements; e++) {
                    float dx = inv * (projected[t][e] - total - values[t][e] * squares);
                    result.value[e] = X(dx);
                    if constexpr (HAS_CONV_BIAS)
                        if (valid && v < vectors) accum_cb[t][e] += dx;
                }
                if (valid && v < vectors)
                    result.store(out, row * vectors + v);
            }
        }
    }
    if constexpr (backward) {
        __shared__ float partial_w[threads], partial_b[threads], partial_cb[threads];
        for (int t = 0; t < tiles; t++) {
            int v = t * group + lane;
            for (int e = 0; e < elements; e++) {
                partial_w[threadIdx.x] = accum_w[t][e];
                partial_b[threadIdx.x] = accum_b[t][e];
                if constexpr (HAS_CONV_BIAS) partial_cb[threadIdx.x] = accum_cb[t][e];
                __syncthreads();
                if (local_row == 0 && v < vectors) {
                    float sw = 0.f, sb = 0.f, scb = 0.f;
                    for (int i = 0; i < rows; i++) {
                        sw += partial_w[i * group + lane];
                        sb += partial_b[i * group + lane];
                        if constexpr (HAS_CONV_BIAS) scb += partial_cb[i * group + lane];
                    }
                    int c = v * elements + e;
                    int worker = blockIdx.x * (2 + HAS_CONV_BIAS) * C;
                    partial[worker + c] = sw;
                    partial[worker + C + c] = sb;
                    if constexpr (HAS_CONV_BIAS) partial[worker + 2 * C + c] = scb;
                }
                __syncthreads();
            }
        }
    }
}

extern "C" __global__ void stem_forward(
    const X* x, const W* w, const B* b, const CB* cb, Y* y, long long n, float eps
) {
    pixel_rows<false>(x, w, b, cb, nullptr, y, nullptr, n, eps);
}

extern "C" __global__ void stem_backward(
    const X* x, const W* w, const B* b, const CB* cb, const Y* dy,
    X* dx, float* partial, long long n, float eps
) {
    pixel_rows<true>(x, w, b, cb, dy, dx, partial, n, eps);
}

extern "C" __global__ void stem_reduce(
    const float* partial, W* dw, B* db, CB* dcb, int workers
) {
    constexpr int fields = 2 + HAS_CONV_BIAS;
    constexpr int rows_per_tile = threads / REDUCE_CHANNELS;
    __shared__ float sums[fields][threads];
    int lane = threadIdx.x % REDUCE_CHANNELS, row = threadIdx.x / REDUCE_CHANNELS;
    int c = blockIdx.x * REDUCE_CHANNELS + lane;
    float acc[fields] = {};
    if (c < C) {
        for (int i = row; i < workers; i += rows_per_tile) {
            #pragma unroll
            for (int f = 0; f < fields; f++)
                acc[f] += partial[(i * fields + f) * C + c];
        }
    }
    #pragma unroll
    for (int f = 0; f < fields; f++)
        sums[f][threadIdx.x] = acc[f];
    __syncthreads();
    if (row == 0 && c < C) {
        #pragma unroll
        for (int f = 0; f < fields; f++) {
            acc[f] = 0.f;
            #pragma unroll
            for (int i = 0; i < rows_per_tile; i++)
                acc[f] += sums[f][i * REDUCE_CHANNELS + lane];
        }
        dw[c] = W(acc[0]);
        db[c] = B(acc[1]);
        if constexpr (HAS_CONV_BIAS) dcb[c] = CB(acc[2]);
    }
}
"""


# NVRTC compilation and driver launch
@cache
def _compile_ptx(
    capability: tuple[int, int],
    channels: int,
    types: tuple[str, str, str, str],
    *,
    vector: bool,
    conv_bias_type: str | None = None,
    round_affine: bool = True,
) -> ct.Array:
    header = {
        "C": channels,
        "BLOCK_THREADS": _THREADS,
        "VECTOR_BYTES": _VECTOR_BYTES,
        "REDUCE_CHANNELS": _REDUCE_CHANNELS,
        "WARP_SIZE": _WARP_SIZE,
        "INPUT": types[0],
        "WEIGHT": types[1],
        "BIAS": types[2],
        "OUTPUT": types[3],
        "CONV_BIAS": conv_bias_type or "float",
        "HAS_CONV_BIAS": int(conv_bias_type is not None),
        "ROUND_AFFINE": int(round_affine),
        "BF16": int("__nv_bfloat16" in (*types, conv_bias_type)),
        "VEC": int(vector),
    }
    source = "".join(f"#define {name} {value}\n" for name, value in header.items()) + _CUDA
    return _compile_source(capability, source)


def _compile_source(capability: tuple[int, int], source: str) -> ct.Array:
    nvrtc = _nvrtc()
    program = ct.c_void_p()
    _check_nvrtc(
        nvrtc,
        nvrtc.nvrtcCreateProgram(
            ct.byref(program),
            source.encode(),
            b"fused_stem.cu",
            0,
            None,
            None,
        ),
        program,
    )
    flags = [
        f"--gpu-architecture=compute_{capability[0]}{capability[1]}".encode(),
        b"--std=c++17",
        *(f"--include-path={path}".encode() for path in _include_dirs()),
    ]
    options = (ct.c_char_p * len(flags))(*flags)
    try:
        _check_nvrtc(nvrtc, nvrtc.nvrtcCompileProgram(program, len(flags), options), program)
        size = ct.c_size_t()
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), program)
        image = ct.create_string_buffer(size.value)
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTX(program, image), program)
    finally:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
    return image


@cache
def _module(
    device: int,
    channels: int,
    types: tuple[str, str, str, str],
    *,
    vector: bool,
    conv_bias_type: str | None,
    round_affine: bool,
) -> _CudaModule:
    image = _compile_ptx(
        torch.cuda.get_device_capability(device),
        channels,
        types,
        vector=vector,
        conv_bias_type=conv_bias_type,
        round_affine=round_affine,
    )
    module = _CudaModule(image, device)
    for name in ("stem_forward", "stem_backward", "stem_reduce"):
        module.load_function(name, 0)
    return module


def _geometry(x: Tensor, channels: int, *aligned: Tensor) -> tuple[bool, int]:
    elements = _VECTOR_BYTES // x.element_size()
    vector = (
        channels <= _MAX_VECTOR_CHANNELS
        and channels % elements == 0
        and all(
            t.data_ptr() % _VECTOR_BYTES == 0 and t.element_size() >= x.element_size()
            for t in (x, *aligned)
        )
    )
    if not vector:
        elements = 1
    group = min(_WARP_SIZE, 1 << (channels // elements - 1).bit_length())
    return vector, _THREADS // group


def _launch(
    module: _CudaModule,
    stream: int,
    name: str,
    blocks: int,
    tensors: tuple[Tensor | None, ...],
    *sizes: ct.c_int | ct.c_longlong | ct.c_float,
) -> None:
    module.launch(
        name,
        (blocks, 1, 1),
        (_THREADS, 1, 1),
        0,
        stream,
        [*(ct.c_void_p(t.data_ptr() if t is not None else 0) for t in tensors), *sizes],
    )


def _kernel_module(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    cb: Tensor | None,
    output_dtype: torch.dtype,
    *,
    vector: bool,
    round_affine: bool,
) -> _CudaModule:
    types = (_TYPES[x.dtype], _TYPES[w.dtype], _TYPES[b.dtype], _TYPES[output_dtype])
    return _module(
        x.device.index or 0,
        w.numel(),
        types,
        vector=vector,
        conv_bias_type=_TYPES[cb.dtype] if cb is not None else None,
        round_affine=round_affine,
    )


# Custom ops and autograd
@torch.library.custom_op("va_stem::forward", mutates_args=(), device_types="cuda")
def _forward(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    cb: Tensor | None,
    eps: float,
    output_dtype: torch.dtype,
    round_affine: bool,
) -> Tensor:
    channels = w.numel()
    n = x.numel() // channels
    y = torch.empty_like(x, dtype=output_dtype)
    if n:
        vector, rows = _geometry(x, channels, y)
        blocks = min(_MAX_FORWARD_BLOCKS, (n + rows - 1) // rows)
        with torch.cuda.device(x.device):
            module = _kernel_module(
                x, w, b, cb, output_dtype, vector=vector, round_affine=round_affine
            )
            _ensure_context(module.driver, x.device.index or 0)
            stream = torch.cuda.current_stream().cuda_stream
            _launch(
                module,
                stream,
                "stem_forward",
                blocks,
                (x, w, b, cb, y),
                ct.c_longlong(n),
                ct.c_float(eps),
            )
    return y


@_forward.register_fake
def _forward_fake(
    x: Tensor,
    w: Tensor,  # noqa: ARG001
    b: Tensor,  # noqa: ARG001
    cb: Tensor | None,  # noqa: ARG001
    eps: float,  # noqa: ARG001
    output_dtype: torch.dtype,
    round_affine: bool,  # noqa: ARG001
) -> Tensor:
    return torch.empty_like(x, dtype=output_dtype)


@cache
def _worker_limit(module: _CudaModule, device: torch.device) -> int:
    # The register footprint depends on the channel tile, dtypes and bias fusion.
    # Size the persistent grid for this kernel's occupancy on the current GPU.
    # Extra waves would also add affine partials and reduction traffic.
    resident = ct.c_int()
    _check_cu(
        module.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ct.byref(resident), module.functions["stem_backward"], _THREADS, ct.c_size_t(0)
        ),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    )
    return resident.value * torch.cuda.get_device_properties(device).multi_processor_count


@torch.library.custom_op("va_stem::backward", mutates_args=(), device_types="cuda")
def _backward(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    cb: Tensor | None,
    dy: Tensor,
    eps: float,
    round_affine: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    channels = w.numel()
    n = x.numel() // channels
    if not n:
        dcb = torch.zeros_like(cb) if cb is not None else x.new_empty(0)
        return torch.empty_like(x), torch.zeros_like(w), torch.zeros_like(b), dcb
    dx, dw, db, dcb = _backward_fake(x, w, b, cb, dy, eps, round_affine)
    vector, rows = _geometry(x, channels, dy, dx)
    with torch.cuda.device(x.device):
        module = _kernel_module(x, w, b, cb, dy.dtype, vector=vector, round_affine=round_affine)
        _ensure_context(module.driver, x.device.index or 0)
        workers = min(_worker_limit(module, x.device), (n + rows - 1) // rows)
        partial = x.new_empty((workers, 2 + int(cb is not None), channels), dtype=torch.float32)
        stream = torch.cuda.current_stream().cuda_stream
        _launch(
            module,
            stream,
            "stem_backward",
            workers,
            (x, w, b, cb, dy, dx, partial),
            ct.c_longlong(n),
            ct.c_float(eps),
        )
        _launch(
            module,
            stream,
            "stem_reduce",
            (channels + _REDUCE_CHANNELS - 1) // _REDUCE_CHANNELS,
            (partial, dw, db, dcb),
            ct.c_int(workers),
        )
    return dx, dw, db, dcb


@_backward.register_fake
def _backward_fake(
    x: Tensor,
    w: Tensor,
    b: Tensor,
    cb: Tensor | None,
    dy: Tensor,  # noqa: ARG001
    eps: float,  # noqa: ARG001
    round_affine: bool,  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dcb = torch.empty_like(cb) if cb is not None else x.new_empty(0)
    return torch.empty_like(x), torch.empty_like(w), torch.empty_like(b), dcb


def _save(ctx, inputs, output) -> None:  # noqa: ARG001
    x, w, b, cb, ctx.eps, _, ctx.round_affine = inputs
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x, w, b, cb)


def _autograd_backward(
    ctx,
    dy: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, None, None, None]:
    if dy is None:
        return None, None, None, None, None, None, None
    x, w, b, cb = ctx.saved_tensors
    dx, dw, db, dcb = _backward(x, w, b, cb, dy.contiguous(), ctx.eps, ctx.round_affine)
    return dx, dw, db, dcb if cb is not None else None, None, None, None


_forward.register_autograd(_autograd_backward, setup_context=_save)


# Public API and PyTorch fallback
def _fallback_norm(
    pixels: Tensor,
    weight: Tensor,
    bias: Tensor,
    conv_bias: Tensor | None,
    eps: float,
    *,
    full_precision: bool,
    autocast: bool,
) -> Tensor:
    if full_precision:
        compute_dtype = torch.float64 if pixels.dtype == torch.float64 else torch.float32
        values = pixels.to(compute_dtype)
        if conv_bias is not None:
            values = values + conv_bias.to(compute_dtype)
        norm = functional.layer_norm(
            values, weight.shape, weight.to(compute_dtype), bias.to(compute_dtype), eps
        )
    elif (
        pixels.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype in (torch.float32, pixels.dtype)
        and bias.dtype in (torch.float32, pixels.dtype)
        and (weight.dtype != pixels.dtype or bias.dtype != pixels.dtype)
    ):
        norm = functional.layer_norm(
            pixels.float(), weight.shape, weight.float(), bias.float(), eps
        )
        if not autocast:
            norm = norm.to(pixels.dtype)
    else:
        norm = functional.layer_norm(pixels, weight.shape, weight, bias, eps)
    return norm


@torch.compiler.assume_constant_result
def _stem_cuda_major(device: torch.device) -> int:
    return torch.cuda.get_device_capability(device)[0]


def fused_nhwc_layernorm_gelu(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
    *,
    nchw: bool | None = None,
    conv_bias: Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    nhwc_output: bool = False,
) -> Tensor:
    """Apply channel-wise LayerNorm and exact GELU, preserving the input layout.

    By default, infer NCHW from channels-last strides and the affine width.
    Specify nchw explicitly if singleton dimensions make the layouts ambiguous.
    With convolution bias or an explicit output dtype, compute bias addition,
    normalization and GELU in FP32 before casting the result. Legacy calls
    use FP32 on CUDA autocast and round LayerNorm before GELU otherwise.
    Set nhwc_output to return contiguous NHWC regardless of input layout.
    """
    if x.ndim != 4 or weight.ndim != 1 or bias.shape != weight.shape or not weight.numel():
        raise ValueError("expected rank-four input and nonempty, equally sized affine vectors")
    if nchw is None:
        nchw = (
            x.shape[1] == weight.numel()
            and x.is_contiguous(memory_format=torch.channels_last)
            and (not x.is_contiguous() or x.shape[-1] != weight.numel())
        )
    pixels = x.permute(0, 2, 3, 1) if nchw else x
    if pixels.shape[-1] != weight.numel():
        raise ValueError("input channels must match the affine vectors")
    if conv_bias is not None and (conv_bias.shape != weight.shape or conv_bias.device != x.device):
        raise ValueError("convolution bias must match the channel width and input device")
    autocast = x.is_cuda and torch.is_autocast_enabled("cuda")
    full_precision = conv_bias is not None or output_dtype is not None
    round_affine = not full_precision and not autocast
    if output_dtype is None:
        output_dtype = torch.float32 if autocast and x.dtype in _TYPES else x.dtype
    if (
        pixels.is_cuda
        and pixels.is_contiguous()
        and x.dtype in _TYPES
        and output_dtype in _TYPES
        and weight.dtype in (torch.float32, x.dtype)
        and bias.dtype in (torch.float32, x.dtype)
        and weight.device == x.device
        and bias.device == x.device
        and weight.is_contiguous()
        and bias.is_contiguous()
        and (
            conv_bias is None
            or (conv_bias.dtype in (torch.float32, x.dtype) and conv_bias.is_contiguous())
        )
    ):
        needs_bf16 = torch.bfloat16 in (x.dtype, output_dtype)
        if _stem_cuda_major(x.device) >= (8 if needs_bf16 else 7):
            result = _forward(pixels, weight, bias, conv_bias, eps, output_dtype, round_affine)
            return result.permute(0, 3, 1, 2) if nchw and not nhwc_output else result

    norm = _fallback_norm(
        pixels, weight, bias, conv_bias, eps, full_precision=full_precision, autocast=autocast
    )
    result = functional.gelu(norm).to(output_dtype)
    if nhwc_output:
        return result.contiguous()
    return result.permute(0, 3, 1, 2) if nchw else result


_CONV_CUDA = r"""
#include <cuda_fp16.h>
#if BF16
#include <cuda_bf16.h>
#endif
using T = SCALAR;
constexpr int kernel_values = 9 * CI;
constexpr int tiles = CO / 32;
constexpr int fields = kernel_values + 2 + HAS_BIAS;

__device__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset; offset /= 2)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

template<bool backward>
__device__ void conv_pixels(
    const T* x, const T* w, const float* gamma, const float* beta, const float* bias,
    const T* dy, T* out, float2* saved, const float2* stats, float* partial,
    int height, int width, int oh, int ow, long long n, float eps
) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float weights[tiles][kernel_values], dg[tiles] = {}, db[tiles] = {}, dc[tiles] = {};
    float dw[tiles][kernel_values] = {};
    float scale[tiles], shift[tiles], offset[tiles];
    #pragma unroll
    for (int t = 0; t < tiles; ++t) {
        int c = t * 32 + lane;
        scale[t] = gamma[c];
        shift[t] = beta[c];
        if constexpr (HAS_BIAS) offset[t] = bias[c];
        #pragma unroll
        for (int k = 0; k < kernel_values; ++k)
            weights[t][k] = float(w[k * CO + c]);
    }
    for (long long pixel = blockIdx.x * (BLOCK_THREADS / 32) + warp; pixel < n;
         pixel += gridDim.x * (BLOCK_THREADS / 32)) {
        int col = pixel % ow, row = (pixel / ow) % oh;
        long long batch = pixel / (static_cast<long long>(oh) * ow);
        // Each lane preloads one patch value. CI=4 needs four more values.
        float low = 0.f, high = 0.f;
        #pragma unroll
        for (int tile = 0; tile < (kernel_values + 31) / 32; ++tile) {
            int k = tile * 32 + lane;
            int i = 2 * row + k / (3 * CI) - 1;
            int j = 2 * col + (k / CI) % 3 - 1;
            float value = 0.f;
            if (k < kernel_values && i >= 0 && i < height && j >= 0 && j < width) {
                long long index = CHANNELS_LAST
                    ? ((batch * height + i) * width + j) * CI + k % CI
                    : ((batch * CI + k % CI) * height + i) * width + j;
                value = float(x[index]);
            }
            if (tile == 0) low = value;
            else high = value;
        }
        float values[tiles] = {};
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < kernel_values; ++k) {
            float patch = __shfl_sync(0xffffffff, k < 32 ? low : high, k % 32);
            #pragma unroll
            for (int t = 0; t < tiles; ++t)
                values[t] = fmaf(weights[t][k], patch, values[t]);
        }
        #pragma unroll
        for (int t = 0; t < tiles; ++t) {
            values[t] = float(T(values[t]));
            if constexpr (HAS_BIAS) values[t] += offset[t];
            sum += values[t];
        }
        float mean, inv;
        if constexpr (backward) {
            float2 pair = stats[pixel];
            mean = pair.x;
            inv = pair.y;
        } else {
            mean = warp_sum(sum) / CO;
            float squares = 0.f;
            #pragma unroll
            for (int t = 0; t < tiles; ++t) {
                float d = values[t] - mean;
                squares = fmaf(d, d, squares);
            }
            inv = rsqrtf(warp_sum(squares) / CO + eps);
            if (lane == 0) saved[pixel] = make_float2(mean, inv);
        }
        float s1 = 0.f, s2 = 0.f, projected[tiles];
        #pragma unroll
        for (int t = 0; t < tiles; ++t) {
            int c = t * 32 + lane;
            float norm = (values[t] - mean) * inv;
            float a = fmaf(norm, scale[t], shift[t]);
            float cdf = 0.5f * (1.f + erff(a * 0.7071067811865475244f));
            if constexpr (backward) {
                float derivative = cdf + a * 0.39894228040143267794f * expf(-0.5f * a * a);
                float grad = float(dy[pixel * CO + c]) * derivative;
                dg[t] = fmaf(grad, norm, dg[t]);
                db[t] += grad;
                projected[t] = grad * scale[t];
                s1 += projected[t];
                s2 = fmaf(projected[t], norm, s2);
                values[t] = norm;
            } else {
                out[pixel * CO + c] = T(a * cdf);
            }
        }
        if constexpr (backward) {
            s1 = warp_sum(s1) / CO;
            s2 = warp_sum(s2) / CO;
            #pragma unroll
            for (int t = 0; t < tiles; ++t) {
                float grad = inv * (projected[t] - s1 - values[t] * s2);
                if constexpr (HAS_BIAS) dc[t] += grad;
                projected[t] = float(T(grad));
            }
            #pragma unroll
            for (int k = 0; k < kernel_values; ++k) {
                float patch = __shfl_sync(0xffffffff, k < 32 ? low : high, k % 32);
                #pragma unroll
                for (int t = 0; t < tiles; ++t)
                    dw[t][k] = fmaf(projected[t], patch, dw[t][k]);
            }
        }
    }
    if constexpr (backward) {
        // Each warp owns a contiguous partial. The final reduction has fixed order.
        int worker = blockIdx.x * (BLOCK_THREADS / 32) + warp;
        float* dst = partial + worker * fields * CO;
        #pragma unroll
        for (int t = 0; t < tiles; ++t) {
            int c = t * 32 + lane;
            #pragma unroll
            for (int k = 0; k < kernel_values; ++k)
                dst[k * CO + c] = dw[t][k];
            dst[kernel_values * CO + c] = dg[t];
            dst[(kernel_values + 1) * CO + c] = db[t];
            if constexpr (HAS_BIAS) dst[(kernel_values + 2) * CO + c] = dc[t];
        }
    }
}

extern "C" __global__ void conv_stem_forward(
    const T* x, const T* w, const float* g, const float* b, const float* cb,
    T* out, float2* stats, int h, int w_in, int oh, int ow, long long n, float eps
) {
    conv_pixels<false>(x, w, g, b, cb, nullptr, out, stats, nullptr, nullptr,
                       h, w_in, oh, ow, n, eps);
}

extern "C" __global__ void conv_stem_backward(
    const T* x, const T* w, const float* g, const float* b, const float* cb,
    const T* dy, const float2* stats, float* partial,
    int h, int w_in, int oh, int ow, long long n, float eps
) {
    conv_pixels<true>(x, w, g, b, cb, dy, nullptr, nullptr, stats, partial,
                      h, w_in, oh, ow, n, eps);
}

extern "C" __global__ void conv_stem_reduce(
    const float* partial, T* dw, float* dg, float* db, float* dc, int workers
) {
    int i = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (i >= fields * CO) return;
    float sum = 0.f;
    for (int worker = 0; worker < workers; ++worker)
        sum += partial[worker * fields * CO + i];
    int field = i / CO, channel = i % CO;
    if (field < kernel_values) dw[i] = T(sum);
    else if (field == kernel_values) dg[channel] = sum;
    else if (field == kernel_values + 1) db[channel] = sum;
    else if constexpr (HAS_BIAS) dc[channel] = sum;
}
"""


@cache
def _compile_conv_stem_ptx(
    capability: tuple[int, int],
    dtype: torch.dtype,
    ci: int,
    co: int,
    channels_last: bool,
    has_bias: bool,
) -> ct.Array:
    defines = {
        "BLOCK_THREADS": _THREADS,
        "CI": ci,
        "CO": co,
        "SCALAR": _TYPES[dtype],
        "BF16": int(dtype == torch.bfloat16),
        "CHANNELS_LAST": int(channels_last),
        "HAS_BIAS": int(has_bias),
    }
    header = "".join(f"#define {name} {value}\n" for name, value in defines.items())
    return _compile_source(capability, header + _CONV_CUDA)


@cache
def _conv_stem_module(
    device: int, dtype: torch.dtype, ci: int, co: int, channels_last: bool, has_bias: bool
) -> _CudaModule:
    image = _compile_conv_stem_ptx(
        torch.cuda.get_device_capability(device), dtype, ci, co, channels_last, has_bias
    )
    module = _CudaModule(image, device)
    for name in ("conv_stem_forward", "conv_stem_backward", "conv_stem_reduce"):
        module.load_function(name, 0)
    return module


def _conv_shape(x: Tensor, w: Tensor) -> tuple[int, int, int, int]:
    return x.shape[0], (x.shape[2] + 1) // 2, (x.shape[3] + 1) // 2, w.shape[-1]


def _conv_launch_setup(
    x: Tensor, w: Tensor, cb: Tensor | None, eps: float
) -> tuple[_CudaModule, int, int, tuple[ct.c_int | ct.c_longlong | ct.c_float, ...]]:
    batch, oh, ow, co = _conv_shape(x, w)
    n = batch * oh * ow
    device = x.device.index or 0
    module = _conv_stem_module(
        device,
        x.dtype,
        x.shape[1],
        co,
        x.is_contiguous(memory_format=torch.channels_last),
        cb is not None,
    )
    _ensure_context(module.driver, device)
    stream = torch.cuda.current_stream().cuda_stream
    sizes = (
        ct.c_int(x.shape[2]),
        ct.c_int(x.shape[3]),
        ct.c_int(oh),
        ct.c_int(ow),
        ct.c_longlong(n),
        ct.c_float(eps),
    )
    return module, stream, n, sizes


@triton.jit
def _tensor_core_patch(  # noqa: ANN202
    pixel,
    k,
    H: tl.constexpr,
    W: tl.constexpr,
    CI: tl.constexpr,
    CL: tl.constexpr,
    CHW: tl.constexpr,
):
    oh: tl.constexpr = (H + 1) // 2
    ow: tl.constexpr = (W + 1) // 2
    batch = pixel // (oh * ow)
    row = 2 * (pixel // ow % oh) - 1
    col = 2 * (pixel % ow) - 1
    if CHW:
        i = row[:, None] + (k // 3 % 3)[None, :]
        j = col[:, None] + (k % 3)[None, :]
        ch = k // 9
    else:
        i = row[:, None] + (k // (3 * CI))[None, :]
        j = col[:, None] + (k // CI % 3)[None, :]
        ch = k % CI
    if CL:
        offset = ((batch[:, None] * H + i) * W + j) * CI + ch[None, :]
    else:
        offset = ((batch[:, None] * CI + ch[None, :]) * H + i) * W + j
    valid = (i >= 0) & (i < H) & (j >= 0) & (j < W) & (k[None, :] < 9 * CI)
    return offset, valid


@triton.jit
def _stage0_row_patch(  # noqa: ANN202
    x, row, H: tl.constexpr, W: tl.constexpr, M: tl.constexpr, CL: tl.constexpr,
):
    oh: tl.constexpr = (H + 1) // 2
    ow: tl.constexpr = (W + 1) // 2
    batch = row // oh
    iy = 2 * (row % oh) - 1
    # Stage each physical row with contiguous lanes. The gather then assembles
    # overlapping windows from this CTA-local tile without rereading global x.
    row_size: tl.constexpr = W * 3 if CL else W
    pitch: tl.constexpr = triton.next_power_of_2(row_size)
    size: tl.constexpr = pitch * (4 if CL else 16)
    lane = tl.arange(0, size)
    line = lane // pitch
    col = lane % pitch
    if CL:
        y = iy + line
        address = (batch * H + y) * row_size + col
        lines: tl.constexpr = 3
    else:
        y = iy + line % 3
        address = ((batch * 3 + line // 3) * H + y) * W + col
        lines: tl.constexpr = 9
    tile = tl.load(x + address, (line < lines) & (y >= 0) & (y < H) & (col < row_size), 0)

    element = tl.arange(0, M * 32)
    k = element % 32
    ox = element // 32
    ix = 2 * ox - 1 + k % 3
    index = (k // 3 % 3) * pitch + 3 * ix + k // 9 if CL else (k // 3) * pitch + ix
    valid = (k < 27) & (ox < ow) & (ix >= 0) & (ix < W)
    patch = tl.gather(tile, tl.where(valid, index, 0), axis=0)
    return tl.reshape(tl.where(valid, patch, 0), (M, 32))


@triton.jit
def stem_conv_tensor_stages(  # noqa: ANN201, C901, PLR0912, PLR0915
    X,
    KERNEL,
    G,
    B,
    CB,
    Y,
    DY,
    STAT,
    PART,
    H: tl.constexpr,
    W: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    N: tl.constexpr,
    EP: tl.constexpr,
    CL: tl.constexpr,
    BIAS: tl.constexpr,
    BWD: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    TILE_REDUCE: tl.constexpr,
    CHW: tl.constexpr,
    ROW: tl.constexpr,
):
    k = tl.arange(0, K)
    c = tl.arange(0, CO)
    m = tl.arange(0, M)
    worker = tl.program_id(0)
    weights = tl.load(KERNEL + k[:, None] * CO + c[None, :], k[:, None] < 9 * CI, 0)
    gamma = tl.load(G + c)
    beta = tl.load(B + c)
    if BIAS:
        bias = tl.load(CB + c)
    if BWD:
        dw = tl.full((K, CO), 0, tl.float32)
        if TILE_REDUCE:
            dg = tl.full((M, CO), 0, tl.float32)
            db = tl.full((M, CO), 0, tl.float32)
            dc = tl.full((M, CO), 0, tl.float32)
        else:
            dg = tl.full((CO,), 0, tl.float32)
            db = tl.full((CO,), 0, tl.float32)
            dc = tl.full((CO,), 0, tl.float32)
    ow: tl.constexpr = (W + 1) // 2
    end: tl.constexpr = N // ow if ROW else N
    step: tl.constexpr = 1 if ROW else M
    stride = tl.num_programs(0) * step if BWD else end
    for start in range(worker * step, end, stride):
        if ROW:
            pixel = start * ow + m
            live = m < ow
            a = _stage0_row_patch(X, start, H, W, M, CL)
        else:
            pixel = start + m
            live = pixel < N
            offset, valid = _tensor_core_patch(pixel, k, H, W, CI, CL, CHW)
            a = tl.load(X + offset, valid & live[:, None], 0)
        values = tl.dot(a, weights).to(X.dtype.element_ty).to(tl.float32)
        if BIAS:
            values += bias[None, :]
        if BWD:
            mean = tl.load(STAT + 2 * pixel, live, 0)
            inv = tl.load(STAT + 2 * pixel + 1, live, 0)
            centered = values - mean[:, None]
        else:
            mean = tl.sum(values, 1) / CO
            centered = values - mean[:, None]
            inv = tl.rsqrt(tl.sum(centered * centered, 1) / CO + EP)
            tl.store(STAT + 2 * pixel, mean, live)
            tl.store(STAT + 2 * pixel + 1, inv, live)
        norm = centered * inv[:, None]
        affine = tl.fma(norm, gamma[None, :], beta[None, :])
        cdf = 0.5 * (1.0 + libdevice.erf(affine * 0.7071067811865476))
        output_offsets = pixel[:, None] * CO + c[None, :]
        if BWD:
            dy = tl.load(DY + output_offsets, live[:, None], 0).to(tl.float32)
            grad = dy * (cdf + affine * 0.3989422804014327 * tl.exp(-0.5 * affine * affine))
            projected = grad * gamma[None, :]
            v = (
                projected
                - tl.sum(projected, 1)[:, None] / CO
                - norm * (tl.sum(projected * norm, 1)[:, None] / CO)
            ) * inv[:, None]
            dw = tl.dot(tl.trans(a), v.to(X.dtype.element_ty), dw)
            if TILE_REDUCE:
                dg += grad * norm
                db += grad
                if BIAS:
                    dc += v
            else:
                dg += tl.sum(grad * norm, 0)
                db += tl.sum(grad, 0)
                if BIAS:
                    dc += tl.sum(v, 0)
        else:
            tl.store(Y + output_offsets, affine * cdf, live[:, None])
    if BWD:
        fields: tl.constexpr = 9 * CI + 2 + BIAS
        dst = PART + worker * fields * CO
        tl.store(dst + k[:, None] * CO + c[None, :], dw, k[:, None] < 9 * CI)
        tl.store(dst + 9 * CI * CO + c, tl.sum(dg, 0) if TILE_REDUCE else dg)
        tl.store(dst + (9 * CI + 1) * CO + c, tl.sum(db, 0) if TILE_REDUCE else db)
        if BIAS:
            tl.store(dst + (9 * CI + 2) * CO + c, tl.sum(dc, 0) if TILE_REDUCE else dc)


@triton.jit
def stem_conv_tensor_reduce(  # noqa: ANN201
    P,
    DW,
    DG,
    DB,
    DC,
    CI: tl.constexpr,
    CO: tl.constexpr,
    BIAS: tl.constexpr,
    WORKERS: tl.constexpr,
    R: tl.constexpr,
):
    fields: tl.constexpr = 9 * CI + 2 + BIAS
    k = tl.program_id(0)
    worker = tl.arange(0, R)
    c = tl.arange(0, CO)
    v = tl.sum(
        tl.load(P + (worker[:, None] * fields + k) * CO + c[None, :], worker[:, None] < WORKERS, 0),
        0,
    )
    if k < CI * 9:
        tl.store(DW + k * CO + c, v)
    elif k == CI * 9:
        tl.store(DG + c, v)
    elif k == CI * 9 + 1:
        tl.store(DB + c, v)
    elif BIAS:
        tl.store(DC + c, v)


def _launch_tensor_core_stem(
    x: Tensor,
    w: Tensor,
    g: Tensor,
    b: Tensor,
    cb: Tensor | None,
    y: Tensor,
    dy: Tensor,
    stats: Tensor,
    part: Tensor,
    eps: float,
    workers: int,
    *,
    backward: bool,
    block: int = 32,
    warps: int = 4,
    tile_reduce: bool = True,
    mode: str = "off",
) -> None:
    ci, h, width = x.shape[1:]
    co = w.shape[-1]
    n = x.shape[0] * ((h + 1) // 2) * ((width + 1) // 2)
    cl = x.is_contiguous(memory_format=torch.channels_last)
    stem_conv_tensor_stages[(workers,)](
        x,
        w,
        g,
        b,
        cb if cb is not None else g,
        y,
        dy,
        stats,
        part,
        h,
        width,
        ci,
        co,
        n,
        eps,
        cl,
        cb is not None,
        backward,
        block,
        triton.next_power_of_2(ci * 9),
        tile_reduce,
        mode != "off",
        mode == "triton_row_coalesced",
        num_warps=warps,
    )


def _tensor_core_tiles(x: Tensor, mode: str, *, backward: bool) -> tuple[int, int]:
    ow = (x.shape[3] + 1) // 2
    rows = x.shape[0] * ((x.shape[2] + 1) // 2)
    if mode == "triton_row_coalesced":
        return triton.next_power_of_2(ow), rows
    block = (64 if backward else 32) if mode == "off" else 128
    return block, triton.cdiv(rows * ow, block)


def _use_tensor_core_stem(x: Tensor, w: Tensor) -> bool:
    # This geometry is the first block in the three target models.
    return x.shape[1] == 3 and w.shape[-1] == 32


@torch.library.custom_op("va_stem::conv_forward", mutates_args=(), device_types="cuda")
def _conv_forward(
    x: Tensor, w: Tensor, g: Tensor, b: Tensor, cb: Tensor | None, eps: float, mode: str = "off"
) -> tuple[Tensor, Tensor]:
    y, stats = _conv_forward_fake(x, w, g, b, cb, eps)
    if not y.numel():
        return y, stats
    with torch.cuda.device(x.device):
        if _use_tensor_core_stem(x, w):
            block, workers = _tensor_core_tiles(x, mode, backward=False)
            _launch_tensor_core_stem(
                x, w, g, b, cb, y, y, stats, stats, eps, workers,
                backward=False, block=block, mode=mode,
            )
        else:
            module, stream, n, sizes = _conv_launch_setup(x, w, cb, eps)
            blocks = min(_MAX_FORWARD_BLOCKS, (n + 3) // 4)
            _launch(module, stream, "conv_stem_forward", blocks, (x, w, g, b, cb, y, stats), *sizes)
    return y, stats


@_conv_forward.register_fake
def _conv_forward_fake(
    x: Tensor,
    w: Tensor,
    g: Tensor,  # noqa: ARG001
    b: Tensor,  # noqa: ARG001
    cb: Tensor | None,  # noqa: ARG001
    eps: float,  # noqa: ARG001
    mode: str = "off",  # noqa: ARG001
) -> tuple[Tensor, Tensor]:
    shape = _conv_shape(x, w)
    return x.new_empty(shape), x.new_empty((*shape[:-1], 2), dtype=torch.float32)


@cache
def _conv_workers(module: _CudaModule, device: torch.device) -> int:
    resident = ct.c_int()
    _check_cu(
        module.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ct.byref(resident), module.functions["conv_stem_backward"], _THREADS, ct.c_size_t(0)
        ),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    )
    return resident.value * torch.cuda.get_device_properties(device).multi_processor_count


def _tensor_core_conv_backward(
    x: Tensor,
    w: Tensor,
    g: Tensor,
    b: Tensor,
    cb: Tensor | None,
    dy: Tensor,
    stats: Tensor,
    eps: float,
    dw: Tensor,
    dg: Tensor,
    db: Tensor,
    dc: Tensor,
    mode: str,
) -> None:
    block, tiles = _tensor_core_tiles(x, mode, backward=True)
    workers = min(1024 if mode == "off" else 528, tiles)
    ci, co = x.shape[1], g.numel()
    fields = ci * 9 + 2 + int(cb is not None)
    part = g.new_empty((workers, fields, co))
    _launch_tensor_core_stem(
        x,
        w,
        g,
        b,
        cb,
        dy,
        dy,
        stats,
        part,
        eps,
        workers,
        backward=True,
        block=block,
        tile_reduce=mode == "off",
        mode=mode,
    )
    stem_conv_tensor_reduce[(fields,)](
        part,
        dw,
        dg,
        db,
        dc,
        ci,
        co,
        cb is not None,
        workers,
        triton.next_power_of_2(workers),
    )


@torch.library.custom_op("va_stem::conv_backward", mutates_args=(), device_types="cuda")
def _conv_backward(
    x: Tensor,
    w: Tensor,
    g: Tensor,
    b: Tensor,
    cb: Tensor | None,
    dy: Tensor,
    stats: Tensor,
    eps: float,
    mode: str = "off",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not dy.numel():
        return (
            torch.zeros_like(w),
            torch.zeros_like(g),
            torch.zeros_like(b),
            torch.zeros_like(cb) if cb is not None else g.new_empty(0),
        )
    dw, dg, db, dc = _conv_backward_fake(x, w, g, b, cb, dy, stats, eps)
    with torch.cuda.device(x.device):
        if _use_tensor_core_stem(x, w):
            _tensor_core_conv_backward(x, w, g, b, cb, dy, stats, eps, dw, dg, db, dc, mode)
            return dw, dg, db, dc
        module, stream, n, sizes = _conv_launch_setup(x, w, cb, eps)
        blocks = min(_conv_workers(module, x.device), (n + 3) // 4)
        workers = blocks * 4
        fields = w.numel() + (2 + int(cb is not None)) * g.numel()
        partial = g.new_empty((workers, fields))
        _launch(
            module,
            stream,
            "conv_stem_backward",
            blocks,
            (x, w, g, b, cb, dy, stats, partial),
            *sizes,
        )
        _launch(
            module,
            stream,
            "conv_stem_reduce",
            (fields + _THREADS - 1) // _THREADS,
            (partial, dw, dg, db, dc),
            ct.c_int(workers),
        )
    return dw, dg, db, dc


@_conv_backward.register_fake
def _conv_backward_fake(
    x: Tensor,  # noqa: ARG001
    w: Tensor,
    g: Tensor,
    b: Tensor,
    cb: Tensor | None,
    dy: Tensor,  # noqa: ARG001
    stats: Tensor,  # noqa: ARG001
    eps: float,  # noqa: ARG001
    mode: str = "off",  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(w),
        torch.empty_like(g),
        torch.empty_like(b),
        torch.empty_like(cb) if cb is not None else g.new_empty(0),
    )


def _conv_save(ctx, inputs, output) -> None:
    x, w, g, b, cb, ctx.eps, ctx.mode = inputs
    _, stats = output
    ctx.mark_non_differentiable(stats)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x, w, g, b, cb, stats)


def _conv_autograd(
    ctx,
    dy: Tensor | None,
    dstats: Tensor | None,  # noqa: ARG001
) -> tuple[None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, None, None]:
    if dy is None:
        return None, None, None, None, None, None, None
    x, w, g, b, cb, stats = ctx.saved_tensors
    dw, dg, db, dc = _conv_backward(x, w, g, b, cb, dy.contiguous(), stats, ctx.eps, ctx.mode)
    return None, dw, dg, db, dc if cb is not None else None, None, None


_conv_forward.register_autograd(_conv_autograd, setup_context=_conv_save)


@torch.library.custom_op("va_stem::stage0_cudnn_conv", mutates_args=(), device_types="cuda")
def _stage0_cudnn_conv(x: Tensor, w: Tensor, bias: Tensor | None) -> Tensor:
    result = functional.conv2d(x, w, bias, stride=2, padding=1)
    return result.contiguous(memory_format=torch.channels_last)


@_stage0_cudnn_conv.register_fake
def _stage0_cudnn_conv_fake(x: Tensor, w: Tensor, bias: Tensor | None) -> Tensor:  # noqa: ARG001
    shape = (x.shape[0], w.shape[0], (x.shape[2] + 1) // 2, (x.shape[3] + 1) // 2)
    return torch.empty(shape, dtype=x.dtype, device=x.device, memory_format=torch.channels_last)


@torch.library.custom_op("va_stem::stage0_cudnn_grad", mutates_args=(), device_types="cuda")
def _stage0_cudnn_grad(
    dy: Tensor, x: Tensor, w: Tensor, needs_dx: bool, needs_dw: bool, needs_dc: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    dx, dw, dc = torch.ops.aten.convolution_backward.default(
        dy, x, w, [w.shape[0]] if needs_dc else None, [2, 2], [1, 1],
        [1, 1], False, [0, 0], 1, (needs_dx, needs_dw, needs_dc),  # noqa: FBT003
    )
    empty = x.new_empty(0)
    return (
        dx.contiguous(memory_format=torch.channels_last) if needs_dx else empty,
        dw.contiguous(memory_format=torch.channels_last) if needs_dw else x.new_empty(0),
        dc if needs_dc else x.new_empty(0),
    )


@_stage0_cudnn_grad.register_fake
def _stage0_cudnn_grad_fake(
    dy: Tensor,  # noqa: ARG001
    x: Tensor, w: Tensor, needs_dx: bool, needs_dw: bool, needs_dc: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(x, memory_format=torch.channels_last) if needs_dx else x.new_empty(0),
        torch.empty_like(w) if needs_dw else x.new_empty(0),
        x.new_empty(w.shape[0] if needs_dc else 0),
    )


def _stage0_cudnn_save(ctx, inputs, output) -> None:  # noqa: ARG001
    x, w, _ = inputs
    ctx.save_for_backward(x, w)


def _stage0_cudnn_autograd(ctx, dy: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    x, w = ctx.saved_tensors
    needs_dx, needs_dw, needs_dc = ctx.needs_input_grad
    dx, dw, dc = _stage0_cudnn_grad(dy, x, w, needs_dx, needs_dw, needs_dc)
    return dx if needs_dx else None, dw if needs_dw else None, dc if needs_dc else None


_stage0_cudnn_conv.register_autograd(_stage0_cudnn_autograd, setup_context=_stage0_cudnn_save)


def _stage0_cudnn_fused(
    x: Tensor, w: Tensor, cb: Tensor | None, gamma: Tensor, beta: Tensor,
    eps: float, dtype: torch.dtype,
) -> Tensor:
    weight = w.to(dtype=dtype, memory_format=torch.channels_last)
    values = _stage0_cudnn_conv(x.to(dtype), weight, cb.to(dtype) if cb is not None else None)
    pixels = values.permute(0, 2, 3, 1).contiguous()
    result = fused_nhwc_layernorm_gelu(pixels, gamma, beta, eps, nchw=False, output_dtype=dtype)
    return result.permute(0, 3, 1, 2)


def fused_conv_stem(
    inputs: Tensor,
    weight: Tensor,
    conv_bias: Tensor | None,
    gamma: Tensor,
    beta: Tensor,
    eps: float,
    dtype: torch.dtype,
    *,
    tiled_geometry: tuple[int, int, int, int] | None = None,
    mode: str = "off",
) -> Tensor:
    """Run the small-input stem with Wave 21C rounding boundaries."""
    mode = resolve_stage0_candidate(mode, needs_dx=inputs.requires_grad, width=inputs.shape[3])
    if mode == "cudnn_fused":
        return _stage0_cudnn_fused(inputs, weight, conv_bias, gamma, beta, eps, dtype)
    co, ci = weight.shape[:2]
    order = (1, 2, 3, 0) if mode != "off" else (2, 3, 1, 0)
    packed = weight.to(dtype).permute(order).reshape(9 * ci, co).contiguous()
    args = (
        inputs.to(dtype),
        packed,
        gamma.float().contiguous(),
        beta.float().contiguous(),
        conv_bias.float().contiguous() if conv_bias is not None else None,
        eps,
    )
    if tiled_geometry is None:
        y, _ = _conv_forward(*args, mode)
    else:
        y, _ = _tiled_forward(*args, *tiled_geometry)
    return y.permute(0, 3, 1, 2)


# Four warps share a 4x8 tile. Each lane owns one output channel and its
# weight-gradient accumulators throughout the persistent backward loop.
_TILED_CONV_CUDA = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T = SCALAR;
constexpr int th = SH <= 4 ? 4 : 1, tw = SW <= 4 ? 8 : 1;
constexpr int hh = (th - 1) * SH + 3, hw = (tw - 1) * SW + 3;
constexpr int taps = 9 * CI, fields = taps + 2 + HAS_BIAS;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset; offset /= 2)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

template<bool Backward>
__device__ __forceinline__ void tiled(
    const T* x, const T* w, const float* g, const float* b, const float* cb,
    const T* dy, T* y, float2* stats, float* partial,
    int h, int wi, int oh, int ow, long long n, float eps, int ph, int pw
) {
    __shared__ float halo[hh * hw * CI];
    __shared__ float weights[taps][32];
    __shared__ float scratch[4][32];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    for (int i = tid; i < taps * 32; i += 128)
        weights[i / 32][i % 32] = float(w[i]);
    const float gamma = g[lane], beta = b[lane];
    const float bias = HAS_BIAS ? cb[lane] : 0.f;
    float dw[taps] = {}, dg = 0.f, db = 0.f, dc = 0.f;
    const int tiles_h = (oh + th - 1) / th, tiles_w = (ow + tw - 1) / tw;
    const int image_tiles = tiles_h * tiles_w;
    const long long tiles = n / (static_cast<long long>(oh) * ow) * image_tiles;
    for (long long tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const long long batch = tile / image_tiles;
        const int local = tile % image_tiles;
        const int row = local / tiles_w * th, col = local % tiles_w * tw;
        for (int i = tid; i < hh * hw * CI; i += 128) {
            const int r = SH * row - ph + i / (hw * CI);
            const int s = SW * col - pw + i / CI % hw;
            const int ci = i % CI;
            const long long offset = CHANNELS_LAST
                ? ((batch * h + r) * wi + s) * CI + ci
                : ((batch * CI + ci) * h + r) * wi + s;
            halo[i] = r >= 0 && s >= 0 && r < h && s < wi ? float(x[offset]) : 0.f;
        }
        __syncthreads();
        for (int p = warp; p < th * tw; p += 4) {
            const int r = row + p / tw, s = col + p % tw;
            if (r >= oh || s >= ow) continue;
            const long long pixel = (batch * oh + r) * ow + s;
            const int origin = (SH * (p / tw) * hw + SW * (p % tw)) * CI;
            float value = 0.f;
            #pragma unroll
            for (int k = 0; k < taps; ++k) {
                const int offset = (k / (3 * CI) * hw + k / CI % 3) * CI + k % CI;
                value = fmaf(halo[origin + offset], weights[k][lane], value);
            }
            value = float(T(value)) + bias;
            float mean, inverse;
            if constexpr (Backward) {
                const float2 stat = stats[pixel];
                mean = stat.x;
                inverse = stat.y;
            } else {
                mean = warp_sum(value) / 32.f;
                const float centered = value - mean;
                inverse = rsqrtf(warp_sum(centered * centered) / 32.f + eps);
                if (lane == 0) stats[pixel] = make_float2(mean, inverse);
            }
            const float norm = (value - mean) * inverse;
            const float affine = fmaf(norm, gamma, beta);
            const float cdf = .5f * (1.f + erff(affine * .7071067811865476f));
            if constexpr (Backward) {
                const float grad = float(dy[pixel * 32 + lane])
                    * (cdf + affine * .3989422804014327f * expf(-.5f * affine * affine));
                const float projected = grad * gamma;
                const float delta = (projected - warp_sum(projected) / 32.f
                    - norm * (warp_sum(projected * norm) / 32.f)) * inverse;
                dg = fmaf(grad, norm, dg);
                db += grad;
                dc += delta;
                const float rounded = float(T(delta));
                #pragma unroll
                for (int k = 0; k < taps; ++k) {
                    const int offset = (k / (3 * CI) * hw + k / CI % 3) * CI + k % CI;
                    dw[k] = fmaf(halo[origin + offset], rounded, dw[k]);
                }
            } else {
                y[pixel * 32 + lane] = T(affine * cdf);
            }
        }
        __syncthreads();
    }
    if constexpr (Backward) {
        #pragma unroll
        for (int k = 0; k < fields; ++k) {
            scratch[warp][lane] = k < taps ? dw[k] : (k == taps ? dg : (k == taps + 1 ? db : dc));
            __syncthreads();
            if (warp == 0)
                partial[(blockIdx.x * fields + k) * 32 + lane] =
                    (scratch[0][lane] + scratch[1][lane]) + (scratch[2][lane] + scratch[3][lane]);
            __syncthreads();
        }
    }
}

extern "C" __global__ void stem_tiled_forward(
    const T* x, const T* w, const float* g, const float* b, const float* cb,
    T* y, float2* stats, int h, int wi, int oh, int ow, long long n, float eps, int ph, int pw
) {
    tiled<false>(x, w, g, b, cb, nullptr, y, stats, nullptr, h, wi, oh, ow, n, eps, ph, pw);
}

extern "C" __global__ void stem_tiled_backward(
    const T* x, const T* w, const float* g, const float* b, const float* cb,
    const T* dy, float2* stats, float* partial,
    int h, int wi, int oh, int ow, long long n, float eps, int ph, int pw
) {
    tiled<true>(x, w, g, b, cb, dy, nullptr, stats, partial, h, wi, oh, ow, n, eps, ph, pw);
}

extern "C" __global__ void stem_tiled_reduce(
    const float* partial, T* dw, float* dg, float* db, float* dc, int workers
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fields * 32) return;
    float sum = 0.f;
    for (int j = 0; j < workers; ++j)
        sum += partial[j * fields * 32 + i];
    const int k = i / 32, c = i % 32;
    if (k < taps) dw[i] = T(sum);
    else if (k == taps) dg[c] = sum;
    else if (k == taps + 1) db[c] = sum;
    else if constexpr (HAS_BIAS) dc[c] = sum;
}
"""


def _compile_tiled_stem_ptx(
    capability: tuple[int, int], dtype: torch.dtype, ci: int,
    channels_last: bool, has_bias: bool, sh: int = 2, sw: int = 2,
) -> ct.Array:
    header = (
        f"#define SCALAR {_TYPES[dtype]}\n#define CI {ci}\n"
        f"#define CHANNELS_LAST {int(channels_last)}\n#define HAS_BIAS {int(has_bias)}\n"
        f"#define SH {sh}\n#define SW {sw}\n"
    )
    return _compile_source(capability, header + _TILED_CONV_CUDA)


@cache
def _tiled_stem_module(
    device: int, dtype: torch.dtype, ci: int,
    channels_last: bool, has_bias: bool, sh: int, sw: int,
) -> _CudaModule:
    image = _compile_tiled_stem_ptx(
        torch.cuda.get_device_capability(device), dtype, ci, channels_last, has_bias, sh, sw
    )
    module = _CudaModule(image, device)
    for name in ("stem_tiled_forward", "stem_tiled_backward", "stem_tiled_reduce"):
        module.load_function(name, 0)
    return module


STAGE0_CANDIDATES = ("cudnn_fused", "triton_row_coalesced", "triton_tuned_v2")


def stem_stage0_mode() -> str:
    mode = os.environ.get("LNET_STEM_STAGE0_MODE", "off")
    modes = ("off", "cuda_tiled", "auto", *STAGE0_CANDIDATES)
    if mode not in modes:
        raise ValueError(f"LNET_STEM_STAGE0_MODE must be one of {', '.join(modes)}")
    return mode


def tiled_stem_enabled() -> bool:
    # #854: the tiled stage-0 stem cost +6.31 ms on H200.
    return stem_stage0_mode() == "cuda_tiled"


def resolve_stage0_candidate(mode: str, *, needs_dx: bool, width: int) -> str:
    if mode in STAGE0_CANDIDATES[1:] and needs_dx:
        return "cudnn_fused"
    if mode == "triton_row_coalesced" and not 31 < width <= 256:
        return "triton_tuned_v2"
    return mode


def _tiled_stem_shape(x: Tensor, sh: int, sw: int, ph: int, pw: int) -> tuple:
    oh = (x.shape[2] + 2 * ph - 3) // sh + 1
    ow = (x.shape[3] + 2 * pw - 3) // sw + 1
    return x.shape[0], oh, ow, 32


def _tiled_stem_setup(
    x: Tensor, cb: Tensor | None, eps: float, sh: int, sw: int, ph: int, pw: int,
) -> tuple:
    batch, oh, ow, _ = _tiled_stem_shape(x, sh, sw, ph, pw)
    device = x.device.index or 0
    module = _tiled_stem_module(
        device, x.dtype, x.shape[1], x.is_contiguous(memory_format=torch.channels_last),
        cb is not None, sh, sw,
    )
    _ensure_context(module.driver, device)
    tiles = batch * triton.cdiv(oh, 4 if sh <= 4 else 1) * triton.cdiv(ow, 8 if sw <= 4 else 1)
    sizes = (
        ct.c_int(x.shape[2]), ct.c_int(x.shape[3]), ct.c_int(oh), ct.c_int(ow),
        ct.c_longlong(batch * oh * ow), ct.c_float(eps), ct.c_int(ph), ct.c_int(pw),
    )
    return module, torch.cuda.current_stream(x.device).cuda_stream, tiles, sizes


@torch.library.custom_op("va_stem::tiled_forward", mutates_args=(), device_types="cuda")
def _tiled_forward(
    x: Tensor, w: Tensor, g: Tensor, b: Tensor, cb: Tensor | None,
    eps: float, sh: int, sw: int, ph: int, pw: int,
) -> tuple[Tensor, Tensor]:
    y, stats = _tiled_forward_fake(x, w, g, b, cb, eps, sh, sw, ph, pw)
    if y.numel():
        with torch.cuda.device(x.device):
            module, stream, tiles, sizes = _tiled_stem_setup(x, cb, eps, sh, sw, ph, pw)
            _launch(
                module, stream, "stem_tiled_forward", min(tiles, _MAX_FORWARD_BLOCKS),
                (x, w, g, b, cb, y, stats), *sizes,
            )
    return y, stats


@_tiled_forward.register_fake
def _tiled_forward_fake(
    x: Tensor, w: Tensor, g: Tensor, b: Tensor, cb: Tensor | None,  # noqa: ARG001
    eps: float, sh: int, sw: int, ph: int, pw: int,  # noqa: ARG001
) -> tuple[Tensor, Tensor]:
    shape = _tiled_stem_shape(x, sh, sw, ph, pw)
    return x.new_empty(shape), x.new_empty((*shape[:-1], 2), dtype=torch.float32)


@torch.library.custom_op("va_stem::tiled_backward", mutates_args=(), device_types="cuda")
def _tiled_backward(
    x: Tensor, w: Tensor, g: Tensor, b: Tensor, cb: Tensor | None,
    dy: Tensor, stats: Tensor, eps: float, sh: int, sw: int, ph: int, pw: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not dy.numel():
        return (
            torch.zeros_like(w), torch.zeros_like(g), torch.zeros_like(b),
            torch.zeros_like(cb) if cb is not None else g.new_empty(0),
        )
    dw, dg, db, dc = _conv_backward_fake(x, w, g, b, cb, dy, stats, eps)
    with torch.cuda.device(x.device):
        module, stream, tiles, sizes = _tiled_stem_setup(x, cb, eps, sh, sw, ph, pw)
        workers = min(tiles, 1024)
        fields = 9 * x.shape[1] + 2 + int(cb is not None)
        partial = g.new_empty((workers, fields, 32))
        _launch(
            module, stream, "stem_tiled_backward", workers,
            (x, w, g, b, cb, dy, stats, partial), *sizes,
        )
        _launch(
            module, stream, "stem_tiled_reduce", triton.cdiv(fields * 32, _THREADS),
            (partial, dw, dg, db, dc), ct.c_int(workers),
        )
    return dw, dg, db, dc


@_tiled_backward.register_fake
def _tiled_backward_fake(
    x: Tensor, w: Tensor, g: Tensor, b: Tensor, cb: Tensor | None,
    dy: Tensor, stats: Tensor, eps: float, sh: int, sw: int, ph: int, pw: int,  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _conv_backward_fake(x, w, g, b, cb, dy, stats, eps)


def _tiled_save(ctx, inputs, output) -> None:
    x, w, g, b, cb, *ctx.geometry = inputs
    _, stats = output
    ctx.mark_non_differentiable(stats)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x, w, g, b, cb, stats)


def _tiled_autograd(ctx, dy: Tensor | None, dstats: Tensor | None) -> tuple:  # noqa: ARG001
    if dy is None:
        return (None,) * 10
    x, w, g, b, cb, stats = ctx.saved_tensors
    dw, dg, db, dc = _tiled_backward(x, w, g, b, cb, dy.contiguous(), stats, *ctx.geometry)
    return None, dw, dg, db, dc if cb is not None else None, None, None, None, None, None


_tiled_forward.register_autograd(_tiled_autograd, setup_context=_tiled_save)
