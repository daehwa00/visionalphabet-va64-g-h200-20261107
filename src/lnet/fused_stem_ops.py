"""Fused, exact-GELU LayerNorm over physically contiguous pixel channels."""

from __future__ import annotations

# Custom op schemas use positional flags.
# ruff: noqa: ANN001, FBT001, N803
import ctypes as ct
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
    # One resident wave covers the SMs while bounding the affine partial buffer.
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
        major, _ = torch.cuda.get_device_capability(x.device)
        needs_bf16 = torch.bfloat16 in (x.dtype, output_dtype)
        if major >= (8 if needs_bf16 else 7):
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
):
    oh: tl.constexpr = (H + 1) // 2
    ow: tl.constexpr = (W + 1) // 2
    batch = pixel // (oh * ow)
    row = 2 * (pixel // ow % oh) - 1
    col = 2 * (pixel % ow) - 1
    # HWC patch order makes adjacent K lanes coalesced for training images.
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
):
    hybrid: tl.constexpr = CI == 32
    k = tl.arange(0, K)
    c = tl.arange(0, CO)
    m = tl.arange(0, M)
    worker = tl.program_id(0)
    if not hybrid:
        weights = tl.load(KERNEL + k[:, None] * CO + c[None, :], k[:, None] < 9 * CI, 0)
    gamma = tl.load(G + c)
    beta = tl.load(B + c)
    if BIAS:
        bias = tl.load(CB + c)
    if BWD:
        if not hybrid:
            dw = tl.full((K, CO), 0, tl.float32)
        if TILE_REDUCE:
            dg = tl.full((M, CO), 0, tl.float32)
            db = tl.full((M, CO), 0, tl.float32)
            dc = tl.full((M, CO), 0, tl.float32)
        else:
            dg = tl.full((CO,), 0, tl.float32)
            db = tl.full((CO,), 0, tl.float32)
            dc = tl.full((CO,), 0, tl.float32)
        stride = tl.num_programs(0) * M
    else:
        stride = N
    for start in range(worker * M, N, stride):
        pixel = start + m
        if hybrid:
            oh: tl.constexpr = (H + 1) // 2
            ow: tl.constexpr = (W + 1) // 2
            batch = pixel // (oh * ow)
            row = 2 * (pixel // ow % oh) - 1
            col = 2 * (pixel % ow) - 1
            acc = tl.full((M, CO), 0, tl.float32)
            for tap in range(9):
                i, j = row + tap // 3, col + tap % 3
                valid = (pixel < N) & (i >= 0) & (i < H) & (j >= 0) & (j < W)
                if CL:
                    offset = ((batch * H + i) * W + j)[:, None] * CI + k[None, :]
                else:
                    offset = ((batch[:, None] * CI + k[None, :]) * H + i[:, None]) * W + j[:, None]
                a = tl.load(X + offset, valid[:, None], 0)
                weights = tl.load(KERNEL + (tap * CI + k[:, None]) * CO + c[None, :])
                acc = tl.dot(a, weights, acc)
        else:
            offset, valid = _tensor_core_patch(pixel, k, H, W, CI, CL)
            a = tl.load(X + offset, valid & (pixel[:, None] < N), 0)
            acc = tl.dot(a, weights)
        values = acc.to(X.dtype.element_ty).to(tl.float32)
        if BIAS:
            values += bias[None, :]
        if BWD:
            mean = tl.load(STAT + 2 * pixel, pixel < N, 0)
            inv = tl.load(STAT + 2 * pixel + 1, pixel < N, 0)
            centered = values - mean[:, None]
        else:
            mean = tl.sum(values, 1) / CO
            centered = values - mean[:, None]
            inv = tl.rsqrt(tl.sum(centered * centered, 1) / CO + EP)
            tl.store(STAT + 2 * pixel, mean, pixel < N)
            tl.store(STAT + 2 * pixel + 1, inv, pixel < N)
        norm = centered * inv[:, None]
        affine = tl.fma(norm, gamma[None, :], beta[None, :])
        cdf = 0.5 * (1.0 + libdevice.erf(affine * 0.7071067811865476))
        output_offsets = pixel[:, None] * CO + c[None, :]
        if BWD:
            dy = tl.load(DY + output_offsets, pixel[:, None] < N, 0).to(tl.float32)
            grad = dy * (cdf + affine * 0.3989422804014327 * tl.exp(-0.5 * affine * affine))
            projected = grad * gamma[None, :]
            v = (
                projected
                - tl.sum(projected, 1)[:, None] / CO
                - norm * (tl.sum(projected * norm, 1)[:, None] / CO)
            ) * inv[:, None]
            if hybrid:
                tl.store(Y + output_offsets, v, pixel[:, None] < N)
                grad = tl.where(pixel[:, None] < N, grad, 0)
                v = tl.where(pixel[:, None] < N, v, 0)
            else:
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
            tl.store(Y + output_offsets, affine * cdf, pixel[:, None] < N)
    if BWD:
        taps: tl.constexpr = 0 if hybrid else 9 * CI
        fields: tl.constexpr = taps + 2 + BIAS
        dst = PART + worker * fields * CO
        if not hybrid:
            tl.store(dst + k[:, None] * CO + c[None, :], dw, k[:, None] < taps)
        tl.store(dst + taps * CO + c, tl.sum(dg, 0) if TILE_REDUCE else dg)
        tl.store(dst + (taps + 1) * CO + c, tl.sum(db, 0) if TILE_REDUCE else db)
        if BIAS:
            tl.store(dst + (taps + 2) * CO + c, tl.sum(dc, 0) if TILE_REDUCE else dc)


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
    stages: int = 3,
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
        32 if ci == 32 else triton.next_power_of_2(ci * 9),
        tile_reduce,
        num_warps=warps,
        num_stages=stages,
    )


def _use_tensor_core_stem(x: Tensor, w: Tensor) -> bool:
    # This geometry is the first block in the three target models.
    return x.shape[1] == 3 and w.shape[-1] == 32


def _stem_schedule(
    mode: str, *, backward: bool, channels_last: bool = True
) -> tuple[int, int, int, bool, int]:
    # M tile, warps, software pipeline stages, lane-local reduction, target waves.
    if mode == "stage0_v2":
        return (32, 2, 2, False, 8) if backward else (128, 8, 3, True, 0)
    if mode in ("stage0_opt", "stage01"):
        return (32, 4, 2, False, 6) if backward else (64, 4, 2, True, 0)
    if mode == "hybrid":
        # A shorter NCHW forward pipeline removes Hopper's 8-byte stack frame.
        return (32, 4, 3, False, 8) if backward else (32, 4, 3 if channels_last else 2, True, 0)
    return (64, 4, 3, True, 0) if backward else (32, 4, 3, True, 0)


@torch.library.custom_op("va_stem::conv_forward", mutates_args=(), device_types="cuda")
def _conv_forward(
    x: Tensor, w: Tensor, g: Tensor, b: Tensor, cb: Tensor | None, eps: float, mode: str
) -> tuple[Tensor, Tensor]:
    y, stats = _conv_forward_fake(x, w, g, b, cb, eps, mode)
    if not y.numel():
        return y, stats
    with torch.cuda.device(x.device):
        if mode == "hybrid" or _use_tensor_core_stem(x, w):
            block, warps, stages, _, _ = _stem_schedule(
                mode,
                backward=False,
                channels_last=x.is_contiguous(memory_format=torch.channels_last),
            )
            workers = triton.cdiv(stats.numel() // 2, block)
            _launch_tensor_core_stem(
                x,
                w,
                g,
                b,
                cb,
                y,
                y,
                stats,
                stats,
                eps,
                workers,
                backward=False,
                block=block,
                warps=warps,
                stages=stages,
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
    mode: str,  # noqa: ARG001
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
    dconv: Tensor | None = None,
) -> None:
    block, warps, stages, tile_reduce, waves = _stem_schedule(mode, backward=True)
    limit = (
        waves * torch.cuda.get_device_properties(x.device).multi_processor_count if waves else 1024
    )
    workers = min(limit, triton.cdiv(stats.numel() // 2, block))
    ci, co = (0 if dconv is not None else x.shape[1]), g.numel()
    fields = ci * 9 + 2 + int(cb is not None)
    part = g.new_empty((workers, fields, co))
    _launch_tensor_core_stem(
        x,
        w,
        g,
        b,
        cb,
        dconv if dconv is not None else dy,
        dy,
        stats,
        part,
        eps,
        workers,
        backward=True,
        block=block,
        warps=warps,
        stages=stages,
        tile_reduce=tile_reduce,
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
        num_warps=8 if workers > 1024 else 4,
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
    mode: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not dy.numel():
        return (
            torch.zeros_like(w),
            torch.zeros_like(g),
            torch.zeros_like(b),
            torch.zeros_like(cb) if cb is not None else g.new_empty(0),
        )
    dw, dg, db, dc = _conv_backward_fake(x, w, g, b, cb, dy, stats, eps, mode)
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
    mode: str,  # noqa: ARG001
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


@torch.library.custom_op("va_stem::hybrid_backward", mutates_args=(), device_types="cuda")
def _hybrid_backward(
    x: Tensor,
    w: Tensor,
    g: Tensor,
    b: Tensor,
    cb: Tensor | None,
    dy: Tensor,
    stats: Tensor,
    eps: float,
    input_grad: bool,
    weight_grad: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if not dy.numel():
        return (
            torch.zeros_like(x) if input_grad else x.new_empty(0),
            torch.zeros_like(w) if weight_grad else w.new_empty(0),
            torch.zeros_like(g),
            torch.zeros_like(b),
            torch.zeros_like(cb) if cb is not None else g.new_empty(0),
        )
    dg, db = torch.empty_like(g), torch.empty_like(b)
    dc = torch.empty_like(cb) if cb is not None else g.new_empty(0)
    dconv = torch.empty_like(dy)
    with torch.cuda.device(x.device):
        _tensor_core_conv_backward(x, w, g, b, cb, dy, stats, eps, dg, dg, db, dc, "hybrid", dconv)
    native_weight = w.view(3, 3, x.shape[1], g.numel()).permute(3, 2, 0, 1)
    dx = dw = None
    if input_grad or weight_grad:
        dx, dw, _ = torch.ops.aten.convolution_backward.default(
            dconv.permute(0, 3, 1, 2),
            x,
            native_weight,
            None,
            [2, 2],
            [1, 1],
            [1, 1],
            False,  # noqa: FBT003
            [0, 0],
            1,
            [input_grad, weight_grad, False],
        )
    memory_format = (
        torch.channels_last
        if x.is_contiguous(memory_format=torch.channels_last)
        else torch.contiguous_format
    )
    return (
        dx.contiguous(memory_format=memory_format) if input_grad else x.new_empty(0),
        dw.permute(2, 3, 1, 0).reshape_as(w).contiguous() if weight_grad else w.new_empty(0),
        dg,
        db,
        dc,
    )


@_hybrid_backward.register_fake
def _hybrid_backward_fake(
    x: Tensor,
    w: Tensor,
    g: Tensor,
    b: Tensor,
    cb: Tensor | None,
    dy: Tensor,  # noqa: ARG001
    stats: Tensor,  # noqa: ARG001
    eps: float,  # noqa: ARG001
    input_grad: bool,
    weight_grad: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(x) if input_grad else x.new_empty(0),
        torch.empty_like(w) if weight_grad else w.new_empty(0),
        torch.empty_like(g),
        torch.empty_like(b),
        torch.empty_like(cb) if cb is not None else g.new_empty(0),
    )


def _conv_autograd(
    ctx,
    dy: Tensor | None,
    dstats: Tensor | None,  # noqa: ARG001
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, None, None]:
    if dy is None:
        return None, None, None, None, None, None, None
    x, w, g, b, cb, stats = ctx.saved_tensors
    dx = None
    if ctx.mode == "hybrid":
        input_grad, weight_grad = ctx.needs_input_grad[:2]
        dx, dw, dg, db, dc = _hybrid_backward(
            x, w, g, b, cb, dy.contiguous(), stats, ctx.eps, input_grad, weight_grad
        )
        dx = dx if input_grad else None
        dw = dw if weight_grad else None
    else:
        dw, dg, db, dc = _conv_backward(x, w, g, b, cb, dy.contiguous(), stats, ctx.eps, ctx.mode)
    return dx, dw, dg, db, dc if cb is not None else None, None, None


_conv_forward.register_autograd(_conv_autograd, setup_context=_conv_save)


def fused_conv_stem(
    inputs: Tensor,
    weight: Tensor,
    conv_bias: Tensor | None,
    gamma: Tensor,
    beta: Tensor,
    eps: float,
    dtype: torch.dtype,
    *,
    mode: str = "off",
    nhwc_output: bool = False,
) -> Tensor:
    """Run the small-input stride-two stem with Wave 21C rounding boundaries."""
    co, ci = weight.shape[:2]
    packed = weight.to(dtype).permute(2, 3, 1, 0).reshape(9 * ci, co).contiguous()
    y, _ = _conv_forward(
        inputs.to(dtype),
        packed,
        gamma.float().contiguous(),
        beta.float().contiguous(),
        conv_bias.float().contiguous() if conv_bias is not None else None,
        eps,
        mode,
    )
    return y if nhwc_output else y.permute(0, 3, 1, 2)
