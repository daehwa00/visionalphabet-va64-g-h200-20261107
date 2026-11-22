"""Fused, exact-GELU LayerNorm over physically contiguous pixel channels."""

from __future__ import annotations

# Custom op schemas use positional flags.
# ruff: noqa: FBT001
import ctypes as ct
from functools import cache

import torch
from torch import Tensor
from torch.nn import functional

from .cuda_nvrtc import _check_nvrtc, _CudaModule, _ensure_context, _include_dirs, _nvrtc

_TYPES = {torch.float32: "float", torch.float16: "__half", torch.bfloat16: "__nv_bfloat16"}
_THREADS = 128
_VECTOR_BYTES = 16
_MAX_VECTOR_CHANNELS = 1024
_WARP_SIZE = 32
_MAX_FORWARD_BLOCKS = 65535
_BACKWARD_BLOCKS_PER_SM = 2


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
    int c = blockIdx.x * threads + threadIdx.x;
    if (c >= C) return;
    float w = 0.f, b = 0.f, cb = 0.f;
    for (int i = 0; i < workers; i++) {
        int worker = i * (2 + HAS_CONV_BIAS) * C;
        w += partial[worker + c];
        b += partial[worker + C + c];
        if constexpr (HAS_CONV_BIAS) cb += partial[worker + 2 * C + c];
    }
    dw[c] = W(w);
    db[c] = B(b);
    if constexpr (HAS_CONV_BIAS) dcb[c] = CB(cb);
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
def _worker_limit(device: torch.device) -> int:
    return _BACKWARD_BLOCKS_PER_SM * torch.cuda.get_device_properties(device).multi_processor_count


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
    workers = min(_worker_limit(x.device), (n + rows - 1) // rows)
    partial = x.new_empty((workers, 2 + int(cb is not None), channels), dtype=torch.float32)
    with torch.cuda.device(x.device):
        module = _kernel_module(x, w, b, cb, dy.dtype, vector=vector, round_affine=round_affine)
        _ensure_context(module.driver, x.device.index or 0)
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
            (channels + _THREADS - 1) // _THREADS,
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


def _save(ctx, inputs, output) -> None:  # noqa: ANN001, ARG001
    x, w, b, cb, ctx.eps, _, ctx.round_affine = inputs
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x, w, b, cb)


def _autograd_backward(
    ctx,  # noqa: ANN001
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
