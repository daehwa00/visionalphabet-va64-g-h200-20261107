"""Row-wise RMSNorm with deterministic, fused input and scale gradients."""

from __future__ import annotations

import ctypes as ct
import math
import os
from functools import cache

import torch
from torch import Tensor

from .cuda_nvrtc import _check_nvrtc, _CudaModule, _include_dirs, _nvrtc

_TYPES = {torch.bfloat16: "__nv_bfloat16", torch.float16: "__half", torch.float32: "float"}
_THREADS = 128
_MAX_CHANNELS = 1024
_CUDA = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using T = DTYPE;
constexpr int Packs = (CHANNELS + LANES * VEC - 1) / (LANES * VEC);
constexpr int Rows = 128 / LANES;
union Pack {
    uint4 bits;
    T values[VEC];
};

__device__ Pack load(const T* src, int c) {
    Pack p;
    if constexpr (VEC > 1) p.bits = *reinterpret_cast<const uint4*>(src + c);
    else p.values[0] = c < CHANNELS ? src[c] : T(0.f);
    return p;
}

__device__ void store(T* dst, int c, Pack p) {
    if constexpr (VEC > 1) {
        // Preserve the 128-bit store after per-element conversion.
        asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::
            "l"(dst + c), "r"(p.bits.x), "r"(p.bits.y), "r"(p.bits.z), "r"(p.bits.w) : "memory");
    } else if (c < CHANNELS) dst[c] = p.values[0];
}

__device__ float sum_row(float v) {
    #pragma unroll
    for (int delta = LANES / 2; delta; delta /= 2)
        v += __shfl_xor_sync(0xffffffff, v, delta, LANES);
    return v;
}

extern "C" __global__ void rms_forward(
    const T* x, const T* w, T* y, float* inverse, const T* right, T* residual,
    long long n, float eps
) {
    int lane = threadIdx.x % LANES;
    long long row = (long long)blockIdx.x * Rows + threadIdx.x / LANES;
    bool valid = row < n;
    Pack values[Packs];
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < Packs; ++i) {
        int c = (i * LANES + lane) * VEC;
        if (valid && c < CHANNELS) {
            values[i] = load(x + row * CHANNELS, c);
            if (right) {
                Pack b = load(right + row * CHANNELS, c);
                #pragma unroll
                for (int v = 0; v < VEC; ++v)
                    values[i].values[v] = T(float(values[i].values[v]) + float(b.values[v]));
                store(residual + row * CHANNELS, c, values[i]);
            }
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                float a = float(values[i].values[v]);
                sum += a * a;
            }
        }
    }
    float scale = rsqrtf(sum_row(sum) / CHANNELS + eps);
    if (!valid) return;
    if (lane == 0) inverse[row] = scale;
    #pragma unroll
    for (int i = 0; i < Packs; ++i) {
        int c = (i * LANES + lane) * VEC;
        if (c < CHANNELS) {
            Pack p, weight = load(w, c);
            #pragma unroll
            for (int v = 0; v < VEC; ++v)
                p.values[v] = T(float(values[i].values[v]) * scale * float(weight.values[v]));
            store(y + row * CHANNELS, c, p);
        }
    }
}

extern "C" __global__ void rms_backward(
    const T* x, const T* w, const T* dy, const float* inverse,
    T* dx, float* partial, long long n
) {
    int lane = threadIdx.x % LANES;
    int group = threadIdx.x / LANES;
    long long start = ((long long)blockIdx.x * n / gridDim.x) / Rows * Rows;
    long long end = blockIdx.x + 1 == gridDim.x ? n :
        (((long long)blockIdx.x + 1) * n / gridDim.x) / Rows * Rows;
    float dw[Packs][VEC] = {};
    Pack weight[Packs];
    #pragma unroll
    for (int i = 0; i < Packs; ++i) {
        int c = (i * LANES + lane) * VEC;
        if (c < CHANNELS) weight[i] = load(w, c);
    }
    for (long long base = start; base < end; base += Rows) {
        long long row = base + group;
        bool valid = row < end;
        float scale = valid ? inverse[row] : 0.f;
        float normalized[Packs][VEC], grad[Packs][VEC];
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < Packs; ++i) {
            int c = (i * LANES + lane) * VEC;
            if (c < CHANNELS && valid) {
                Pack a = load(x + row * CHANNELS, c);
                Pack g = load(dy + row * CHANNELS, c);
                #pragma unroll
                for (int v = 0; v < VEC; ++v) {
                    normalized[i][v] = float(a.values[v]) * scale;
                    grad[i][v] = float(g.values[v]) * float(weight[i].values[v]);
                    sum += normalized[i][v] * grad[i][v];
                    dw[i][v] += float(g.values[v]) * normalized[i][v];
                }
            }
        }
        float projection = sum_row(sum) / CHANNELS;
        if (valid) {
            #pragma unroll
            for (int i = 0; i < Packs; ++i) {
                int c = (i * LANES + lane) * VEC;
                if (c < CHANNELS) {
                    Pack p;
                    #pragma unroll
                    for (int v = 0; v < VEC; ++v)
                        p.values[v] = T((grad[i][v] - normalized[i][v] * projection) * scale);
                    store(dx + row * CHANNELS, c, p);
                }
            }
        }
    }
    __shared__ float workspace[Rows][CHANNELS];
    #pragma unroll
    for (int i = 0; i < Packs; ++i) {
        int c = (i * LANES + lane) * VEC;
        if (c < CHANNELS) {
            #pragma unroll
            for (int v = 0; v < VEC; ++v)
                workspace[group][c + v] = dw[i][v];
        }
    }
    __syncthreads();
    for (int c = threadIdx.x; c < CHANNELS; c += 128) {
        float sum = 0.f;
        #pragma unroll
        for (int r = 0; r < Rows; ++r) sum += workspace[r][c];
        partial[(long long)blockIdx.x * CHANNELS + c] = sum;
    }
}

extern "C" __global__ void rms_dw_reduce(const float* partial, T* dw, int blocks) {
    __shared__ float sums[4][32];
    int c = blockIdx.x * 32 + threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    float sum = 0.f;
    if (c < CHANNELS)
        for (int i = warp; i < blocks; i += 4) sum += partial[i * CHANNELS + c];
    sums[warp][threadIdx.x % 32] = sum;
    __syncthreads();
    if (warp == 0 && c < CHANNELS)
        dw[c] = T(sum + sums[1][threadIdx.x] + sums[2][threadIdx.x] + sums[3][threadIdx.x]);
}
"""


def _lanes(channels: int, vector: int) -> int:
    return min(32, 1 << (channels // vector - 1).bit_length())


def _compile_ptx(
    capability: tuple[int, int], channels: int, dtype: torch.dtype, vector: int
) -> ct.Array:
    header = {
        "DTYPE": _TYPES[dtype],
        "CHANNELS": channels,
        "VEC": vector,
        "LANES": _lanes(channels, vector),
    }
    source = "".join(f"#define {key} {value}\n" for key, value in header.items()) + _CUDA
    nvrtc = _nvrtc()
    program = ct.c_void_p()
    _check_nvrtc(
        nvrtc,
        nvrtc.nvrtcCreateProgram(ct.byref(program), source.encode(), b"rmsnorm.cu", 0, None, None),
        program,
    )
    flags = [
        f"--gpu-architecture=compute_{capability[0]}{capability[1]}".encode(),
        b"--std=c++17",
        *(f"--include-path={path}".encode() for path in _include_dirs()),
    ]
    try:
        options = (ct.c_char_p * len(flags))(*flags)
        _check_nvrtc(nvrtc, nvrtc.nvrtcCompileProgram(program, len(flags), options), program)
        size = ct.c_size_t()
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), program)
        image = ct.create_string_buffer(size.value)
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTX(program, image), program)
        return image
    finally:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))


@cache
def _module(device: int, channels: int, dtype: torch.dtype, vector: int) -> _CudaModule:
    module = _CudaModule(
        _compile_ptx(torch.cuda.get_device_capability(device), channels, dtype, vector), device
    )
    for name in ("rms_forward", "rms_backward", "rms_dw_reduce"):
        module.load_function(name, 0)
    return module


def _geometry(x: Tensor, *aligned: Tensor) -> tuple[int, int]:
    vector = 16 // x.element_size()
    if x.shape[-1] % vector or any(t.data_ptr() % 16 for t in (x, *aligned)):
        vector = 1
    return vector, _THREADS // _lanes(x.shape[-1], vector)


def _launch(
    module: _CudaModule,
    name: str,
    blocks: int,
    tensors: tuple[Tensor | None, ...],
    *sizes: ct.c_longlong | ct.c_float | ct.c_int,
) -> None:
    module.launch(
        name,
        (blocks, 1, 1),
        (_THREADS, 1, 1),
        0,
        torch.cuda.current_stream().cuda_stream,
        [*(ct.c_void_p(t.data_ptr() if t is not None else 0) for t in tensors), *sizes],
    )


def _run_forward(
    x: Tensor, weight: Tensor, eps: float,
    right: Tensor | None = None, residual: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    y, inverse = _forward_fake(x, weight, eps)
    channels = x.shape[-1]
    n = x.numel() // channels
    vector, rows = _geometry(x, weight, y, *((right,) if right is not None else ()))
    device = x.get_device()
    with torch.cuda.device(device):
        module = _module(device, channels, x.dtype, vector)
        _launch(
            module, "rms_forward", math.ceil(n / rows),
            (x, weight, y, inverse, right, residual), ct.c_longlong(n), ct.c_float(eps),
        )
    return y, inverse


@torch.library.custom_op("va_rmsnorm::forward", mutates_args=(), device_types="cuda")
def _forward(x: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    return _run_forward(x, weight, eps)


@_forward.register_fake
def _forward_fake(x: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:  # noqa: ARG001
    return torch.empty_like(x), x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)


@torch.library.custom_op("va_rmsnorm::residual_forward", mutates_args=(), device_types="cuda")
def _residual_forward(
    left: Tensor, right: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor, Tensor]:
    residual = torch.empty_like(left)
    y, inverse = _run_forward(left, weight, eps, right, residual)
    return residual, y, inverse


@_residual_forward.register_fake
def _residual_forward_fake(
    left: Tensor, right: Tensor, weight: Tensor, eps: float  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor]:
    y, inverse = _forward_fake(left, weight, eps)
    return torch.empty_like(left), y, inverse


@torch.library.custom_op("va_rmsnorm::backward", mutates_args=(), device_types="cuda")
def _backward(x: Tensor, weight: Tensor, grad: Tensor, inverse: Tensor) -> tuple[Tensor, Tensor]:
    dx, dw = _backward_fake(x, weight, grad, inverse)
    channels = x.shape[-1]
    n = x.numel() // channels
    device = x.get_device()
    vector, rows = _geometry(x, weight, grad, dx)
    with torch.cuda.device(device):
        module = _module(device, channels, x.dtype, vector)
        blocks = min(
            math.ceil(n / rows), 4 * torch.cuda.get_device_properties(device).multi_processor_count
        )
        partial = x.new_empty((blocks, channels), dtype=torch.float32)
        _launch(
            module, "rms_backward", blocks, (x, weight, grad, inverse, dx, partial),
            ct.c_longlong(n),
        )
        _launch(module, "rms_dw_reduce", math.ceil(channels / 32), (partial, dw), ct.c_int(blocks))
    return dx, dw


@_backward.register_fake
def _backward_fake(
    x: Tensor, weight: Tensor, grad: Tensor, inverse: Tensor  # noqa: ARG001
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_context(
    ctx,  # noqa: ANN001
    inputs: tuple[Tensor, Tensor, float],
    output: tuple[Tensor, Tensor],
) -> None:
    x, weight, ctx.eps = inputs
    _, inverse = output
    ctx.mark_non_differentiable(inverse)
    ctx.save_for_backward(x, weight, inverse)


def _autograd_backward(ctx, grad: Tensor, _grad_inverse: Tensor) -> tuple[Tensor, Tensor, None]:  # noqa: ANN001
    x, weight, inverse = ctx.saved_tensors
    dx, dw = _derivatives(x, weight, grad, inverse, ctx.eps)
    return dx, dw, None


def _derivatives(
    x: Tensor, weight: Tensor, grad: Tensor, inverse: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    if grad.is_contiguous() and not torch.is_grad_enabled():
        dx, dw = _backward(x, weight, grad, inverse)
    else:
        if torch.is_grad_enabled():
            inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        normalized = x * inverse
        scaled = grad * weight.float()
        projection = (normalized * scaled).mean(-1, keepdim=True)
        dx = ((scaled - normalized * projection) * inverse).to(x.dtype)
        dw = (grad * normalized).reshape(-1, weight.numel()).sum(0).to(weight.dtype)
    return dx, dw


_forward.register_autograd(_autograd_backward, setup_context=_setup_context)


def _setup_residual_context(
    ctx,  # noqa: ANN001
    inputs: tuple[Tensor, Tensor, Tensor, float],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    _, _, weight, ctx.eps = inputs
    residual, _, inverse = output
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(inverse)
    ctx.save_for_backward(residual, weight, inverse)


def _residual_backward(
    ctx,  # noqa: ANN001
    grad_residual: Tensor | None,
    grad: Tensor | None,
    _grad_inverse: Tensor | None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, None]:
    if grad is None:
        return grad_residual, grad_residual, None, None
    residual, weight, inverse = ctx.saved_tensors
    dx, dw = _derivatives(residual, weight, grad, inverse, ctx.eps)
    if grad_residual is not None:
        dx = dx + grad_residual
    return dx, dx, dw, None


_residual_forward.register_autograd(_residual_backward, setup_context=_setup_residual_context)


def rmsnorm_mode() -> str:
    mode = os.environ.get("LNET_RMSNORM_OPT", "auto")
    if mode not in {"off", "auto", "on"}:
        raise ValueError("LNET_RMSNORM_OPT must be off, auto or on")
    return mode


def supports_fused_rmsnorm(x: Tensor, weight: Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype in _TYPES
        and x.dtype == weight.dtype
        and x.device == weight.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and 0 < x.shape[-1] == weight.numel() <= _MAX_CHANNELS
        and x.numel() > 0
    )


def fused_rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    if supports_fused_rmsnorm(x, weight):
        return _forward(x, weight, eps)[0]
    if x.dtype in (torch.bfloat16, torch.float16):
        return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps).to(
            x.dtype
        )
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def supports_fused_residual_rmsnorm(left: Tensor, right: Tensor, weight: Tensor) -> bool:
    return (
        supports_fused_rmsnorm(left, weight)
        and right.shape == left.shape
        and right.dtype == left.dtype
        and right.device == left.device
        and right.is_contiguous()
    )


def fused_residual_rmsnorm(
    left: Tensor, right: Tensor, weight: Tensor, eps: float
) -> tuple[Tensor, Tensor]:
    if supports_fused_residual_rmsnorm(left, right, weight):
        residual, y, _ = _residual_forward(left, right, weight, eps)
        return residual, y
    residual = left + right
    return residual, fused_rmsnorm(residual, weight, eps)
