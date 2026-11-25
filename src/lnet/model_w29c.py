"""Row-tiled SwiGLU and channel RMSNorm kernels."""

# ruff: noqa: ANN001
from __future__ import annotations

import ctypes as ct
import os
from functools import cache

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .cuda_nvrtc import _CudaModule, _ensure_context
from .fused_stem_ops import _TYPES, _compile_source, _launch
from .model_w28c import _differentiate_swiglu, _save_joint, model_cuda_supported, model_opt_mode
from .model_w36b import inductor_norms_enabled


def model_w29c_enabled(device: torch.device, feature: str) -> bool:
    if feature in ("rmsnorm_bwd", "rmsnorm_full") and _fast_rmsnorm_enabled():
        return model_cuda_supported(device)
    mode = model_opt_mode("LNET_MODEL_W29C")
    if mode == "auto":
        if feature == "swiglu_vec2d" and model_opt_mode("LNET_MODEL_W28C") in (
            "swiglu", "combined",
        ):
            return False
        mode = "combined_bwd"
    selected = mode in {
        "swiglu_vec2d": ("swiglu_vec2d", "combined_bwd", "combined_full"),
        "rmsnorm_bwd": ("rmsnorm_bwd", "rmsnorm_full", "combined_bwd", "combined_full"),
        "rmsnorm_full": ("rmsnorm_full", "combined_full"),
        "nr4": ("combined_bwd", "combined_full"),
    }[feature]
    if selected and feature in ("swiglu_vec2d", "rmsnorm_bwd", "rmsnorm_full"):
        selected = not inductor_norms_enabled()
    return selected and model_cuda_supported(device)


@triton.jit
def _swiglu_vec2d_kernel(
    joint, gradient, output, rows, hidden: tl.constexpr,
    block_rows: tl.constexpr, block_cols: tl.constexpr, backward: tl.constexpr,
    divisor_tiled: tl.constexpr = False,  # noqa: FBT002
) -> None:
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    if divisor_tiled:
        tiles_per_row: tl.constexpr = hidden // block_cols
        col_tile = row_tile % tiles_per_row
        row_tile = row_tile // tiles_per_row
    row = row_tile * block_rows + tl.arange(0, block_rows)
    col = col_tile * block_cols + tl.arange(0, block_cols)
    packed = row[:, None] * (2 * hidden) + col[None, :]
    dense = row[:, None] * hidden + col[None, :]
    active = row[:, None] < rows
    if not divisor_tiled:
        active = active & (col[None, :] < hidden)
    dtype: tl.constexpr = joint.dtype.element_ty
    value = tl.load(joint + packed, active, 0).to(tl.float32)
    gate = tl.load(joint + packed + hidden, active, 0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    activated = (gate * sigmoid).to(dtype).to(tl.float32)
    if backward:
        grad = tl.load(gradient + dense, active, 0).to(tl.float32)
        gate_grad = (grad * value).to(dtype).to(tl.float32)
        tl.store(output + packed, grad * activated, active)
        derivative = sigmoid * (1 + gate * (1 - sigmoid))
        tl.store(output + packed + hidden, gate_grad * derivative, active)
    else:
        tl.store(output + dense, value * activated, active)


def _launch_swiglu(joint: Tensor, grad: Tensor, output: Tensor, *, backward: bool) -> None:
    hidden = joint.shape[-1] // 2
    rows = joint.numel() // (2 * hidden)
    block_cols = min(512, triton.next_power_of_2(hidden))
    block_rows = 8
    wrap_triton(_swiglu_vec2d_kernel)[
        (triton.cdiv(rows, block_rows), triton.cdiv(hidden, block_cols))
    ](
        joint, grad, output, rows, hidden, block_rows, block_cols, backward,
        divisor_tiled=False, enable_fp_fusion=False,
    )


@triton_op("lnet::w29c_swiglu", mutates_args=())
def packed_swiglu_vec2d(joint: Tensor) -> Tensor:
    output = joint.new_empty((*joint.shape[:-1], joint.shape[-1] // 2))
    _launch_swiglu(joint, joint, output, backward=False)
    return output


@triton_op("lnet::w29c_swiglu_backward", mutates_args=())
def _swiglu_vec2d_backward(joint: Tensor, grad: Tensor) -> Tensor:
    output = torch.empty_like(joint)
    _launch_swiglu(joint, grad, output, backward=True)
    return output


def _differentiate_swiglu_vec2d(ctx, grad: Tensor) -> Tensor:
    if grad.is_contiguous() and not torch.is_grad_enabled():
        return _swiglu_vec2d_backward(ctx.saved_tensors[0], grad)
    return _differentiate_swiglu(ctx, grad)


packed_swiglu_vec2d.register_autograd(_differentiate_swiglu_vec2d, setup_context=_save_joint)


def _row_pitch(tensor: Tensor) -> int:
    """Return a constant pitch for a row-major view, or zero for other layouts."""
    if tensor.is_contiguous():
        return tensor.shape[-1]
    if tensor.ndim < 2 or tensor.stride(-1) != 1:
        return 0
    pitch = tensor.stride(-2)
    if pitch < tensor.shape[-1]:
        return 0
    expected = pitch
    for dim in range(tensor.ndim - 2, -1, -1):
        if tensor.shape[dim] != 1 and tensor.stride(dim) != expected:
            return 0
        expected *= tensor.shape[dim]
    return pitch


def rmsnorm_eligible(x: Tensor, weight: Tensor) -> bool:
    if not (0 < x.shape[-1] <= 1024 and 0 < x.numel() < 2**31 and weight.is_contiguous()):
        return False
    pitch = _row_pitch(x)
    return pitch != 0 and x.numel() // x.shape[-1] * pitch < 2**31


@triton.jit
def _rmsnorm_forward_kernel(
    x, weight, y, inverse, rows, width: tl.constexpr, pitch,
    eps: tl.constexpr, block_rows: tl.constexpr, block_cols: tl.constexpr,
    save_inverse: tl.constexpr,
) -> None:
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    col = tl.arange(0, block_cols)
    active = (row[:, None] < rows) & (col[None, :] < width)
    value = tl.load(x + row[:, None] * pitch + col[None, :], active, 0).to(tl.float32)
    scale = tl.load(weight + col, col < width, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, 1) / width + eps)
    tl.store(y + row[:, None] * width + col[None, :], value * inv[:, None] * scale[None, :], active)
    if save_inverse:
        tl.store(inverse + row, inv, row < rows)


@triton.jit
def _rmsnorm_backward_kernel(
    x, weight, inverse, dy, dx, partial, rows,
    width: tl.constexpr, x_pitch, dy_pitch,
    block_rows: tl.constexpr, block_cols: tl.constexpr,
    need_dx: tl.constexpr, need_dw: tl.constexpr,
    # Only the archived offline compiler probe varies this constant.
    reduce_in_loop: tl.constexpr = False,  # noqa: FBT002
) -> None:
    pid = tl.program_id(0)
    col = tl.arange(0, block_cols)
    if need_dx:
        scale = tl.load(weight + col, col < width, 0).to(tl.float32)
    if need_dw:
        if reduce_in_loop:
            accum = tl.zeros((block_cols,), tl.float32)
        else:
            accum = tl.full((block_rows, block_cols), 0, tl.float32)
    for start in range(pid * block_rows, rows, tl.num_programs(0) * block_rows):
        row = start + tl.arange(0, block_rows)
        active = (row[:, None] < rows) & (col[None, :] < width)
        inv = tl.load(inverse + row, row < rows, 0)
        value = tl.load(x + row[:, None] * x_pitch + col[None, :], active, 0).to(tl.float32)
        grad = tl.load(dy + row[:, None] * dy_pitch + col[None, :], active, 0).to(tl.float32)
        normalized = value * inv[:, None]
        if need_dx:
            scaled = grad * scale[None, :]
            projection = tl.sum(normalized * scaled, 1)
            delta = (scaled - (normalized / width) * projection[:, None]) * inv[:, None]
            tl.store(dx + row[:, None] * width + col[None, :], delta, active)
        if need_dw:
            if reduce_in_loop:
                accum += tl.sum(grad * normalized, 0)
            else:
                accum += grad * normalized
    if need_dw:
        result = accum if reduce_in_loop else tl.sum(accum, 0)
        tl.store(partial + pid * width + col, result, col < width)


def channel_rmsnorm(
    x: Tensor, weight: Tensor, epsilon: float, save_inverse: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor]:
    if _fast_rmsnorm_enabled():
        return _fast_rmsnorm(x, weight, epsilon, save_inverse)
    return _triton_rmsnorm(x, weight, epsilon, save_inverse)


@triton_op("lnet::w29c_rmsnorm_forward", mutates_args=())
def _triton_rmsnorm(
    x: Tensor, weight: Tensor, epsilon: float, save_inverse: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor]:
    width = x.shape[-1]
    rows = x.numel() // width
    output = x.new_empty(x.shape)
    inverse = x.new_empty((*x.shape[:-1], 1) if save_inverse else (0,), dtype=torch.float32)
    block_rows = 8
    wrap_triton(_rmsnorm_forward_kernel)[(triton.cdiv(rows, block_rows),)](
        x, weight, output, inverse, rows, width, _row_pitch(x), epsilon,
        block_rows, triton.next_power_of_2(width), save_inverse, enable_fp_fusion=False,
    )
    return output, inverse


def _rmsnorm_backward_rows(width: int, *, v2: bool = False) -> int:
    # Scalar tails use smaller tiles to bound register pressure.
    max_rows = 16 if v2 else 32
    max_elements = 2048 if v2 and width <= 256 else 4096
    return min(max_rows if width % 16 == 0 else 8, max_elements // triton.next_power_of_2(width))


@triton_op("lnet::w29c_rmsnorm_backward", mutates_args=())
def _channel_rmsnorm_backward(
    x: Tensor, weight: Tensor, inverse: Tensor, grad: Tensor, *,
    need_dx: bool, need_dw: bool,
) -> tuple[Tensor, Tensor]:
    width = x.shape[-1]
    rows = x.numel() // width
    dx = x.new_empty(x.shape if need_dx else (0,))
    block_rows = _rmsnorm_backward_rows(width)
    # Two CTAs per SM preserve the H200 launch geometry and bound the reduction.
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    ctas = min(2 * sms, triton.cdiv(rows, block_rows))
    partial = x.new_empty((ctas, width) if need_dw else (0,), dtype=torch.float32)
    wrap_triton(_rmsnorm_backward_kernel)[(ctas,)](
        x, weight, inverse, grad, dx, partial, rows, width, _row_pitch(x), _row_pitch(grad),
        block_rows, triton.next_power_of_2(width), need_dx, need_dw,
        reduce_in_loop=False, enable_fp_fusion=False,
    )
    dw = partial.sum(0).to(weight.dtype) if need_dw else weight.new_empty((0,))
    return dx, dw


def channel_rmsnorm_vjp(
    x: Tensor, weight: Tensor, inverse: Tensor, grad: Tensor, *,
    need_dx: bool, need_dw: bool,
) -> tuple[Tensor | None, Tensor | None]:
    if (
        not torch.is_grad_enabled() and (need_dx or need_dw)
        and inverse.is_contiguous() and rmsnorm_eligible(x, weight)
    ):
        pitch = _row_pitch(grad)
        if pitch != 0 and grad.numel() // grad.shape[-1] * pitch < 2**31:
            dx, dw = _channel_rmsnorm_backward(
                x, weight, inverse, grad, need_dx=need_dx, need_dw=need_dw,
            )
            return dx if need_dx else None, dw if need_dw else None

    normalized = x * inverse
    dx = dw = None
    if need_dx:
        scaled_grad = grad * weight.float()
        projection = (normalized * scaled_grad).sum(-1, keepdim=True)
        dx = ((scaled_grad - (normalized / x.shape[-1]) * projection) * inverse).to(x.dtype)
    if need_dw:
        # Reduce the original axes directly so unusual layouts never need a flattening copy.
        weighted = grad * normalized
        dims = tuple(range(x.ndim - 1))
        dw = (weighted.sum(dims) if dims else weighted).to(weight.dtype)
    return dx, dw


def _save_rmsnorm(ctx, inputs, output) -> None:
    x, weight, _, _ = inputs
    _, inverse = output
    ctx.mark_non_differentiable(inverse)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x, weight, inverse)


def _differentiate_rmsnorm(ctx, grad: Tensor, inverse_grad: Tensor | None) -> tuple:  # noqa: ARG001
    x, weight, inverse = ctx.saved_tensors
    dx, dw = channel_rmsnorm_vjp(
        x, weight, inverse, grad,
        need_dx=ctx.needs_input_grad[0], need_dw=ctx.needs_input_grad[1],
    )
    return dx, dw, None, None


_triton_rmsnorm.register_autograd(_differentiate_rmsnorm, setup_context=_save_rmsnorm)


class _CompiledChannelRMSNorm(torch.autograd.Function):
    """Keep strided activations and output gradients in their original layouts."""

    @staticmethod
    def forward(
        ctx, x: Tensor, weight: Tensor, epsilon: float | None, fused: bool,  # noqa: FBT001
    ) -> Tensor:
        result, inverse = torch.ops.aten._fused_rms_norm(  # noqa: SLF001
            x, [x.shape[-1]], weight, epsilon,
        )
        ctx.fused = fused
        ctx.save_for_backward(x, weight, inverse.reshape(*x.shape[:-1], 1))
        return result

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor | None, Tensor | None, None, None]:
        x, weight, inverse = ctx.saved_tensors
        if ctx.fused:
            dx, dw = channel_rmsnorm_vjp(
                x, weight, inverse, grad,
                need_dx=ctx.needs_input_grad[0], need_dw=ctx.needs_input_grad[1],
            )
            return dx, dw, None, None
        # Promotion keeps the reductions in FP32 without requesting contiguous copies
        # of the activation and output gradient in the ATen backward decomposition.
        normalized = x * inverse
        dx = dw = None
        if ctx.needs_input_grad[0]:
            scaled_grad = grad * weight.float()
            projection = (normalized * scaled_grad).sum(-1, keepdim=True)
            dx = ((scaled_grad - (normalized / x.shape[-1]) * projection) * inverse).to(x.dtype)
        if ctx.needs_input_grad[1]:
            dw = (grad * normalized).reshape(-1, x.shape[-1]).sum(0).to(weight.dtype)
        return dx, dw, None, None


_FAST_RMS_CUDA = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T = SCALAR;
constexpr int elements = 16 / sizeof(T);

union Pack {
    uint4 vector;
    T values[elements];
    __device__ Pack() {}
};

template<bool Vector, bool Wide = false>
__device__ __forceinline__ void rms(
    const T* x, const T* w, T* y, float* inverse,
    int rows, int width, int pitch, float eps
) {
    const int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= rows) return;
    x += row * pitch;
    y += row * width;
    float sum = 0.f;
    if constexpr (Vector) {
        auto vx = reinterpret_cast<const uint4*>(x);
        auto vw = reinterpret_cast<const uint4*>(w);
        auto vy = reinterpret_cast<uint4*>(y);
        const int packs = width / elements;
        Pack input;
        for (int p = lane; p < packs; p += 32) {
            input.vector = vx[p];
            #pragma unroll
            for (int i = 0; i < elements; ++i) {
                const float v = float(input.values[i]);
                sum += v * v;
            }
            if constexpr (!Wide) break;
        }
        #pragma unroll
        for (int offset = 16; offset; offset /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        const float inv = rsqrtf(sum / width + eps);
        if (lane == 0 && inverse) inverse[row] = inv;
        for (int p = lane; p < packs; p += 32) {
            Pack scale, output;
            if constexpr (Wide) input.vector = vx[p];
            scale.vector = vw[p];
            #pragma unroll
            for (int i = 0; i < elements; ++i)
                output.values[i] = T(float(input.values[i]) * inv * float(scale.values[i]));
            vy[p] = output.vector;
            if constexpr (!Wide) break;
        }
    } else {
        for (int c = lane; c < width; c += 32) {
            const float v = float(x[c]);
            sum += v * v;
        }
        #pragma unroll
        for (int offset = 16; offset; offset /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        const float inv = rsqrtf(sum / width + eps);
        if (lane == 0 && inverse) inverse[row] = inv;
        for (int c = lane; c < width; c += 32)
            y[c] = T(float(x[c]) * inv * float(w[c]));
    }
}

extern "C" __global__ void rms_vector(
    const T* x, const T* w, T* y, float* inverse, int rows, int width, int pitch, float eps
) {
    rms<true>(x, w, y, inverse, rows, width, pitch, eps);
}

extern "C" __global__ void rms_vector_wide(
    const T* x, const T* w, T* y, float* inverse, int rows, int width, int pitch, float eps
) {
    rms<true, true>(x, w, y, inverse, rows, width, pitch, eps);
}

extern "C" __global__ void rms_scalar(
    const T* x, const T* w, T* y, float* inverse, int rows, int width, int pitch, float eps
) {
    rms<false>(x, w, y, inverse, rows, width, pitch, eps);
}
"""


def _fast_rmsnorm_enabled() -> bool:
    mode = os.environ.get("LNET_NORM_W50_MODE", "off")
    if mode not in ("off", "fast_fwd"):
        raise ValueError("LNET_NORM_W50_MODE must be off or fast_fwd")
    return mode == "fast_fwd"


def _compile_fast_rms_ptx(capability, dtype: torch.dtype) -> ct.Array:
    return _compile_source(capability, f"#define SCALAR {_TYPES[dtype]}\n" + _FAST_RMS_CUDA)


@cache
def _fast_rms_module(device: int, dtype: torch.dtype) -> _CudaModule:
    image = _compile_fast_rms_ptx(torch.cuda.get_device_capability(device), dtype)
    module = _CudaModule(image, device)
    for name in ("rms_vector", "rms_vector_wide", "rms_scalar"):
        module.load_function(name, 0)
    return module


@torch.library.custom_op("lnet::w50_rmsnorm_forward", mutates_args=(), device_types="cuda")
def _fast_rmsnorm(
    x: Tensor, weight: Tensor, epsilon: float, save_inverse: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor]:
    output, inverse = _fast_rmsnorm_fake(x, weight, epsilon, save_inverse)
    width = x.shape[-1]
    rows = x.numel() // width
    pitch = _row_pitch(x)
    elements = 16 // x.element_size()
    vector = (
        width % elements == 0 and pitch % elements == 0
        and x.data_ptr() % 16 == 0 and weight.data_ptr() % 16 == 0
    )
    name = "rms_scalar"
    if vector:
        name = "rms_vector" if width <= elements * 32 else "rms_vector_wide"
    device = x.device.index or 0
    with torch.cuda.device(x.device):
        module = _fast_rms_module(device, x.dtype)
        _ensure_context(module.driver, device)
        _launch(
            module, torch.cuda.current_stream(x.device).cuda_stream,
            name, triton.cdiv(rows, 4),
            (x, weight, output, inverse if save_inverse else None),
            ct.c_int(rows), ct.c_int(width), ct.c_int(pitch), ct.c_float(epsilon),
        )
    return output, inverse


@_fast_rmsnorm.register_fake
def _fast_rmsnorm_fake(
    x: Tensor, weight: Tensor, epsilon: float, save_inverse: bool,  # noqa: ARG001, FBT001
) -> tuple[Tensor, Tensor]:
    return x.new_empty(x.shape), x.new_empty(
        (*x.shape[:-1], 1) if save_inverse else (0,), dtype=torch.float32
    )


_fast_rmsnorm.register_autograd(_differentiate_rmsnorm, setup_context=_save_rmsnorm)
