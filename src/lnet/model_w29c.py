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
from .fused_stem_ops import _TYPES, _compile_source
from .model_w28c import _differentiate_swiglu, _save_joint, model_cuda_supported, model_opt_mode
from .model_w36b import inductor_norms_enabled


def _wave56b_mode() -> str:
    mode = os.environ.get("LNET_MODEL_W56B", "off")
    if mode not in ("off", "swiglu_vec8_tanh"):
        raise ValueError(f"invalid LNET_MODEL_W56B={mode!r}")
    return mode


def model_w29c_enabled(device: torch.device, feature: str) -> bool:
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
) -> None:
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    row = row_tile * block_rows + tl.arange(0, block_rows)
    col = col_tile * block_cols + tl.arange(0, block_cols)
    packed = row[:, None] * (2 * hidden) + col[None, :]
    dense = row[:, None] * hidden + col[None, :]
    active = (row[:, None] < rows) & (col[None, :] < hidden)
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
        enable_fp_fusion=False,
    )


def packed_swiglu_vec2d(joint: Tensor) -> Tensor:
    mode = _wave56b_mode()
    if mode == "off":
        return _off_swiglu(joint)
    return _tuned_swiglu(joint)


@triton_op("lnet::w29c_swiglu", mutates_args=())
def _off_swiglu(joint: Tensor) -> Tensor:
    output = joint.new_empty((*joint.shape[:-1], joint.shape[-1] // 2))
    _launch_swiglu(joint, joint, output, backward=False)
    return output


@triton_op("lnet::w56b_swiglu_forward", mutates_args=())
def _tuned_swiglu(joint: Tensor) -> Tensor:
    return _vec_swiglu(joint, joint, False)


def _swiglu_vec2d_backward(joint: Tensor, grad: Tensor) -> Tensor:
    mode = _wave56b_mode()
    if mode == "off":
        return _off_swiglu_backward(joint, grad)
    return _vec_swiglu(joint, grad, True)


@triton_op("lnet::w29c_swiglu_backward", mutates_args=())
def _off_swiglu_backward(joint: Tensor, grad: Tensor) -> Tensor:
    output = torch.empty_like(joint)
    _launch_swiglu(joint, grad, output, backward=True)
    return output


def _differentiate_swiglu_vec2d(ctx, grad: Tensor) -> Tensor:
    if grad.is_contiguous() and not torch.is_grad_enabled():
        return _off_swiglu_backward(ctx.saved_tensors[0], grad)
    return _differentiate_swiglu(ctx, grad)


_off_swiglu.register_autograd(_differentiate_swiglu_vec2d, setup_context=_save_joint)


def _differentiate_tuned_swiglu(ctx, grad):
    if grad.is_contiguous() and not torch.is_grad_enabled():
        return _vec_swiglu(ctx.saved_tensors[0], grad, True)
    return _differentiate_swiglu(ctx, grad)


_tuned_swiglu.register_autograd(_differentiate_tuned_swiglu, setup_context=_save_joint)


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
) -> None:
    pid = tl.program_id(0)
    col = tl.arange(0, block_cols)
    if need_dx:
        scale = tl.load(weight + col, col < width, 0).to(tl.float32)
    if need_dw:
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
            accum += grad * normalized
    if need_dw:
        result = tl.sum(accum, 0)
        tl.store(partial + pid * width + col, result, col < width)


def channel_rmsnorm(
    x: Tensor, weight: Tensor, epsilon: float, save_inverse: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor]:
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


def _channel_rmsnorm_backward(
    x: Tensor, weight: Tensor, inverse: Tensor, grad: Tensor, *,
    need_dx: bool, need_dw: bool,
) -> tuple[Tensor, Tensor]:
    sp = _wave57b_mode()
    if sp != "off":
        return _sp_rms(x, weight, inverse, grad, need_dx, need_dw, sp)
    return _off_rms_bwd(x, weight, inverse, grad, need_dx=need_dx, need_dw=need_dw)


@triton_op("lnet::w29c_rmsnorm_backward", mutates_args=())
def _off_rms_bwd(x: Tensor, weight: Tensor, inverse: Tensor, grad: Tensor, *,
                 need_dx: bool, need_dw: bool) -> tuple[Tensor, Tensor]:
    return _launch_rms_bwd(x, weight, inverse, grad, need_dx, need_dw)


def _launch_rms_bwd(x, weight, inverse, grad, need_dx, need_dw):
    width = x.shape[-1]
    rows = x.numel() // width
    dx = x.new_empty(x.shape if need_dx else (0,))
    block_rows = _rmsnorm_backward_rows(width)
    sms = (_vec_sms(x.device.index) if x.is_cuda
           else torch.cuda.get_device_properties(x.device).multi_processor_count)
    ctas = min(2 * sms, triton.cdiv(rows, block_rows))
    partial = x.new_empty((ctas, width) if need_dw else (0,), dtype=torch.float32)
    wrap_triton(_rmsnorm_backward_kernel)[(ctas,)](
        x, weight, inverse, grad, dx, partial, rows, width, _row_pitch(x), _row_pitch(grad),
        block_rows, triton.next_power_of_2(width), need_dx, need_dw,
        enable_fp_fusion=False,
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


_VEC_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using T = SCALAR;
union Pack {
    uint4 bits;
    T v[8];
    __device__ Pack() {}
};

"""

_SWIGLU_BODY = r"""
extern "C" __global__ void swiglu_vec8_kernel(
    const uint4* joint, const uint4* grad, uint4* out, int rows
) {
    constexpr int packs = WIDTH / 8;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * packs) return;
    int j = idx + (idx / packs) * packs;
    Pack a, b, d, u, v;
    a.bits = joint[j];
    b.bits = joint[j + packs];
    if (BACKWARD) d.bits = grad[idx];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float x = float(a.v[i]), g = float(b.v[i]), sig;
        {
            float t;
            asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f*g));
            sig = 0.5f*t + 0.5f;
        }
        float act = float(T(g*sig));
        if (BACKWARD) {
            float dy = float(d.v[i]);
            u.v[i] = T(dy * act);
            v.v[i] = T(float(T(dy*x)) * (sig * (1.f + g*(1.f-sig))));
        } else u.v[i] = T(x*act);
    }
    out[BACKWARD ? j : idx] = u.bits;
    if (BACKWARD) out[j+packs] = v.bits;
}

"""


def _vec_source(dtype, width, backward):
    return (f"#define SCALAR {_TYPES[dtype]}\n#define WIDTH {width}\n"
            f"#define BACKWARD {int(backward)}\n" + _VEC_SOURCE + _SWIGLU_BODY)


@cache
def _vec_module(device, dtype, width, backward):
    source = _vec_source(dtype, width, backward)
    if torch.cuda.is_current_stream_capturing():
        # An uncached capture uses portable PTX without querying device properties.
        capability = (7, 5)
    else:
        capability = torch.cuda.get_device_capability(device)
        if capability == (9, 0):
            # _compile_source concatenates the two architecture components.
            capability = (9, "0a")
    image = _compile_source(capability, source)
    module = _CudaModule(image, device)
    name = "swiglu_vec8_kernel"
    module.load_function(name, 0)
    return module, name


def _run_vec(tensors, sizes, blocks, key):
    x = tensors[0]
    with torch.cuda.device(x.device):
        module, name = _vec_module(x.device.index, x.dtype, *key)
        _ensure_context(module.driver, x.device.index)
        args = [ct.c_void_p(t.data_ptr()) for t in tensors] + [ct.c_int(n) for n in sizes]
        module.launch(name, (blocks, 1, 1), (256, 1, 1), 0,
                      torch.cuda.current_stream(x.device).cuda_stream, args)


@torch.library.custom_op("lnet::w56b_swiglu", mutates_args=(), device_types="cuda")
def _vec_swiglu(joint: Tensor, grad: Tensor, backward: bool) -> Tensor:
    width = joint.shape[-1] // 2
    out = _vec_swiglu_fake(joint, grad, backward)
    if (width % 8 or joint.dtype not in (torch.float16, torch.bfloat16)
            or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (joint, grad))):
        _launch_swiglu(joint.contiguous(), grad.contiguous(), out, backward=backward)
        return out
    rows = joint.numel() // (2 * width)
    _run_vec((joint, grad, out), (rows,), triton.cdiv(rows * (width // 8), 256),
             (width, backward))
    return out


@_vec_swiglu.register_fake
def _vec_swiglu_fake(joint, grad, backward):
    return joint.new_empty(joint.shape if backward else (*joint.shape[:-1], joint.shape[-1] // 2))


@cache
def _vec_sms(device):
    if torch.cuda.is_current_stream_capturing():
        return 132
    return torch.cuda.get_device_properties(device).multi_processor_count



_W57_MODES = ("off", "cuda_sp_vec8_4cta")


def _wave57b_mode():
    mode = os.environ.get("LNET_MODEL_W57B_RMS", "off")
    if mode not in _W57_MODES:
        raise ValueError(f"invalid LNET_MODEL_W57B_RMS={mode!r}")
    return mode


_SP_BODY = r"""
extern "C" __global__ void rms_sp(const uint4* x, const uint4* w, const float* inverse,
        const uint4* dy, uint4* dx, float* partial, int rows, int xp, int yp) {
    constexpr int groups = 256/G, packs = WIDTH/8;
    int t = threadIdx.x, lane = t%G, group = t/G;
    float weight[8], acc[8] = {};
    if (DX && lane < packs) {
        Pack raw;
        raw.bits = w[lane];
        #pragma unroll
        for (int i=0; i<8; ++i) weight[i] = float(raw.v[i]);
    }
    for (int base=blockIdx.x*groups; base<rows; base+=gridDim.x*groups) {
        int row = base+group;
        bool active = row<rows && lane<packs;
        float norm[8], scaled[8], proj=0.f;
        float inv = row<rows ? inverse[row] : 0.f;
        if (active) {
            Pack a,b;
            a.bits = x[row*xp+lane];
            b.bits = dy[row*yp+lane];
            #pragma unroll
            for (int i=0; i<8; ++i) {
                norm[i] = float(a.v[i])*inv;
                float grad = float(b.v[i]);
                if (DW) acc[i] += grad*norm[i];
                if (DX) {
                    scaled[i] = grad*weight[i];
                    proj += scaled[i]*norm[i];
                }
            }
        }
        if (DX) {
            #pragma unroll
            for (int k=G/2; k; k/=2) proj += __shfl_xor_sync(0xffffffff,proj,k,G);
            if (active) {
                Pack out;
                #pragma unroll
                for (int i=0; i<8; ++i)
                    out.v[i] = T((scaled[i]-norm[i]*(proj/WIDTH))*inv);
                dx[row*packs+lane] = out.bits;
            }
        }
    }
    if (DW) {
        __shared__ float scratch[groups*WIDTH];
        if (lane<packs) {
            #pragma unroll
            for (int i=0; i<8; ++i) scratch[group*WIDTH+lane*8+i] = acc[i];
        }
        __syncthreads();
        if (t<WIDTH/4) {
            float4 sum = {};
            #pragma unroll
            for (int g=0; g<groups; ++g) {
                float4 v = reinterpret_cast<float4*>(scratch+g*WIDTH)[t];
                sum.x+=v.x; sum.y+=v.y; sum.z+=v.z; sum.w+=v.w;
            }
            reinterpret_cast<float4*>(partial+blockIdx.x*WIDTH)[t] = sum;
        }
    }
}
"""


def _sp_group(width):
    return 1 << (width // 8 - 1).bit_length()


def _sp_source(dtype, width, dx, dw):
    header = _VEC_SOURCE
    return (f"#define SCALAR {_TYPES[dtype]}\n#define WIDTH {width}\n#define G {_sp_group(width)}\n"
            f"#define DX {int(dx)}\n#define DW {int(dw)}\n" + header + _SP_BODY)


@cache
def _sp_module(device, dtype, width, dx, dw):
    cap = torch.cuda.get_device_capability(device)
    source = _sp_source(dtype, width, dx, dw)
    module = _CudaModule(_compile_source((9, "0a") if cap == (9, 0) else cap, source), device)
    module.load_function("rms_sp", 0)
    return module


@torch.library.custom_op("lnet::w57b_sp_rms", mutates_args=(), device_types="cuda")
def _sp_rms(x: Tensor, w: Tensor, inv: Tensor, dy: Tensor, dx: bool,
            dw: bool, mode: str) -> tuple[Tensor, Tensor]:
    width = x.shape[-1]
    xp, yp = _row_pitch(x), _row_pitch(dy)
    if (not (0 < width <= 256 and width % 8 == 0)
            or not xp or not yp or xp % 8 or yp % 8 or w.dtype != x.dtype
            or x.dtype not in (torch.bfloat16, torch.float16) or dy.dtype != x.dtype
            or inv.dtype != torch.float32 or not inv.is_contiguous() or not w.is_contiguous()
            or any(t.data_ptr() % 16 for t in (x, dy, w))):
        return _launch_rms_bwd(x if xp else x.contiguous(), w.contiguous(), inv.contiguous(),
                               dy if yp else dy.contiguous(), dx, dw)
    rows = x.numel() // width
    blocks = min(4 * _vec_sms(x.device.index),
                 triton.cdiv(rows, 256 // _sp_group(width)))
    out = x.new_empty(x.shape if dx else (0,))
    partial = inv.new_empty((blocks, width) if dw else (0,))
    with torch.cuda.device(x.device):
        module = _sp_module(x.device.index, x.dtype, width, dx, dw)
        _ensure_context(module.driver, x.device.index)
        args = [ct.c_void_p(t.data_ptr()) for t in (x, w, inv, dy, out, partial)]
        args.extend(ct.c_int(n) for n in (rows, xp // 8, yp // 8))
        module.launch("rms_sp", (blocks, 1, 1), (256, 1, 1), 0,
                      torch.cuda.current_stream(x.device).cuda_stream, args)
    return out, partial.sum(0).to(w.dtype) if dw else w.new_empty((0,))


@_sp_rms.register_fake
def _sp_fake(x, w, inv, dy, dx, dw, mode):
    return x.new_empty(x.shape if dx else (0,)), w.new_empty(w.shape if dw else (0,))
