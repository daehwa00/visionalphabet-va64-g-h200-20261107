"""Opt-in row-tiled model kernels for Wave 29C."""

# ruff: noqa: ANN001
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .model_w28c import _differentiate_swiglu, _save_joint
from .model_w30c import _swiglu_vec2d_kernel, model_w30c_enabled


def model_w29c_enabled(device: torch.device, feature: str) -> bool:
    if feature == "nr4" and model_w30c_enabled(device, "nr4"):
        return True
    mode = os.environ.get("LNET_MODEL_W29C", "off")
    if mode == "auto":
        mode = "combined_full" if torch.cuda.get_device_capability(device) == (9, 0) else "off"
    return mode in {
        "swiglu_vec2d": ("swiglu_vec2d", "combined_bwd", "combined_full"),
        "rmsnorm_bwd": ("rmsnorm_bwd", "rmsnorm_full", "combined_bwd", "combined_full"),
        "rmsnorm_full": ("rmsnorm_full", "combined_full"),
        "nr4": ("combined_bwd", "combined_full"),
    }[feature]


def _launch_swiglu(joint: Tensor, grad: Tensor, output: Tensor, *, backward: bool) -> None:
    hidden = joint.shape[-1] // 2
    rows = joint.numel() // (2 * hidden)
    block_cols = min(512, triton.next_power_of_2(hidden))
    block_rows = 8
    wrap_triton(_swiglu_vec2d_kernel)[
        (triton.cdiv(rows, block_rows), triton.cdiv(hidden, block_cols))
    ](joint, grad, output, rows, hidden, block_rows, block_cols, backward, enable_fp_fusion=False)


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


@triton_op("lnet::w29c_rmsnorm_forward", mutates_args=())
def channel_rmsnorm(x: Tensor, weight: Tensor, epsilon: float, save_inverse: bool) -> tuple[Tensor, Tensor]:  # noqa: E501, FBT001
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
    need_dx: bool, need_dw: bool, v2: bool = False,
) -> tuple[Tensor, Tensor]:
    width = x.shape[-1]
    rows = x.numel() // width
    dx = x.new_empty(x.shape if need_dx else (0,))
    block_rows = _rmsnorm_backward_rows(width, v2=v2)
    ctas = min(1056 if v2 else 264, triton.cdiv(rows, block_rows))
    partial = x.new_empty((ctas, width) if need_dw else (0,), dtype=torch.float32)
    wrap_triton(_rmsnorm_backward_kernel)[(ctas,)](
        x, weight, inverse, grad, dx, partial, rows, width, _row_pitch(x), _row_pitch(grad),
        block_rows, triton.next_power_of_2(width), need_dx, need_dw, v2, enable_fp_fusion=False,
    )
    dw = partial.sum(0).to(weight.dtype) if need_dw else weight.new_empty((0,))
    return dx, dw


def channel_rmsnorm_vjp(
    x: Tensor, weight: Tensor, inverse: Tensor, grad: Tensor, *,
    need_dx: bool, need_dw: bool, v2: bool = False,
) -> tuple[Tensor | None, Tensor | None]:
    if (
        not torch.is_grad_enabled() and (need_dx or need_dw)
        and inverse.is_contiguous() and rmsnorm_eligible(x, weight)
    ):
        pitch = _row_pitch(grad)
        if pitch != 0 and grad.numel() // grad.shape[-1] * pitch < 2**31:
            dx, dw = _channel_rmsnorm_backward(
                x, weight, inverse, grad, need_dx=need_dx, need_dw=need_dw, v2=v2,
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


channel_rmsnorm.register_autograd(_differentiate_rmsnorm, setup_context=_save_rmsnorm)


class _CompiledChannelRMSNorm(torch.autograd.Function):
    """Keep strided activations and output gradients in their original layouts."""

    @staticmethod
    def forward(
        ctx, x: Tensor, weight: Tensor, epsilon: float | None, fused: bool, v2: bool,  # noqa: FBT001
    ) -> Tensor:
        result, inverse = torch.ops.aten._fused_rms_norm(  # noqa: SLF001
            x, [x.shape[-1]], weight, epsilon,
        )
        ctx.fused = fused
        ctx.v2 = v2
        ctx.save_for_backward(x, weight, inverse.reshape(*x.shape[:-1], 1))
        return result

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor | None, Tensor | None, None, None, None]:
        x, weight, inverse = ctx.saved_tensors
        if ctx.fused:
            dx, dw = channel_rmsnorm_vjp(
                x, weight, inverse, grad,
                need_dx=ctx.needs_input_grad[0], need_dw=ctx.needs_input_grad[1], v2=ctx.v2,
            )
            return dx, dw, None, None, None
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
        return dx, dw, None, None, None
