"""Channel RMSNorm and residual fusion for packed, channels-last features."""

# ruff: noqa: ANN001
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .model_w28c import model_cuda_supported
from .model_w29c import _row_pitch, channel_rmsnorm_vjp, rmsnorm_eligible


def rmsnorm_w34c_supported(x: Tensor, weight: Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[-1] == weight.numel()
        and rmsnorm_eligible(x, weight)
        and model_cuda_supported(x.device)
    )


@torch.compiler.assume_constant_result
def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _forward_tile(width: int, rows: int, sms: int) -> tuple[int, int, int]:
    cols = triton.next_power_of_2(width)
    warps = 4 if cols >= 512 else 2
    block_rows = max(2, 1024 // cols)
    if rows < 2 * sms * block_rows:
        block_rows = max(1, block_rows // 2)
    return block_rows, warps, 1


@triton.jit
def _rmsnorm_w34c_forward_kernel(
    x, residual, weight, y, x_sum, inverse, rows,
    width: tl.constexpr, x_pitch: tl.constexpr, residual_pitch: tl.constexpr,
    eps: tl.constexpr, block_rows: tl.constexpr, block_cols: tl.constexpr,
    add_residual: tl.constexpr, save_inverse: tl.constexpr,
) -> None:
    row = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    col = tl.arange(0, block_cols)
    mask = (row[:, None] < rows) & (col[None, :] < width)
    dense = row[:, None] * width + col[None, :]
    dtype: tl.constexpr = x.dtype.element_ty
    value = tl.load(x + row[:, None] * x_pitch + col[None, :], mask, 0).to(tl.float32)
    if add_residual:
        other = tl.load(
            residual + row[:, None] * residual_pitch + col[None, :], mask, 0,
        ).to(tl.float32)
        rounded = (value + other).to(dtype)
        tl.store(x_sum + dense, rounded, mask)
        value = rounded.to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, axis=1) / width + eps)
    scale = tl.load(weight + col, col < width, 0).to(tl.float32)
    tl.store(y + dense, (value * inv[:, None]) * scale[None, :], mask)
    if save_inverse:
        tl.store(inverse + row, inv, row < rows)


@triton.jit
def _residual_rmsnorm_backward_kernel(
    x, weight, inverse, dy, skip, dx, partial, rows,
    width: tl.constexpr, dy_pitch: tl.constexpr, skip_pitch: tl.constexpr,
    block_rows: tl.constexpr, block_cols: tl.constexpr,
    need_dx: tl.constexpr, need_dw: tl.constexpr, has_skip: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    col = tl.arange(0, block_cols)
    col_mask = col < width
    if need_dx:
        scale = tl.load(weight + col, col_mask, 0).to(tl.float32)
    if need_dw:
        accum = tl.zeros((block_rows, block_cols), tl.float32)
    for start in range(pid * block_rows, rows, tl.num_programs(0) * block_rows):
        row = start + tl.arange(0, block_rows)
        mask = (row[:, None] < rows) & col_mask[None, :]
        dense = row[:, None] * width + col[None, :]
        inv = tl.load(inverse + row, row < rows, 0)
        value = tl.load(x + dense, mask, 0).to(tl.float32)
        grad = tl.load(dy + row[:, None] * dy_pitch + col[None, :], mask, 0).to(tl.float32)
        normalized = value * inv[:, None]
        if need_dx:
            scaled = grad * scale[None, :]
            projection = tl.sum(normalized * scaled, 1)
            delta = (scaled - (normalized / width) * projection[:, None]) * inv[:, None]
            if has_skip:
                direct = tl.load(skip + row[:, None] * skip_pitch + col[None, :], mask, 0)
                delta = delta.to(dx.dtype.element_ty).to(tl.float32) + direct.to(tl.float32)
            tl.store(dx + dense, delta, mask)
        if need_dw:
            accum += grad * normalized
    if need_dw:
        tl.store(partial + pid * width + col, tl.sum(accum, 0), col_mask)


@triton_op("lnet::w34c_rmsnorm_forward", mutates_args=())
def rmsnorm_w34c(
    x: Tensor, residual: Tensor | None, weight: Tensor, epsilon: float, *, save_inverse: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    width = x.shape[-1]
    rows = x.numel() // width
    output = x.new_empty(x.shape)
    added = x.new_empty(x.shape if residual is not None else (0,))
    inverse = x.new_empty((*x.shape[:-1], 1) if save_inverse else (0,), dtype=torch.float32)
    block_rows, warps, stages = _forward_tile(width, rows, _sm_count(x.device))
    wrap_triton(_rmsnorm_w34c_forward_kernel)[(triton.cdiv(rows, block_rows),)](
        x, residual if residual is not None else x, weight, output, added, inverse, rows,
        width, _row_pitch(x), _row_pitch(residual) if residual is not None else 0,
        epsilon, block_rows, triton.next_power_of_2(width), residual is not None, save_inverse,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False,
    )
    return output, added, inverse


@triton_op("lnet::w34c_residual_rmsnorm_backward", mutates_args=())
def _residual_rmsnorm_backward(
    x: Tensor, weight: Tensor, inverse: Tensor, grad: Tensor, skip: Tensor, *,
    need_dx: bool, need_dw: bool,
) -> tuple[Tensor, Tensor]:
    width = x.shape[-1]
    rows = x.numel() // width
    cols = triton.next_power_of_2(width)
    block_rows = max(2, 1024 // cols)
    ctas = min(4 * _sm_count(x.device), triton.cdiv(rows, block_rows))
    dx = x.new_empty(x.shape if need_dx else (0,))
    partial = x.new_empty((ctas, width) if need_dw else (0,), dtype=torch.float32)
    wrap_triton(_residual_rmsnorm_backward_kernel)[(ctas,)](
        x, weight, inverse, grad, skip, dx, partial, rows, width,
        _row_pitch(grad), _row_pitch(skip), block_rows, cols, need_dx, need_dw, True,  # noqa: FBT003
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    dw = partial.sum(0).to(weight.dtype) if need_dw else weight.new_empty((0,))
    return dx, dw


def _save_rmsnorm_w34c(ctx, inputs, keyword_only_inputs, output) -> None:  # noqa: ARG001
    x, residual, weight, epsilon = inputs
    _, added, inverse = output
    ctx.mark_non_differentiable(inverse)
    ctx.set_materialize_grads(False)
    ctx.epsilon = epsilon
    ctx.save_for_backward(x if residual is None else added, weight, inverse)


def _differentiate_rmsnorm_w34c(
    ctx, grad: Tensor | None, skip: Tensor | None, inverse_grad: Tensor | None,  # noqa: ARG001
) -> tuple:
    x, weight, inverse = ctx.saved_tensors
    need_x, need_residual, need_weight, _ = ctx.needs_input_grad
    need_dx = need_x or need_residual
    dx, dw = skip, None
    if grad is not None:
        if (
            skip is not None and need_dx and not torch.is_grad_enabled()
            and rmsnorm_eligible(grad, weight) and rmsnorm_eligible(skip, weight)
        ):
            dx, dw = _residual_rmsnorm_backward(
                x, weight, inverse, grad, skip, need_dx=True, need_dw=need_weight,
            )
        else:
            if torch.is_grad_enabled():
                inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + ctx.epsilon)
            dx, dw = channel_rmsnorm_vjp(
                x, weight, inverse, grad, need_dx=need_dx, need_dw=need_weight,
            )
            if dx is not None and skip is not None:
                dx = dx + skip
    return dx if need_x else None, dx if need_residual else None, dw if need_weight else None, None


rmsnorm_w34c.register_autograd(_differentiate_rmsnorm_w34c, setup_context=_save_rmsnorm_w34c)
