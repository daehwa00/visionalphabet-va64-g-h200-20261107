"""Wave 30C dispatch and divisor-aligned SwiGLU tiling."""

# ruff: noqa: ANN001
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .model_w28c import _save_joint


def model_w30c_enabled(device: torch.device, feature: str) -> bool:
    mode = os.environ.get("LNET_MODEL_W30C", "off")
    if mode == "auto":
        return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0)
    return mode == "combined_v2" or (feature != "nr4" and mode == feature)


def swiglu_tile(hidden: int) -> tuple[int, int]:
    cols = min(64, hidden & -hidden)
    return 4096 // cols, cols


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
    block_rows, block_cols = swiglu_tile(hidden)
    wrap_triton(_swiglu_vec2d_kernel)[
        (triton.cdiv(rows, block_rows) * (hidden // block_cols),)
    ](
        joint, grad, output, rows, hidden, block_rows, block_cols, backward,
        divisor_tiled=True, num_warps=8, enable_fp_fusion=False,
    )


@triton_op("lnet::w30c_swiglu", mutates_args=())
def packed_swiglu_v2(joint: Tensor) -> Tensor:
    output = joint.new_empty((*joint.shape[:-1], joint.shape[-1] // 2))
    _launch_swiglu(joint, joint, output, backward=False)
    return output


@triton_op("lnet::w30c_swiglu_backward", mutates_args=())
def _swiglu_backward_v2(joint: Tensor, grad: Tensor) -> Tensor:
    output = torch.empty_like(joint)
    _launch_swiglu(joint, grad, output, backward=True)
    return output


def _differentiate_swiglu_v2(ctx, grad: Tensor) -> Tensor:
    (joint,) = ctx.saved_tensors
    if grad.is_contiguous() and not torch.is_grad_enabled():
        return _swiglu_backward_v2(joint, grad)
    value, gate = joint.chunk(2, -1)
    return torch.cat((
        grad * torch.nn.functional.silu(gate),
        torch.ops.aten.silu_backward(grad * value, gate),
    ), dim=-1)


packed_swiglu_v2.register_autograd(_differentiate_swiglu_v2, setup_context=_save_joint)
