"""Triton kernels for packed SwiGLU activations and channel RMSNorm gradients."""

from __future__ import annotations

# Kernel pointer types are inferred by Triton.
# ruff: noqa: ANN001
import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


@triton.jit
def _gate_kernel(
    packed, dy, output, groups, hidden_groups: tl.constexpr, block: tl.constexpr,
    vector: tl.constexpr, backward: tl.constexpr,
) -> None:
    group = tl.program_id(0).to(tl.int64) * (block // vector) + tl.arange(0, block // vector)
    lane = tl.arange(0, vector)
    index = group[:, None] * vector + lane[None, :]
    value_group = group + group // hidden_groups * hidden_groups
    value_offset = value_group[:, None] * vector + lane[None, :]
    hidden: tl.constexpr = hidden_groups * vector
    valid = group[:, None] < groups
    value = tl.load(packed + value_offset, valid, other=0).to(tl.float32)
    gate = tl.load(packed + value_offset + hidden, valid, other=0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    dtype: tl.constexpr = packed.dtype.element_ty
    # Preserve the rounding boundaries of aten.silu and aten.mul in low precision.
    silu = (gate * sigmoid).to(dtype).to(tl.float32)
    if backward:
        grad = tl.load(dy + index, valid, other=0).to(tl.float32)
        dsilu = (grad * value).to(dtype).to(tl.float32)
        tl.store(output + value_offset, grad * silu, valid)
        dgate = dsilu * sigmoid * (1 + gate * (1 - sigmoid))
        tl.store(output + value_offset + hidden, dgate, valid)
    else:
        tl.store(output + index, value * silu, valid)


@triton_op("lnet::packed_swiglu_backward", mutates_args=())
def _packed_swiglu_backward(packed: Tensor, dy: Tensor) -> Tensor:
    dy = dy.contiguous()
    dx = torch.empty_like(packed)
    _launch_gate(packed, dy, dx, dy.numel(), dy.shape[-1], backward=True)
    return dx


@triton_op("lnet::packed_swiglu", mutates_args=())
def packed_swiglu(packed: Tensor) -> Tensor:
    hidden = packed.shape[-1] // 2
    output = packed.new_empty((*packed.shape[:-1], hidden))
    _launch_gate(packed, output, output, output.numel(), hidden, backward=False)
    return output


def _launch_gate(
    packed: Tensor, dy: Tensor, output: Tensor, size: int, hidden: int, *,
    backward: bool,
) -> None:
    block = 1024
    vector = math.gcd(hidden, 16 // packed.element_size())
    wrap_triton(_gate_kernel)[(triton.cdiv(size, block),)](
        packed, dy, output, size // vector, hidden // vector, block, vector, backward,
    )


def _save_gate(ctx, inputs, output) -> None:  # noqa: ARG001
    ctx.save_for_backward(inputs[0])


def _gate_backward(ctx, grad: Tensor) -> Tensor:
    (packed,) = ctx.saved_tensors
    return _packed_swiglu_backward(packed, grad)


packed_swiglu.register_autograd(_gate_backward, setup_context=_save_gate)


@triton.jit
def _rms_backward_kernel(
    x, dy, inverse, weight, dx, partial, rows, width: tl.constexpr,
    block_rows: tl.constexpr, block_columns: tl.constexpr, tiles: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    local_row = tl.arange(0, block_rows)
    col = tl.arange(0, block_columns)
    scale = tl.load(weight + col, col < width, other=0).to(tl.float32)
    weight_grad = tl.full((block_columns,), 0, tl.float32)
    for tile in range(tiles):
        row = (group.to(tl.int64) + tile * tl.num_programs(0)) * block_rows + local_row
        offset = row[:, None] * width + col[None, :]
        valid = (row[:, None] < rows) & (col[None, :] < width)
        reciprocal = tl.load(inverse + row, row < rows, other=0)
        normalized = tl.load(x + offset, valid, other=0).to(tl.float32) * reciprocal[:, None]
        grad = tl.load(dy + offset, valid, other=0).to(tl.float32)
        scaled = grad * scale[None, :]
        projection = tl.sum(normalized * scaled, 1)
        input_grad = (scaled - (normalized / width) * projection[:, None]) * reciprocal[:, None]
        tl.store(dx + offset, input_grad, valid)
        weight_grad += tl.sum(grad * normalized, 0)
    tl.store(partial + group.to(tl.int64) * width + col, weight_grad, col < width)


@triton_op("lnet::channel_rms_backward", mutates_args=())
def channel_rms_backward(
    x: Tensor, grad: Tensor, weight: Tensor, inverse: Tensor,
) -> tuple[Tensor, Tensor]:
    width = x.shape[-1]
    rows = x.numel() // width
    block_columns = triton.next_power_of_2(width)
    # Bound register pressure as the channel width grows.
    block_rows = min(16, 4096 // block_columns)
    # Reduce four row tiles into each partial, without global atomics.
    tiles = 4
    groups = triton.cdiv(rows, block_rows * tiles)
    partial = x.new_empty((groups, width), dtype=torch.float32)
    dx = torch.empty_like(x)
    wrap_triton(_rms_backward_kernel)[(groups,)](
        x, grad, inverse, weight, dx, partial, rows, width, block_rows, block_columns, tiles,
        enable_fp_fusion=False,
    )
    return dx, partial.sum(0).to(weight.dtype)
