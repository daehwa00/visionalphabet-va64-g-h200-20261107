"""Opt-in Wave 28C packed activation and terminal-correlation dispatch."""

# ruff: noqa: ANN001
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


def model_w28c_enabled(device: torch.device, preset: str) -> bool:
    mode = os.environ.get("LNET_MODEL_W28C", "off")
    if mode == "auto":
        return torch.cuda.get_device_capability(device) == (9, 0)
    return mode in (preset, "combined")


@triton.jit
def triton_w28c_packed_swiglu_kernel(
    joint, gradient, output,
    elements, hidden: tl.constexpr, block: tl.constexpr, backward: tl.constexpr,
) -> None:
    index = tl.program_id(0) * block + tl.arange(0, block)
    value_offset = (index // hidden) * (2 * hidden) + index % hidden
    gate_offset = value_offset + hidden
    active = index < elements
    dtype: tl.constexpr = joint.dtype.element_ty
    value = tl.load(joint + value_offset, active, 0).to(tl.float32)
    gate = tl.load(joint + gate_offset, active, 0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    activated = (gate * sigmoid).to(dtype).to(tl.float32)
    if backward:
        grad = tl.load(gradient + index, active, 0).to(tl.float32)
        gate_grad = (grad * value).to(dtype).to(tl.float32)
        tl.store(output + value_offset, grad * activated, active)
        tl.store(output + gate_offset, gate_grad * (sigmoid * (1 + gate * (1 - sigmoid))), active)
    else:
        tl.store(output + index, value * activated, active)


@triton_op("lnet::w28c_swiglu", mutates_args=())
def packed_swiglu(joint: Tensor) -> Tensor:
    hidden = joint.shape[-1] // 2
    output = joint.new_empty((*joint.shape[:-1], hidden))
    block = 256
    wrap_triton(triton_w28c_packed_swiglu_kernel)[(triton.cdiv(output.numel(), block),)](
        joint, joint, output, output.numel(), hidden, block, False,  # noqa: FBT003
        enable_fp_fusion=False,
    )
    return output


@triton_op("lnet::w28c_swiglu_backward", mutates_args=())
def _packed_swiglu_backward(joint: Tensor, grad: Tensor) -> Tensor:
    output = torch.empty_like(joint)
    block = 256
    wrap_triton(triton_w28c_packed_swiglu_kernel)[(triton.cdiv(grad.numel(), block),)](
        joint, grad, output, grad.numel(), grad.shape[-1], block, True,  # noqa: FBT003
        enable_fp_fusion=False,
    )
    return output


def _save_joint(ctx, inputs, output) -> None:  # noqa: ARG001
    ctx.save_for_backward(*inputs)


def _differentiate_swiglu(ctx, grad: Tensor) -> Tensor:
    (joint,) = ctx.saved_tensors
    if grad.is_contiguous() and not torch.is_grad_enabled():
        return _packed_swiglu_backward(joint, grad)
    value, gate = joint.chunk(2, dim=-1)
    return torch.cat((grad * torch.nn.functional.silu(gate),
                      torch.ops.aten.silu_backward(grad * value, gate)), dim=-1)


packed_swiglu.register_autograd(_differentiate_swiglu, setup_context=_save_joint)
