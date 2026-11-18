"""Experimental opt-in FP32 RMSNorm with fused input/weight-gradient reduction.

The original PyTorch forward formula remains traceable and fusible; only its
first-order backward uses custom Triton kernels.

The SM-sized partial-gradient strategy was inspired by Liger-Kernel rms_norm.py
at commit 83746f4007d18a8ecc16786e9848cc1e5fc4eae0 (version 0.8.5).
This implementation is
written for RealRMSNorm's FP32 semantics and does not import Liger. Liger is BSD
2-Clause, copyright 2024 LinkedIn; its RMSNorm also credits Apache-2.0 Unsloth.
Exact upstream provenance is recorded beside the experiment results.
"""

from __future__ import annotations

# Triton pointer and constexpr argument conventions match existing project kernels.
# ruff: noqa: ANN001, N803, SLF001
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor, nn
from torch.library import triton_op, wrap_triton

from .real_memory_backbone import RealRMSNorm


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": rows}, num_warps=warps)
        for rows in (1, 4, 8, 16)
        for warps in (4, 8)
    ],
    key=["ROWS", "WIDTH"],
)
@triton.jit
def _backward_kernel(
    X,
    W,
    R,
    DY,
    DX,
    PARTIAL,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    col = tl.arange(0, COLS)
    local_row = tl.arange(0, BLOCK_ROWS)
    w = tl.load(W + col, col < WIDTH, 0)
    accumulated = tl.full((BLOCK_ROWS, COLS), 0, tl.float32)
    for start in range(group * BLOCK_ROWS, ROWS, GROUPS * BLOCK_ROWS):
        row = start + local_row
        mask = (row[:, None] < ROWS) & (col[None, :] < WIDTH)
        x = tl.load(X + row[:, None] * WIDTH + col[None, :], mask, 0)
        dy = tl.load(DY + row[:, None] * WIDTH + col[None, :], mask, 0)
        r = tl.load(R + row, row < ROWS, 0)
        weighted = dy * w[None, :]
        # Preserve the native autograd graph's FP32 rounding boundaries:
        # rsqrt grad = (sum_grad * -0.5) * r.pow(3), then mean and square grads.
        # Keep /WIDTH after rsqrt_grad instead of algebraically reassociating it.
        sum_grad = tl.sum(weighted * x, 1)
        r_cubed = (r * r) * r
        rsqrt_grad = (sum_grad * -0.5) * r_cubed
        mean_grad = rsqrt_grad / WIDTH
        square_grad = mean_grad[:, None] * (2.0 * x)
        dx = weighted * r[:, None] + square_grad
        tl.store(DX + row[:, None] * WIDTH + col[None, :], dx, mask)
        accumulated += dy * (x * r[:, None])
    partial = tl.sum(accumulated, 0)
    tl.store(PARTIAL + group * WIDTH + col, partial, col < WIDTH)


@triton.jit
def _reduce_kernel(
    PARTIAL,
    DW,
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    REDUCE_ROWS: tl.constexpr,
    COLS: tl.constexpr,
) -> None:
    col = tl.program_id(0) * COLS + tl.arange(0, COLS)
    group = tl.arange(0, REDUCE_ROWS)
    value = tl.load(
        PARTIAL + group[:, None] * WIDTH + col[None, :],
        (group[:, None] < GROUPS) & (col[None, :] < WIDTH),
        0,
    )
    tl.store(DW + col, tl.sum(value, 0), col < WIDTH)


@triton_op("lnet::real_rms_norm_candidate_backward", mutates_args={})
def _backward_op(
    x: Tensor, weight: Tensor, r: Tensor, dy: Tensor, groups: int
) -> tuple[Tensor, Tensor]:
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    dy = dy.contiguous()
    dx = torch.empty_like(x)
    partial = torch.empty((groups, width), device=x.device, dtype=torch.float32)
    dw = torch.empty_like(weight)
    wrap_triton(_backward_kernel)[(groups,)](
        x,
        weight,
        r,
        dy,
        dx,
        partial,
        rows,
        width,
        groups,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    wrap_triton(_reduce_kernel)[(triton.cdiv(width, 32),)](
        partial,
        dw,
        width,
        groups,
        triton.next_power_of_2(groups),
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return dx, dw


class _AutogradContext(Protocol):
    groups: int
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class _FusedBackwardNorm(torch.autograd.Function):
    """Keep the exact reference forward graph available for Inductor fusion."""

    @staticmethod
    def forward(
        ctx: _AutogradContext, x: Tensor, weight: Tensor, epsilon: float, groups: int
    ) -> Tensor:
        inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon).to(x.dtype)
        ctx.save_for_backward(x, weight, inverse)
        ctx.groups = groups
        return x * inverse * weight.to(x.dtype)

    @staticmethod
    def backward(ctx: _AutogradContext, dy: Tensor) -> tuple[Tensor, Tensor, None, None]:
        x, weight, inverse = ctx.saved_tensors
        dx, dw = _backward_op(x, weight, inverse, dy, ctx.groups)
        return dx, dw, None, None


class _CandidateRealRMSNorm(RealRMSNorm):
    """Same module and parameter objects; unsupported inputs use the reference."""

    _norm_candidate_groups: int

    def forward(self, x: Tensor) -> Tensor:
        if (
            x.is_cuda
            and x.dtype == torch.float32
            and self.weight.dtype == torch.float32
            and x.is_contiguous()
            and self.weight.is_contiguous()
            and x.numel() > 0
            and x.shape[-1] == self.weight.numel()
            and x.shape[-1] <= 8192
        ):
            return _FusedBackwardNorm.apply(
                x, self.weight, self.epsilon, self._norm_candidate_groups
            )
        return super().forward(x)


def enable_real_memory_norm_candidate_(model: nn.Module) -> dict[str, int]:
    """Install before torch.compile, after moving the model to its CUDA device.

    This is an experimental first-order training candidate. It preserves module,
    Parameter, optimizer references and state_dict keys. It never caches weights.
    BF16, CPU, noncontiguous and oversized inputs retain the reference formula.
    Higher-order gradients are not supported by the custom backward operator.
    """
    modules = [module for module in model.modules() if isinstance(module, RealRMSNorm)]
    if not modules:
        raise ValueError("Expected RealRMSNorm modules")
    for module in modules:
        if not module.weight.is_cuda:
            raise ValueError("Move the model to CUDA before installing the candidate")
    for module in modules:
        sms = torch.cuda.get_device_properties(module.weight.device).multi_processor_count
        module._norm_candidate_groups = sms
        module.__class__ = _CandidateRealRMSNorm
    return {"norms": len(modules)}
