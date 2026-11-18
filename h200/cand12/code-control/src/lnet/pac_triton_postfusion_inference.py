"""Forward-only Tensor Core candidates for Cartesian gated PostFusion.

Every GEMM output, SiLU output, and gated product retains the original BF16
rounding boundary. The kernels change storage and scheduling, not the learned
widely-linear map. They deliberately do not implement backward.
"""

from __future__ import annotations

# Triton pointer arguments and constexpr expressions are resolved by its compiler.
# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


@triton.jit
def _gated_coordinates(real, imag, logits):
    real = real.to(tl.bfloat16).to(tl.float32)
    imag = imag.to(tl.bfloat16).to(tl.float32)
    logits = logits.to(tl.bfloat16).to(tl.float32)
    gate = (logits / (1.0 + tl.exp(-logits))).to(tl.bfloat16).to(tl.float32)
    return (real * gate).to(tl.bfloat16), (imag * gate).to(tl.bfloat16)


@triton.jit
def _joint_gated_kernel(
    X,
    WEIGHT,
    GATED,
    ROWS: tl.constexpr,
    INPUT: tl.constexpr,
    HIDDEN: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    hidden = tl.program_id(1) * BH + tl.arange(0, BH)
    reduction = tl.arange(0, BK)
    real = tl.full((BM, BH), 0.0, tl.float32)
    imag = tl.full((BM, BH), 0.0, tl.float32)
    logits = tl.full((BM, BH), 0.0, tl.float32)
    for start in range(tl.cdiv(INPUT, BK)):
        k = start * BK + reduction
        x = tl.load(X + rows[:, None] * INPUT + k[None, :],
                    (rows[:, None] < ROWS) & (k[None, :] < INPUT), 0)
        offset = hidden[None, :] * INPUT + k[:, None]
        mask = (hidden[None, :] < HIDDEN) & (k[:, None] < INPUT)
        wr = tl.load(WEIGHT + offset, mask, 0)
        wi = tl.load(WEIGHT + offset + HIDDEN * INPUT, mask, 0)
        wg = tl.load(WEIGHT + offset + 2 * HIDDEN * INPUT, mask, 0)
        real = tl.dot(x, wr, real)
        imag = tl.dot(x, wi, imag)
        logits = tl.dot(x, wg, logits)
    gated_real, gated_imag = _gated_coordinates(real, imag, logits)
    offset = rows[:, None] * (2 * HIDDEN) + hidden[None, :]
    mask = (rows[:, None] < ROWS) & (hidden[None, :] < HIDDEN)
    tl.store(GATED + offset, gated_real, mask)
    tl.store(GATED + offset + HIDDEN, gated_imag, mask)


@triton.jit
def _output_residual_kernel(
    X,
    WEIGHT,
    RESIDUAL,
    OUTPUT,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    HIDDEN: tl.constexpr,
    FUSE_GATE: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    output = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    accumulator = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tl.cdiv(2 * HIDDEN, BK)):
        k = start * BK + reduction
        stride: tl.constexpr = 3 * HIDDEN if FUSE_GATE else 2 * HIDDEN
        value = tl.load(X + rows[:, None] * stride + k[None, :],
                        (rows[:, None] < ROWS) & (k[None, :] < 2 * HIDDEN), 0)
        if FUSE_GATE:
            logits = tl.load(X + rows[:, None] * stride + 2 * HIDDEN + (k % HIDDEN)[None, :],
                             (rows[:, None] < ROWS) & (k[None, :] < 2 * HIDDEN), 0)
            logits = logits.to(tl.float32)
            gate = (logits / (1.0 + tl.exp(-logits))).to(tl.bfloat16).to(tl.float32)
            value = (value.to(tl.float32) * gate).to(tl.bfloat16)
        weight = tl.load(WEIGHT + output[None, :] * (2 * HIDDEN) + k[:, None],
                         (output[None, :] < WIDTH) & (k[:, None] < 2 * HIDDEN), 0)
        accumulator = tl.dot(value, weight, accumulator)
    offset = rows[:, None] * WIDTH + output[None, :]
    mask = (rows[:, None] < ROWS) & (output[None, :] < WIDTH)
    # A BF16 linear result is rounded before the residual addition in the model.
    update = accumulator.to(tl.bfloat16).to(tl.float32)
    if ADD_RESIDUAL:
        update += tl.load(RESIDUAL + offset, mask, 0).to(tl.float32)
    tl.store(OUTPUT + offset, update, mask)


@triton.jit
def _full_kernel(
    X,
    JOINT,
    OUT_WEIGHT,
    RESIDUAL,
    OUTPUT,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    HIDDEN: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    output = tl.program_id(1) * BN + tl.arange(0, BN)
    hidden = tl.arange(0, BH)
    reduction = tl.arange(0, BK)
    real = tl.full((BM, BH), 0.0, tl.float32)
    imag = tl.full((BM, BH), 0.0, tl.float32)
    logits = tl.full((BM, BH), 0.0, tl.float32)
    for start in range(tl.cdiv(WIDTH, BK)):
        k = start * BK + reduction
        x = tl.load(X + rows[:, None] * WIDTH + k[None, :],
                    (rows[:, None] < ROWS) & (k[None, :] < WIDTH), 0)
        offset = hidden[None, :] * WIDTH + k[:, None]
        mask = (hidden[None, :] < HIDDEN) & (k[:, None] < WIDTH)
        wr = tl.load(JOINT + offset, mask, 0)
        wi = tl.load(JOINT + offset + HIDDEN * WIDTH, mask, 0)
        wg = tl.load(JOINT + offset + 2 * HIDDEN * WIDTH, mask, 0)
        real = tl.dot(x, wr, real)
        imag = tl.dot(x, wi, imag)
        logits = tl.dot(x, wg, logits)
    gated_real, gated_imag = _gated_coordinates(real, imag, logits)
    # Pad each Cartesian half independently; map weight rows around this padding.
    gated = tl.reshape(tl.permute(tl.join(gated_real, gated_imag), (0, 2, 1)), (BM, 2 * BH))
    packed_hidden = tl.arange(0, 2 * BH)
    h = packed_hidden % BH
    weight_column = (packed_hidden // BH) * HIDDEN + h
    weights = tl.load(OUT_WEIGHT + output[None, :] * (2 * HIDDEN) + weight_column[:, None],
                      (output[None, :] < WIDTH) & (h[:, None] < HIDDEN), 0)
    update = tl.dot(gated, weights).to(tl.bfloat16).to(tl.float32)
    offset = rows[:, None] * WIDTH + output[None, :]
    mask = (rows[:, None] < ROWS) & (output[None, :] < WIDTH)
    if ADD_RESIDUAL:
        update += tl.load(RESIDUAL + offset, mask, 0).to(tl.float32)
    tl.store(OUTPUT + offset, update, mask)


def _validate_bf16_rows(value: Tensor, weight: Tensor, width: int) -> None:
    if (not value.is_cuda or value.ndim < 1 or value.numel() == 0
            or value.shape[-1] != width or not value.is_contiguous()
            or value.dtype is not torch.bfloat16 or weight.device != value.device
            or weight.dtype is not torch.bfloat16 or not weight.is_contiguous()):
        raise ValueError("PostFusion requires contiguous nonempty CUDA BF16 rows and weights")


@triton_op("lnet::postfusion_joint_gated_inference", mutates_args={})
def joint_gated_inference(normalized: Tensor, joint_weight: Tensor) -> Tensor:
    """Compute value, SiLU gate, and their product without storing joint rows."""
    width = normalized.shape[-1]
    _validate_bf16_rows(normalized, joint_weight, width)
    if joint_weight.ndim != 2 or joint_weight.shape[1] != width or joint_weight.shape[0] % 3:
        raise ValueError("PostFusion joint projection has incompatible dimensions")
    hidden = joint_weight.shape[0] // 3
    rows = normalized.numel() // width
    gated = normalized.new_empty((*normalized.shape[:-1], 2 * hidden))
    wrap_triton(_joint_gated_kernel)[(triton.cdiv(rows, 32), triton.cdiv(hidden, 64))](
        normalized, joint_weight, gated, rows, width, hidden,
        BM=32, BH=64, BK=32, num_warps=4,
    )
    return gated


@triton_op("lnet::postfusion_output_residual_inference", mutates_args={})
def output_residual_inference(
    hidden_input: Tensor, out_weight: Tensor, residual: Tensor, fuse_gate: bool,
    add_residual: bool,
) -> Tensor:
    """Fuse the output GEMM and residual, optionally reading raw joint rows."""
    if out_weight.ndim != 2 or out_weight.shape[1] % 2:
        raise ValueError("PostFusion output projection has incompatible dimensions")
    width, doubled_hidden = out_weight.shape
    hidden = doubled_hidden // 2
    _validate_bf16_rows(hidden_input, out_weight, (3 if fuse_gate else 2) * hidden)
    if (residual.shape != (*hidden_input.shape[:-1], width) or not residual.is_contiguous()
            or residual.device != hidden_input.device
            or residual.dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError("PostFusion residual must have matching contiguous CUDA rows")
    output = torch.empty_like(residual)
    rows = residual.numel() // width
    wrap_triton(_output_residual_kernel)[(triton.cdiv(rows, 32), triton.cdiv(width, 64))](
        hidden_input, out_weight, residual, output, rows, width, hidden, fuse_gate, add_residual,
        BM=32, BN=64, BK=32, num_warps=4,
    )
    return output


@triton_op("lnet::postfusion_full_inference", mutates_args={})
def full_postfusion_inference(
    normalized: Tensor, joint_weight: Tensor, out_weight: Tensor, residual: Tensor,
    add_residual: bool,
) -> Tensor:
    """Run both GEMMs and gating in one CTA, with no global hidden activation."""
    width = normalized.shape[-1]
    _validate_bf16_rows(normalized, joint_weight, width)
    if (joint_weight.ndim != 2 or joint_weight.shape[1] != width
            or joint_weight.shape[0] % 3 or out_weight.ndim != 2):
        raise ValueError("PostFusion joint projection has incompatible dimensions")
    hidden = joint_weight.shape[0] // 3
    if (out_weight.shape != (width, 2 * hidden) or not out_weight.is_contiguous()
            or out_weight.dtype is not torch.bfloat16 or out_weight.device != normalized.device
            or residual.shape != normalized.shape or not residual.is_contiguous()
            or residual.device != normalized.device
            or residual.dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError("PostFusion full kernel has incompatible output or residual tensors")
    output = torch.empty_like(residual)
    rows = normalized.numel() // width
    wrap_triton(_full_kernel)[(triton.cdiv(rows, 16), triton.cdiv(width, 64))](
        normalized, joint_weight, out_weight, residual, output, rows, width, hidden, add_residual,
        BM=16, BN=64, BH=max(16, triton.next_power_of_2(hidden)), BK=32, num_warps=4,
    )
    return output
