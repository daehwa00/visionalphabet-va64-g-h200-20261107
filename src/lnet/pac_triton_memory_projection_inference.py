"""Inference candidates after planar scan/path collapse.

The scan is independently tiled over poles, whereas a memory projection reduces
over *all* poles. This module keeps that necessary tensor boundary and fuses the
following four coordinate GEMMs, BF16 coordinate arithmetic and residual add.
The optional full-output tile also fuses the following complex RMS norm. Both
candidates retain the four rounded GEMM results used by ``ComplexLinear``;
replacing them with one packed GEMM changes the BF16 model equation.
"""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING, Literal

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from torch.nn import functional

if TYPE_CHECKING:
    from .pac_complex_layers import ComplexLinear
    from .pac_gated_post_fusion import _GatedPostFusionBase

ProjectionFusion = Literal["projection_residual", "projection_residual_norm"]


def memory_projection_residual_norm_reference(
    memory: Tensor,
    weight_real: Tensor,
    weight_imag: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    epsilon: float = 1e-6,
    *,
    normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Preserve each coordinate GEMM, merge and RMS multiplication boundary."""
    real, imag = memory.unbind(0)
    wr, wi = weight_real.to(memory.dtype), weight_imag.to(memory.dtype)
    projected_real = functional.linear(real, wr) - functional.linear(imag, wi)
    projected_imag = functional.linear(real, wi) + functional.linear(imag, wr)
    merged = residual + torch.cat((projected_real, projected_imag), dim=-1)
    if not normalize:
        return merged, merged.new_empty((0,))
    real, imag = merged.chunk(2, dim=-1)
    energy = (real.float().square() + imag.float().square()).mean(-1, keepdim=True)
    inverse = torch.rsqrt(energy + epsilon).to(merged.dtype)
    weight = norm_weight.to(merged.dtype)
    normalized = torch.cat((real * inverse * weight, imag * inverse * weight), dim=-1)
    # CUDA packed RMSNorm feeds BF16 GEMMs even when the residual stays FP32.
    return merged, normalized.to(torch.bfloat16) if memory.is_cuda else normalized


@triton.jit
def _projection_merge_kernel(
    memory,
    weight_real,
    weight_imag,
    residual,
    norm_weight,
    merged,
    normalized,
    ROWS: tl.constexpr,
    INPUTS: tl.constexpr,
    OUTPUTS: tl.constexpr,
    EPSILON: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    out = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    rr = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    ir = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    ri = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    ii = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for start in range(tl.cdiv(INPUTS, BLOCK_K)):
        k = start * BLOCK_K + inner
        x_offset = row[:, None] * INPUTS + k[None, :]
        x_mask = (row[:, None] < ROWS) & (k[None, :] < INPUTS)
        xr = tl.load(memory + x_offset, x_mask, 0).to(tl.bfloat16)
        xi = tl.load(memory + ROWS * INPUTS + x_offset, x_mask, 0).to(tl.bfloat16)
        w_offset = out[None, :] * INPUTS + k[:, None]
        w_mask = (out[None, :] < OUTPUTS) & (k[:, None] < INPUTS)
        wr = tl.load(weight_real + w_offset, w_mask, 0).to(tl.bfloat16)
        wi = tl.load(weight_imag + w_offset, w_mask, 0).to(tl.bfloat16)
        rr = tl.dot(xr, wr, rr)
        ir = tl.dot(xr, wi, ir)
        ri = tl.dot(xi, wr, ri)
        ii = tl.dot(xi, wi, ii)
    # Each cuBLAS coordinate GEMM stores BF16 before the coordinate add/sub.
    pr = (rr.to(tl.bfloat16).to(tl.float32) - ii.to(tl.bfloat16).to(tl.float32)).to(
        tl.bfloat16,
    )
    pi = (ir.to(tl.bfloat16).to(tl.float32) + ri.to(tl.bfloat16).to(tl.float32)).to(
        tl.bfloat16,
    )
    offset = row[:, None] * (2 * OUTPUTS) + out[None, :]
    active = (row[:, None] < ROWS) & (out[None, :] < OUTPUTS)
    sr = tl.load(residual + offset, active, 0).to(tl.float32)
    si = tl.load(residual + offset + OUTPUTS, active, 0).to(tl.float32)
    # A BF16 projection added to an FP32 carry/source remains FP32. Only the
    # projection coordinates have the preceding BF16 storage boundaries.
    mr = (sr + pr.to(tl.float32)).to(merged.dtype.element_ty)
    mi = (si + pi.to(tl.float32)).to(merged.dtype.element_ty)
    tl.store(merged + offset, mr, active)
    tl.store(merged + offset + OUTPUTS, mi, active)
    if NORMALIZE:
        # NORMALIZE always uses one tile covering the complete output width.
        rf = tl.where(active, mr.to(tl.float32), 0.0)
        imf = tl.where(active, mi.to(tl.float32), 0.0)
        energy = tl.sum(rf * rf + imf * imf, axis=1) / OUTPUTS
        inverse = tl.rsqrt(energy + EPSILON)
        scale = tl.load(norm_weight + out, out < OUTPUTS, 0).to(tl.float32)
        if merged.dtype.element_ty == tl.bfloat16:
            inverse_bf16 = inverse.to(tl.bfloat16)
            scale_bf16 = scale.to(tl.bfloat16)
            nr = (mr * inverse_bf16[:, None]).to(tl.bfloat16) * scale_bf16[None, :]
            ni = (mi * inverse_bf16[:, None]).to(tl.bfloat16) * scale_bf16[None, :]
        else:
            nr = mr * inverse[:, None] * scale[None, :]
            ni = mi * inverse[:, None] * scale[None, :]
        tl.store(normalized + offset, nr, active)
        tl.store(normalized + offset + OUTPUTS, ni, active)


@triton_op("lnet::memory_projection_residual_norm_inference", mutates_args={})
def _projection_merge_op(
    memory: Tensor,
    weight_real: Tensor,
    weight_imag: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    epsilon: float,
    normalize: bool,
    block_rows: int,
    output_tile: int,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    if not memory.is_cuda:
        return memory_projection_residual_norm_reference(
            memory, weight_real, weight_imag, residual, norm_weight, epsilon,
            normalize=normalize,
        )
    inputs, outputs = weight_real.shape[1], weight_real.shape[0]
    rows = residual.numel() // (2 * outputs)
    merged = torch.empty_like(residual, dtype=torch.promote_types(memory.dtype, residual.dtype))
    normalized = (
        torch.empty_like(residual, dtype=torch.bfloat16) if normalize else residual.new_empty((0,))
    )
    if rows == 0:
        return merged, normalized
    block_n = max(16, triton.next_power_of_2(outputs)) if normalize else output_tile
    wrap_triton(_projection_merge_kernel)[
        (triton.cdiv(rows, block_rows), triton.cdiv(outputs, block_n))
    ](
        memory, weight_real, weight_imag, residual, norm_weight, merged, normalized,
        rows, inputs, outputs, epsilon, normalize, block_rows, block_n, 32,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return merged, normalized


def memory_projection_residual_norm(  # noqa: C901, PLR0912
    memory: Tensor,
    weight_real: Tensor,
    weight_imag: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    epsilon: float = 1e-6,
    *,
    normalize: bool = True,
    block_rows: int = 16,
    output_tile: int = 64,
    num_warps: int = 4,
) -> tuple[Tensor, Tensor]:
    """Fuse planar complex projection, residual, and optionally complex RMSNorm.

    Inputs are contiguous ``[2, ..., I]`` and ``[..., 2O]``. CUDA memory is BF16;
    residuals and weights may be FP32 or BF16. The returned ``merged`` is needed
    by the PostFusion residual. ``normalize=False`` returns an empty second
    tensor and permits smaller output tiles than the complete mode reduction.
    Merging promotes the coordinate/residual dtype; CUDA normalization produces
    BF16 using the promoted merge's arithmetic, matching packed RMSNorm. The CPU
    reference retains the merge dtype, as does CPU RMSNorm. No autograd is provided.
    """
    if memory.ndim < 2 or memory.shape[0] != 2 or memory.shape[-1] <= 0:
        raise ValueError("memory must have contiguous [2, ..., input_modes] shape")
    if weight_real.ndim != 2 or weight_real.shape != weight_imag.shape:
        raise ValueError("projection weights must have matching matrix shapes")
    outputs, inputs = weight_real.shape
    if outputs <= 0 or inputs != memory.shape[-1]:
        raise ValueError("projection weight dimensions do not match memory")
    if residual.shape != (*memory.shape[1:-1], 2 * outputs):
        raise ValueError("residual shape must match projected packed memory")
    if norm_weight.shape != (outputs,):
        raise ValueError("normalization weight must match output_modes")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    tensors = (memory, weight_real, weight_imag, residual, norm_weight)
    if not all(tensor.device == memory.device for tensor in tensors):
        raise ValueError("all tensors must share a device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous")
    if memory.dtype not in (torch.float32, torch.bfloat16) or residual.dtype not in (
        torch.float32, torch.bfloat16,
    ):
        raise TypeError("memory and residual require FP32 or BF16 dtype")
    if any(tensor.dtype not in (torch.float32, torch.bfloat16) for tensor in tensors):
        raise TypeError("projection and normalization weights require FP32 or BF16 dtype")
    if memory.is_cuda and memory.dtype is not torch.bfloat16:
        raise TypeError("CUDA projection fusion requires BF16 memory")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("memory projection fusion is inference only")
    if block_rows not in (16, 32) or output_tile not in (32, 64, 128) or num_warps not in (4, 8):
        raise ValueError("unsupported projection launch geometry")
    return _projection_merge_op(
        memory, weight_real, weight_imag, residual, norm_weight, epsilon,
        normalize, block_rows, output_tile, num_warps,
    )


def memory_projection_post_fusion(
    memory: Tensor,
    residual: Tensor,
    projection: ComplexLinear,
    post_fusion: _GatedPostFusionBase,
    *,
    fusion: ProjectionFusion = "projection_residual",
    block_rows: int = 16,
    output_tile: int = 64,
    num_warps: int = 4,
) -> Tensor:
    """Replace planar memory projection + ``forward_residual_packed``.

    Install at the same-resolution boundary *after* scan/path collapse, or in a
    transition after its carry projection. This helper owns no parameters and
    makes no mutation, so it composes with independent Reader/scan installers.
    """
    if fusion not in ("projection_residual", "projection_residual_norm"):
        raise ValueError("unknown memory projection fusion")
    merged, normalized = memory_projection_residual_norm(
        memory, projection.weight_real, projection.weight_imag, residual,
        post_fusion.norm.weight, post_fusion.norm.epsilon,
        normalize=fusion == "projection_residual_norm",
        block_rows=block_rows, output_tile=output_tile, num_warps=num_warps,
    )
    if fusion == "projection_residual":
        return post_fusion.forward_packed(merged)
    return merged + post_fusion._packed_update(normalized)
