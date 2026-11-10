"""Optional fused Triton kernels for head-wise LRCP inference and training.

The canonical PyTorch path remains the training/reference implementation.  The
kernel here fuses residual formation, both low-rank projections, scaling, and
the final residual add without materializing intermediate full-size tensors.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - optional acceleration
    triton = None
    tl = None


__all__ = ["fused_lrcp_bshd", "fused_lrcp_bshd_autograd", "triton_lrcp_available"]

# This changes only the explicitly selected Triton training path, and is
# separately opt-in. The original PyTorch reductions remain the default.
_FAST_FACTOR_GRADS = os.environ.get("APPLE_FUSED_FACTOR_GRADS", "0") == "1"
_FUSED_INPUT_FACTOR_GRADS = os.environ.get("APPLE_FUSED_INPUT_FACTOR_GRADS", "0") == "1"


def triton_lrcp_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _fused_lrcp_bshd_kernel(
        av_ptr,
        value_ptr,
        down_ptr,
        up_ptr,
        output_ptr,
        stride_ab: tl.constexpr,
        stride_as: tl.constexpr,
        stride_ah: tl.constexpr,
        stride_ad: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_dh: tl.constexpr,
        stride_dd: tl.constexpr,
        stride_dr: tl.constexpr,
        stride_uh: tl.constexpr,
        stride_ur: tl.constexpr,
        stride_ud: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        rank: tl.constexpr,
        gamma: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = row % heads
        token = row // heads
        seq = token % sequence
        batch = token // sequence
        d = tl.arange(0, BLOCK_D)
        r = tl.arange(0, BLOCK_R)
        d_mask = d < head_dim
        r_mask = r < rank

        av_offsets = batch * stride_ab + seq * stride_as + head * stride_ah + d * stride_ad
        value_offsets = batch * stride_vb + seq * stride_vs + head * stride_vh + d * stride_vd
        av = tl.load(av_ptr + av_offsets, mask=d_mask, other=0.0)
        value = tl.load(value_ptr + value_offsets, mask=d_mask, other=0.0)
        residual = (value - av).to(av_ptr.dtype.element_ty)

        down_offsets = head * stride_dh + d[:, None] * stride_dd + r[None, :] * stride_dr
        down = tl.load(down_ptr + down_offsets, mask=d_mask[:, None] & r_mask[None, :], other=0.0)
        raw = tl.sum(residual[:, None] * down, axis=0)
        # The reference writes the first einsum result in FP16/BF16 before the
        # second projection.  Preserve that quantization point explicitly.
        raw = raw.to(av_ptr.dtype.element_ty)

        up_offsets = head * stride_uh + r[:, None] * stride_ur + d[None, :] * stride_ud
        up = tl.load(up_ptr + up_offsets, mask=r_mask[:, None] & d_mask[None, :], other=0.0)
        branch = tl.sum(raw[:, None] * up, axis=0).to(av_ptr.dtype.element_ty)
        scaled_branch = (branch * gamma).to(av_ptr.dtype.element_ty)
        output = (av + scaled_branch).to(av_ptr.dtype.element_ty)
        output_offsets = batch * stride_ob + seq * stride_os + head * stride_oh + d * stride_od
        tl.store(output_ptr + output_offsets, output, mask=d_mask)


    @triton.jit
    def _fused_lrcp_bshd_tiled_kernel(
        av_ptr,
        value_ptr,
        down_ptr,
        up_ptr,
        output_ptr,
        raw_ptr,
        stride_ab: tl.constexpr,
        stride_as: tl.constexpr,
        stride_ah: tl.constexpr,
        stride_ad: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_dh: tl.constexpr,
        stride_dd: tl.constexpr,
        stride_dr: tl.constexpr,
        stride_uh: tl.constexpr,
        stride_ur: tl.constexpr,
        stride_ud: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_os: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        stride_rb: tl.constexpr,
        stride_rs: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_rr: tl.constexpr,
        tokens: tl.constexpr,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        rank: tl.constexpr,
        gamma: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        STORE_RAW: tl.constexpr,
    ):
        token = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        head = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        r = tl.arange(0, BLOCK_R)
        batch = token // sequence
        seq = token % sequence
        token_mask = token < tokens
        d_mask = d < head_dim
        r_mask = r < rank

        av_offsets = (
            batch[:, None] * stride_ab
            + seq[:, None] * stride_as
            + head * stride_ah
            + d[None, :] * stride_ad
        )
        value_offsets = (
            batch[:, None] * stride_vb
            + seq[:, None] * stride_vs
            + head * stride_vh
            + d[None, :] * stride_vd
        )
        mask_md = token_mask[:, None] & d_mask[None, :]
        av = tl.load(av_ptr + av_offsets, mask=mask_md, other=0.0)
        value = tl.load(value_ptr + value_offsets, mask=mask_md, other=0.0)
        residual = (value - av).to(av_ptr.dtype.element_ty)

        down_offsets = head * stride_dh + d[:, None] * stride_dd + r[None, :] * stride_dr
        down = tl.load(down_ptr + down_offsets, mask=d_mask[:, None] & r_mask[None, :], other=0.0)
        raw = tl.dot(residual, down, out_dtype=tl.float32).to(av_ptr.dtype.element_ty)
        if STORE_RAW:
            raw_offsets = (
                batch[:, None] * stride_rb
                + seq[:, None] * stride_rs
                + head * stride_rh
                + r[None, :] * stride_rr
            )
            tl.store(raw_ptr + raw_offsets, raw, mask=token_mask[:, None] & r_mask[None, :])

        up_offsets = head * stride_uh + r[:, None] * stride_ur + d[None, :] * stride_ud
        up = tl.load(up_ptr + up_offsets, mask=r_mask[:, None] & d_mask[None, :], other=0.0)
        branch = tl.dot(raw, up, out_dtype=tl.float32).to(av_ptr.dtype.element_ty)
        scaled_branch = (branch * gamma).to(av_ptr.dtype.element_ty)
        output = (av + scaled_branch).to(av_ptr.dtype.element_ty)
        output_offsets = (
            batch[:, None] * stride_ob
            + seq[:, None] * stride_os
            + head * stride_oh
            + d[None, :] * stride_od
        )
        tl.store(output_ptr + output_offsets, output, mask=mask_md)


    @triton.jit
    def _fused_lrcp_backward_inputs_kernel(
        grad_ptr,
        down_ptr,
        up_ptr,
        grad_av_ptr,
        grad_value_ptr,
        grad_raw_ptr,
        stride_gb: tl.constexpr,
        stride_gs: tl.constexpr,
        stride_gh: tl.constexpr,
        stride_gd: tl.constexpr,
        stride_dh: tl.constexpr,
        stride_dd: tl.constexpr,
        stride_dr: tl.constexpr,
        stride_uh: tl.constexpr,
        stride_ur: tl.constexpr,
        stride_ud: tl.constexpr,
        stride_ab: tl.constexpr,
        stride_as: tl.constexpr,
        stride_ah: tl.constexpr,
        stride_ad: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_rb: tl.constexpr,
        stride_rs: tl.constexpr,
        stride_rh: tl.constexpr,
        stride_rr: tl.constexpr,
        tokens: tl.constexpr,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        rank: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        token = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        head = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        r = tl.arange(0, BLOCK_R)
        batch = token // sequence
        seq = token % sequence
        token_mask = token < tokens
        d_mask = d < head_dim
        r_mask = r < rank
        mask_md = token_mask[:, None] & d_mask[None, :]

        grad_offsets = (
            batch[:, None] * stride_gb
            + seq[:, None] * stride_gs
            + head * stride_gh
            + d[None, :] * stride_gd
        )
        grad = tl.load(grad_ptr + grad_offsets, mask=mask_md, other=0.0)
        up_transpose_offsets = head * stride_uh + r[None, :] * stride_ur + d[:, None] * stride_ud
        up_transpose = tl.load(
            up_ptr + up_transpose_offsets,
            mask=d_mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        grad_raw = tl.dot(grad, up_transpose, out_dtype=tl.float32).to(grad_ptr.dtype.element_ty)
        raw_offsets = (
            batch[:, None] * stride_rb
            + seq[:, None] * stride_rs
            + head * stride_rh
            + r[None, :] * stride_rr
        )
        tl.store(grad_raw_ptr + raw_offsets, grad_raw, mask=token_mask[:, None] & r_mask[None, :])

        down_transpose_offsets = head * stride_dh + d[None, :] * stride_dd + r[:, None] * stride_dr
        down_transpose = tl.load(
            down_ptr + down_transpose_offsets,
            mask=r_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        grad_residual = tl.dot(grad_raw, down_transpose, out_dtype=tl.float32).to(grad_ptr.dtype.element_ty)
        grad_av = (grad - grad_residual).to(grad_ptr.dtype.element_ty)
        av_offsets = (
            batch[:, None] * stride_ab
            + seq[:, None] * stride_as
            + head * stride_ah
            + d[None, :] * stride_ad
        )
        value_offsets = (
            batch[:, None] * stride_vb
            + seq[:, None] * stride_vs
            + head * stride_vh
            + d[None, :] * stride_vd
        )
        tl.store(grad_av_ptr + av_offsets, grad_av, mask=mask_md)
        tl.store(grad_value_ptr + value_offsets, grad_residual, mask=mask_md)


def fused_lrcp_bshd(
    av: torch.Tensor,
    value_states: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Fused inference LRCP for ``[B,S,H,D]`` FP16/BF16 CUDA tensors.

    This function is intentionally inference-only.  Training and unsupported
    dtypes/layouts should use :meth:`lrcp.headwise.LRCP.forward_bshd`.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed.")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (av, value_states, down, up)):
        raise RuntimeError("fused_lrcp_bshd is inference-only and does not implement autograd.")
    if not (av.is_cuda and value_states.is_cuda and down.is_cuda and up.is_cuda):
        raise ValueError("fused_lrcp_bshd requires CUDA tensors.")
    if av.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != av.dtype for t in (value_states, down, up)):
        raise ValueError("fused_lrcp_bshd requires matching FP16 or BF16 tensors.")
    if any(t.device != av.device for t in (value_states, down, up)):
        raise ValueError("fused_lrcp_bshd requires tensors on the same device.")
    if av.ndim != 4 or av.shape != value_states.shape:
        raise ValueError(f"Expected matching [B,S,H,D] inputs, got {av.shape} and {value_states.shape}.")
    batch, sequence, heads, head_dim = av.shape
    if batch == 0 or sequence == 0:
        raise ValueError("Fused APPLE requires nonempty batch and sequence dimensions")
    if down.ndim != 3 or up.ndim != 3:
        raise ValueError(f"Expected rank-3 factors, got {down.shape} and {up.shape}.")
    rank = down.shape[-1]
    if down.shape != (heads, head_dim, rank) or up.shape != (heads, rank, head_dim):
        raise ValueError(f"Factor shapes do not match input: down={down.shape}, up={up.shape}, av={av.shape}.")
    if not (16 <= head_dim <= 256 and 0 < rank <= 32):
        raise ValueError(f"Unsupported fused LRCP shape D={head_dim}, rank={rank}.")

    return _fused_lrcp_bshd_unchecked(av, value_states, down, up, gamma)


def _fused_lrcp_bshd_unchecked(
    av: torch.Tensor,
    value_states: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Internal fast launcher after :class:`LRCP` has validated the tensors."""

    batch, sequence, heads, head_dim = av.shape
    rank = down.shape[-1]
    if batch * sequence == 1:
        # A one-row GEMV may round differently from a padded tensor-core GEMM.
        # Use the same ordinary PyTorch operator on every GPU for this case.
        residual = value_states - av
        raw = torch.bmm(residual.reshape(heads, 1, head_dim), down)
        branch = torch.bmm(raw, up).reshape_as(av)
        if gamma != 1.0:
            branch = branch * gamma
        return av + branch
    output = torch.empty(av.shape, device=av.device, dtype=av.dtype)
    block_d = triton.next_power_of_2(head_dim)
    # Tensor-core dot products require dimensions of at least 16.  Rank is
    # padded in registers only; no padded activation tensor is materialized.
    block_r = max(16, triton.next_power_of_2(rank))
    tokens = batch * sequence
    block_m = 16 if tokens <= 16 else 64
    warps, stages = 4, 3
    grid = (triton.cdiv(tokens, block_m), heads)
    _fused_lrcp_bshd_tiled_kernel[grid](
        av,
        value_states,
        down,
        up,
        output,
        output,
        *av.stride(),
        *value_states.stride(),
        *down.stride(),
        *up.stride(),
        *output.stride(),
        0,
        0,
        0,
        0,
        tokens=tokens,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        rank=rank,
        gamma=float(gamma),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        STORE_RAW=False,
        num_warps=warps,
        num_stages=stages,
    )
    return output


def _fused_lrcp_bshd_training_forward(
    av: torch.Tensor,
    value_states: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, sequence, heads, head_dim = av.shape
    rank = down.shape[-1]
    output = torch.empty(av.shape, device=av.device, dtype=av.dtype)
    raw = torch.empty((batch, sequence, heads, rank), device=av.device, dtype=av.dtype)
    block_m = 16 if batch * sequence <= 16 else 64
    block_d = triton.next_power_of_2(head_dim)
    block_r = max(16, triton.next_power_of_2(rank))
    tokens = batch * sequence
    grid = (triton.cdiv(tokens, block_m), heads)
    _fused_lrcp_bshd_tiled_kernel[grid](
        av,
        value_states,
        down,
        up,
        output,
        raw,
        *av.stride(),
        *value_states.stride(),
        *down.stride(),
        *up.stride(),
        *output.stride(),
        *raw.stride(),
        tokens=tokens,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        rank=rank,
        gamma=1.0,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        STORE_RAW=True,
        num_warps=4,
        num_stages=3,
    )
    return output, raw


def _fused_lrcp_bshd_backward_inputs(
    grad_output: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, sequence, heads, head_dim = grad_output.shape
    rank = down.shape[-1]
    grad_av = torch.empty_like(grad_output, memory_format=torch.contiguous_format)
    grad_value = torch.empty_like(grad_output, memory_format=torch.contiguous_format)
    grad_raw = torch.empty((batch, sequence, heads, rank), device=grad_output.device, dtype=grad_output.dtype)
    block_m = 16 if batch * sequence <= 16 else 64
    block_d = triton.next_power_of_2(head_dim)
    block_r = max(16, triton.next_power_of_2(rank))
    tokens = batch * sequence
    grid = (triton.cdiv(tokens, block_m), heads)
    _fused_lrcp_backward_inputs_kernel[grid](
        grad_output,
        down,
        up,
        grad_av,
        grad_value,
        grad_raw,
        *grad_output.stride(),
        *down.stride(),
        *up.stride(),
        *grad_av.stride(),
        *grad_value.stride(),
        *grad_raw.stride(),
        tokens=tokens,
        sequence=sequence,
        heads=heads,
        head_dim=head_dim,
        rank=rank,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_R=block_r,
        num_warps=4,
        num_stages=3,
    )
    return grad_av, grad_value, grad_raw


class _FusedLRCPAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, av, value_states, down, up):
        output, raw = _fused_lrcp_bshd_training_forward(av, value_states, down, up)
        ctx.save_for_backward(av, value_states, down, up, raw)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        av, value_states, down, up, raw = ctx.saved_tensors
        if _FAST_FACTOR_GRADS and _FUSED_INPUT_FACTOR_GRADS:
            from .triton_backward import fused_backward_and_factor_grads_bshd
            return fused_backward_and_factor_grads_bshd(
                av, value_states, raw, grad_output, down, up
            )
        # All inputs are consumed through strides, without a contiguous copy.
        grad_av, grad_value, grad_raw = _fused_lrcp_bshd_backward_inputs(grad_output, down, up)
        if _FAST_FACTOR_GRADS:
            from .triton_backward import factor_grads_bshd
            grad_down, grad_up = factor_grads_bshd(
                av, value_states, raw, grad_output, grad_raw
            )
        else:
            grad_up = torch.einsum("bshr,bshd->hrd", raw, grad_output)
            grad_down = torch.einsum("bshd,bshr->hdr", value_states - av, grad_raw)
        return grad_av, grad_value, grad_down, grad_up


def fused_lrcp_bshd_autograd(
    av: torch.Tensor,
    value_states: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Autograd-enabled fused LRCP for gamma=1 FP16/BF16 CUDA training."""

    if triton is None:
        raise RuntimeError("Triton is not installed.")
    # Reuse public shape/device/dtype validation without launching a kernel.
    if av.ndim != 4 or value_states.shape != av.shape:
        raise ValueError("Expected matching [B,S,H,D] inputs")
    if down.ndim != 3 or up.ndim != 3 or av.shape[0] == 0 or av.shape[1] == 0:
        raise ValueError("Expected rank-3 factors and nonempty inputs")
    h, d, r = av.shape[2], av.shape[3], down.shape[-1]
    if down.shape != (h, d, r) or up.shape != (h, r, d) or not (16 <= d <= 256 and 0 < r <= 32):
        raise ValueError("Unsupported fused APPLE factor shapes")
    if not av.is_cuda or av.dtype not in (torch.float16, torch.bfloat16) or any(
        t.device != av.device or t.dtype != av.dtype for t in (value_states, down, up)
    ):
        raise ValueError("Fused APPLE requires matching FP16/BF16 CUDA tensors")
    return _fused_lrcp_bshd_autograd_unchecked(av, value_states, down, up)


def _fused_lrcp_bshd_autograd_unchecked(av, value_states, down, up):
    """Internal dispatch after functional.apple_bshd has checked all inputs."""
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in (av, value_states, down, up)):
        return _fused_lrcp_bshd_unchecked(av, value_states, down, up, 1.0)
    return _FusedLRCPAutograd.apply(av, value_states, down, up)
