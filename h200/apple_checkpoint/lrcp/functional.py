"""Stateless APPLE implementations; parameter names and checkpoint format stay put.

The low-rank intermediate is rounded in the GEMM dtype before the up projection
(the autocast dtype under AMP, otherwise the input dtype).
Do not replace this with R @ (D @ U): it changes both cost and low-precision math.
"""
from __future__ import annotations

import torch


def project_bshd(residual, down, up):
    """Two explicit batched GEMMs, shared over tokens but never over heads."""
    b, s, h, d = residual.shape
    rows = residual.reshape(b * s, h, d).transpose(0, 1)
    raw = torch.bmm(rows, down)
    branch = torch.bmm(raw, up)
    return branch.transpose(0, 1).reshape(b, s, h, d)


def apple_bshd(av, value, down, up, gamma=1.0, *, backend="torch", residual_detach=False):
    """Safe PyTorch default; opt-in Triton with fallback for unsupported inputs.

Training, including FP32 master weights under AMP, uses ordinary autograd unless
Triton is explicitly requested. No trainable state or per-shape caches are added.
Compiler/runtime errors in an explicitly requested kernel are surfaced to callers.
"""
    if backend not in ("torch", "reference", "triton"):
        raise ValueError(f"Unknown APPLE backend: {backend}")
    if av.ndim != 4 or av.shape != value.shape:
        raise ValueError("APPLE expects matching [B,S,H,D] inputs")
    if down.ndim != 3 or up.ndim != 3:
        raise ValueError("APPLE factors must have shapes [H,D,R] and [H,R,D]")
    h, d, r = av.shape[-2], av.shape[-1], down.shape[-1]
    if down.shape != (h, d, r) or up.shape != (h, r, d):
        raise ValueError("APPLE factors do not match the head dimensions")
    # A detached residual has the same forward values. Inference may use the
    # fused kernel; training must retain the detached-gradient PyTorch path.
    if (backend == "triton" and (not residual_detach or not torch.is_grad_enabled())
            and gamma == 1.0
            and d in (32, 64, 128, 256) and 0 < r <= 32
            and av.shape[0] > 0 and av.shape[1] > 0
            and av.is_cuda and av.dtype in (torch.float16, torch.bfloat16)
            and value.dtype == av.dtype
            and all(t.device == av.device for t in (value, down, up))):
        from .triton_headwise import triton_lrcp_available, _fused_lrcp_bshd_autograd_unchecked
        if triton_lrcp_available():
            kernel_down, kernel_up = down, up
            autocast_enabled = torch.is_autocast_enabled("cuda")
            autocast_matches = not autocast_enabled or torch.get_autocast_dtype("cuda") == av.dtype
            if autocast_enabled and autocast_matches:
                # Match einsum/bmm autocast, preserving FP32 master-weight
                # gradient edges. FP64 is not eligible for autocasting.
                eligible = (torch.float16, torch.bfloat16, torch.float32)
                if down.dtype in eligible and up.dtype in eligible:
                    kernel_down, kernel_up = down.to(av.dtype), up.to(av.dtype)
            if autocast_matches and kernel_down.dtype == av.dtype and kernel_up.dtype == av.dtype:
                return _fused_lrcp_bshd_autograd_unchecked(av, value, kernel_down, kernel_up)
    residual = value - av
    if residual_detach:
        residual = residual.detach()
    if backend == "reference":
        raw = torch.einsum("bshd,hdr->bshr", residual, down)
        branch = torch.einsum("bshr,hrd->bshd", raw, up)
    else:
        branch = project_bshd(residual, down, up)
    # The common gamma=1 case needs neither a full tensor nor a GPU launch.
    if gamma != 1.0:
        branch = branch * gamma
    return av + branch
