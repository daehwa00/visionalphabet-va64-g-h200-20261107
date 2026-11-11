"""Model-independent APPLE backward fusion with bounded FP32 partial storage.

No global residual or (in joint mode) grad_raw tensor is materialized. Dtype
rounding boundaries are preserved. This optional first-order path complements
the ordinary PyTorch reference; there is no model/GPU dispatch table.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _backward_partials(
    av_ptr, value_ptr, raw_ptr, grad_ptr, q_ptr, down_ptr, up_ptr,
    grad_av_ptr, grad_value_ptr, partial_ptr, grad_down_ptr, grad_up_ptr,
    ab: tl.constexpr, ass: tl.constexpr, ah: tl.constexpr, ad: tl.constexpr,
    vb: tl.constexpr, vs: tl.constexpr, vh: tl.constexpr, vd: tl.constexpr,
    rb: tl.constexpr, rs: tl.constexpr, rh: tl.constexpr, rr: tl.constexpr,
    gb: tl.constexpr, gs: tl.constexpr, gh: tl.constexpr, gd: tl.constexpr,
    qb: tl.constexpr, qs: tl.constexpr, qh: tl.constexpr, qr: tl.constexpr,
    dh: tl.constexpr, dd: tl.constexpr, dr: tl.constexpr,
    uh: tl.constexpr, ur: tl.constexpr, ud: tl.constexpr,
    TOKENS: tl.constexpr, SEQUENCE: tl.constexpr, HEADS: tl.constexpr,
    DIM: tl.constexpr, RANK: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr,
    JOINT: tl.constexpr,
):
    split, head = tl.program_id(0), tl.program_id(1)
    d, r = tl.arange(0, BLOCK_D), tl.arange(0, BLOCK_R)
    acc_down = tl.full((BLOCK_D, BLOCK_R), 0, tl.float32)
    acc_up = tl.full((BLOCK_R, BLOCK_D), 0, tl.float32)
    if JOINT:
        up_t = tl.load(up_ptr + head * uh + r[None, :] * ur + d[:, None] * ud,
                       (d[:, None] < DIM) & (r[None, :] < RANK), 0)
        down_t = tl.load(down_ptr + head * dh + d[None, :] * dd + r[:, None] * dr,
                         (d[None, :] < DIM) & (r[:, None] < RANK), 0)
    # Every split executes the same compile-time-known count. A pid-dependent
    # loop end otherwise retains unnecessary control flow even for one tile.
    for iteration in range(tl.cdiv(TOKENS, SPLITS * BLOCK_T)):
        start = (iteration * SPLITS + split) * BLOCK_T
        token = start + tl.arange(0, BLOCK_T)
        batch, seq = token // SEQUENCE, token % SEQUENCE
        md = (token[:, None] < TOKENS) & (d[None, :] < DIM)
        mr = (token[:, None] < TOKENS) & (r[None, :] < RANK)
        grad = tl.load(grad_ptr + batch[:, None]*gb + seq[:, None]*gs
                       + head*gh + d[None, :]*gd, md, 0)
        if JOINT:
            q = tl.dot(grad, up_t).to(grad_ptr.dtype.element_ty)
            grad_residual = tl.dot(q, down_t).to(grad_ptr.dtype.element_ty)
            out = (token[:, None] * HEADS + head) * DIM + d[None, :]
            tl.store(grad_value_ptr + out, grad_residual, md)
            tl.store(grad_av_ptr + out, (grad - grad_residual).to(grad_ptr.dtype.element_ty), md)
        else:
            q = tl.load(q_ptr + batch[:, None]*qb + seq[:, None]*qs
                        + head*qh + r[None, :]*qr, mr, 0)
        av = tl.load(av_ptr + batch[:, None]*ab + seq[:, None]*ass
                     + head*ah + d[None, :]*ad, md, 0)
        value = tl.load(value_ptr + batch[:, None]*vb + seq[:, None]*vs
                        + head*vh + d[None, :]*vd, md, 0)
        residual = (value - av).to(av_ptr.dtype.element_ty)
        raw = tl.load(raw_ptr + batch[:, None]*rb + seq[:, None]*rs
                      + head*rh + r[None, :]*rr, mr, 0)
        acc_down = tl.dot(tl.trans(residual), q, acc_down)
        acc_up = tl.dot(tl.trans(raw), grad, acc_up)
    offset = (head * SPLITS + split) * DIM * RANK + d[:, None] * RANK + r[None, :]
    valid = (d[:, None] < DIM) & (r[None, :] < RANK)
    if SPLITS == 1:
        # A single partial is already the final gradient. Keep the identical
        # FP32 accumulation and final cast, without scratch traffic or a launch.
        tl.store(grad_down_ptr + head*DIM*RANK + d[:, None]*RANK + r[None, :], acc_down, valid)
        tl.store(grad_up_ptr + head*RANK*DIM + r[:, None]*DIM + d[None, :], acc_up, tl.trans(valid))
    else:
        tl.store(partial_ptr + offset, acc_down, valid)
        tl.store(partial_ptr + HEADS * SPLITS * DIM * RANK + offset, tl.trans(acc_up), valid)


@triton.jit
def _reduce_partials(
    partial_ptr, grad_down_ptr, grad_up_ptr,
    HEADS: tl.constexpr, SPLITS: tl.constexpr, DIM: tl.constexpr, RANK: tl.constexpr,
    K: tl.constexpr, BLOCK_P: tl.constexpr,
):
    head, kind, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    k = tl.arange(0, K)
    p = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    offset = ((kind * HEADS + head) * SPLITS + k[:, None]) * DIM * RANK + p[None, :]
    part = tl.load(partial_ptr + offset,
                   (k[:, None] < SPLITS) & (p[None, :] < DIM * RANK), 0)
    result = tl.sum(part, axis=0)
    if kind == 0:
        tl.store(grad_down_ptr + head * DIM * RANK + p, result, p < DIM * RANK)
    else:
        tl.store(grad_up_ptr + head * DIM * RANK + (p % RANK) * DIM + p // RANK,
                 result, p < DIM * RANK)


def _launch(av, value, raw, grad_output, grad_raw=None, down=None, up=None):
    batch, sequence, heads, dim = av.shape
    rank = raw.shape[-1]
    if not (batch > 0 and sequence > 0 and heads > 0 and 16 <= dim <= 256 and 0 < rank <= 32):
        raise ValueError("Unsupported APPLE backward dimensions")
    if av.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("APPLE backward fusion requires FP16/BF16")
    joint = grad_raw is None
    # A uniform policy; bound scratch space independently of sequence length.
    block_tokens = 64
    splits = min(triton.cdiv(batch * sequence, block_tokens), 128)
    # The unused pointer is eliminated by the compile-time SPLITS condition.
    partial = (av if splits == 1 else
               torch.empty((2, heads, splits, dim, rank), device=av.device, dtype=torch.float32))
    grad_down = torch.empty((heads, dim, rank), device=av.device, dtype=av.dtype)
    grad_up = torch.empty((heads, rank, dim), device=av.device, dtype=av.dtype)
    grad_av = torch.empty(av.shape, device=av.device, dtype=av.dtype) if joint else av
    grad_value = torch.empty(av.shape, device=av.device, dtype=av.dtype) if joint else value
    q = raw if joint else grad_raw
    # Unused pointers/strides are eliminated by the compile-time JOINT flag.
    dstr, ustr = (down.stride(), up.stride()) if joint else ((0, 0, 0), (0, 0, 0))
    _backward_partials[(splits, heads)](
        av, value, raw, grad_output, q, down if joint else raw, up if joint else raw,
        grad_av, grad_value, partial, grad_down, grad_up,
        *av.stride(), *value.stride(), *raw.stride(), *grad_output.stride(), *q.stride(),
        *dstr, *ustr, batch * sequence, sequence, heads, dim, rank, splits,
        block_tokens, triton.next_power_of_2(dim), max(16, triton.next_power_of_2(rank)),
        joint, num_warps=4, num_stages=3,
    )
    if splits > 1:
        _reduce_partials[(heads, 2, triton.cdiv(dim * rank, 64))](
            partial, grad_down, grad_up, heads, splits, dim, rank,
            triton.next_power_of_2(splits), 64, num_warps=4,
        )
    return (grad_av, grad_value, grad_down, grad_up) if joint else (grad_down, grad_up)


def factor_grads_bshd(av, value, raw, grad_output, grad_raw):
    """Factor gradients with strided inputs and bounded FP32 workspace."""
    return _launch(av, value, raw, grad_output, grad_raw)


def fused_backward_and_factor_grads_bshd(av, value, raw, grad_output, down, up):
    """Joint backward; independent of batch, sequence and head counts."""
    return _launch(av, value, raw, grad_output, down=down, up=up)
