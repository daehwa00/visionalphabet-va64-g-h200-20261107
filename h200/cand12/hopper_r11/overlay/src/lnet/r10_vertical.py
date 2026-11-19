"""Segmented retained 56² vertical forward."""

import triton
import triton.language as tl

from .effective_structural_scan import (
    _first_layer, _gains, _pack_fwd_once, _structural_fast_forward_2out, _take_low8,
)
from .r6_segment import _segment_pair


@triton.jit
def _scan_states_segmented(ar, ai, gr, gi, sar, sai, sbr, sbi):
    pr, pi, qr, qi = _segment_pair(
        ar, ai, gr * sar - gi * sai, gr * sai + gi * sar,
        gr * sbr - gi * sbi, gr * sbi + gi * sbr, False, False,
    )
    nr, ni, sr, si = _segment_pair(
        ar, -ai, gr * sar + gi * sai, gr * sai - gi * sar,
        gr * sbr + gi * sbi, gr * sbi - gi * sbr, True, False,
    )
    return pr, pi, qr, qi, nr, ni, sr, si


@triton.jit
def r10_vfwd56(
    AR,
    AI,
    GR,
    GI,
    SA,
    IA,
    SB,
    IB,
    VX,
    VY,
    WI,
    BI,
    WO,
    BO,
    OR,
    OI,
    H: tl.constexpr,
    W: tl.constexpr,
    M: tl.constexpr,
    EPS: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    DOUBLING: tl.constexpr,
):
    mode = _structural_fast_forward_2out(tl.program_id(0) // GROUP * BM + tl.arange(0, BM))
    line = tl.program_id(1) * GROUP + tl.program_id(0) % GROUP
    x = line % W
    batch = line // W
    y = tl.arange(0, 64)[:, None, None]
    m = mode[None, None, :]
    mask = y < H
    off = ((batch * H + y) * W + x) * M + m
    ar = tl.load(AR + m)
    ai = tl.load(AI + m)
    gr = tl.load(GR + m)
    gi = tl.load(GI + m)
    sa = tl.load(SA + off, mask, 0).to(tl.float32)
    ia = tl.load(IA + off, mask, 0).to(tl.float32)
    sb = tl.load(SB + off, mask, 0).to(tl.float32)
    ib = tl.load(IB + off, mask, 0).to(tl.float32)
    r0, i0, r1, i1, r2, i2, r3, i3 = _scan_states_segmented(ar, ai, gr, gi, sa, ia, sb, ib)
    g0, g1, g2, g3 = _gains(VX, VY, x, x, y, m, H, W, M, EPS)
    packed = _pack_fwd_once(r0, r1, r2, r3, i0, i1, i2, i3, g0, g1, g2, g3, mask, BM)
    coord = tl.arange(0, 16)
    _, hidden, _silu = _first_layer(packed, WI, BI, mode, coord)
    lanes = tl.arange(0, 8)
    h8 = _take_low8(hidden, BM, 64).to(tl.float32)
    wo_r = tl.load(WO + mode[:, None] * 16 + lanes[None, :]).to(tl.bfloat16).to(tl.float32)
    wo_i = tl.load(WO + mode[:, None] * 16 + 8 + lanes[None, :]).to(tl.bfloat16).to(tl.float32)
    bo_r = tl.load(BO + mode * 2).to(tl.bfloat16).to(tl.float32)
    bo_i = tl.load(BO + mode * 2 + 1).to(tl.bfloat16).to(tl.float32)
    out_r = (tl.sum(h8 * wo_r[:, None, :], 2) + bo_r[:, None]).to(tl.bfloat16)
    out_i = (tl.sum(h8 * wo_i[:, None, :], 2) + bo_i[:, None]).to(tl.bfloat16)
    flat = tl.reshape(tl.permute(off, (2, 0, 1)), (BM, 64))

    tl.store(
        OR + flat,
        out_r,
        tl.arange(0, 64)[None, :] < H,
    )
    tl.store(
        OI + flat,
        out_i,
        tl.arange(0, 64)[None, :] < H,
    )
