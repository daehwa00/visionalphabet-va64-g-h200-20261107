"""Horizontal complex adjoint for retained 56² paths."""
import triton
import triton.language as tl
from .r6_segment import _compose_complex, _segment


@triton.jit
def hback56(
    AR,
    AI,
    GR,
    GI,
    SR,
    SI,
    STATES,
    POS_R,
    POS_I,
    NEG_R,
    NEG_I,
    DAR,
    DAI,
    DGR,
    DGI,
    DS,
    partials,
    SEGMENT: tl.constexpr,
    PARTIAL: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
    STRIDE: tl.constexpr,
    BM: tl.constexpr,
    COARSE: tl.constexpr,
    GROUP: tl.constexpr,
):
    # The x grid dimension traverses adjacent mode tiles of the same horizontal line.
    line = tl.program_id(1)
    mode = tl.program_id(0) * BM + tl.arange(0, BM)[None, :]
    x = tl.arange(0, 64)[:, None]
    active = x < W
    offset = (line * W + x) * M + mode
    packed = (line * W + x) * STRIDE + mode
    pos = active & (x % 2 == 1) if COARSE else active
    neg = active & (x % 2 == 0) if COARSE else active
    ar = tl.load(AR + mode)
    ai = tl.load(AI + mode)
    gr = tl.load(GR + mode)
    gi = tl.load(GI + mode)
    pr = tl.load(POS_R + offset, pos, 0).to(tl.float32)
    pi = tl.load(POS_I + offset, pos, 0).to(tl.float32)
    nr = tl.load(NEG_R + offset, neg, 0).to(tl.float32)
    ni = tl.load(NEG_I + offset, neg, 0).to(tl.float32)
    if SEGMENT:
        lr, li = _segment(ar, -ai, pr, pi, True)
        rr, ri = _segment(ar, ai, nr, ni, False)
    else:
        ax = tl.where(active, ar, 1.0)
        ay = tl.where(active, ai, 0.0)
        pp = tl.associative_scan((ax, -ay, pr, pi), 0, _compose_complex, reverse=True)
        nn = tl.associative_scan((ax, ay, nr, ni), 0, _compose_complex)
        lr, li = pp[2], pp[3]
        rr, ri = nn[2], nn[3]
    sr = tl.load(SR + packed, active, 0).to(tl.float32)
    si = tl.load(SI + packed, active, 0).to(tl.float32)
    plane: tl.constexpr = B * H * W * M
    pr = tl.load(STATES + offset - M, active & (x > 0), 0).to(tl.float32)
    pi = tl.load(STATES + plane + offset - M, active & (x > 0), 0).to(tl.float32)
    nr = tl.load(STATES + 2 * plane + offset + M, x < W - 1, 0).to(tl.float32)
    ni = tl.load(STATES + 3 * plane + offset + M, x < W - 1, 0).to(tl.float32)
    out = (line * W + x) * (2 * M) + mode
    tl.store(DS + out, lr * gr + li * gi + rr * gr - ri * gi, active)
    tl.store(DS + out + M, -lr * gi + li * gr + rr * gi + ri * gr, active)
    if PARTIAL:
        prd = tl.sum(tl.where(active, lr * pr + li * pi + rr * nr + ri * ni, 0.0), 0)
        pid = tl.sum(tl.where(active, -lr * pi + li * pr + rr * ni - ri * nr, 0.0), 0)
        prg = tl.sum(tl.where(active, lr * sr + li * si + rr * sr + ri * si, 0.0), 0)
        pig = tl.sum(tl.where(active, -lr * si + li * sr + rr * si - ri * sr, 0.0), 0)
        base = (line * M + mode) * 4
        tl.store(partials + base, prd[None, :])
        tl.store(partials + base + 1, pid[None, :])
        tl.store(partials + base + 2, prg[None, :])
        tl.store(partials + base + 3, pig[None, :])
    else:
        tl.atomic_add(
            DAR + mode,
            tl.sum(tl.where(active, lr * pr + li * pi + rr * nr + ri * ni, 0.0), 0)[None, :],
            sem="relaxed",
        )
        tl.atomic_add(
            DAI + mode,
            tl.sum(tl.where(active, -lr * pi + li * pr + rr * ni - ri * nr, 0.0), 0)[None, :],
            sem="relaxed",
        )
        tl.atomic_add(
            DGR + mode,
            tl.sum(tl.where(active, lr * sr + li * si + rr * sr + ri * si, 0.0), 0)[None, :],
            sem="relaxed",
        )
        tl.atomic_add(
            DGI + mode,
            tl.sum(tl.where(active, -lr * si + li * sr + rr * si - ri * sr, 0.0), 0)[None, :],
            sem="relaxed",
        )
