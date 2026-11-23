"""Tensor Core alternative for the forward and direct backward experiment."""

# ruff: noqa: ANN001, ANN201, ANN202, ANN205, N803, N806, PLR0915
import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def patch(pixel, k, H: tl.constexpr, W: tl.constexpr, CI: tl.constexpr, CL: tl.constexpr):
    OH: tl.constexpr = (H + 1) // 2
    OW: tl.constexpr = (W + 1) // 2
    batch = pixel // (OH * OW)
    row = 2 * (pixel // OW % OH) - 1
    col = 2 * (pixel % OW) - 1
    # HWC patch order makes adjacent K lanes coalesced for training images.
    i = row[:, None] + (k // (3 * CI))[None, :]
    j = col[:, None] + (k // CI % 3)[None, :]
    ch = k % CI
    if CL:
        offset = ((batch[:, None] * H + i) * W + j) * CI + ch[None, :]
    else:
        offset = ((batch[:, None] * CI + ch[None, :]) * H + i) * W + j
    valid = (i >= 0) & (i < H) & (j >= 0) & (j < W) & (k[None, :] < 9 * CI)
    return offset, valid


@triton.jit
def _stages(  # noqa: C901, PLR0912
    X,
    KERNEL,
    G,
    B,
    CB,
    Y,
    DY,
    STAT,
    PART,
    H: tl.constexpr,
    W: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    N: tl.constexpr,
    EP: tl.constexpr,
    CL: tl.constexpr,
    BIAS: tl.constexpr,
    BWD: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    TILE_REDUCE: tl.constexpr,
):
    k = tl.arange(0, K)
    c = tl.arange(0, CO)
    m = tl.arange(0, M)
    worker = tl.program_id(0)
    weights = tl.load(KERNEL + k[:, None] * CO + c[None, :], k[:, None] < 9 * CI, 0)
    gamma = tl.load(G + c)
    beta = tl.load(B + c)
    if BIAS:
        bias = tl.load(CB + c)
    if BWD:
        dw = tl.full((K, CO), 0, tl.float32)
        if TILE_REDUCE:
            dg = tl.full((M, CO), 0, tl.float32)
            db = tl.full((M, CO), 0, tl.float32)
            dc = tl.full((M, CO), 0, tl.float32)
        else:
            dg = tl.full((CO,), 0, tl.float32)
            db = tl.full((CO,), 0, tl.float32)
            dc = tl.full((CO,), 0, tl.float32)
        stride = tl.num_programs(0) * M
    else:
        stride = N
    for start in range(worker * M, N, stride):
        pixel = start + m
        offset, valid = patch(pixel, k, H, W, CI, CL)
        a = tl.load(X + offset, valid & (pixel[:, None] < N), 0)
        values = tl.dot(a, weights).to(X.dtype.element_ty).to(tl.float32)
        if BIAS:
            values += bias[None, :]
        if BWD:
            mean = tl.load(STAT + 2 * pixel, pixel < N, 0)
            inv = tl.load(STAT + 2 * pixel + 1, pixel < N, 0)
            centered = values - mean[:, None]
        else:
            mean = tl.sum(values, 1) / CO
            centered = values - mean[:, None]
            inv = tl.rsqrt(tl.sum(centered * centered, 1) / CO + EP)
            tl.store(STAT + 2 * pixel, mean, pixel < N)
            tl.store(STAT + 2 * pixel + 1, inv, pixel < N)
        norm = centered * inv[:, None]
        affine = tl.fma(norm, gamma[None, :], beta[None, :])
        cdf = 0.5 * (1.0 + libdevice.erf(affine * 0.7071067811865476))
        output_offsets = pixel[:, None] * CO + c[None, :]
        if BWD:
            dy = tl.load(DY + output_offsets, pixel[:, None] < N, 0).to(tl.float32)
            grad = dy * (cdf + affine * 0.3989422804014327 * tl.exp(-0.5 * affine * affine))
            projected = grad * gamma[None, :]
            v = (
                projected
                - tl.sum(projected, 1)[:, None] / CO
                - norm * (tl.sum(projected * norm, 1)[:, None] / CO)
            ) * inv[:, None]
            dw = tl.dot(tl.trans(a), v.to(X.dtype.element_ty), dw)
            if TILE_REDUCE:
                dg += grad * norm
                db += grad
                if BIAS:
                    dc += v
            else:
                dg += tl.sum(grad * norm, 0)
                db += tl.sum(grad, 0)
                if BIAS:
                    dc += tl.sum(v, 0)
        else:
            tl.store(Y + output_offsets, affine * cdf, pixel[:, None] < N)
    if BWD:
        fields: tl.constexpr = 9 * CI + 2 + BIAS
        dst = PART + worker * fields * CO
        tl.store(dst + k[:, None] * CO + c[None, :], dw, k[:, None] < 9 * CI)
        tl.store(dst + 9 * CI * CO + c, tl.sum(dg, 0) if TILE_REDUCE else dg)
        tl.store(dst + (9 * CI + 1) * CO + c, tl.sum(db, 0) if TILE_REDUCE else db)
        if BIAS:
            tl.store(dst + (9 * CI + 2) * CO + c, tl.sum(dc, 0) if TILE_REDUCE else dc)


@triton.jit
def _reduce(
    P,
    DW,
    DG,
    DB,
    DC,
    CI: tl.constexpr,
    CO: tl.constexpr,
    BIAS: tl.constexpr,
    WORKERS: tl.constexpr,
    R: tl.constexpr,
):
    fields: tl.constexpr = 9 * CI + 2 + BIAS
    k = tl.program_id(0)
    worker = tl.arange(0, R)
    c = tl.arange(0, CO)
    v = tl.sum(
        tl.load(P + (worker[:, None] * fields + k) * CO + c[None, :], worker[:, None] < WORKERS, 0),
        0,
    )
    if k < CI * 9:
        tl.store(DW + k * CO + c, v)
    elif k == CI * 9:
        tl.store(DG + c, v)
    elif k == CI * 9 + 1:
        tl.store(DB + c, v)
    elif BIAS:
        tl.store(DC + c, v)


def _launch(x, w, g, b, cb, y, dy, stats, part, eps, workers, *, backward):
    ci, h, width = x.shape[1:]
    co = w.shape[-1]
    n = x.shape[0] * ((h + 1) // 2) * ((width + 1) // 2)
    cl = x.is_contiguous(memory_format=torch.channels_last)
    _stages[(workers,)](
        x,
        w,
        g,
        b,
        cb if cb is not None else g,
        y,
        dy,
        stats,
        part,
        h,
        width,
        ci,
        co,
        n,
        eps,
        cl,
        cb is not None,
        backward,
        int(os.environ.get("WAVE25C_TC_M", "32")) if backward else 32,
        triton.next_power_of_2(ci * 9),
        os.environ.get("WAVE25C_TC_TILE_REDUCE") == "1",
        num_warps=int(os.environ.get("WAVE25C_TC_WARPS", "4")) if backward else 4,
    )


class TensorCoreStem(torch.autograd.Function):
    """Reuse each input tile for convolution and parameter-gradient GEMMs."""

    @staticmethod
    def forward(ctx, x, w, g, b, cb, eps):
        batch, _ci, h, width = x.shape
        shape = (batch, (h + 1) // 2, (width + 1) // 2)
        y = x.new_empty((*shape, g.numel()))
        stats = g.new_empty((*shape, 2))
        _launch(
            x,
            w,
            g,
            b,
            cb,
            y,
            y,
            stats,
            stats,
            eps,
            triton.cdiv(stats.numel() // 2, 32),
            backward=False,
        )
        ctx.save_for_backward(x, w, g, b, cb, stats)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, g, b, cb, stats = ctx.saved_tensors
        block = int(os.environ.get("WAVE25C_TC_M", "32"))
        workers = min(
            int(os.environ.get("WAVE25C_TC_CTA", "512")), triton.cdiv(stats.numel() // 2, block)
        )
        ci, co = x.shape[1], g.numel()
        fields = ci * 9 + 2 + int(cb is not None)
        dw, dg, db = torch.empty_like(w), torch.empty_like(g), torch.empty_like(b)
        dc = torch.empty_like(cb) if cb is not None else None
        part = g.new_empty((workers, fields, co))
        _launch(x, w, g, b, cb, dy, dy.contiguous(), stats, part, ctx.eps, workers, backward=True)
        _reduce[(fields,)](
            part,
            dw,
            dg,
            db,
            dc if dc is not None else dg,
            ci,
            co,
            dc is not None,
            workers,
            triton.next_power_of_2(workers),
        )
        return None, dw, dg, db, dc, None


def run(
    x: torch.Tensor,
    w: torch.Tensor,
    cb: torch.Tensor | None,
    g: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    co, ci = w.shape[:2]
    packed = w.to(dtype).permute(2, 3, 1, 0).reshape(9 * ci, co).contiguous()
    return TensorCoreStem.apply(
        x.to(dtype),
        packed,
        g.float(),
        b.float(),
        cb.float() if cb is not None else None,
        eps,
    ).permute(0, 3, 1, 2)
