"""Dual-pointer read projection and first-order gradients."""

# ruff: noqa: ANN001, ANN202, N803

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

MODES = (
    "off",
    "dual_mma_triton",
    "dual_fwd_only",
    "dual_all",
    "cublas_cat_ablation",
)
SPLIT_SIZE = 1024


if triton is not None:

    @triton.jit
    def _project_tile(
        X,
        W,
        acc,
        rows,
        cols,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        XS: tl.constexpr,
        WS: tl.constexpr,
    ):
        lane = tl.arange(0, 32)
        for tile in range(tl.cdiv(K, 32)):
            k = tile * 32 + lane
            x = tl.load(
                X + rows[:, None] * XS + k[None, :], (rows[:, None] < M) & (k[None, :] < K), 0
            )
            w = tl.load(
                W + cols[None, :] * WS + k[:, None], (cols[None, :] < N) & (k[:, None] < K), 0
            )
            acc = tl.dot(x, w, acc, input_precision="ieee")
        return acc

    @triton.jit
    def _forward_kernel(
        X,
        Y,
        W,
        V,
        B,
        OUT,
        M: tl.constexpr,
        N: tl.constexpr,
        KX: tl.constexpr,
        KY: tl.constexpr,
        XS: tl.constexpr,
        YS: tl.constexpr,
        WS: tl.constexpr,
        VS: tl.constexpr,
        BIAS: tl.constexpr,
        EXACT: tl.constexpr,
    ):
        rows = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
        cols = tl.program_id(1) * 64 + tl.arange(0, 64)
        zero = tl.full((32, 64), 0, tl.float32)
        left = _project_tile(X, W, zero, rows, cols, M, KX, N, XS, WS)
        dtype = OUT.dtype.element_ty
        if EXACT:
            right = _project_tile(Y, V, zero, rows, cols, M, KY, N, YS, VS)
            value = left.to(dtype).to(tl.float32) + right.to(dtype).to(tl.float32)
        else:
            value = _project_tile(Y, V, left, rows, cols, M, KY, N, YS, VS)
        if BIAS:
            if EXACT:
                value = value.to(dtype).to(tl.float32)
            bias = tl.load(B + cols, cols < N, 0)
            value = value + bias.to(tl.float32)[None, :]
        tl.store(
            OUT + rows[:, None] * N + cols[None, :],
            value,
            (rows[:, None] < M) & (cols[None, :] < N),
        )

    @triton.jit
    def _dual_grad_kernel(
        G,
        X,
        Y,
        DX,
        DY,
        P: tl.constexpr,
        R: tl.constexpr,
        KX: tl.constexpr,
        KY: tl.constexpr,
        GS0: tl.constexpr,
        GS1: tl.constexpr,
        XS: tl.constexpr,
        YS: tl.constexpr,
        OX: tl.constexpr,
        OY: tl.constexpr,
        SPLIT: tl.constexpr,
    ):
        tiles_k: tl.constexpr = tl.cdiv(max(KX, KY), 32)
        row = (tl.program_id(0) // tiles_k).to(tl.int64) * 32 + tl.arange(0, 32)
        col = tl.program_id(0) % tiles_k * 32 + tl.arange(0, 32)
        split = tl.program_id(1).to(tl.int64)
        lane = tl.arange(0, 32)
        left = tl.full((32, 32), 0, tl.float32)
        right = tl.full((32, 32), 0, tl.float32)
        for tile in range(tl.cdiv(SPLIT, 32)):
            r = split * SPLIT + tile * 32 + lane
            g = tl.load(
                G + row[:, None] * GS0 + r[None, :] * GS1, (row[:, None] < P) & (r[None, :] < R), 0
            )
            x = tl.load(
                X + r[:, None] * XS + col[None, :], (r[:, None] < R) & (col[None, :] < KX), 0
            )
            y = tl.load(
                Y + r[:, None] * YS + col[None, :], (r[:, None] < R) & (col[None, :] < KY), 0
            )
            left = tl.dot(g, x, left, input_precision="ieee")
            right = tl.dot(g, y, right, input_precision="ieee")
        row_offset = split * P + row[:, None]
        tl.store(
            DX + row_offset * OX + col[None, :], left, (row[:, None] < P) & (col[None, :] < KX)
        )
        tl.store(
            DY + row_offset * OY + col[None, :], right, (row[:, None] < P) & (col[None, :] < KY)
        )

    @triton.jit
    def _sum_splits(
        PARTIAL, DW, DV, N: tl.constexpr, KX: tl.constexpr, KY: tl.constexpr, SPLITS: tl.constexpr
    ):
        plane = tl.program_id(1).to(tl.int64)
        width: tl.constexpr = max(KX, KY)
        k = tl.where(plane == 0, KX, KY)
        index = tl.program_id(0) * 32 + tl.arange(0, 32)
        row = index // k
        col = index % k
        splits = tl.arange(0, 64).to(tl.int64)
        acc = tl.full((64, 32), 0, tl.float32)
        for block in range(tl.cdiv(SPLITS, 64)):
            split = block * 64 + splits
            offset = ((plane * SPLITS + split[:, None]) * N + row[None, :]) * width
            acc += tl.load(
                PARTIAL + offset + col[None, :], (split[:, None] < SPLITS) & (row[None, :] < N), 0
            )
        value = tl.sum(acc, axis=0)
        if plane == 0:
            tl.store(DW + index, value, row < N)
        else:
            tl.store(DV + index, value, row < N)


def _baseline(x: Tensor, y: Tensor, w: Tensor, v: Tensor, bias: Tensor | None) -> Tensor:
    output = functional.linear(x, w) + functional.linear(y, v)
    return output if bias is None else output + bias


def _can_launch(x: Tensor, y: Tensor, w: Tensor, v: Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and all(t.numel() > 0 and t.stride(-1) == 1 for t in (x, y, w, v))
    )


@torch.library.custom_op("lnet::readproj_forward", mutates_args=())
def _forward(
    x: Tensor,
    y: Tensor,
    w: Tensor,
    v: Tensor,
    bias: Tensor | None,
    mode: str,
) -> Tensor:
    if not _can_launch(x, y, w, v):
        if mode == "dual_mma_triton":
            with torch.autocast(x.device.type, enabled=False):
                value = x.float() @ w.float().T + y.float() @ v.float().T
                return (value if bias is None else value + bias.float()).to(x.dtype)
        return _baseline(x, y, w, v, bias)

    m, kx = x.shape
    n, ky = v.shape
    out = x.new_empty((m, n))
    with torch.cuda.device(x.device):
        _forward_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            x,
            y,
            w,
            v,
            bias if bias is not None else out,
            out,
            m,
            n,
            kx,
            ky,
            x.stride(0),
            y.stride(0),
            w.stride(0),
            v.stride(0),
            bias is not None,
            mode != "dual_mma_triton",
        )
    return out


@_forward.register_fake
def _fake_forward(
    x: Tensor,
    _y: Tensor,
    w: Tensor,
    _v: Tensor,
    _bias: Tensor | None,
    _mode: str,
) -> Tensor:
    return x.new_empty((x.shape[0], w.shape[0]))


def _dual_grad(
    g: Tensor,
    x: Tensor,
    y: Tensor,
    kx: int,
    ky: int,
    *,
    weight_grad: bool,
) -> tuple[Tensor, Tensor]:
    p, r = g.shape
    split = SPLIT_SIZE if weight_grad else r
    splits = triton.cdiv(r, split)
    if weight_grad:
        width = max(kx, ky)
        workspace = g.new_empty((2, splits, p, width), dtype=torch.float32)
        dx, dy = workspace.unbind(0)
    else:
        dx, dy = g.new_empty((p, kx)), g.new_empty((p, ky))
    with torch.cuda.device(g.device):
        _dual_grad_kernel[(triton.cdiv(p, 32) * triton.cdiv(max(kx, ky), 32), splits)](
            g,
            x,
            y,
            dx,
            dy,
            p,
            r,
            kx,
            ky,
            *g.stride(),
            x.stride(0),
            y.stride(0),
            dx.shape[-1],
            dy.shape[-1],
            split,
            num_stages=1 if g.dtype == torch.float32 else 3,
        )
    if weight_grad:
        left, right = g.new_empty((p, kx)), g.new_empty((p, ky))
        with torch.cuda.device(g.device):
            _sum_splits[(triton.cdiv(p * width, 32), 2)](
                workspace,
                left,
                right,
                p,
                kx,
                ky,
                splits,
            )
        return left, right
    return dx, dy


@torch.library.custom_op("lnet::readproj_backward", mutates_args=())
def _backward(
    g: Tensor,
    x: Tensor,
    y: Tensor,
    w: Tensor,
    v: Tensor,
    mode: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    kx, ky = w.shape[1], v.shape[1]
    fused = _can_launch(x, y, w, v)
    if mode == "dual_all" and fused:
        dx, dy = _dual_grad(g, w, v, kx, ky, weight_grad=False)
    else:
        dx, dy = g @ w, g @ v

    if mode in ("dual_all", "dual_mma_triton") and fused:
        dw, dv = _dual_grad(g.T, x, y, kx, ky, weight_grad=True)
    else:
        dw, dv = g.T @ x, g.T @ y
    return dx, dy, dw, dv


@_backward.register_fake
def _fake_backward(
    g: Tensor,
    x: Tensor,
    y: Tensor,
    w: Tensor,
    v: Tensor,
    _mode: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        g.new_empty(x.shape),
        g.new_empty(y.shape),
        g.new_empty(w.shape),
        g.new_empty(v.shape),
    )


def _setup_context(ctx: object, inputs: tuple, output: Tensor) -> None:  # noqa: ARG001
    x, y, w, v, bias, mode = inputs
    ctx.save_for_backward(x, y, w, v)
    ctx.mode = mode
    ctx.has_bias = bias is not None


def _autograd(
    ctx: object,
    grad: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, None]:
    mode = ctx.mode
    if mode == "cublas_cat_ablation":
        # Keep the GEMM and its views outside the custom op's alias-free boundary.
        x, y, w, v = ctx.saved_tensors
        dx, dy = (grad @ torch.cat((w, v), dim=1)).split((w.shape[1], v.shape[1]), dim=1)
        grads = dx, dy, grad.T @ x, grad.T @ y
    else:
        grads = _backward(grad, *ctx.saved_tensors, mode)
    return (*grads, grad.sum(0) if ctx.has_bias else None, None)


_forward.register_autograd(_autograd, setup_context=_setup_context)


def readproj(
    x: Tensor,
    y: Tensor,
    w: Tensor,
    v: Tensor,
    bias: Tensor | None,
    mode: str,
) -> Tensor:
    """Apply a read projection to tensors with equal leading dimensions."""
    if mode not in MODES:
        raise ValueError(f"Unknown LNET_READPROJ_OPT mode {mode!r}")
    if (
        mode == "off"
        or x.ndim == 0
        or x.shape[:-1] != y.shape[:-1]
        or x.device.type not in ("cpu", "cuda")
        or w.ndim != 2
        or v.ndim != 2
        or x.shape[-1] != w.shape[1]
        or y.shape[-1] != v.shape[1]
        or w.shape[0] != v.shape[0]
        or (bias is not None and bias.shape != (w.shape[0],))
        or any(t.device != x.device for t in (y, w, v))
        or (bias is not None and bias.device != x.device)
    ):
        return _baseline(x, y, w, v, bias)
    if torch.is_autocast_enabled(x.device.type) and all(
        t.dtype in (torch.bfloat16, torch.float16, torch.float32) for t in (x, y, w, v)
    ):
        dtype = torch.get_autocast_dtype(x.device.type)
        if bias is not None and bias.dtype != dtype:
            return _baseline(x, y, w, v, bias)
        x, y, w, v = x.to(dtype), y.to(dtype), w.to(dtype), v.to(dtype)
    if not (x.dtype == y.dtype == w.dtype == v.dtype) or x.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        return _baseline(x, y, w, v, bias)
    x2 = x.unsqueeze(0) if x.ndim == 1 else x.flatten(0, -2) if x.ndim > 2 else x
    y2 = y.unsqueeze(0) if y.ndim == 1 else y.flatten(0, -2) if y.ndim > 2 else y
    if bias is not None and (bias.dtype != x.dtype or bias.stride(0) != 1):
        return _baseline(x, y, w, v, bias)
    return _forward(x2, y2, w, v, bias, mode).reshape((*x.shape[:-1], w.shape[0]))
