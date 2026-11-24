"""Sweep Tensor Core launch geometries for the Stage 0 experiment."""

# ruff: noqa: ANN001, ANN205
import os

import torch
import triton

from lnet.fused_stem_ops import _launch_tensor_core_stem, stem_conv_tensor_reduce


class TensorCoreStem(torch.autograd.Function):
    """Reuse each input tile for convolution and parameter-gradient GEMMs."""

    @staticmethod
    def forward(ctx, x, w, g, b, cb, eps):
        batch, _ci, h, width = x.shape
        shape = (batch, (h + 1) // 2, (width + 1) // 2)
        y = x.new_empty((*shape, g.numel()))
        stats = g.new_empty((*shape, 2))
        _launch_tensor_core_stem(
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
        _launch_tensor_core_stem(
            x,
            w,
            g,
            b,
            cb,
            dy,
            dy.contiguous(),
            stats,
            part,
            ctx.eps,
            workers,
            backward=True,
            block=block,
            warps=int(os.environ.get("WAVE25C_TC_WARPS", "4")),
            tile_reduce=os.environ.get("WAVE25C_TC_TILE_REDUCE") == "1",
        )
        stem_conv_tensor_reduce[(fields,)](
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
