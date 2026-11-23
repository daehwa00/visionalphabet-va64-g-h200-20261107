"""CUDA Graph screening for candidate backbone kernels."""

# ruff: noqa: ANN001, ANN201, ANN202, B023, B905, C901, FBT003, N803, N812, T201
import statistics

import torch
import triton
import triton.language as tl
from direct import FusedStage
from direct import run as direct
from torch.nn import functional as F
from triton.language.extra.cuda import libdevice

from lnet.fused_stem_ops import fused_nhwc_layernorm_gelu


@triton.jit
def swiglu(X, DY, OUT, N: tl.constexpr, H: tl.constexpr, BWD: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = i // H * (2 * H) + i % H
    if H % 8 == 0:
        j = tl.max_contiguous(tl.multiple_of(j, 8), 8)
        valid = i // 8 < N // 8
    else:
        valid = i < N
    v = tl.load(X + j, valid, 0).to(tl.float32)
    g = tl.load(X + j + H, valid, 0).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-g))
    dtype: tl.constexpr = OUT.dtype.element_ty
    activated = (g * sigmoid).to(dtype).to(tl.float32)
    if BWD:
        dy = tl.load(DY + i, valid, 0).to(tl.float32)
        dact = (dy * v).to(dtype).to(tl.float32)
        tl.store(OUT + j, dy * activated, valid)
        tl.store(OUT + j + H, (dact * sigmoid) * (1.0 + g * (1.0 - sigmoid)), valid)
    else:
        tl.store(OUT + i, v * activated, valid)


def candidate(x, dy=None):
    h = x.shape[-1] // 2
    n = x.numel() // 2
    out = torch.empty_like(x) if dy is not None else x.new_empty((*x.shape[:-1], h))
    swiglu[(triton.cdiv(n, 1024),)](
        x,
        x if dy is None else dy,
        out,
        n,
        h,
        dy is not None,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def eager(x):
    v, g = x.chunk(2, -1)
    return v * F.silu(g)


def compare(arms):
    graphs = {}
    for name, run in arms.items():
        for _ in range(3):
            result = run()
            del result
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()
        graphs[name] = (graph, result)
    measurements = {name: [] for name in arms}
    for trial in range(8):
        for name in list(arms) if trial % 2 == 0 else list(reversed(arms)):
            graph, _ = graphs[name]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            measurements[name].append(start.elapsed_time(end) / 20)
    return {k: round(statistics.median(v), 5) for k, v in measurements.items()}


def stem(dtype, input_grad, *, autocast=False):
    parameter_dtype = torch.float32 if autocast else dtype
    x = torch.randn(128, 3, 224, 224, device="cuda", dtype=parameter_dtype)
    x = x.contiguous(memory_format=torch.channels_last).requires_grad_(input_grad)
    w = torch.randn(32, 3, 3, 3, device="cuda", dtype=parameter_dtype) / 27**0.5
    if autocast:
        w = w.contiguous(memory_format=torch.channels_last)
    w.requires_grad_()
    b = torch.zeros(32, device="cuda", requires_grad=True)
    gamma = torch.ones(32, device="cuda", requires_grad=True)
    beta = torch.zeros_like(gamma, requires_grad=True)
    params = (w, b, gamma, beta, x) if input_grad else (w, b, gamma, beta)

    @torch.autocast("cuda", dtype=dtype, enabled=autocast)
    def forward(kind):
        if kind.startswith("direct"):
            source = x.to(dtype)
            weight = w.to(dtype).contiguous()
        if kind == "direct-vjp":
            return FusedStage.apply(source, weight, b, gamma, beta)
        if kind == "direct-fused":
            return direct(source, weight, b, gamma, beta, True).permute(0, 3, 1, 2)
        if kind in {"direct", "direct-fma"}:
            y = direct(
                source,
                weight,
                b,
                gamma,
                beta,
                False,
                tensor_cores=kind == "direct",
            ).permute(0, 3, 1, 2)
        elif kind == "padded":
            source = F.pad(x, (0, 0, 0, 0, 0, 5))
            weight = F.pad(w, (0, 0, 0, 0, 0, 5)).contiguous(memory_format=torch.channels_last)
            y = F.conv2d(source, weight, None, 2, 1)
        elif kind == "im2col":
            rows = F.unfold(x, 3, padding=1, stride=2).transpose(1, 2)
            y = (rows @ w.flatten(1).T).reshape(128, 112, 112, 32).permute(0, 3, 1, 2)
        else:
            y = F.conv2d(x, w, None, 2, 1)
        return fused_nhwc_layernorm_gelu(
            y,
            gamma,
            beta,
            nchw=True,
            conv_bias=b,
            output_dtype=dtype,
        )

    dy = torch.randn_like(forward("baseline"))

    def step(kind):
        y = forward(kind)
        return y, torch.autograd.grad(y, params, dy)

    base, base_grad = step("baseline")
    trained = ("direct-vjp",) if autocast else ("padded", "im2col", "direct-vjp")
    for kind in trained:
        result, grads = step(kind)
        print(
            "stem parity",
            dtype,
            input_grad,
            kind,
            (result - base).abs().max().item(),
            [(a - b).abs().max().item() for a, b in zip(grads, base_grad)],
            flush=True,
        )
    for kind in ("direct", "direct-fused", "direct-fma"):
        result = forward(kind)
        print(
            "stem forward-only parity",
            dtype,
            input_grad,
            kind,
            (result - base).abs().max().item(),
            flush=True,
        )
    del base, base_grad, result, grads
    for mode, fn in (("fw", forward), ("step", step)):
        forwards = (
            ("direct", "direct-fused", "direct-fma")
            if autocast
            else ("padded", "im2col", "direct", "direct-fused", "direct-fma")
        )
        kinds = forwards if mode == "fw" else trained
        for kind in kinds:
            print(
                "stem ms",
                dtype,
                input_grad,
                mode,
                compare({"baseline": lambda: fn("baseline"), kind: lambda: fn(kind)}),
                flush=True,
            )


def activation():
    compiled = torch.compile(eager, fullgraph=True)
    shapes = (
        (401408, 384),
        (100352, 576),
        (25088, 768),
        (6272, 1536),
        (97, 383),
        (401408, 96),
        (100352, 144),
        (25088, 192),
        (6272, 384),
    )
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for n, h in shapes:
            x = torch.randn(n, 2 * h, device="cuda", dtype=dtype, requires_grad=True)
            y = eager(x)
            dy = torch.randn_like(y)
            ref = torch.autograd.grad(y, x, dy)[0]
            y2 = compiled(x)
            ref2 = torch.autograd.grad(y2, x, dy)[0]
            z, dz = candidate(x), candidate(x, dy)
            print(
                "swiglu parity",
                dtype,
                n,
                h,
                [
                    ((a != b).sum().item(), (a - b).abs().max().item())
                    for a, b in ((z, y), (dz, ref), (z, y2), (dz, ref2))
                ],
                flush=True,
            )
            del y, y2, ref, ref2, z, dz
            if dtype != torch.float32:

                def step():
                    out = compiled(x)
                    return out, torch.autograd.grad(out, x, dy)[0]

                print(
                    "swiglu ms",
                    dtype,
                    n,
                    h,
                    compare(
                        {"compiled": step, "candidate": lambda: (candidate(x), candidate(x, dy))}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    import sys

    torch.manual_seed(501)
    if sys.argv[1] in {"stem", "stem-autocast"}:
        for dtype in (torch.bfloat16, torch.float16):
            for input_grad in (False, True):
                stem(dtype, input_grad, autocast=sys.argv[1] == "stem-autocast")
    else:
        activation()
