"""Screen the four Wave 52B image stem modes on CUDA and emit JSON."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

from lnet.fused_stem_ops import STAGE0_CANDIDATES
from lnet.image_layers import LayerNorm2d, _fused_stem_block

if TYPE_CHECKING:
    from torch import Tensor

MODES = ("off", *STAGE0_CANDIDATES)
DTYPES = (torch.bfloat16, torch.float16)
LAYOUTS = {"nchw": torch.contiguous_format, "channels_last": torch.channels_last}


def stem_case(
    shape: tuple[int, int, int, int],
    layout: torch.memory_format,
    dtype: torch.dtype,
    *,
    needs_dx: bool,
) -> tuple[Tensor, nn.Conv2d, LayerNorm2d, Tensor]:
    torch.manual_seed(185)
    x = torch.randn(shape, device="cuda").to(dtype).contiguous(memory_format=layout)
    x.requires_grad_(needs_dx)
    conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, device="cuda")
    norm = LayerNorm2d(32).cuda()
    dy = torch.randn_like(x[:, :1, ::2, ::2].expand(-1, 32, -1, -1))
    return x, conv, norm, dy


def stem_forward(conv: nn.Conv2d, norm: LayerNorm2d, dtype: torch.dtype) -> Callable:
    def forward(x: Tensor) -> Tensor:
        with torch.autocast("cuda", dtype=dtype):
            return _fused_stem_block(x, conv, norm)

    return forward


def stem_parameters(
    x: Tensor, conv: nn.Conv2d, norm: LayerNorm2d,
) -> tuple[Tensor, ...]:
    params = (conv.weight, norm.norm.weight, norm.norm.bias)
    if conv.bias is not None:
        params += (conv.bias,)
    return (*params, x) if x.requires_grad else params


def reference_gradients(
    x: Tensor, conv: nn.Conv2d, norm: LayerNorm2d, dy: Tensor,
) -> tuple[Tensor, ...]:
    x_fp32 = x.detach().float().requires_grad_(x.requires_grad)
    z = functional.conv2d(x_fp32, conv.weight, conv.bias, stride=2, padding=1)
    z = functional.layer_norm(
        z.permute(0, 2, 3, 1), (32,), norm.norm.weight, norm.norm.bias, norm.norm.eps,
    )
    y = functional.gelu(z).permute(0, 3, 1, 2)
    return (y, *torch.autograd.grad(y, stem_parameters(x_fp32, conv, norm), dy.float()))


def run_step(
    forward: Callable, x: Tensor, params: tuple[Tensor, ...], dy: Tensor,
) -> tuple[Tensor, ...]:
    y = forward(x)
    return (y, *torch.autograd.grad(y, params, dy))


def parity_errors(actual: tuple[Tensor, ...], ref: tuple[Tensor, ...]) -> dict[str, float]:
    return {
        name: float((got.float() - expected).norm() / expected.norm().clamp_min(1e-8))
        for name, got, expected in zip(
            ("y", "dw", "dg", "db", "dc", "dx")[:len(actual)], actual, ref, strict=True,
        )
    }


def graph_finite(fn: Callable[[], tuple[Tensor, ...]]) -> bool:
    torch.cuda.synchronize()
    warmup = torch.cuda.Stream()
    with torch.cuda.stream(warmup):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=warmup):
        result = fn()
    graph.replay()
    torch.cuda.synchronize()
    return all(bool(t.isfinite().all()) for t in result)


def benchmark(
    batch: int, layout: torch.memory_format, iterations: int,
) -> dict[str, float | bool]:
    x, conv, norm, dy = stem_case((batch, 3, 224, 224), layout, torch.bfloat16, needs_dx=False)
    forward = stem_forward(conv, norm, torch.bfloat16)
    params = stem_parameters(x, conv, norm)
    for _ in range(3):
        run_step(forward, x, params, dy)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events = [
        tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        for _ in range(iterations)
    ]
    for start, midpoint, end in events:
        start.record()
        y = forward(x)
        midpoint.record()
        torch.autograd.grad(y, params, dy)
        end.record()
        del y
    torch.cuda.synchronize()
    fwd = sum(a.elapsed_time(b) for a, b, _ in events) / iterations
    bwd = sum(b.elapsed_time(c) for _, b, c in events) / iterations
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return {
        "fwd_ms": fwd, "bwd_ms": bwd, "total_ms": fwd + bwd, "peak_vram_gib": peak,
        "cuda_graph_finite": graph_finite(lambda: run_step(forward, x, params, dy)),
    }


def parity_and_compile(layout: torch.memory_format) -> list[dict]:
    results = []
    for dtype in DTYPES:
        for needs_dx in (False, True):
            x, conv, norm, dy = stem_case((4, 3, 224, 224), layout, dtype, needs_dx=needs_dx)
            forward = stem_forward(conv, norm, dtype)
            params = stem_parameters(x, conv, norm)
            eager = run_step(forward, x, params, dy)
            errors = parity_errors(eager, reference_gradients(x, conv, norm, dy))
            torch._dynamo.reset()
            compiled = torch.compile(forward, fullgraph=True)
            compiled_out = run_step(compiled, x, params, dy)
            for got, expected in zip(compiled_out, eager, strict=True):
                torch.testing.assert_close(got, expected)
            results.append({
                "dtype": str(dtype), "needs_dx": needs_dx, "relative_l2": errors,
                "compile_fullgraph": True,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    if args.batch <= 0 or args.iterations <= 0:
        parser.error("batch and iterations must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    torch.backends.cudnn.allow_tf32 = False
    os.environ["LNET_FUSED_STEM_CONV"] = "on"
    report = {"device": torch.cuda.get_device_name(), "batch": args.batch, "results": []}
    for label in args.layouts:
        layout = LAYOUTS[label]
        timings = {}
        for mode in dict.fromkeys(("off", *args.modes)):
            os.environ["LNET_STEM_STAGE0_MODE"] = mode
            torch._dynamo.reset()
            timings[mode] = benchmark(args.batch, layout, args.iterations)
        for mode in args.modes:
            os.environ["LNET_STEM_STAGE0_MODE"] = mode
            report["results"].append({
                "layout": label, "mode": mode, **timings[mode],
                "delta_vs_off_ms": timings[mode]["total_ms"] - timings["off"]["total_ms"],
                "parity": parity_and_compile(layout),
            })
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
