"""Screen the Wave 53B image stem modes on CUDA and emit JSON."""

from __future__ import annotations

import argparse
import json
import os

import torch
from bench_wave52b_stem_type1 import (
    DTYPES,
    LAYOUTS,
    graph_finite,
    run_step,
    stem_case,
    stem_forward,
    stem_parameters,
)
from torch.nn import functional

from lnet.fused_stem_ops import STAGE0_CANDIDATES, stem_conv_tensor_reduce

MODES = ("off", "triton_tuned_v2", *(m for m in STAGE0_CANDIDATES if m.startswith("triton_v3_")))


def benchmark(
    batch: int,
    layout: torch.memory_format,
    iterations: int,
) -> dict[str, float | bool]:
    x, conv, norm, dy = stem_case((batch, 3, 224, 224), layout, torch.bfloat16, needs_dx=False)
    forward = stem_forward(conv, norm, torch.bfloat16)
    params = stem_parameters(x, conv, norm)
    for _ in range(5):
        run_step(forward, x, params, dy)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events = [
        tuple(torch.cuda.Event(enable_timing=True) for _ in range(3)) for _ in range(iterations)
    ]
    flush = torch.empty(120 * 1024**2, device=x.device, dtype=torch.uint8)
    for start, midpoint, end in events:
        flush.zero_()
        start.record()
        y = forward(x)
        midpoint.record()
        torch.autograd.grad(y, params, dy)
        end.record()
        del y
    torch.cuda.synchronize()
    fwd = sum(a.elapsed_time(b) for a, b, _ in events) / iterations
    bwd = sum(b.elapsed_time(c) for _, b, c in events) / iterations
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return {
        "fwd_ms": fwd,
        "bwd_ms": bwd,
        "total_ms": fwd + bwd,
        "peak_vram_gib": peak,
        "cuda_graph_finite": graph_finite(lambda: run_step(forward, x, params, dy)),
    }


def check_large_reductions() -> None:
    # Integer-valued FP32 partials have an exact sum for all worker counts.
    for workers in (1056, 2112, 4224):
        for bias in (False, True):
            fields = 29 + int(bias)
            partial = torch.randint(-2, 3, (workers, fields, 32), device="cuda").float()
            dw = partial.new_empty((27, 32), dtype=torch.bfloat16)
            dg, db, dc = (partial.new_empty(32) for _ in range(3))
            stem_conv_tensor_reduce[(fields,)](partial, dw, dg, db, dc, 3, 32, bias, workers, 256)
            ref = partial.sum(0)
            torch.testing.assert_close(dw, ref[:27].to(dw.dtype), rtol=0, atol=0)
            torch.testing.assert_close(dg, ref[27], rtol=0, atol=0)
            torch.testing.assert_close(db, ref[28], rtol=0, atol=0)
            if bias:
                torch.testing.assert_close(dc, ref[29], rtol=0, atol=0)


def check_case(
    layout: torch.memory_format, dtype: torch.dtype, *, needs_dx: bool, bias: bool
) -> dict:
    x, conv, norm, dy = stem_case((4, 3, 33, 63), layout, dtype, needs_dx=needs_dx)
    if not bias:
        conv.bias = None
    forward = stem_forward(conv, norm, dtype)
    params = stem_parameters(x, conv, norm)
    x_ref = x.detach().float().requires_grad_(needs_dx)
    z = functional.conv2d(x_ref, conv.weight, conv.bias, stride=2, padding=1)
    z = functional.layer_norm(
        z.permute(0, 2, 3, 1), (32,), norm.norm.weight, norm.norm.bias, norm.norm.eps
    )
    y = functional.gelu(z).permute(0, 3, 1, 2)
    ref_params = stem_parameters(x_ref, conv, norm)
    ref_grads = torch.autograd.grad(y, ref_params, dy.float())
    expected = (y.detach(), *ref_grads)
    del y, z, x_ref, ref_grads, ref_params
    eager = run_step(forward, x, params, dy)
    names = ["y", "dw", "dg", "db"] + (["dc"] if bias else []) + (["dx"] if needs_dx else [])

    def check_relative_l2(actual, reference, limit, comparison):
        errors = {
            name: float((a.float() - r.float()).norm() / r.float().norm().clamp_min(1e-8))
            for name, a, r in zip(names, actual, reference, strict=True)
        }
        if not all(v < limit for v in errors.values()):
            raise AssertionError(f"{comparison} (limit {limit}): {errors}")
        return errors

    limit = 0.04 if dtype == torch.bfloat16 else 0.008
    errors = check_relative_l2(eager, expected, limit, "eager vs FP32")
    torch._dynamo.reset()
    compiled_aot = torch.compile(forward, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(run_step(compiled_aot, x, params, dy), eager)
    if x.is_cuda:
        compiled = torch.compile(forward, fullgraph=True)
        compiled_outputs = run_step(compiled, x, params, dy)
        check_relative_l2(compiled_outputs, expected, limit, "Inductor vs FP32")
        eager_limit = 0.015 if dtype == torch.bfloat16 else 0.005
        check_relative_l2(compiled_outputs, eager, eager_limit, "Inductor vs eager")
        del compiled_outputs
    if not graph_finite(lambda: run_step(forward, x, params, dy)):
        raise AssertionError("Nonfinite CUDA Graph replay")
    return {
        "dtype": str(dtype),
        "needs_dx": needs_dx,
        "bias": bias,
        "relative_l2": errors,
        "compile_fullgraph": True,
        "cuda_graph_finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--selftest", action="store_true", help="Run all CUDA assertions without timing"
    )
    args = parser.parse_args()
    if args.batch <= 0 or args.iterations <= 0:
        parser.error("batch and iterations must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    torch.backends.cudnn.allow_tf32 = False
    os.environ["LNET_FUSED_STEM_CONV"] = "on"
    check_large_reductions()
    report = {"device": torch.cuda.get_device_name(), "batch": args.batch, "results": []}
    for label in args.layouts:
        layout = LAYOUTS[label]
        timings = {}
        for mode in dict.fromkeys(("off", *args.modes)):
            os.environ["LNET_STEM_STAGE0_MODE"] = mode
            torch._dynamo.reset()
            if not args.selftest:
                timings[mode] = benchmark(args.batch, layout, args.iterations)
                if not timings[mode]["cuda_graph_finite"]:
                    raise AssertionError(f"Nonfinite CUDA Graph replay for {mode} on {label}")
                timings[mode]["delta_vs_off_ms"] = (
                    timings[mode]["total_ms"] - timings["off"]["total_ms"]
                )
        for mode in args.modes:
            os.environ["LNET_STEM_STAGE0_MODE"] = mode
            report["results"].append(
                {
                    "layout": label,
                    "mode": mode,
                    **timings.get(mode, {}),
                    "parity": [
                        check_case(layout, dtype, needs_dx=dx, bias=bias)
                        for dtype in DTYPES
                        for dx in (False, True)
                        for bias in (False, True)
                    ],
                }
            )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
