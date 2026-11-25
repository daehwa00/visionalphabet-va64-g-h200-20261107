#!/usr/bin/env python3
"""L2-cold read-projection screening. Run on a CUDA host with PyTorch and Triton."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

MODEL_CASES = (
    (3211264, 64, 160, 160, 1),
    (802816, 96, 160, 160, 2),
    (200704, 128, 160, 160, 6),
    (50176, 256, 160, 160, 1),
)
EDGES = (
    (1027, 37, 51, 77, torch.bfloat16),
    (513, 65, 29, 19, torch.float16),
    (257, 31, 17, 33, torch.float32),
)
NAMES = ("out", "dx_real", "dx_imag", "dw_real", "dw_imag", "dbias")


def _load_projection() -> ModuleType:
    # Load the standalone module without the model package's Python 3.13 dependencies.
    path = Path(__file__).resolve().parents[1] / "src" / "lnet" / "fused_readproj_ops.py"
    spec = importlib.util.spec_from_file_location("fused_readproj_ops", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="off,dual_fwd_only,dual_all")
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--edge-only", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    return args


def _inputs(shape: tuple[int, int, int, int], dtype: torch.dtype, *, bias: bool) -> tuple:
    m, n, kr, ki = shape
    dims = ((m, kr), (m, ki), (n, kr + ki), (m, n))
    x, y, w, g = (torch.randn(d, device="cuda", dtype=dtype) for d in dims)
    b = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True) if bias else None
    return x.requires_grad_(), y.requires_grad_(), w.requires_grad_(), g, b


def _fwd(projection: Callable, inputs: tuple, mode: str) -> torch.Tensor:
    x, y, w, _, bias = inputs
    wr, wi = w.split((x.shape[1], y.shape[1]), dim=1)
    return projection(x, y, wr, wi, bias, mode)


def _bwd(out: torch.Tensor, inputs: tuple) -> tuple:
    x, y, w, g, bias = inputs
    args = (x, y, w) if bias is None else (x, y, w, bias)
    return torch.autograd.grad(out, args, g)


def _measure(
    projection: Callable, inputs: tuple, mode: str, iterations: int, flush: Callable[[], None]
) -> dict:
    out = _fwd(projection, inputs, mode)
    _bwd(out, inputs)
    del out
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(2 * iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
    for index in range(iterations):
        flush()
        starts[2 * index].record()
        out = _fwd(projection, inputs, mode)
        ends[2 * index].record()
        flush()
        starts[2 * index + 1].record()
        grads = _bwd(out, inputs)
        ends[2 * index + 1].record()
        del out, grads
    torch.cuda.synchronize()
    times = [a.elapsed_time(b) for a, b in zip(starts, ends, strict=True)]
    fwd, bwd = sorted(times[::2]), sorted(times[1::2])
    return {"fwd_ms": fwd[len(fwd) // 2], "bwd_ms": bwd[len(bwd) // 2]}


def _values(projection: Callable, inputs: tuple, mode: str) -> tuple:
    out = _fwd(projection, inputs, mode)
    dx, dy, dw, *db = _bwd(out, inputs)
    dr, di = dw.split((dx.shape[1], dy.shape[1]), dim=1)
    return (out.detach(), dx, dy, dr, di, *db)


def _error(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    # Slice large rows to bound transient memory while accumulating on the GPU.
    stats = []
    for a, chunk in zip(actual.split(65536), reference.split(65536), strict=True):
        b = chunk.float()
        diff = a.float() - b
        stats.append(torch.stack((diff.abs().amax(), diff.square().sum(), b.square().sum())))
    stacked = torch.stack(stats)
    norm = stacked[:, 1].sum().sqrt() / stacked[:, 2].sum().sqrt().clamp_min(1e-12)
    max_abs, relative_l2 = stacked[:, 0].max().item(), norm.item()
    return {
        "max_abs": max_abs if math.isfinite(max_abs) else None,
        "relative_l2": relative_l2 if math.isfinite(relative_l2) else None,
    }


def _parity(actual: tuple, baseline: tuple, reference: tuple, dtype: torch.dtype) -> dict:
    result = {}
    tolerance = 0.008 if dtype == torch.bfloat16 else 0.002 if dtype == torch.float16 else 0.0003
    for name, got, off, fp32 in zip(NAMES[: len(actual)], actual, baseline, reference, strict=True):
        errors = {"off": _error(got, off), "fp32": _error(got, fp32)}
        errors["passed"] = all(
            metrics["max_abs"] is not None
            and metrics["relative_l2"] is not None
            and metrics["relative_l2"] <= tolerance
            for metrics in errors.values()
        )
        result[name] = errors
    return result


def main() -> int:
    args = _args()
    module = _load_projection()
    modes = tuple(dict.fromkeys(mode.strip() for mode in args.modes.split(",")))
    if not modes or any(mode not in module.MODES for mode in modes):
        raise SystemExit(f"Expected comma-separated modes from {module.MODES}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for L2-cold timing")
    if any(mode != "off" for mode in modes) and module.triton is None:
        raise SystemExit("Triton is required for fused modes")

    props = torch.cuda.get_device_properties(0)
    flush_size = max(64 * 1024**2, 2 * props.L2_cache_size) if args.flush_l2 else 0
    evict = torch.zeros(flush_size // 4, device="cuda", dtype=torch.int32)

    def flush() -> None:
        if flush_size:
            evict.add_(1)

    torch.manual_seed(161)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_cases = (
        ()
        if args.edge_only
        else tuple((m, n, kr, ki, torch.bfloat16, copies) for m, n, kr, ki, copies in MODEL_CASES)
    )
    cases = (*model_cases, *((*edge, 0) for edge in EDGES))
    rows = []
    totals = {mode: {"fwd_ms": 0.0, "bwd_ms": 0.0, "total_ms": 0.0} for mode in modes}
    passed = True
    for m, n, kr, ki, dtype, copies in cases:
        shape = (m, n, kr, ki)
        inputs = _inputs(shape, dtype, bias=not copies)
        print(f"shape={shape} dtype={dtype} model_calls={copies}", file=sys.stderr, flush=True)
        if args.parity:
            baseline = _values(module.readproj, inputs, "off")
            fp32_inputs = tuple(
                t.detach().float().requires_grad_(t.requires_grad) if t is not None else None
                for t in inputs
            )
            reference = _values(module.readproj, fp32_inputs, "off")
            del fp32_inputs
        for mode in modes:
            print(f"  {mode}: warmup and timing", file=sys.stderr, flush=True)
            times = _measure(module.readproj, inputs, mode, args.iterations, flush)
            row = dict(shape=shape, dtype=str(dtype), model_calls=copies, mode=mode, **times)
            if args.parity:
                actual = baseline if mode == "off" else _values(module.readproj, inputs, mode)
                row["parity"] = _parity(actual, baseline, reference, dtype)
                passed &= all(item["passed"] for item in row["parity"].values())
                del actual
            rows.append(row)
            for key, value in times.items():
                totals[mode][key] += copies * value
            totals[mode]["total_ms"] += copies * sum(times.values())
            print(f"  {mode}: {times}", file=sys.stderr, flush=True)
        if args.parity:
            del baseline, reference
        del inputs
    result = {
        "device": props.name,
        "flush_bytes": flush_size,
        "iterations": args.iterations,
        "rows": rows,
        "weighted_model_ms": totals,
        "parity_passed": passed if args.parity else None,
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.json_out is not None:
        args.json_out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
