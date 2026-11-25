"""Screen Wave 48B reader policies with eager CUDA launches and cold caches."""

from __future__ import annotations

import argparse
import ctypes
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lnet import factorized_reader_kernels as reader

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

MODEL = ((56, 56, 160, 4, 2), (28, 28, 160, 4, 2), (14, 14, 160, 4, 6), (7, 7, 128, 4, 2))
EDGE = ((13, 19, 63, 4, 0), (9, 11, 34, 4, 0), (7, 7, 128, 2, 0),
        (1, 1, 1, 1, 0), (37, 17, 97, 4, 0))
OPS = ("fwd", "dinput", "dweight")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def record_module_plans() -> tuple[AbstractContextManager, dict, dict]:
    compile_module = reader._reader_cuda_module
    signature = inspect.signature(compile_module)
    plans = {}
    compilation = {"count": 0, "seconds": 0.0}

    def record(*args: object, **kwargs: object) -> reader._CudaModule:
        misses = compile_module.cache_info().misses
        started = time.perf_counter()
        module = compile_module(*args, **kwargs)
        if compile_module.cache_info().misses > misses:
            compilation["count"] += 1
            compilation["seconds"] += time.perf_counter() - started
        if module not in plans:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            plans[module] = bound.arguments
        return module

    return patch.object(reader, "_reader_cuda_module", record), plans, compilation


def progress(mode: str, shape: str, op: str, compile_s: float, *,
             timing_s: float = 0.0, parity_s: float = 0.0) -> None:
    print(
        f"[{mode}] {shape} {op} compile={compile_s:.3f}s "
        f"timing={timing_s:.3f}s parity={parity_s:.3f}s",
        file=sys.stderr, flush=True,
    )


def make_inputs(batch: int, shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    h, w, modes, rank, _ = shape
    kernel = 1 if h == w == modes == rank == 1 else 3
    sizes = ((batch, h, w, rank * modes), *((modes, rank, kernel, kernel),) * 2,
             (batch, h, w, 2 * modes))
    return tuple(torch.randn(size, device="cuda", dtype=dtype).mul_(0.125) for size in sizes)


def prepare(
    op: str, inputs: tuple[torch.Tensor, ...], module_plans: dict,
) -> tuple[Callable[[], None], torch.Tensor, dict]:
    projected, real, imag, grad = inputs
    batch, height, width, _ = projected.shape
    modes, rank, kernel, _ = real.shape
    sizes = (height, width, modes, rank)
    if op == "dweight":
        plan = reader._reader_weight_plan(projected, real, grad, strict=False)
        partial = projected.new_empty((plan.grid[1], 2, rank, kernel, kernel, modes),
                                      dtype=torch.float32)
        output = real.new_empty((2, *real.shape), dtype=torch.float32)
        reduce_grid = (reader._ceildiv(output.numel(), plan.reduce_columns), 1, 1)
        stream = reader._reader_stream(plan.module, projected.device)

        def weight() -> None:
            reader._launch_reader_cuda(
                plan.module, plan.operation, plan.grid, (projected, grad, partial),
                (height, width, modes, batch, rank), stream=stream, threads=plan.warps * 32,
            )
            reader._launch_reader_cuda(
                plan.module, plan.reduce, reduce_grid, (partial, output),
                (modes, rank, plan.grid[1]), stream=stream,
            )

        config = module_plans[plan.module]
        tile, rows, _ = config["weight_geometry"] or reader._DEFAULT_WEIGHT_GEOMETRY
        return weight, output, {
            "vec": config["weight_vec"], "rank_tile": tile, "rows": rows,
            "columns": config["columns"], "warps": plan.warps, "grid": plan.grid,
            "reduce": plan.reduce, "reduce_grid": reduce_grid,
            "convert": plan.operation == "dweight_cvt", "kernel": config,
        }

    backward = op == "dinput"
    output = torch.empty_like(projected if backward else grad)
    launch, module = reader._reader_spatial_plan(projected, real, grad if backward else None, strict=False)
    grid = reader._reader_spatial_grid(
        batch, height, width, modes, launch.rows, launch.warps, launch.vec, launch.columns,
    )
    stream = reader._reader_stream(module, projected.device)

    def spatial() -> None:
        reader._launch_reader_cuda(
            module, "dinput" if backward else "forward", grid,
            (grad if backward else projected, real, imag, output),
            sizes, stream=stream, threads=launch.warps * 32,
        )

    config = module_plans[module]
    return spatial, output, {
        "vec": launch.vec, "rank_tile": rank, "rows": launch.rows, "columns": launch.columns,
        "warps": launch.warps, "grid": grid, "reduce": None,
        "convert": config["convert_spatial"], "updated": config["updated_spatial"],
        "kernel": config,
    }


def l2_buffer() -> torch.Tensor:
    from lnet.cuda_nvrtc import _load_cuda_driver

    driver = _load_cuda_driver()
    size = ctypes.c_int()
    # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
    reader._check_cu(driver.cuDeviceGetAttribute(ctypes.byref(size), 38, torch.cuda.current_device()),
                     "cuDeviceGetAttribute(L2_CACHE_SIZE)")
    return torch.zeros(max(64 << 20, 2 * size.value), device="cuda", dtype=torch.uint8)


def measure(run: Callable[[], None], warmup: int, iters: int, flush: torch.Tensor | None) -> float:
    for _ in range(warmup):
        run()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, stop in events:
        if flush is not None:
            flush.zero_()
        start.record()
        run()
        stop.record()
    events[-1][1].synchronize()
    return statistics.median(start.elapsed_time(stop) for start, stop in events)


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    atol, rtol = (0.005, 0.02) if actual.dtype == torch.bfloat16 else (0.0005, 0.003)
    maxima = torch.zeros(2, device=actual.device)
    exact = close = torch.ones((), dtype=torch.bool, device=actual.device)
    for observed, reference in zip(
        actual.flatten().split(1 << 20), expected.flatten().split(1 << 20), strict=True,
    ):
        a, b = observed.float(), reference.float()
        delta = (a - b).abs()
        maxima = torch.maximum(maxima, torch.stack(
            (delta.amax(), (delta / b.abs().clamp_min(1e-8)).amax()),
        ))
        exact = exact & (a == b).all()
        close = close & torch.isclose(a, b, atol=atol, rtol=rtol).all()
    max_abs, max_rel = maxima.tolist()
    return {"max_abs": max_abs, "max_rel": max_rel,
            "exact_match": exact.item(), "tol_pass": close.item()}


def timing_pass(
    args: argparse.Namespace, flush: torch.Tensor | None, module_plans: dict, compilation: dict,
) -> dict:
    rows = []
    plans = {}
    totals = {mode: dict.fromkeys(OPS, 0.0) for mode in args.modes}
    shapes = {}
    for shape in MODEL:
        h, w, modes, rank, calls = shape
        name = f"{h}x{w}x{modes}x{rank}"
        per_shape = {mode: {} for mode in args.modes}
        inputs = make_inputs(args.batch, shape, DTYPES["bf16"])
        for op in OPS:
            for mode in args.modes:
                os.environ["LNET_READER_OPT"] = mode
                compile_before = compilation["seconds"]
                run, output, plan = prepare(op, inputs, module_plans)
                compile_s = compilation["seconds"] - compile_before
                started = time.perf_counter()
                ms = measure(run, args.warmup, args.iters, flush)
                progress(mode, name, op, compile_s, timing_s=time.perf_counter() - started)
                plans[shape, op, mode] = plan
                rows.append({"shape": [args.batch, h, w, modes, rank], "calls": calls,
                             "dtype": "bf16", "op": op, "mode": mode, "ms": ms, "plan": plan})
                per_shape[mode][op] = ms
                totals[mode][op] += calls * ms
                del run, output
        shapes[name] = with_deltas(per_shape)
        del inputs
    return {"timing_rows": rows, "weighted_ms": with_deltas(totals), "per_shape_ms": shapes,
            "timing_plans": plans}


def with_deltas(timings: dict) -> dict:
    for values in timings.values():
        values["total"] = sum(values.values())
    baseline = timings.get("off")
    if baseline is not None:
        for values in timings.values():
            values["delta_vs_off_ms"] = {op: values[op] - baseline[op] for op in (*OPS, "total")}
    return timings


def parity_pass(
    args: argparse.Namespace, timing_plans: dict, module_plans: dict, compilation: dict,
) -> list[dict]:
    rows = []
    for shape in (*MODEL, *EDGE):
        h, w, modes, rank, calls = shape
        # Model parity uses the timing batch to preserve the exact specialization
        # and worker partition. Edge shapes only exercise correctness.
        batch = args.batch if calls else (1 if h == w == modes == rank == 1 else min(args.batch, 16))
        for dtype, torch_dtype in DTYPES.items():
            inputs = make_inputs(batch, shape, torch_dtype)
            for op in OPS:
                os.environ["LNET_READER_OPT"] = "off"
                compile_before = compilation["seconds"]
                baseline, expected, _ = prepare(op, inputs, module_plans)
                baseline_compile = compilation["seconds"] - compile_before
                started = time.perf_counter()
                baseline()
                torch.cuda.synchronize()
                name = f"{h}x{w}x{modes}x{rank}/{dtype}"
                progress("off", name, op, baseline_compile, parity_s=time.perf_counter() - started)
                del baseline
                for mode in args.modes:
                    os.environ["LNET_READER_OPT"] = mode
                    compile_before = compilation["seconds"]
                    run, output, plan = prepare(op, inputs, module_plans)
                    compile_s = compilation["seconds"] - compile_before
                    started = time.perf_counter()
                    run()
                    timing = timing_plans.get((shape, op, mode))
                    match = None if timing is None else plan == timing
                    same_geometry = None if timing is None else (
                        {**plan, "kernel": {**plan["kernel"], "types": timing["kernel"]["types"]}}
                        == timing
                    )
                    rows.append({"shape": [batch, h, w, modes, rank], "calls": calls, "dtype": dtype,
                                 "op": op, "mode": mode, "plan": plan, "timing_plan_match": match if dtype == "bf16" else None,
                                 "bf16_timing_geometry_match": same_geometry,
                                 **metrics(output, expected)})
                    progress(mode, name, op, compile_s, parity_s=time.perf_counter() - started)
                    del run, output
                del expected
            del inputs
    return rows


def screen(args: argparse.Namespace) -> dict:
    flush = l2_buffer() if args.flush_l2 else None
    capture, module_plans, compilation = record_module_plans()
    with capture, patch.dict(os.environ, {"LNET_READER_W44": "off"}):
        result = timing_pass(args, flush, module_plans, compilation)
        timing_plans = result.pop("timing_plans")
        rows = parity_pass(args, timing_plans, module_plans, compilation) if args.parity else []
    return {"flush_l2": args.flush_l2, "flush_bytes": flush.numel() if flush is not None else 0,
            **result, "rows": rows, "compilation": compilation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--modes", choices=reader._READER_MODES, nargs="+", default=reader._READER_MODES)
    args = parser.parse_args()
    if args.batch < 1 or args.warmup < 0 or args.iters < 1:
        parser.error("batch and iters must be positive and warmup must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.manual_seed(44)
    result = screen(args)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("weighted_ms", "per_shape_ms")}, indent=2))
    if any(not row["tol_pass"] or row["timing_plan_match"] is False for row in result["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
