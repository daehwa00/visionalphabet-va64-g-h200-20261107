#!/usr/bin/env python3
"""Time isolated eager CUDA factorized reader passes, including weight reduction."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lnet import factorized_reader_kernels as reader

if TYPE_CHECKING:
    from collections.abc import Callable

Shape = tuple[int, int, int, int, int]
Tensors = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

SHAPES = ((56, 56, 160, 4, 2), (28, 28, 160, 4, 2), (14, 14, 160, 4, 6), (7, 7, 128, 4, 2))
EDGES = ((13, 19, 63, 3, 0), (37, 13, 160, 1, 0), (13, 37, 128, 6, 0))
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
PASSES = ("fwd", "dinput", "dweight")
OUTPUTS = dict(zip(PASSES, ("out", "d_proj", "d_w"), strict=True))


def inputs(batch: int, shape: Shape, dtype: torch.dtype) -> Tensors:
    h, w, modes, rank, _ = shape
    def sample(size: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(size, device="cuda", dtype=dtype).normal_(std=0.125)

    return (sample((batch, h, w, rank * modes)), sample((modes, rank, 3, 3)),
            sample((modes, rank, 3, 3)), sample((batch, h, w, 2 * modes)))


def prepare(op: str, tensors: Tensors) -> tuple[Callable[[], None], torch.Tensor, dict]:
    projected, real, imag, grad = tensors
    batch, h, w, _ = projected.shape
    modes, rank, kernel, _ = real.shape
    if op == "dweight":
        plan = reader._reader_weight_plan(projected, real, grad, strict=False)
        workers = plan.grid[1]
        partial = projected.new_empty((workers, 2, rank, kernel, kernel, modes), dtype=torch.float32)
        output = real.new_empty((2, *real.shape), dtype=torch.float32)
        stream = reader._reader_stream(plan.module, projected.device)
        reduce_grid = (reader._ceildiv(output.numel(), plan.reduce_columns), 1, 1)

        def launch() -> None:
            reader._launch_reader_cuda(
                plan.module, op, plan.grid, (projected, grad, partial), (h, w, modes, batch, rank),
                stream=stream, threads=plan.warps * 32,
            )
            reader._launch_reader_cuda(
                plan.module, plan.reduce, reduce_grid, (partial, output), (modes, rank, workers),
                stream=stream,
            )

        geometry = {key: value for key, value in plan._asdict().items() if key != "module"}
        return launch, output, {**geometry, "reduce_grid": reduce_grid}

    forward = op == "fwd"
    plan, module = reader._reader_spatial_plan(projected, real, None if forward else grad,
                                               strict=False)
    grid = reader._reader_spatial_grid(batch, h, w, modes, plan.rows, plan.warps, plan.vec,
                                       plan.columns)
    output = torch.empty_like(grad if forward else projected)
    source = projected if forward else grad
    stream = reader._reader_stream(module, projected.device)

    def launch() -> None:
        reader._launch_reader_cuda(
            module, "forward" if forward else op, grid, (source, real, imag, output),
            (h, w, modes, rank), stream=stream, threads=plan.warps * 32,
        )

    return launch, output, {**plan._asdict(), "grid": grid}


def timed(launch: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        launch()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def parity(actual: torch.Tensor, expected: torch.Tensor, name: str) -> dict:
    # Near zero, a relative error has no useful scale below float32 epsilon.
    if name == "d_w":
        atol, rtol = 0.004, 5e-4
    elif actual.dtype == torch.bfloat16:
        atol, rtol = 0.002, 0.008
    else:
        atol, rtol = 0.0003, 0.001
    stats = []
    epsilon = torch.finfo(torch.float32).eps
    for tile, reference in zip(actual.split(32), expected.split(32), strict=True):
        a, b = tile.float(), reference.float()
        error = (a - b).abs()
        stats.append(torch.stack((error.amax(), (error / b.abs().clamp_min(epsilon)).amax(),
                                  (error / (atol + rtol * b.abs())).amax())))
    max_abs, max_rel, tolerance_ratio = torch.stack(stats).amax(dim=0).tolist()
    finite = all(math.isfinite(value) for value in (max_abs, max_rel, tolerance_ratio))
    return {"exact_match": finite and max_abs == 0, "max_abs": max_abs if finite else None,
            "max_rel": max_rel if finite else None, "tol_pass": finite and tolerance_ratio <= 1,
            "atol": atol, "rtol": rtol}


def shape_key(shape: Shape) -> str:
    return "x".join(map(str, shape[:4]))


def screen(args: argparse.Namespace, modes: list[str]) -> dict:
    timings = {mode: {} for mode in modes}
    diagnostics = {mode: {} for mode in modes} if args.parity else {}
    for dtype_name, dtype in DTYPES.items():
        measure = dtype_name == args.dtype
        if not measure and not args.parity:
            continue
        shapes = SHAPES + EDGES if args.parity else SHAPES
        for shape in shapes:
            h, w, _, _, calls = shape
            batch = args.batch if calls else reader._ceildiv(reader._SPECIALIZE_MIN_PIXELS, h * w)
            tensors = inputs(batch, shape, dtype)
            key = shape_key(shape)
            for op in PASSES:
                expected = None
                for mode in modes:
                    os.environ["LNET_READER_W42"] = mode
                    launch, output, plan = prepare(op, tensors)
                    if measure and calls:
                        elapsed = timed(launch, args.warmup, args.iters)
                        timings[mode].setdefault(key, {"calls": calls})[op] = {
                            "ms": elapsed, "plan": plan,
                        }
                    if args.parity:
                        if not (measure and calls):
                            launch()
                        if mode == "off":
                            expected = output
                        name = OUTPUTS[op]
                        diagnostics[mode].setdefault(dtype_name, {}).setdefault(key, {})[name] = (
                            parity(output, expected, name)
                        )
                    del launch, output
            del tensors
    weighted = {}
    for mode, shapes in timings.items():
        values = {op: sum(s["calls"] * s[op]["ms"] for s in shapes.values()) for op in PASSES}
        weighted[mode] = {**values, "total": sum(values.values())}

    def qualifies(mode: str, op: str) -> bool:
        return all(record[name]["tol_pass"] for dtype in diagnostics.get(mode, {}).values()
                   for record in dtype.values()
                   for name in (OUTPUTS.values() if op == "total" else (OUTPUTS[op],)))

    return {
        "per_shape": timings, "weighted_ms": weighted,
        "delta_vs_off_ms": {mode: {op: ms - weighted["off"][op] for op, ms in values.items()}
                            for mode, values in weighted.items()},
        "parity": diagnostics,
        "best_qualifying": {
            op: min((mode for mode in modes if qualifies(mode, op)),
                    key=lambda mode: weighted[mode][op], default=None)
            for op in (*PASSES, "total")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="off,ro,fwd_hiocc,dw_hiocc,combo")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reader_micro.json"))
    args = parser.parse_args()
    modes = list(dict.fromkeys(["off", *(mode.strip() for mode in args.modes.split(","))]))
    if unknown := set(modes) - reader._READER_W42_MODES:
        parser.error(f"unknown modes {sorted(unknown)}")
    if not (1 <= args.batch <= reader._MAX_CUDA_GRID_DIM and args.iters > 0 and args.warmup >= 0):
        parser.error("batch and iterations must be positive and warmup must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("a CUDA device is required")
    torch.manual_seed(147)
    report = screen(args, modes)
    report["settings"] = {**vars(args), "out": str(args.out), "modes": modes,
                          "device": torch.cuda.get_device_name()}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["best_qualifying"]))


if __name__ == "__main__":
    main()
