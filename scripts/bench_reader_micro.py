"""Screen Wave 44 reader policies with eager CUDA launches and cold caches."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lnet import factorized_reader_kernels as reader

if TYPE_CHECKING:
    from collections.abc import Callable

MODEL = ((56, 56, 160, 4, 2), (28, 28, 160, 4, 2), (14, 14, 160, 4, 6), (7, 7, 128, 4, 2))
EDGE = ((13, 19, 63, 4, 0), (9, 11, 34, 4, 0), (7, 7, 128, 2, 0),
        (1, 1, 1, 1, 0), (37, 17, 97, 4, 0))
OPS = ("fwd", "dinput", "dweight")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def make_inputs(batch: int, shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    h, w, modes, rank, _ = shape
    kernel = 1 if h == w == modes == rank == 1 else 3
    sizes = ((batch, h, w, rank * modes), *((modes, rank, kernel, kernel),) * 2,
             (batch, h, w, 2 * modes))
    return tuple(torch.randn(size, device="cuda", dtype=dtype).mul_(0.125) for size in sizes)


def prepare(
    op: str, inputs: tuple[torch.Tensor, ...],
) -> tuple[Callable[[], None], torch.Tensor]:
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
                plan.module, op, plan.grid, (projected, grad, partial),
                (height, width, modes, batch, rank), stream=stream, threads=plan.warps * 32,
            )
            reader._launch_reader_cuda(
                plan.module, plan.reduce, reduce_grid, (partial, output),
                (modes, rank, plan.grid[1]), stream=stream,
            )

        return weight, output

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

    return spatial, output


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
            flush.add_(1)
        start.record()
        run()
        stop.record()
    events[-1][1].synchronize()
    return statistics.median(start.elapsed_time(stop) for start, stop in events)


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    max_abs = max_rel = 0.0
    exact = passed = True
    atol, rtol = (0.005, 0.02) if actual.dtype == torch.bfloat16 else (0.0005, 0.003)
    chunks = zip(actual.flatten().split(1 << 22), expected.flatten().split(1 << 22), strict=True)
    for actual_chunk, expected_chunk in chunks:
        a, b = actual_chunk.float(), expected_chunk.float()
        delta = (a - b).abs()
        max_abs = max(max_abs, delta.max().item())
        max_rel = max(max_rel, (delta / b.abs().clamp_min(1e-8)).max().item())
        exact &= torch.equal(a, b)
        passed &= bool(torch.isclose(a, b, atol=atol, rtol=rtol).all().item())
    return {"max_abs": max_abs, "max_rel": max_rel, "exact_match": exact, "tol_pass": passed}


def screen(args: argparse.Namespace) -> dict:
    flush = l2_buffer() if args.flush_l2 else None
    rows = []
    totals = {mode: dict.fromkeys(OPS, 0.0) for mode in args.modes}
    for shape in (*MODEL, *EDGE) if args.parity else MODEL:
        h, w, modes, rank, calls = shape
        batch = args.batch if calls else min(args.batch, 32)
        if h == w == modes == rank == 1:
            batch = 1
        for dtype_name in DTYPES if args.parity else ("bf16",):
            inputs = make_inputs(batch, shape, DTYPES[dtype_name])
            for op in OPS:
                expected = None
                if args.parity:
                    os.environ["LNET_READER_W44"] = "off"
                    baseline, expected = prepare(op, inputs)
                    baseline()
                    del baseline
                for mode in args.modes:
                    os.environ["LNET_READER_W44"] = mode
                    run, output = prepare(op, inputs)
                    row = {"shape": [batch, h, w, modes, rank], "calls": calls, "dtype": dtype_name,
                               "op": op, "mode": mode}
                    if calls and dtype_name == "bf16":
                        row["ms"] = measure(run, args.warmup, args.iters, flush)
                        totals[mode][op] += calls * row["ms"]
                    if expected is not None:
                        run()
                        row.update(metrics(output, expected))
                    rows.append(row)
                    del run, output
    for ops in totals.values():
        ops["total"] = sum(ops.values())
    return {"flush_l2": args.flush_l2, "flush_bytes": flush.numel() if flush is not None else 0,
                "weighted_ms": totals, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--modes", choices=reader._W44_MODES, nargs="+", default=reader._W44_MODES)
    args = parser.parse_args()
    if args.batch < 1 or args.warmup < 0 or args.iters < 1:
        parser.error("batch and iters must be positive and warmup must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.manual_seed(44)
    result = screen(args)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["weighted_ms"], indent=2))
    if any(not row.get("tol_pass", True) for row in result["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
