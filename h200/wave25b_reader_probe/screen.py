"""Measure real-model reader shapes across historical dispatches."""

# ruff: noqa: SLF001
import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import torch
from bench import compare, compute, inputs, record_launches
from weight_window import compile_weight

from lnet import factorized_reader_kernels as reader

REAL = ((56, 56, 160, 4), (28, 28, 160, 4), (14, 14, 160, 4), (7, 7, 128, 4))
GENERAL = ((28, 28, 320, 4), (14, 14, 640, 4), (7, 7, 640, 4), (128, 192, 160, 4),
           (37, 53, 192, 4))
SAMPLES = 24
WARMUP_SAMPLES = 24
PASSES_PER_SAMPLE = 8


def capture(
    module: object, launches: list,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Event, torch.cuda.Event, list]:
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.graph(graph):
        for _ in range(PASSES_PER_SAMPLE):
            for args, options in launches:
                stream = module._reader_stream(args[0], args[3][0].device)
                module._launch_reader_cuda(*args, **{**options, "stream": stream})
    return graph, start, end, [(args[1], args[2], options.get("threads", module._DEFAULT_THREADS))
                               for args, options in launches]


def measure(graphs: dict) -> dict:
    samples = {name: [] for name in graphs}
    names = tuple(graphs)
    stream = torch.cuda.current_stream()
    for _ in range(WARMUP_SAMPLES + SAMPLES):
        names = names[1:] + names[:1]
        for name in names:
            graph, start, end, _ = graphs[name]
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / PASSES_PER_SAMPLE)
    result = {}
    for name, raw in samples.items():
        warmup, timed = raw[:WARMUP_SAMPLES], raw[WARMUP_SAMPLES:]
        q1, _, q3 = statistics.quantiles(timed, n=4, method="inclusive")
        result[name] = {"warmup_ms": warmup, "samples_ms": timed,
                        "median_ms": statistics.median(timed), "iqr_ms": q3 - q1,
                        "launches": graphs[name][3]}
    return result


def parity(shape: tuple, types: tuple, *, strict: bool, batch: int = 2) -> dict:
    dtype, weight_dtype, grad_dtype = types
    p, g, real, imag = inputs(batch, shape, dtype, strict)
    g, real, imag = g.to(grad_dtype), real.to(weight_dtype), imag.to(weight_dtype)
    reference = (
        reader._spatial_reference(p, real, imag, strict_complex=strict),
        *reader._spatial_reference_backward(g, p, real, imag, strict_complex=strict),
    )
    os.environ["LNET_READER_W25B"] = "off"
    baseline = compute(reader, p, g, real, imag, strict)
    os.environ["LNET_READER_W25B"] = "on"
    actual = compute(reader, p, g, real, imag, strict)
    return {"shape": (batch, *shape), "dtypes": list(map(str, types)), "strict": strict,
            "reference": compare(actual, reference, dtype),
            "baseline": compare(actual, baseline, dtype)}


def scalar_weight_variant(call: tuple, shape: tuple, dtype: str, rank: int, device: int) -> dict:
    batch, height, width, modes = shape
    rows, warps = 7, 4
    if height % rows or width > 2 * reader._COMPACT_COLUMNS:
        return {}
    (_, op, grid, tensors, sizes), opts = call
    columns = reader._reader_columns(width)
    workers = min(grid[1], reader._ceildiv(batch * reader._ceildiv(height, rows), warps))
    module = reader._reader_cuda_module(
        device, shape, columns, rank, 3, (dtype,) * 3, strict=False,
        weight_geometry=(rank, rows, warps),
    )
    if not reader._reader_resident(module, op, warps):
        return {}
    launch_grid = reader._ceildiv(modes, 32), workers, 1
    return {"scalar-r7-w4": (
        (module, op, launch_grid, (*tensors[:2], tensors[2][:workers]), sizes),
        {**opts, "threads": 32 * warps},
    )}


def variants(call: tuple) -> dict:
    args, opts = call
    _, op, grid, tensors, sizes = args
    if op not in {"forward", "dinput", "dweight"}:
        return {}
    first = tensors[0]
    batch, height, width = first.shape[:3]
    modes, rank = sizes[2], sizes[-1]
    dtype = reader._CUDA_TYPES[first.dtype]
    shape = (batch, height, width, modes)
    device = first.device.index or 0
    result = {}
    if op == "dweight":
        configs = itertools.product((4, 7, 8, 14), (4, 8), (False,), (False,))
        for rows, warps, readonly, convert in configs:
            vec, tile = 2, min(rank, 2)
            workers = min(grid[1], reader._ceildiv(batch * reader._ceildiv(height, rows), warps))
            mode_rank_tiles = reader._ceildiv(modes, 32 * vec) * reader._ceildiv(rank, tile)
            launch_grid = (mode_rank_tiles, workers, 1)
            module = compile_weight(
                device, shape, reader._reader_columns(width), rank, 3, (dtype,) * 3,
                strict=False, weight_geometry=(tile, rows, warps), weight_vec=vec,
                weight_read_only=readonly, convert=convert,
            )
            if reader._reader_resident(module, op, warps):
                label = f"r{rows}-w{warps}-ldg{int(readonly)}-fp32win{int(convert)}"
                partial = tensors[2][:workers]
                result[label] = ((module, op, launch_grid, (*tensors[:2], partial), sizes),
                                 {**opts, "threads": 32 * warps})
        result.update(scalar_weight_variant(call, shape, dtype, rank, device))
    else:
        for vec, rows, columns in itertools.product((1, 2, 4), (4, 7, 8), (7, 8, 14, 16)):
            if modes % vec or (vec == 4 and op != "dinput"):
                continue
            if vec == 1 and (rows != 7 or height % rows or width > 2 * columns):
                continue
            if columns == 7 and vec != 1:
                continue
            warps = min(2, reader._ceildiv(width, columns))
            module = reader._reader_cuda_module(
                device, shape, columns, rank, 3, (dtype,) * 3, strict=False,
                spatial_geometry=(rows, warps), register_filters=0, keep_real=0,
                spatial_vec=vec, updated_spatial=True,
            )
            if reader._reader_resident(module, op, warps):
                launch_grid = reader._reader_spatial_grid(
                    batch, height, width, modes, rows, warps, vec, columns,
                )
                result[f"v{vec}-r{rows}-c{columns}"] = (
                    (module, op, launch_grid, tensors, sizes), {**opts, "threads": 32 * warps},
                )
    return result


def bench_case(batch: int, shape: tuple, dtype: torch.dtype, *, ablate: bool) -> dict:
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    h, w, modes, rank = shape
    buffers = h * w * modes * (rank + 2) * 6 * dtype.itemsize
    batch = min(batch, max(1, int(free * 0.48) // buffers))
    tensors = inputs(batch, shape, dtype)
    arms = {}
    for name in ("off", "on"):
        os.environ["LNET_READER_W25B"] = name
        calls = record_launches(reader, *tensors)
        arms[name] = {call[0][1].replace("_parallel", ""): call for call in calls}
    ops = {}
    for op, call in arms["off"].items():
        choices = {name: dispatch[op] for name, dispatch in arms.items()}
        if ablate:
            choices.update(variants(call))
        graphs = {name: capture(reader, [launch]) for name, launch in choices.items()}
        timings = measure(graphs)
        for name, timing in timings.items():
            logical = sum(t.numel() * t.element_size() for t in choices[name][0][3])
            timing.update(logical_bytes=logical, effective_gb_s=logical / timing["median_ms"] / 1e6)
        ops[op] = timings
    return {"shape": (batch, *shape), "dtype": str(dtype), "ops": ops}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bench", "parity"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--shape", nargs=4, type=int)
    args = parser.parse_args()
    torch.manual_seed(644)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    dtypes = (torch.bfloat16, torch.float16, torch.float32)
    if args.mode == "bench":
        shapes = (args.shape,) if args.shape else REAL if args.ablate else (*REAL, *GENERAL)
        cases = (bench_case(args.batch, shape, dtype, ablate=args.ablate)
                 for shape, dtype in itertools.product(shapes, dtypes[:2]))
    else:
        shapes = ((9, 11, m, r) for m, r in itertools.product(
            (34, 53, 128, 160, 192, 320, 640), range(1, 5),
        ))
        tuned_shapes = ((0, 64), (1, 128), (3, 128))
        tuned = (
            parity(REAL[index], (dtype,) * 3, strict=False, batch=batch)
            for (index, batch), dtype in itertools.product(tuned_shapes, dtypes[:2])
        )
        small = (
            parity(shape, types, strict=strict) for shape, types, strict in itertools.product(
                (*REAL, *GENERAL, *shapes), itertools.product(dtypes, repeat=3), (False, True),
            )
        )
        cases = itertools.chain(tuned, small)
    failed = False
    with args.out.open("w") as out:
        for case in cases:
            print(json.dumps(case), file=out, flush=True)
            if args.mode == "parity":
                failed |= any(
                    not check["passed"] for arm in ("reference", "baseline")
                    for check in case[arm].values()
                )
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
