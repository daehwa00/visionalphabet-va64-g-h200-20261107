"""Measure real-model reader shapes across historical dispatches."""

# ruff: noqa: SLF001
import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import torch
from bench import compare, compute, inputs, load_baseline, record_launches

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


def parity(shape: tuple, types: tuple, wave17: object, *, strict: bool) -> dict:
    dtype, weight_dtype, grad_dtype = types
    p, g, real, imag = inputs(2, shape, dtype, strict)
    g, real, imag = g.to(grad_dtype), real.to(weight_dtype), imag.to(weight_dtype)
    ref = (reader._spatial_reference(p, real, imag, strict_complex=strict),
           *reader._spatial_reference_backward(g, p, real, imag, strict_complex=strict))
    os.environ["LNET_READER_VEC"] = "auto"
    historical = compute(wave17, p, g, real, imag, strict)
    arms = {}
    for width in ("auto", "4", "8", "4-rank"):
        os.environ["LNET_READER_VEC"] = width
        actual = compute(reader, p, g, real, imag, strict)
        arms[width] = {"reference": compare(actual, ref, dtype),
                       "baseline": compare(actual, historical, dtype)}
    return {"shape": shape, "dtype": list(map(str, types)), "strict": strict, "arms": arms}


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bench", "parity"))
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--general-only", action="store_true")
    parser.add_argument("--arms", choices=("ablate", "release"), default="ablate")
    args = parser.parse_args()
    torch.manual_seed(644)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    wave17 = load_baseline(args.refs / "wave17.py")
    dtypes = (torch.bfloat16, torch.float16)
    shapes = REAL if args.real_only else (*REAL, *GENERAL)
    if args.general_only:
        shapes = GENERAL
    if args.mode == "bench":
        wave20 = load_baseline(args.refs / "wave20.py")

        def bench_case(shape: tuple, dtype: torch.dtype) -> dict:
            h, w, modes, rank = shape
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            # Retain the outputs from all arms for interleaved graph replay.
            buffers = h * w * modes * (rank + 2) * (1 + 5) * dtype.itemsize
            batch = min(args.batch, max(1, int(free * 0.48) // buffers))
            tensors = inputs(batch, shape, dtype)
            graphs = {}
            retained = []
            candidate_arms = (
                (("vec4", "4", reader), ("vec8", "8", reader),
                 ("vec4-rank", "4-rank", reader)) if args.arms == "ablate"
                else (("wave23", "auto", reader),)
            )
            for name, arm, module in (
                ("wave17", "auto", wave17), ("wave20", "auto", wave20), *candidate_arms,
            ):
                os.environ["LNET_READER_VEC"] = arm
                calls = record_launches(module, *tensors)
                retained.extend(calls)
                groups = {"forward": calls[:1], "dinput": calls[-1:],
                          "dweight": calls[1:-1], "total": calls}
                for op, launches in groups.items():
                    graphs.setdefault(op, {})[name] = capture(module, launches)
            arms = {name: {} for name in graphs["total"]}
            for op, arms_graphs in graphs.items():
                for name, timing in measure(arms_graphs).items():
                    payload = batch * h * w * modes * (rank + 2) * dtype.itemsize
                    if op == "total":
                        payload *= 3
                    timing["logical_GBps"] = payload / (timing["median_ms"] * 1e6)
                    timing["percent_of_912_GBps"] = timing["logical_GBps"] / 9.12
                    # Historical roofline uses a complex rank+one payload.
                    timing["complex_roofline_GBps"] = timing["logical_GBps"] * (
                        2 * (rank + 1) / (rank + 2)
                    )
                    arms[name][op] = timing
            return {"shape": (batch, *shape), "dtype": str(dtype),
                    "passes_per_graph": PASSES_PER_SAMPLE, "arms": arms}

        cases = (bench_case(shape, dtype) for shape, dtype in itertools.product(shapes, dtypes))
    else:
        types = (*dtypes, torch.float32)
        edges = tuple((9, 11, m, rank) for m in (34, 53, 128, 160, 192, 320, 640)
                      for rank in range(1, 5))
        cases = (
            parity(shape, types, wave17, strict=strict)
            for shape, types, strict in itertools.product(
                (*REAL, *edges), itertools.product(types, repeat=3), (False, True),
            )
        )
    failed = False
    with args.out.open("w") as output:
        for record in cases:
            print(json.dumps(record), file=output, flush=True)
            if args.mode == "parity":
                failed |= any(
                    not metric["passed"] for arm in record["arms"].values()
                    for comparison in arm.values() for metric in comparison.values()
                )
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
