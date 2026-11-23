"""Measure real-model reader shapes across historical dispatches."""

# ruff: noqa: SLF001
import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import torch
from bench import PROBE_EDGES, STANDARD, inputs, load_baseline, parity_case, record_launches

from lnet import factorized_reader_kernels as reader

REAL = ((56, 56, 160, 4), (28, 28, 160, 4), (14, 14, 160, 4), (7, 7, 128, 4))
GENERAL = ((28, 28, 320, 4), (14, 14, 640, 4), (7, 7, 640, 4), (128, 192, 160, 4),
           (96, 128, 320, 4), (512, 768, 160, 4))
SAMPLES = 24


def capture(
    module: object, launches: list,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Event, torch.cuda.Event, list]:
    graph = torch.cuda.CUDAGraph()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        start.record()
        for args, options in launches:
            stream = module._reader_stream(args[0], args[3][0].device)
            module._launch_reader_cuda(*args, **{**options, "stream": stream})
        end.record()
    return graph, start, end, [(args[1], args[2], options.get("threads", module._DEFAULT_THREADS))
                               for args, options in launches]


def measure(graphs: dict) -> dict:
    samples = {name: [] for name in graphs}
    names = tuple(graphs)
    for _ in range(4):
        for graph, *_ in graphs.values():
            graph.replay()
    stream = torch.cuda.current_stream()
    for _ in range(SAMPLES):
        names = names[1:] + names[:1]
        for name in names:
            graph, start, end, _ = graphs[name]
            graph.replay()
            stream.synchronize()
            samples[name].append(start.elapsed_time(end))
    result = {}
    for name, raw in samples.items():
        q1, _, q3 = statistics.quantiles(raw, n=4, method="inclusive")
        result[name] = {"samples_ms": raw, "median_ms": statistics.median(raw),
                        "iqr_ms": q3 - q1, "launches": graphs[name][3]}
    return result


def parity(shape: tuple, dtype: torch.dtype, strict: bool, wave17: object) -> dict:  # noqa: FBT001
    os.environ["LNET_READER_VEC"] = "auto"
    return parity_case(1, shape, dtype, strict, wave17)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bench", "parity"))
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--general-only", action="store_true")
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
        wave14 = load_baseline(args.refs / "wave14.py")
        wave19 = load_baseline(args.refs / "wave19.py")
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
            for name, arm, module in (
                ("wave14", "1", wave14), ("wave17", "auto", wave17), ("wave19", "auto", wave19),
                ("wave20", "auto", reader), ("wave20-vec1", "1", reader),
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
                    arms[name][op] = timing
            return {"shape": (batch, *shape), "dtype": str(dtype), "arms": arms}

        cases = (bench_case(shape, dtype) for shape, dtype in itertools.product(shapes, dtypes))
    else:
        shapes = tuple(dict.fromkeys((*REAL, *STANDARD)))
        edges = tuple((5 + i % 3, 7 + i % 3, m, 1 + i % 4) for i, m in enumerate(PROBE_EDGES))
        cases = (
            parity(shape, dtype, strict, wave17)
            for shape, dtype, strict in itertools.product(
                [*shapes, *edges],
                (*dtypes, torch.float32), (False, True),
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
