"""Run the matched training benchmark and time Reader launches with CUDA events."""

# ruff: noqa: SLF001
import argparse
import runpy
import sys
from collections import defaultdict
from collections.abc import Callable

import torch

from lnet import factorized_reader_kernels as reader


def profile_reader(step: Callable, profile: Callable) -> dict:
    original = reader._launch_reader_cuda
    launches = []

    def track(*args: object, **kwargs: object) -> None:
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        original(*args, **kwargs)
        end.record()
        launches.append((args[1], start, end))

    reader._launch_reader_cuda = track
    try:
        result = profile(step)
    finally:
        reader._launch_reader_cuda = original
    events = defaultdict(float)
    counts = defaultdict(int)
    for op, start, end in launches:
        events[op] += start.elapsed_time(end)
        counts[op] += 1
    result["reader_cuda_event_ms"] = dict(events)
    result["reader_cuda_event_counts"] = dict(counts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    args, benchmark_args = parser.parse_known_args()
    path = "h200/liverms_v2_opt/bench_h200.py"
    benchmark = runpy.run_path(path)["main"]
    namespace = benchmark.__globals__
    setup, profile, measure = namespace["setup"], namespace["profile_step"], namespace["measure"]

    def measure_bytes(*args: object, **kwargs: object) -> tuple:
        result, outputs = measure(*args, **kwargs)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        return result, outputs

    def configure(batch: int) -> torch.Tensor:
        inputs = setup(batch)
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
        return inputs

    namespace["setup"] = configure
    namespace["measure"] = measure_bytes
    namespace["profile_step"] = lambda step: profile_reader(step, profile)
    sys.argv = [path, *benchmark_args]
    benchmark()


if __name__ == "__main__":
    main()
