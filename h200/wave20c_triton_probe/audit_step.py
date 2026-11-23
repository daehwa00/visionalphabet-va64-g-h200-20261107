"""Profile the baseline full training step and retain every CUDA kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from h200.liverms_v2_opt import bench_h200

if TYPE_CHECKING:
    from collections.abc import Callable


def profile_step(step: Callable[[], torch.Tensor], trace: Path) -> dict:
    """Record launch buckets and a trace with CPU-to-CUDA correlations."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
    ) as profiler:
        step()
        torch.cuda.synchronize()

    profiler.export_chrome_trace(str(trace))
    return summarize_trace(trace)


def summarize_trace(trace: Path) -> dict:
    """Sum CUDA kernel, memcpy and memset durations, excluding annotations."""
    buckets: dict[str, float] = {}
    kernels: dict[str, dict] = {}
    for event in json.loads(trace.read_text())["traceEvents"]:
        if event.get("cat") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        name = event["name"]
        bucket = "triton" if name.startswith("_swiglu_backward") else bench_h200.kernel_bucket(name)
        elapsed = event["dur"] / 1000
        buckets[bucket] = buckets.get(bucket, 0.0) + elapsed
        row = kernels.setdefault(name, {
            "name": name,
            "bucket": bucket,
            "count": 0,
            "self_cuda_ms": 0.0,
        })
        row["count"] += 1
        row["self_cuda_ms"] += elapsed
    return {
        "self_cuda_ms_by_bucket": buckets,
        "total_self_cuda_ms": sum(buckets.values()),
        "kernels": sorted(kernels.values(), key=lambda kernel: -kernel["self_cuda_ms"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=bench_h200.VARIANTS[0], choices=bench_h200.VARIANTS)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args()
    setup = bench_h200.setup

    def configure_inputs(batch: int) -> torch.Tensor:
        inputs = setup(batch)
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        return inputs

    args.profile = True
    bench_h200.setup = configure_inputs
    def capture(step: Callable[[], torch.Tensor]) -> dict:
        return profile_step(step, args.out.with_suffix(".trace.json"))

    bench_h200.profile_step = capture
    result = bench_h200.train_arm(args)
    for name in ("allocated", "reserved"):
        key = f"peak_{name}"
        result[f"{key}_bytes"] = int(result[f"{key}_gib"] * bench_h200.GIB)
    bench_h200.write_json(args.out, result)


if __name__ == "__main__":
    main()
