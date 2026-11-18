"""Measure input-pipeline costs without running or modifying a training job.

The paced probe sleeps on the CPU in place of model compute. It measures supply
headroom, not end-to-end training throughput or contention with real GPU work.
Only the consumer stream is synchronized; waiting for the entire device would
incorrectly charge next-batch prefetch to the current batch.
"""

from __future__ import annotations

# CLI namespaces, loader batches, and profiler JSON expose dynamic external schemas.
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


def summarize_samples(samples: Sequence[float]) -> dict[str, float | int]:
    if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("Samples must be nonempty, finite, nonnegative milliseconds")
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "mean_ms": statistics.mean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "max_ms": ordered[-1],
    }


def summarize_gpu_trace(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gpu = [
        event
        for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
        and "ts" in event
        and "dur" in event
    ]
    if not gpu:
        raise ValueError("Trace contains no timed GPU events")
    devices = {event.get("args", {}).get("device") for event in gpu}
    if len(devices - {None}) > 1:
        raise ValueError("Trace must contain a single GPU; split devices before summarizing")
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in gpu)
    end = intervals[0][0]
    busy = 0.0
    for start, stop in intervals:
        busy += max(0, stop - max(start, end))
        end = max(end, stop)
    span = end - intervals[0][0]
    kernels: dict[str, float] = collections.defaultdict(float)
    for event in gpu:
        if event["cat"] == "kernel":
            kernels[event["name"]] += event["dur"] / 1000
    return {
        "span_ms": span / 1000,
        "busy_union_ms": busy / 1000,
        "idle_within_span_ms": (span - busy) / 1000,
        "kernel_calls": sum(event["cat"] == "kernel" for event in gpu),
        "kernel_ms": dict(sorted(kernels.items(), key=lambda item: item[1], reverse=True)),
        "kernel_time_scope": "Sum of individual kernel durations, not timeline union",
        "scope": "GPU timeline union; profiler overhead included; excludes untimed pipeline work",
    }


@contextlib.contextmanager
def idle_gpu(uuid: str) -> Generator[None]:
    """Use the same advisory GPU lock as the training supervisor."""
    if not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid):
        raise ValueError("Pass the full GPU UUID reported by nvidia-smi")
    lock_path = Path.home() / ".cache/visionalphabet" / f"gpu-{uuid}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            text=True,
        )
        if any(row.startswith(uuid + ",") for row in apps.splitlines()):
            raise RuntimeError(f"GPU is occupied: {uuid}")
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        yield


def measure_loader(loader: Any, *, iterations: int, warmup: int, cadence: float) -> dict[str, Any]:
    import imagenet100_data_runtime as data
    import torch

    count = min(iterations, len(loader))
    if count <= warmup:
        raise ValueError("Loader must contain more batches than warmup")
    samples = []
    batches = cast(
        "Generator[tuple[torch.Tensor, torch.Tensor]]",
        data.device_batches(loader, torch.device("cuda"), channels_last=True),
    )
    try:
        for _ in range(count):
            started = time.perf_counter()
            inputs, targets = next(batches)
            torch.cuda.current_stream().synchronize()
            samples.append((time.perf_counter() - started) * 1000)
            if cadence:
                time.sleep(cadence)
            del inputs, targets
    finally:
        batches.close()
    return {
        "first_batch_ms": samples[0],
        "steady": summarize_samples(samples[warmup:]),
        "samples_ms": samples,
        "warmup_batches": warmup,
        "simulated_compute_ms": cadence * 1000,
    }


def measure_transfer() -> dict[str, Any]:
    import torch

    host = torch.empty((128, 3, 224, 224), pin_memory=True).normal_()
    results = {}
    for name, channels_last in (("h2d", False), ("h2d_plus_layout", True)):
        samples = []
        for iteration in range(25):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            batch = host.to("cuda", non_blocking=True)
            if channels_last:
                batch = batch.contiguous(memory_format=torch.channels_last)
            end.record()
            end.synchronize()
            if iteration >= 5:
                samples.append(begin.elapsed_time(end))
        results[name] = summarize_samples(samples)
    return {"batch_shape": list(host.shape), "dtype": str(host.dtype), **results}


def probe_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    import imagenet100_data_runtime as data
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(501)
    affinity = data.configure_cpu_affinity()
    train, validation = data.loaders(
        args.data_root,
        batch_size=128,
        workers=args.workers,
        training_generator=torch.Generator().manual_seed(501),
    )
    cases = (
        ("train_saturated", train, 0.0),
        ("train_paced", train, args.cadence_ms / 1000),
        ("validation_saturated", validation, 0.0),
    )
    result: dict[str, Any] = {
        "scope": "Actual transforms and prefetch; CPU sleep replaces compute; not training throughput",
        "gpu_uuid": args.gpu_uuid,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cpu_affinity": sorted(affinity),
        "batch_size": 128,
        "workers": train.num_workers,
        "persistent_workers": train.persistent_workers,
        "prefetch_factor": train.prefetch_factor,
        "data_runtime_sha256": hashlib.sha256(Path(data.__file__).read_bytes()).hexdigest(),
        "runs": {},
    }
    for name, loader, cadence in cases:
        row = measure_loader(
            loader, iterations=args.iterations, warmup=args.warmup, cadence=cadence
        )
        result["runs"][name] = row
        print(name, row["steady"], flush=True)
    result["transfer"] = measure_transfer()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--cadence-ms", type=float, default=106.37)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve evidence")
    if not args.trace and not args.data_root:
        parser.error("Provide --trace, --data-root, or both")
    if args.data_root and not args.gpu_uuid:
        parser.error("Pipeline measurement requires --gpu-uuid")
    if args.warmup < 0 or args.iterations <= args.warmup or args.workers < 0:
        parser.error("Require iterations > warmup >= 0 and workers >= 0")
    if not math.isfinite(args.cadence_ms) or args.cadence_ms < 0:
        parser.error("Cadence must be finite and nonnegative")
    result = {}
    if args.trace:
        result["trace"] = summarize_gpu_trace(json.loads(args.trace.read_text())["traceEvents"])
    if args.data_root:
        with idle_gpu(args.gpu_uuid):
            result["pipeline"] = probe_pipeline(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
