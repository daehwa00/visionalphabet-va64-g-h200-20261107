"""Measure the two unchanged stride-2 stems with isolated backward masks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


def measure(op: Callable, samples: int) -> dict:
    for _ in range(5):
        op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(samples):
        start.record()
        op()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as profile:
        op()
        torch.cuda.synchronize()
    return {
        "event_ms": statistics.median(timings),
        "samples_ms": timings,
        "kernels": [
            {"name": event.name, "us": event.device_time_total}
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ],
    }


def clock_sample() -> dict:
    """Compare host clocks and elapsed CUDA events across the same idle interval."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter_ns()
    raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    start.record()
    start.synchronize()
    time.sleep(0.075)
    end.record()
    end.synchronize()
    return {
        "wall_ms": (time.perf_counter_ns() - wall) / 1e6,
        "raw_wall_ms": (time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - raw) / 1e6,
        "cuda_event_ms": start.elapsed_time(end),
    }


def stem(batch: int, in_channels: int, out_channels: int, side: int, samples: int) -> dict:
    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16).contiguous(
            memory_format=torch.channels_last
        )

    inputs = rand(batch, in_channels, side, side)
    weights = rand(out_channels, in_channels, 3, 3)
    upstream = rand(batch, out_channels, side // 2, side // 2)

    def backward(mask: list[bool]) -> tuple[torch.Tensor, ...]:
        return torch.ops.aten.convolution_backward.default(
            upstream,
            inputs,
            weights,
            None,
            [2, 2],
            [1, 1],
            [1, 1],
            False,  # noqa: FBT003
            [0, 0],
            1,
            mask,
        )

    return {
        "channels": [in_channels, out_channels],
        "side": side,
        "forward": measure(
            lambda: torch.nn.functional.conv2d(inputs, weights, stride=2, padding=1), samples
        ),
        "dgrad": measure(lambda: backward([True, False, False]), samples),
        "wgrad": measure(lambda: backward([False, True, False]), samples),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(501)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    result = {
        "batch": args.batch,
        "clock_samples": [clock_sample() for _ in range(3)],
        "cudnn_benchmark": args.cudnn_benchmark,
        "stems": [
            stem(args.batch, cin, cout, side, args.count)
            for cin, cout, side in ((3, 32, 224), (32, 64, 112))
        ],
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
