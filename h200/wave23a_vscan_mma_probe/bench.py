"""Measure the backward planner opt-in against Wave 21C on identical inputs."""

import argparse
import json
import os
import statistics
from collections.abc import Callable
from unittest.mock import patch

import torch

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, T201
SHAPES = (
    (56, 56, 160, False), (56, 56, 160, True),
    (28, 28, 160, False), (28, 28, 160, True),
    (14, 14, 160, False), (14, 14, 160, True), (7, 7, 128, False),
    (64, 96, 160, False), (37, 53, 192, False), (128, 192, 160, False),
)


def measure(
    operation: Callable[[], list[torch.Tensor]], repeats: int,
) -> tuple[dict, list[torch.Tensor]]:
    launch_log = []
    dispatch = scans._dispatch
    start = torch.cuda.Event(enable_timing=True, external=True)
    stop = torch.cuda.Event(enable_timing=True, external=True)

    def track(source: torch.Tensor, launch: scans.Launch, args: tuple) -> None:
        launch_log.append(launch)
        start.record()
        dispatch(source, launch, args)
        stop.record()

    first = operation()
    torch.cuda.synchronize()
    samples = [t.flatten()[::max(t.numel() // 8192, 1)].float().cpu() for t in first]
    del first
    graph = torch.cuda.CUDAGraph()
    torch.cuda.reset_peak_memory_stats()
    with patch.object(scans, "_dispatch", track), torch.cuda.graph(graph):
        outputs = operation()
    times = []
    for _ in range(repeats):
        graph.replay()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    del outputs
    (launch,) = launch_log
    result = {
        "kernel_ms": statistics.median(times),
        "mma": getattr(launch.key, "mma_path", False),
        "plan": launch.plan._asdict(),
        "workers": launch.workers,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }
    return result, samples


def probe(
    shape: tuple[int, int, int, bool], dtype: torch.dtype, batch: int, repeats: int,
    policies: tuple[str, ...],
) -> dict:
    h, w, m, coarse = shape
    step = 1 + int(coarse)
    torch.manual_seed(731)

    def sample(
        shape: tuple[int, ...], scale: float, dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.randn(shape, device="cuda", dtype=dtype).mul_(scale)

    states = sample((4, batch, h, w, m), 0.4, dtype)
    pole = [sample((m,), 0.12) for _ in range(4)]
    aux = [torch.ones((2, length, m), device="cuda") for length in (w, h)]
    path = [sample(size, 0.17) for size in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    dy = [sample((batch, h // step, w // step, m), 0.1, dtype) for _ in range(2)]

    def backward() -> list[torch.Tensor]:
        return scans.vertical_backward(pole, states, aux, path, *dy, coarse, compact=coarse)

    out = {"shape": (batch, *shape), "dtype": str(dtype)}
    for mode in policies:
        os.environ["LNET_VSCAN_MMA"] = mode
        torch.cuda.empty_cache()
        out[mode], values = measure(backward, repeats)
        if mode == policies[0]:
            expected = values
        else:
            out[mode]["max_abs"] = [(a - b).abs().max().item() for a, b in zip(
                values, expected, strict=True
            )]
            out[mode]["finite"] = all(t.isfinite().all().item() for t in values)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--cases", type=int, nargs="+", default=range(len(SHAPES)))
    parser.add_argument("--policies", choices=("off", "on", "auto"), nargs="+",
                        default=("off", "on", "auto"))
    args = parser.parse_args()
    for dtype in (torch.bfloat16, torch.float16):
        for case in args.cases:
            try:
                row = probe(SHAPES[case], dtype, args.batch, args.repeats, tuple(args.policies))
            except torch.OutOfMemoryError as error:
                row = {"shape": SHAPES[case], "dtype": str(dtype), "oom": str(error)}
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
