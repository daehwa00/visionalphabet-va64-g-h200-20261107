"""Compare Wave 14A and spatial kernels in one process with paired graph replays."""

import argparse
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import torch

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, T201

CASES = (
    (128, 56, 56, 160, False), (128, 56, 56, 160, True),
    (128, 28, 28, 160, False), (128, 28, 28, 160, True),
    (128, 14, 14, 160, False), (128, 7, 7, 160, False),
    (4, 128, 192, 160, False), (4, 512, 768, 48, False),
)


def sample(shape: tuple[int, ...], dtype: torch.dtype, scale: float) -> torch.Tensor:
    return torch.empty(shape, device="cuda", dtype=dtype).normal_(std=scale)


type ScanOutput = list[torch.Tensor] | tuple[torch.Tensor, ...]
type Capture = tuple[torch.cuda.CUDAGraph, ScanOutput, torch.cuda.Event, torch.cuda.Event,
                     scans.Launch, list[torch.Tensor]]


def capture(operation: Callable[[], ScanOutput], variant: str) -> Capture:
    original = scans._dispatch
    plans = []
    begin, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))

    def track(source: torch.Tensor, launch: scans.Launch, args: tuple) -> None:
        begin.record()
        original(source, launch, args)
        plans.append(launch)
        end.record()

    os.environ["LNET_VSCAN_SPATIAL"] = variant
    warm = operation()
    torch.cuda.synchronize()
    sample_values = [value.reshape(-1)[::max(1, value.numel() // 4096)].float() for value in warm]
    del warm
    graph = torch.cuda.CUDAGraph()
    with patch.object(scans, "_dispatch", track), torch.cuda.graph(graph):
        outputs = operation()
    (launch,) = plans
    return graph, outputs, begin, end, launch, sample_values


def replay_timing(arm: Capture) -> tuple[float, float]:
    graph, _, begin, end, _, _ = arm
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    stop.record()
    stop.synchronize()
    return begin.elapsed_time(end), start.elapsed_time(stop) / 5


def compare(values: list[torch.Tensor], expected: list[torch.Tensor]) -> list[dict]:
    return [
        {"max_abs": (a - b).abs().max().item(), "finite": bool(a.isfinite().all())}
        for a, b in zip(values, expected, strict=True)
    ]


def probe(case: tuple, dtype: torch.dtype, args: argparse.Namespace) -> dict:
    b, h, w, m, coarse = case
    if b == 128:
        b = args.batch
    torch.manual_seed(731)
    states = sample((4, b, h, w, m), dtype, 0.4)
    pole = [sample((m,), torch.float32, 0.12) for _ in range(4)]
    aux = [torch.ones((2, length, m), device="cuda") for length in (w, h)]
    path = [sample(size, torch.float32, 0.17) for size in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    dy = tuple(sample((b, h // (1 + coarse), w // (1 + coarse), m), dtype, 0.1) for _ in range(2))

    def forward() -> ScanOutput:
        result = scans._vscan(pole, states, aux, path, coarse, 1e-6, precise=False)
        if not isinstance(result, tuple):
            raise TypeError("Expected separate vertical outputs")
        return result

    def backward() -> ScanOutput:
        return scans.vertical_backward(pole, states, aux, path, *dy, coarse, compact=coarse)

    row = {"shape": (b, h, w, m, coarse), "dtype": str(dtype), "results": {}}
    for direction, operation in (("forward", forward), ("backward", backward)):
        results = {}
        for variant in args.variants:
            baseline = capture(operation, "")
            candidate = capture(operation, variant)
            samples = ([], [])
            for iteration in range(args.repeats):
                order = (0, 1) if iteration % 2 == 0 else (1, 0)
                for idx in order:
                    samples[idx].append(replay_timing((baseline, candidate)[idx]))
            entry = {}
            arms = zip(("base", "trial"), (baseline, candidate), samples, strict=True)
            for arm, data, timings in arms:
                plan, key, workers, partials = data[4]
                entry[arm] = {
                    "spatial": key.spatial, "plan": plan._asdict(),
                    "workers": workers, "partials": partials,
                    "raw_ms": timings,
                    "median_kernel_ms": statistics.median(t[0] for t in timings),
                    "median_graph_ms": statistics.median(t[1] for t in timings),
                }
            entry["parity"] = compare(candidate[5], baseline[5])
            results[variant] = entry
            del baseline, candidate
        row["results"][direction] = results
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--variants", nargs="+", default=["4", "8"])
    parser.add_argument("--cases", nargs="+", type=int, default=list(range(len(CASES))))
    parser.add_argument("--dtypes", nargs="+", choices=["float16", "bfloat16"],
                        default=["bfloat16", "float16"])
    args = parser.parse_args()
    os.environ["LNET_VSCAN_DECOUPLED"] = "0"
    with args.out.open("w") as handle, torch.no_grad():
        for name in args.dtypes:
            for case in args.cases:
                row = probe(CASES[case], getattr(torch, name), args)
                print(json.dumps(row), file=handle, flush=True)
                print(name, case, {
                    direction: {
                        variant: round(
                            data["trial"]["median_graph_ms"] / data["base"]["median_graph_ms"], 3,
                        )
                        for variant, data in results.items()
                    }
                    for direction, results in row["results"].items()
                }, flush=True)


if __name__ == "__main__":
    main()
