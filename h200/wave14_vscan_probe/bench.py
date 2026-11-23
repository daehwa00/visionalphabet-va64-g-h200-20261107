"""Measure fused scans one direction at a time with CUDA Graph event nodes."""

import argparse
import json
import os
import statistics
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from variants import BACKWARD, FORWARD, select

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, T201

CASES = (
    (1024, 56, 56, 160, False),
    (1024, 56, 56, 160, True),
    (1024, 28, 28, 160, False),
    (1024, 28, 28, 160, True),
    (1024, 14, 14, 160, False),
    (1024, 7, 7, 160, False),
    (4, 128, 192, 160, False),
    (4, 512, 768, 48, False),
)


def sample(shape: tuple[int, ...], dtype: torch.dtype, scale: float) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=dtype).mul_(scale)


def sample_output(outputs: tuple | list) -> list[torch.Tensor]:
    return [value.flatten()[::max(1, value.numel() // 8192)].float() for value in outputs]


def compare(outputs: list[torch.Tensor], expected: list[torch.Tensor]) -> list[dict]:
    return [
        {
            "max_abs": (a - b).abs().max().item(),
            "max_reference": b.abs().max().item(),
            "rms_error": (a - b).square().mean().sqrt().item(),
            "finite": bool(a.isfinite().all()),
        }
        for a, b in zip(outputs, expected, strict=True)
    ]


def measure(
    operation: Callable, repeats: int, verify_keys: tuple[str, ...], *, vertical: bool = True,
) -> tuple[dict, list[torch.Tensor]]:
    original = scans._dispatch
    begin, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
    plans = []

    def track(source: torch.Tensor, launch: scans.Launch, args: tuple) -> None:
        if launch.key.vertical == vertical:
            begin.record()
            plans.append(launch)
        original(source, launch, args)
        if launch.key.vertical == vertical:
            end.record()

    warm = operation()
    torch.cuda.synchronize()
    values = sample_output(warm)
    del warm
    graph = torch.cuda.CUDAGraph()
    with patch.object(scans, "_dispatch", track), torch.cuda.graph(graph):
        outputs = operation()
    if len(plans) != 1:
        raise RuntimeError(f"Expected one fused scan dispatch, received {len(plans)}")
    plan, key, workers, partials = plans[0]
    if any(not getattr(key, field) for field in verify_keys):
        return {"status": "infeasible"}, values
    graph.replay()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times, totals = [], []
    for _ in range(repeats):
        start.record()
        for _ in range(5):
            graph.replay()
        stop.record()
        stop.synchronize()
        times.append(begin.elapsed_time(end))
        totals.append(start.elapsed_time(stop) / 5)
    del outputs
    return {
        "status": "ok", "kernel_ms": statistics.median(times), "kernel_samples_ms": times,
        "graph_ms": statistics.median(totals), "graph_samples_ms": totals,
        "plan": {**plan._asdict(), "workers": workers, "partials": partials},
        "key": {field: asdict(key).get(field) for field in (
            "vector_weights", "fast_forward", "shared_hidden", "prefetch_rows", "precise",
            "cache_positive", "async_source",
        )},
    }, values


def probe(
    shape: tuple[int, int, int, int, bool], args: argparse.Namespace,
) -> dict:
    b, h, w, m, coarse = shape
    b = max(1, b // args.batch_divisor) if h <= 56 else b
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    torch.manual_seed(731)
    states = sample((4, b, h, w, m), dtype, 0.4)
    pole = [sample((m,), torch.float32, 0.12) for _ in range(4)]
    aux = [torch.ones((2, length, m), device="cuda") for length in (w, h)]
    path = [sample(size, torch.float32, 0.17)
            for size in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    dy = [sample((b, oh, ow, m), dtype, 0.1) for _ in range(2)]

    def forward() -> tuple[torch.Tensor, ...]:
        return scans.vertical_forward_packed(pole, states, aux, path, coarse, 1e-6, True).unbind(0)  # noqa: FBT003

    def backward() -> list[torch.Tensor]:
        return scans.vertical_backward(pole, states, aux, path, *dy, coarse, compact=coarse)

    row = {"shape": [b, h, w, m, coarse], "dtype": str(dtype)}
    states_bytes = states.numel() * states.element_size() // (1 + coarse)
    dy_bytes = sum(t.numel() * t.element_size() for t in dy)
    for direction, operation, options, names, payload in (
        ("forward", forward, FORWARD, args.forward, states_bytes + dy_bytes),
        ("backward", backward, BACKWARD, args.backward, 2 * states_bytes + dy_bytes),
    ):
        results = {}
        expected = None
        for label in ("baseline", *(names if names is not None else options)):
            arm = (label, "baseline") if direction == "forward" else ("baseline", label)
            select(arm, args.warps)
            trial = options.get(label)
            verify = tuple(
                field for field in (
                    "vector_weights", "cache_line", "cache_source", "cache_positive",
                    "async_source",
                )
                if trial is not None and getattr(trial, field)
            )
            result, values = measure(operation, args.repeats, verify)
            if expected is None:
                expected = values
            elif result["status"] == "ok":
                result["parity"] = compare(values, expected)
            if result["status"] == "ok":
                result["payload_bytes"] = payload
                result["GBps"] = payload / (result["kernel_ms"] * 1e6)
            results[label] = result
            torch.cuda.empty_cache()
        row[direction] = results
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-divisor", type=int, default=1)
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)), nargs="+")
    parser.add_argument("--forward", nargs="*", choices=(*FORWARD, "traffic"))
    parser.add_argument("--backward", nargs="*", choices=BACKWARD)
    parser.add_argument("--warps", type=int, choices=range(1, 5))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    os.environ.pop("LNET_VSCAN_DECOUPLED", None)
    with args.out.open("a", encoding="utf-8") as out:
        for index in args.case_index if args.case_index is not None else range(len(CASES)):
            row = probe(CASES[index], args)
            print(json.dumps(row), file=out, flush=True)
            print(CASES[index], {phase: {
                name: round(info["kernel_ms"], 4) if info["status"] == "ok" else "infeasible"
                for name, info in row[phase].items()
            } for phase in ("forward", "backward")}, flush=True)


if __name__ == "__main__":
    main()
