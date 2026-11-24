"""Cycle CUDA Graph timings for fused scan and partial-reduction paths."""

import argparse
import json
import os
import statistics
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001

CASES = (
    (56, 56, 160, False), (56, 56, 160, True),
    (28, 28, 160, False), (28, 28, 160, True),
    (14, 14, 160, False), (14, 14, 160, True),
    (7, 7, 128, False), (64, 96, 160, False),
    (37, 53, 192, False), (128, 192, 160, False),
    (37, 53, 53, False), (38, 54, 53, True),
)
PLAN = scans._plan_launch
Operation = Callable[[], torch.Tensor | list[torch.Tensor]]
Events = tuple[torch.cuda.Event, torch.cuda.Event]
Captured = tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, ...], scans.Launch, Events]
Case = tuple[int, int, int, bool]
DISPATCH = scans._dispatch


def trial_plan(args: tuple, options: dict, name: str) -> scans.Launch | None:
    policy = "auto" if name == "auto" else "off"
    with patch.dict(os.environ, {"LNET_SCAN_W25A": policy}):
        base = PLAN(*args, **options)
    if name in {"baseline", "auto"} or base is None:
        return base
    kind, warp_text, mode = name.split("/")
    warps = int(warp_text)
    plan, key, workers, _ = base
    _, (batch, h, w, _m) = args
    threads = warps * 32
    device = args[0].device.index or 0
    limit = scans.device_limits(device)[1]
    if key.vertical:
        vector = mode.startswith("v")
        batch_path = mode.endswith("4")
        register_batch2 = kind == "reg" and mode.endswith("2")
        if kind == "reg":
            batch_path = batch_path or register_batch2 or mode.endswith("b")
        line_bytes = h // (1 + key.coarse) * 4 * key.dtype.itemsize * threads
        weight_bytes = 11 * 8 * 32 * (4 if vector else 2)
        cache_line = plan.cache_line and line_bytes + weight_bytes <= limit
        smem = (line_bytes if cache_line else 0) + (weight_bytes if kind == "ref" else 0)
        key = replace(
            key, warps=warps, vector_weights=vector, precise=not vector,
            batch_path=batch_path, register_path=kind == "reg", cache_weights=kind == "ref",
            register_batch2=register_batch2,
        )
    else:
        line_bytes = w // (1 + key.compact) * (8 if key.backward else 2 * key.dtype.itemsize)
        line_bytes *= threads
        cache_line = plan.cache_line and line_bytes <= limit
        smem = max(
            16 * threads if key.backward and warps > 1 else 0,
            line_bytes if cache_line else 0,
        )
        forward_unroll = int(kind[1:]) if kind.startswith("u") and not key.backward else 0
        key = replace(
            key, warps=warps, horizontal_unroll8=kind == "u8" and key.backward,
            horizontal_forward_unroll=forward_unroll,
        )
    key = replace(key, cache_line=cache_line)
    plan = plan._replace(
        warps=warps, smem=smem, cache_weights=key.cache_weights, cache_line=cache_line,
    )
    if smem > limit or not scans._resident(device, key, smem):
        return None
    workers = min(workers, batch * (w // (1 + key.coarse) if key.vertical else h))
    return scans.Launch(plan, key, workers, scans._ceil(workers, warps))


def capture(operation: Operation, name: str) -> Captured | None:
    chosen = None

    def pick(*args: object, **options: object) -> scans.Launch:
        nonlocal chosen
        chosen = trial_plan(args, options, name)
        if chosen is None:
            raise ValueError("infeasible")
        return chosen

    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)

    def record(source: torch.Tensor, launch: scans.Launch, args: tuple) -> None:
        begin.record()
        DISPATCH(source, launch, args)
        end.record()

    with patch.object(scans, "_plan_launch", pick):
        try:
            operation()
        except ValueError as error:
            if str(error) == "infeasible":
                return None
            raise
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with patch.object(scans, "_dispatch", record), torch.cuda.graph(graph):
            outputs = operation()
    if chosen is None:
        raise RuntimeError("Capture did not launch a scan")
    return graph, tuple(outputs), chosen, (begin, end)


def compare(a: tuple, b: tuple) -> list[dict]:
    return [
        {"exact": torch.equal(x, y), "finite": bool(y.isfinite().all()),
         "max_abs": (x.float() - y.float()).abs().max().item(),
         "reference_max": x.abs().max().item()}
        for x, y in zip(a, b, strict=True)
    ]


def replay_ms(graph: torch.cuda.CUDAGraph) -> float:
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(3):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 3


def pair(operation: Operation, trials: tuple[str, ...], repeats: int) -> Iterator[dict]:
    for name in trials:
        baseline_name = "baseline" if name == "auto" else "ref/" + name.split("/", 1)[1]
        reference = capture(operation, baseline_name)
        if reference is None:
            yield {"trial": name, "status": "reference infeasible"}
            continue
        baseline, expected, original, base_events = reference
        baseline.replay()
        candidate = capture(operation, name)
        if candidate is None:
            yield {"trial": name, "status": "infeasible"}
            continue
        graph, values, selected, events = candidate
        graph.replay()
        parity = compare(expected, values)
        times = [[], []]
        scan_times = [[], []]
        for i in range(repeats):
            for j in (range(2) if i % 2 == 0 else (1, 0)):
                times[j].append(replay_ms((baseline, graph)[j]))
                start, stop = (base_events, events)[j]
                scan_times[j].append(start.elapsed_time(stop))
        yield {
            "trial": name, "status": "ok", "parity": parity, "samples_ms": times,
            "median_ms": [statistics.median(timings) for timings in times],
            "scan_median_ms": [statistics.median(timings) for timings in scan_times],
            "scan_samples_ms": scan_times,
            "launches": [
                {**vars(p.key), **p.plan._asdict(), "dtype": str(p.key.dtype),
                 "workers": p.workers, "partials": p.partials} for p in (original, selected)
            ],
        }
        del candidate, graph, values, reference, baseline, expected


def probe(
    case: Case, batch: int, dtype: torch.dtype, repeats: int, *,
    production: bool, only: set[str] | None = None,
) -> Iterator[dict]:
    h, w, m, coarse = case
    b = min(batch, 4) if max(h, w) > 96 else batch
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    torch.manual_seed(501)

    def sample(shape: tuple, kind: torch.dtype, scale: float) -> torch.Tensor:
        return torch.randn(shape, device="cuda", dtype=kind).mul_(scale)

    shape = (b, h, w, m)
    pole = [sample((m,), torch.float32, 0.12) for _ in range(4)]
    states = sample((4, *shape), dtype, 0.4)
    auxiliary = [sample((2, length, m), torch.float32, 0.1).add_(1) for length in (w, h)]
    path = [sample(size, torch.float32, 0.17)
            for size in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    dy = [sample((b, oh, ow, m), dtype, 0.1) for _ in range(2)]

    planar = True

    def forward() -> torch.Tensor:
        return scans.vertical_forward_packed(pole, states, auxiliary, path, coarse, 1e-6, planar)


    def backward() -> list[torch.Tensor]:
        return scans.vertical_backward(pole, states, auxiliary, path, *dy, coarse, compact=coarse)

    def rows(kind: str, operation: Operation, trials: tuple[str, ...]) -> Iterator[dict]:
        if only:
            trials = tuple(name for name in trials if name in only)
        if not trials:
            return
        for row in pair(operation, trials, repeats):
            yield {"shape": shape, "coarse": coarse, "dtype": str(dtype), "direction": kind, **row}

    forward_trials = tuple(
        f"reg/{w}/{m}" for w in range(1, 5)
        for m in ("s1", "s2", "s4", "sb", "v1", "v2", "v4", "vb")
    )
    yield from rows("vf", forward, ("auto",) if production else forward_trials)
    if h in {14, 28, 56}:
        planar = False
        yield from rows(
            "vf_interleaved", forward,
            ("auto",) if production else ("reg/1/v4", "reg/4/v1", "reg/4/v2", "reg/4/vb"),
        )
    if production:
        yield from rows("vb", backward, ("auto",))
    real, imag = [sample(shape, dtype, 0.4) for _ in range(2)]
    hs = scans.horizontal_forward(pole, real, imag)
    grad = sample((4, b, h, ow, m), dtype, 0.1)
    yield from rows(
        "hb", lambda: scans.horizontal_backward(pole, [real, imag], hs, grad, coarse),
        ("auto",) if production else tuple(f"u8/{w}/h" for w in range(1, 5)),
    )
    yield from rows(
        "hf", lambda: scans.horizontal_forward(pole, real, imag),
        ("auto",) if production else tuple(f"u{u}/{w}/h" for w in range(1, 5) for u in (4, 8)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(len(CASES))))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), required=True)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--only", nargs="+", type=str)
    args = parser.parse_args()
    with args.out.open("w") as stream:
        for index in args.cases:
            for row in probe(
                CASES[index], args.batch, getattr(torch, args.dtype), args.repeats,
                production=args.production, only=set(args.only) if args.only else None,
            ):
                print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    main()
