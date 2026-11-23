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
from variants import experimental, source

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, FBT003

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


def with_backward_geometry(base: scans.Launch, h: int, name: str) -> scans.Launch | None:
    plan, key, workers, partials = base
    warps, divisor = map(int, name[6:].split("/"))
    workers = max(1, workers // divisor)
    count = scans._ceil(workers, warps)
    if count > partials or key.shared_acc or not key.hidden8:
        return None
    threads = 32 * warps
    itemsize = key.dtype.itemsize
    weights = 11 * 8 * 32 * (4 if key.vector_weights else itemsize) if key.cache_weights else 0
    span = plan.chunk
    trajectory = 16 * span + 4 * itemsize * (h if key.cache_source else span)
    boundary = 0 if key.global_checkpoints else ((h - 1) // span) * 16
    line = (h // (1 + key.coarse)) * 4 * itemsize if key.cache_line else 0
    reduction = 32 * threads * 4 if warps > 1 else 0
    smem = max(reduction, weights + (trajectory + boundary + line) * threads)
    return scans.Launch(plan._replace(warps=warps, smem=smem),
                        replace(key, warps=warps), workers, count)


def trial_plan(args: tuple, options: dict, name: str) -> scans.Launch | None:  # noqa: C901, PLR0911, PLR0912, PLR0915
    policy = "auto" if name == "auto" else "off"
    with patch.dict(os.environ, {"LNET_SCAN_W24A": policy}):
        base = PLAN(*args, **options)
    if name in {"baseline", "auto"} or base is None:
        return base
    plan, key, workers, partials = base
    _, (batch, h, w, _m) = args
    if name.startswith(("reg", "float")):
        if key.dtype == torch.float32:
            return None
        weights = 11 * 8 * 32 * (4 if key.vector_weights else key.dtype.itemsize)
        trial = experimental(key, name.removesuffix("-line"))
        smem = plan.smem - (weights if key.cache_weights else 0)
        line = key.cache_line or name.endswith("-line")
        if line and not key.cache_line:
            smem += (h // (1 + key.coarse)) * 4 * key.dtype.itemsize * key.warps * 32
        plan = plan._replace(smem=smem, cache_weights=False, cache_line=line)
        key = replace(trial, cache_line=line)
    elif name.startswith("packed"):
        if key.dtype == torch.float32 or not key.cache_weights:
            return None
        trial = experimental(key, name)
        chunk = 14 if "14" in name else plan.chunk
        line = key.cache_line or name.endswith("-line")
        if chunk % 2 and key.coarse:
            return None
        threads = key.warps * 32
        itemsize = key.dtype.itemsize
        full = plan.full_trajectory and chunk == h
        boundary = (h - 1) // chunk * threads * 16
        trajectory = (16 * chunk + 4 * itemsize * (h if key.cache_source else chunk)) * threads
        line_bytes = (h // (1 + key.coarse)) * 4 * itemsize * threads if line else 0
        weights = 11 * 8 * 32 * itemsize
        smem = max(32 * threads * 4 if key.warps > 1 else 0,
                   weights + trajectory + boundary + line_bytes)
        if key.shared_acc:
            return None
        plan = plan._replace(smem=smem, chunk=chunk, cache_line=line,
                             full_trajectory=full, global_checkpoints=False)
        key = replace(trial, chunk=0 if full else chunk, cache_line=line, global_checkpoints=False)
    elif name in {"cap192", "hu1", "hu2", "hu8", "hldg"}:
        key = experimental(key, name)
    elif name == "vy":
        if key.shared_acc:
            return None
        plan = plan._replace(smem=plan.smem + 2 * h * 32 * 4)
        key = experimental(key, name)
    elif name.startswith("vbwarp"):
        updated = with_backward_geometry(base, h, name)
        if updated is None:
            return None
        plan, key, workers, partials = updated
    elif name.startswith("warp"):
        warps, divisor = map(int, name[4:].split("/"))
        workers = min(batch * h, workers // divisor)
        partials = scans._ceil(workers, warps)
        if partials > base.partials:
            return None
        smem = max(16 * warps * 32 if warps > 1 else 0,
                   8 * (w // (1 + key.compact)) * warps * 32 if key.cache_line else 0)
        plan = plan._replace(warps=warps, smem=smem)
        key = replace(key, warps=warps)
    device = args[0].device.index or 0
    if plan.smem > scans.device_limits(device)[1] or not scans._resident(device, key, plan.smem):
        return None
    return scans.Launch(plan, key, workers, partials)


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
    reference = capture(operation, "baseline")
    if reference is None:
        raise RuntimeError("Baseline is infeasible")
    baseline, expected, original, base_events = reference
    baseline.replay()
    for name in trials:
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
        del candidate, graph, values


def probe(
    case: Case, batch: int, dtype: torch.dtype, repeats: int, *, production: bool,
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

    def forward() -> torch.Tensor:
        return scans.vertical_forward_packed(pole, states, auxiliary, path, coarse, 1e-6, True)

    def backward() -> list[torch.Tensor]:
        return scans.vertical_backward(pole, states, auxiliary, path, *dy, coarse, compact=coarse)

    def rows(kind: str, operation: Operation, trials: tuple[str, ...]) -> Iterator[dict]:
        for row in pair(operation, trials, repeats):
            yield {"shape": shape, "coarse": coarse, "dtype": str(dtype), "direction": kind, **row}

    forward_trials = ("reg1", "reg2", "reg4", "float1", "float2", "float4", "reg1-line")
    backward_trials = ("packed", "packed-line", "packed14", "packed14-line", "vy", "cap192")
    yield from rows("vf", forward, ("auto",) if production else forward_trials)
    if h in {14, 56}:
        backward_trials += ("vbwarp2/1", "vbwarp2/2", "vbwarp4/1", "vbwarp4/2")
    yield from rows("vb", backward, ("auto",) if production else backward_trials)
    if h == 56 or (production and h in {14, 28}):
        real, imag = [sample(shape, dtype, 0.4) for _ in range(2)]
        hs = scans.horizontal_forward(pole, real, imag)
        grad = sample((4, b, h, ow, m), dtype, 0.1)
        horizontal_trials = tuple(f"warp{w}/{d}" for w in range(1, 5) for d in (1, 2, 4))
        horizontal_trials += ("hu1", "hu8", "hldg") if coarse else ("hu1", "hu2", "hu8", "hldg")
        yield from rows(
            "hb", lambda: scans.horizontal_backward(pole, [real, imag], hs, grad, coarse),
            ("auto",) if production else horizontal_trials,
        )
        yield from rows(
            "hf", lambda: scans.horizontal_forward(pole, real, imag),
            ("auto",) if production else ("baseline",),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(len(CASES))))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), required=True)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    with patch.object(scans, "_kernel_source", source), args.out.open("w") as stream:
        for index in args.cases:
            for row in probe(
                CASES[index], args.batch, getattr(torch, args.dtype), args.repeats,
                production=args.production,
            ):
                print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    main()
