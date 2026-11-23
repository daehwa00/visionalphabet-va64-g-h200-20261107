"""Run the existing P1/NR4 training and correctness probes under a selected VScan policy."""

import argparse
import json
import os
import runpy
import statistics
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from bench import POLICY

from lnet import general_cuda_scans as scans

# ruff: noqa: T201, SLF001

SUITE = "h200/liverms_v2_opt/bench_h200.py"
VARIANT = "17-p1-nr4-1024d-v3"


def measured(
    step: Callable, batch: int, *, warmup: int, count: int, graph: bool,
) -> tuple[dict, list[torch.Tensor]]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    capture = None
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            loss = step()
    raw, outputs = [], []
    start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(max(12, count)):
        if capture is None:
            start = time.perf_counter()
            outputs.append(step())
            torch.cuda.synchronize()
            raw.append((time.perf_counter() - start) * 1e3)
        else:
            start_event.record()
            capture.replay()
            end_event.record()
            end_event.synchronize()
            raw.append(start_event.elapsed_time(end_event))
            outputs.append(loss.clone())
    p25, median, p75 = statistics.quantiles(raw, n=4, method="inclusive")
    return {
        "method": "cuda_graph" if graph else "wall_clock",
        "raw_ms": raw, "median_ms": median, "iqr_ms": p75 - p25,
        "min_ms": min(raw), "max_ms": max(raw), "p25_ms": p25, "p75_ms": p75,
        "images_per_second": batch * 1000 / median,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }, outputs


def profile_step(profile: Callable, step: Callable) -> dict:
    original = scans._launch_kernel
    launches = []

    def launch(source: torch.Tensor, key: scans.Kernel, grid: int, smem: int, args: tuple) -> None:
        if not (key.projection or key.projection_reduction):
            original(source, key, grid, smem, args)
            return
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        original(source, key, grid, smem, args)
        end.record()
        launches.append((scans._kernel_name(key), start, end))

    with patch.object(scans, "_launch_kernel", launch):
        result = profile(step)
    projection_ms: dict[str, float] = {}
    for name, start, end in launches:
        projection_ms[name] = projection_ms.get(name, 0) + start.elapsed_time(end)
    result["projection_cuda_event_ms"] = projection_ms
    result["vscan_with_projection_cuda_event_ms"] = (
        result["scan_axis_cuda_event_ms"]["general_vscan"] + sum(projection_ms.values())
    )
    return result


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train-arm", "dynamo", "infer-fold"))
    parser.add_argument(
        "--policy", choices=("auto", "off", "test_saved_tc", "saved_tc"), required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    os.environ[POLICY] = args.policy
    suite = runpy.run_path(SUITE)
    probe = suite["main"]
    def measure(step: Callable, batch: int, *, warmup: int, count: int) -> tuple:
        return measured(step, batch, warmup=warmup, count=count, graph=args.graph)

    def profile(step: Callable) -> dict:
        return profile_step(suite["profile_step"], step)

    cmd = [SUITE, args.command, "--out", str(args.out)]
    if args.command != "dynamo":
        cmd.extend(("--variant", VARIANT, "--batch", str(args.batch)))
    if args.profile:
        if args.command != "train-arm":
            parser.error("--profile requires train-arm")
        cmd.append("--profile")
    if args.graph and args.command != "train-arm":
        parser.error("--graph requires train-arm")
    with ExitStack() as stack:
        stack.enter_context(patch("sys.argv", cmd))
        overrides = {"measure": measure, "profile_step": profile}
        stack.enter_context(patch.dict(probe.__globals__, overrides))
        if args.graph:
            import va_model_builder as builder  # noqa: PLC0415

            build_optimizer = builder.build_optimizer

            def capturable(*values: object, **kwargs: object) -> torch.optim.Optimizer:
                optimizer = build_optimizer(*values, **kwargs)
                for group in optimizer.param_groups:
                    group["capturable"] = True
                return optimizer

            stack.enter_context(patch.object(builder, "build_optimizer", capturable))
        try:
            probe()
        except SystemExit as exc:
            if exc.code:
                raise
    result = json.loads(args.out.read_text())
    if args.command == "train-arm":
        fields = ("finite_losses", "finite_gradients", "running_ms_finite")
        checks = [result.get(key) for key in fields]
    elif args.command == "dynamo":
        counts = [pair for variants in result["graphs"].values() for pair in variants.values()]
        checks = [len(counts) == 18, all(count == [1, 0] for count in counts)]
    else:
        checks = [
            result["fp32_head_parity_by_precision"]["highest"]["passed"],
            result["strict_reload_passed"],
            result["folded_train_guard"],
        ]
    print(args.policy, args.command, checks, args.out)
    raise SystemExit(result.get("status") != "ok" or not all(checks))


if __name__ == "__main__":
    main()
