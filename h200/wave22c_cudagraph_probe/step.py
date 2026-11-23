"""Compare native, capture-safe, and CUDA Graph training in fresh processes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liverms_v2_opt"))
from bench_h200 import VARIANTS, build_model, setup
from va_model_builder import build_optimizer

from lnet import capture_safe_orthogonal
from lnet.capture_safe_orthogonal import (
    fixed_taylor_matrix_exp,
    prepare_capture_safe_orthogonal_,
)


def autocast_disabled_exp(matrix: torch.Tensor, **options: object) -> torch.Tensor:
    with torch.autocast(matrix.device.type, enabled=False):
        return fixed_taylor_matrix_exp(matrix, **options)


def raw_time() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def trace_summary(path: Path) -> dict:
    events = json.loads(path.read_text())["traceEvents"]
    (marker,) = [
        event
        for event in events
        if event.get("name") == "wave22c.step" and event.get("cat") == "user_annotation"
    ]
    begin, end = marker["ts"], marker["ts"] + marker["dur"]
    runtime = Counter(
        event["name"]
        for event in events
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}
        and begin <= event["ts"] < end
    )
    gpu = [
        event for event in events if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in gpu)
    busy = 0.0
    if intervals:
        left, right = intervals[0]
        for start, stop in intervals[1:]:
            if start > right:
                busy += right - left
                left = start
            right = max(right, stop)
        busy += right - left
    return {
        "runtime": dict(runtime),
        "gpu_events": len(gpu),
        "gpu_sum_us": sum(event["dur"] for event in gpu),
        "gpu_union_us": busy,
        "step_cpu_us": marker["dur"],
    }


def run(args: argparse.Namespace) -> dict:  # noqa: C901, PLR0915
    inputs = setup(args.batch)
    torch.backends.cudnn.benchmark = False
    labels = torch.arange(args.batch, device="cuda") % 100
    model = build_model(VARIANTS[0])
    paths = prepare_capture_safe_orthogonal_(model) if args.arm != "native" else ()
    mode = "reduce-overhead" if args.arm == "reduce-overhead" else "default"
    compiled = torch.compile(model, mode=mode)
    optimizer = build_optimizer(
        model,
        {
            "learning_rate": 0.003,
            "weight_decay": 0.05,
            "modal_learning_rate_multiplier": 1.0,
            "pole_geometry_learning_rate_multiplier": 1.0,
            "fused_optimizer": True,
        },
    )
    explicit = args.arm == "graph"
    for group in optimizer.param_groups:
        group["capturable"] = explicit

    def step() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        with (
            torch.profiler.record_function("wave22c.forward"),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            loss = torch.nn.functional.cross_entropy(compiled(inputs)[0], labels)
        with torch.profiler.record_function("wave22c.backward"):
            loss.backward()
        with torch.profiler.record_function("wave22c.clip"):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gradients = tuple(p.grad for p in model.parameters() if p.grad is not None)
        with torch.profiler.record_function("wave22c.adamw"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return loss.detach(), norm, gradients

    def mark_step() -> None:
        if mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_first = raw_time()
    mark_step()
    step()
    torch.cuda.synchronize()
    first_step_s = raw_time() - start_first
    warmup_stream = torch.cuda.Stream() if explicit else torch.cuda.current_stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(args.warmup - 1):
            mark_step()
            step()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    capture_s = None
    if explicit:
        start_capture = raw_time()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=warmup_stream):
            captured = step()
        capture_s = raw_time() - start_capture

        def execute() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
            graph.replay()
            return captured
    else:
        execute = step

    torch.cuda.synchronize()
    setup_alloc = torch.cuda.max_memory_allocated()
    setup_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    losses = torch.empty(args.count, device="cuda")
    norms = torch.empty_like(losses)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for i in range(args.count):
        mark_step()
        with torch.profiler.record_function("wave22c.step"):
            before = raw_time()
            start.record()
            loss, norm, gradients = execute()
            issued = raw_time()
            end.record()
            end.synchronize()
            after = raw_time()
        samples.append([(after - before) * 1000, start.elapsed_time(end), (issued - before) * 1000])
        losses[i].copy_(loss)
        norms[i].copy_(norm)
        if i < args.count - 1:
            del gradients
    result = {
        "arm": args.arm,
        "batch": args.batch,
        "seed": 501,
        "exp_fp32": args.exp_fp32,
        "warmup": args.warmup,
        "first_step_s": first_step_s,
        "capture_s": capture_s,
        "replaced": paths,
        "median_wall_ms": statistics.median(row[0] for row in samples),
        "median_event_ms": statistics.median(row[1] for row in samples),
        "median_cpu_issue_ms": statistics.median(row[2] for row in samples),
        "median_wall_minus_event_ms": statistics.median(row[0] - row[1] for row in samples),
        "peak_alloc_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "setup_peak_alloc_bytes": setup_alloc,
        "setup_peak_reserved_bytes": setup_reserved,
        "samples_wall_event_issue_ms": samples,
        "losses": losses.tolist(),
        "norms": norms.tolist(),
        "finite_losses": bool(losses.isfinite().all()),
        "finite_grads": bool(gradients) and bool(norms.isfinite().all()) and all(
            bool(grad.isfinite().all()) for grad in gradients
        ),
    }
    del gradients
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.coordinates:
        coordinates = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if name.endswith("analysis.parametrizations.weight.original")
        }
        torch.save(coordinates, args.coordinates)
    if args.trace:
        mark_step()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as profile:
            with torch.profiler.record_function("wave22c.step"):
                execute()
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(args.trace))
        result["trace"] = trace_summary(args.trace)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=["native", "safe", "reduce-overhead", "graph"], required=True
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument(
        "--exp-fp32",
        action="store_true",
        help="Disable BF16 autocast inside the Taylor exponential only",
    )
    args = parser.parse_args()
    if args.exp_fp32:
        with patch.object(
            capture_safe_orthogonal, "fixed_taylor_matrix_exp", autocast_disabled_exp
        ):
            run(args)
    else:
        run(args)
