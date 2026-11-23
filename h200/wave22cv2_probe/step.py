"""Compare native, capture-safe, and CUDA Graph training in fresh processes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liverms_v2_opt"))
from bench_h200 import VARIANTS, build_model, setup
from va_model_builder import build_optimizer

from lnet.capture_safe_orthogonal import CaptureSafeMatrixExpOrthogonal


class FP64MatrixExpOrthogonal(CaptureSafeMatrixExpOrthogonal):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        lower = coordinates.tril()
        skew = lower - lower.mT
        return self.base @ torch.matrix_exp(skew.double()).to(skew.dtype)


def raw_time() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def trace_summary(path: Path) -> dict:
    events = json.loads(path.read_text())["traceEvents"]
    (marker,) = [
        event
        for event in events
        if event.get("name") == "wave22cv2.step" and event.get("cat") == "user_annotation"
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
    torch.backends.cudnn.benchmark = args.benchmark
    torch.set_float32_matmul_precision("highest")
    labels = torch.arange(args.batch, device="cuda") % 100
    os.environ["LNET_CAPTURE_SAFE_ORTHOGONAL"] = "off" if args.arm == "native" else "on"
    model = build_model(args.variant)
    paths = [name for name, mod in model.named_modules() if isinstance(
        mod, CaptureSafeMatrixExpOrthogonal
    )]
    if args.arm == "fp64":
        (path,) = paths
        parent_path, name = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        original = getattr(parent, name)
        if original.shape[0] != original.shape[1]:
            raise ValueError("FP64 reference requires a square analysis map")
        setattr(parent, name, FP64MatrixExpOrthogonal(original.shape, original.base))
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
    def step() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        with (
            torch.profiler.record_function("wave22cv2.forward"),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            loss = torch.nn.functional.cross_entropy(compiled(inputs)[0], labels)
        with torch.profiler.record_function("wave22cv2.backward"):
            loss.backward()
        with torch.profiler.record_function("wave22cv2.clip"):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gradients = tuple(p.grad for p in model.parameters() if p.grad is not None)
        with torch.profiler.record_function("wave22cv2.adamw"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return loss.detach(), norm, gradients

    def mark_step() -> None:
        if mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_first = raw_time()
    warmup = []
    for _ in range(args.warmup):
        mark_step()
        loss, norm, gradients = step()
        warmup.append((loss.clone(), norm.clone()))
        del gradients
        if len(warmup) == 1:
            torch.cuda.synchronize()
            first_step_s = raw_time() - start_first
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
        with torch.profiler.record_function("wave22cv2.step"):
            before = raw_time()
            start.record()
            loss, norm, gradients = step()
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
        "variant": args.variant,
        "benchmark": args.benchmark,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "warmup_loss_norm": [[loss.item(), norm.item()] for loss, norm in warmup],
        "warmup": args.warmup,
        "first_step_s": first_step_s,
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
            with torch.profiler.record_function("wave22cv2.step"):
                step()
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(args.trace))
        result["trace"] = trace_summary(args.trace)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=["native", "safe", "reduce-overhead", "fp64"], required=True
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANTS[0])
    parser.add_argument("--benchmark", action="store_true")
    run(parser.parse_args())
