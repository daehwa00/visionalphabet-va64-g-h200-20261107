"""Isolated training-step ablations and Kineto trace capture."""

# ruff: noqa: PLC0415
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liverms_v2_opt"))
from bench_h200 import VARIANTS, build_model, setup
from va_model_builder import build_optimizer


def run(args: argparse.Namespace) -> dict:  # noqa: C901, PLR0912, PLR0915
    inputs = setup(args.batch)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    labels = torch.arange(args.batch, device="cuda") % 100
    model = build_model(VARIANTS[0])
    options = {}
    if args.arm == "tuned":
        options.update(
            epilogue_fusion=True,
            coordinate_descent_tuning=True,
            max_autotune_gemm=True,
            max_autotune=True,
            **{"triton.cudagraphs": False},
        )
    if args.trace:
        from torch._inductor import config

        config.trace.enabled = True
        config.trace.debug_dir = str(args.trace.resolve().parent / "inductor")
    if args.arm == "reduce-overhead":
        compiled = torch.compile(model, mode="reduce-overhead")
    elif options:
        compiled = torch.compile(model, options=options)
    else:
        compiled = torch.compile(model, mode="default")
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
    sizes = [len(group["params"]) for group in optimizer.param_groups]
    if args.arm == "merge":
        from torch.optim import AdamW

        groups = {}
        for group in optimizer.param_groups:
            key = group["lr"], group["weight_decay"]
            groups.setdefault(key, []).extend(group["params"])
        optimizer = AdamW(
            [
                {"params": params, "lr": lr, "weight_decay": wd}
                for (lr, wd), params in groups.items()
            ],
            fused=True,
        )
    if args.arm == "graph":
        for group in optimizer.param_groups:
            group["capturable"] = True

    def backward() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(compiled(inputs)[0], labels)
        loss.backward()
        return loss.detach()

    def step() -> torch.Tensor:
        with torch.profiler.record_function("wave22a.forward_backward"):
            loss = backward()
        with torch.profiler.record_function("wave22a.clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with torch.profiler.record_function("wave22a.adamw"):
            optimizer.step()
        with torch.profiler.record_function("wave22a.zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        return loss

    torch.cuda.synchronize()
    begin = time.perf_counter()
    step()
    torch.cuda.synchronize()
    first_step_s = time.perf_counter() - begin
    if args.arm == "graph":
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(warmup_stream)
    else:
        for _ in range(3):
            step()
    torch.cuda.synchronize()
    capture_s = None
    if args.arm == "graph":
        capture_begin = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_loss = step()
        capture_s = time.perf_counter() - capture_begin

        def timed_step() -> torch.Tensor:
            graph.replay()
            return captured_loss.clone()
    else:
        timed_step = step

    torch.cuda.reset_peak_memory_stats()
    timings, losses = [], []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(args.count):
        begin = time.perf_counter()
        start.record()
        losses.append(timed_step())
        end.record()
        end.synchronize()
        timings.append(((time.perf_counter() - begin) * 1000, start.elapsed_time(end)))
    result = {
        "arm": args.arm,
        "batch": args.batch,
        "cudnn_benchmark": args.cudnn_benchmark,
        "first_step_s": first_step_s,
        "capture_s": capture_s,
        "samples_wall_event_ms": timings,
        "median_wall_ms": statistics.median(wall for wall, _ in timings),
        "median_event_ms": statistics.median(event for _, event in timings),
        "median_gap_ms": statistics.median(wall - event for wall, event in timings),
        "peak_alloc_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "original_param_group_sizes": sizes,
        "active_param_group_sizes": [len(group["params"]) for group in optimizer.param_groups],
        "losses": torch.stack(losses).tolist(),
    }
    if args.arm == "graph":
        del timed_step, graph, captured_loss
    backward()
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    result["finite_grads"] = bool(gradients) and bool(
        torch.stack([grad.isfinite().all() for grad in gradients]).all()
    )
    result["finite_losses"] = bool(torch.stack(losses).isfinite().all())
    optimizer.zero_grad(set_to_none=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.trace:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
        ) as profile:
            step()
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(args.trace))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=["default", "merge", "tuned", "reduce-overhead", "graph"],
        default="default",
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace", type=Path)
    run(parser.parse_args())
