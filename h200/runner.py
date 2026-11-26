"""Run isolated synthetic ImageNet-100 benchmarks on the provisioned GPU."""
# ruff: noqa: PLC0415, SLF001

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

GIB = 1024**3


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def setup(batch: int) -> torch.Tensor:
    import torch

    torch.manual_seed(501)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = os.getenv("LNET_CUDNN_BENCHMARK", "1") != "0"
    return torch.randn(batch, 3, 224, 224, device="cuda").contiguous(
        memory_format=torch.channels_last
    )


def build_model(variant: str) -> torch.nn.Module:
    import torch
    from run_va_eight_ablations import build_variant

    from lnet.model_w36b import prepare_model_w36b_

    model = build_variant(variant).to(device="cuda", memory_format=torch.channels_last)
    return prepare_model_w36b_(model)


def measure(step: Callable, batch: int, *, warmup: int, count: int) -> tuple[dict, list]:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    step()
    torch.cuda.synchronize()
    first_step_s = time.perf_counter() - start
    for _ in range(warmup - 1):
        step()
    torch.cuda.synchronize()
    compile_warmup_s = time.perf_counter() - start
    loop_start = time.perf_counter()
    timings, outputs = [], []
    for _ in range(count):
        start = time.perf_counter()
        outputs.append(step())
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    timed_loop_s = time.perf_counter() - loop_start
    median = statistics.median(timings)
    p25, _, p75 = statistics.quantiles(timings, n=4, method="inclusive")
    return {
        "median_ms": median,
        "mean_ms": statistics.mean(timings),
        "p90_ms": statistics.quantiles(timings, n=10, method="inclusive")[-1],
        "p95_ms": statistics.quantiles(timings, n=20, method="inclusive")[-1],
        "compile_time_s": first_step_s,
        "compile_warmup_s": compile_warmup_s,
        "timed_loop_s": timed_loop_s,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "p25_ms": p25,
        "p75_ms": p75,
        "images_per_second": batch * 1000 / median,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }, outputs


def kernel_bucket(name: str) -> str:
    scan_kernels = {
        "scan": "general_scan_shared",
        "general_vscan": "general_vscan",
        "general_hscan": "general_hscan",
        "vscan_sgs": "general_vscan",
        "reduce_hscan_partials": "general_hscan",
        "reduce_vscan_partials": "general_vscan",
    }
    if name in scan_kernels:
        return scan_kernels[name]
    for prefix, bucket in (
        ("legacy_vscan", "general_vscan"),
        ("legacy_hscan", "general_hscan"),
        ("vscan_bwd_cta_", "general_vscan"),
        ("reader_", "factorized_reader"),
        ("stem_", "fused_stem"),
        ("triton", "triton"),
        ("_terminal_vertical_scan", "triton"),
        ("_nr4_correlation", "triton"),
    ):
        if name.startswith(prefix):
            return bucket
    for pattern, bucket in (
        (r"cudnn|convol(?:ve|ution)|conv[123]d|(?:^|_)(?:fprop|dgrad|wgrad)(?:_|\b)", "cudnn_conv"),
        (r"nvjet_|gemm|cublas|cutlass|xmma", "cublas_gemm"),
    ):
        if re.search(pattern, name, re.IGNORECASE):
            return bucket
    return "other"


def train_arm(  # noqa: C901, PLR0915 - keep the timing and diagnostic phases together
    args: argparse.Namespace, *, profiler: Callable[[Callable, Callable], dict],
) -> dict:
    import torch
    import va_model_builder as builder

    from lnet.model_w36b import (
        AUTOTUNE_FEATURES,
        compile_model_w36b,
        compile_model_w36b_update,
        count_cudagraph_replays,
        mark_model_w36b_step_begin,
        model_w36b_features,
    )

    inputs = setup(args.batch)
    labels = torch.arange(args.batch, device="cuda") % 100
    model = build_model(args.variant)
    features = model_w36b_features()
    if "cuda_graphs" in features:
        # The synthetic batch stays at this address for every timed iteration.
        torch._dynamo.mark_static_address(inputs)
    compiled = (
        compile_model_w36b(model)
        if features & (AUTOTUNE_FEATURES | {"cuda_graphs"})
        else torch.compile(model, mode="default")
    )
    optimizer = builder.build_optimizer(
        model,
        {
            "learning_rate": 3e-3,
            "weight_decay": 0.05,
            "modal_learning_rate_multiplier": 1.0,
            "pole_geometry_learning_rate_multiplier": 1.0,
            "fused_optimizer": True,
        },
    )

    def forward() -> torch.Tensor:
        mark_model_w36b_step_begin()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.nn.functional.cross_entropy(compiled(inputs)[0], labels)

    def backward() -> torch.Tensor:
        loss = forward()
        loss.backward()
        return loss.detach()

    def eager_update() -> None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    compiled_update = compile_model_w36b_update(model, optimizer)
    update = compiled_update or eager_update

    def step() -> torch.Tensor:
        loss = backward()
        update()
        return loss

    result, losses = measure(step, args.batch, warmup=3, count=8)
    rms = model.classifier.affine.standardizer
    loss_values = torch.stack(losses)
    result.update(
        finite_losses=bool(loss_values.isfinite().all()),
        losses={"first": float(loss_values[0]), "last": float(loss_values[-1])},
        ema_weight_after=float(rms.ema_weight),
        running_ms_finite=bool(rms.running_ms.isfinite().all()),
    )
    if "cuda_graphs" in features:
        with count_cudagraph_replays() as forward_replays:
            diagnostic_loss = forward()
        with count_cudagraph_replays() as backward_replays:
            diagnostic_loss.backward()
        result["cudagraph_replays_per_forward_backward"] = (
            len(forward_replays) + len(backward_replays)
        )
        result["cudagraph_replays_by_phase"] = {
            "forward": len(forward_replays), "backward": len(backward_replays),
        }
        del diagnostic_loss
    else:
        backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite_grad_count = sum(bool(grad.isfinite().all()) for grad in gradients)
    terminal = getattr(model.backbone, "terminal_s3", None)
    result.update(
        finite_gradients=finite_grad_count == len(gradients) and bool(gradients),
        finite_grad_count=finite_grad_count,
        grad_tensor_count=len(gradients),
        terminal_s3_nonzero_grads=(
            sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in terminal.parameters())
            if terminal is not None
            else None
        ),
    )
    if compiled_update is None:
        optimizer.zero_grad(set_to_none=True)
    else:
        with count_cudagraph_replays() as update_replays:
            compiled_update()
        result["cudagraph_replays_per_update"] = len(update_replays)
        result.setdefault("cudagraph_replays_by_phase", {})["update"] = len(update_replays)

    def eager_scan_diagnostic() -> None:
        # A replay bypasses Python launch hooks. Observe the same scan policy
        # through eager dispatch without changing the timed graph or profiler.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(inputs)[0], labels)
        try:
            loss.backward()
        finally:
            optimizer.zero_grad(set_to_none=True)

    if args.profile:
        write_json(args.out, {**result, "status": "profiling"})
        try:
            result["profiler"] = profiler(step, eager_scan_diagnostic)
        except Exception:  # noqa: BLE001 - preserve timing after a profiler failure
            result["profiler_error"] = traceback.format_exc()
    return result
