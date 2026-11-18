"""Matched qlab GPU1 A/RawG training benchmark; never mutates the live run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import run_va_a_imcorr as target_runner
import torch
import triton
from benchmark_va128_real_memory_inference import error_metrics

runner = target_runner.baseline


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--mode", default="default")
    p.add_argument("--spatial", action="store_true")
    p.add_argument("--reference", type=Path)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--vector", action="store_true")
    p.add_argument("--reset-each-step", action="store_true", help="Restore checkpoint/optimizer outside each timed step to hold the workload fixed")
    p.add_argument("--dense-reader", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic PyTorch/cuDNN algorithms for both comparison sides")
    a = p.parse_args()
    if min(a.batch, a.steps, a.repeats, a.warmup, a.cpu_threads) < 1:
        raise ValueError("Batch and measurement counts must be positive")
    if a.vector and not a.spatial:
        raise ValueError("--vector requires --spatial")
    if a.dense_reader and a.spatial:
        raise ValueError("Dense and spatial candidates are exclusive")
    if a.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(mode=True)
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(a.cpu_threads)
    torch.manual_seed(809)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = not a.deterministic
    c = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    contract = json.loads(a.contract.read_text())
    if c["variant"] != target_runner.VARIANT or c["seed"] not in contract["seeds"]:
        raise ValueError("Checkpoint does not match the live qlab GPU1 model")
    if not 0 < c["epoch"] <= contract["recipe"]["epochs"]:
        raise ValueError("Invalid completed checkpoint epoch")
    recipe = contract["recipe"]
    torch.manual_seed(c["seed"])
    model = target_runner._build(target_runner.VARIANT, runner.runtime.model_config()).cuda().train()
    if sum(v.numel() for v in model.parameters()) != 2_988_004:
        raise ValueError("Target parameter count changed")
    if model.classifier.affine.linear.in_features != 1024:
        raise ValueError("Expected the actual 1024-dimensional raw correlation head")
    model.load_state_dict(c["model"], strict=True)
    if model.classifier.affine.linear.out_features != 100:
        raise ValueError("Expected the original 100-class head")
    model = runner.prepare._prepare_model(model, recipe)
    optimizer = runner.runtime.build_optimizer(model, recipe)
    optimizer.load_state_dict(c["optimizer"])

    # Keep the actual checkpoint LR for timing. The separately labelled update
    # check uses recorded initial LRs so epoch-98 near-zero updates do not hide
    # errors. Only this isolated benchmark model is changed.
    def restore_optimizer(*, validation: bool = False) -> None:
        optimizer.load_state_dict(c["optimizer"])
        if validation:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"]

    restore_optimizer(validation=True)
    if a.dense_reader:
        from lnet.real_memory_backbone import RealFactorizedMemoryReader

        for reader in model.modules():
            if isinstance(reader, RealFactorizedMemoryReader):
                reader.execution_mode = "dense"
    identities = [id(v) for v in model.parameters()]
    optimization_info = {}
    if a.spatial:
        from lnet.real_memory_training import enable_real_memory_training_

        optimization_info = enable_real_memory_training_(
            model, spatial=a.spatial, vector_weight_gradient=a.vector
        )
    if identities != [id(v) for v in model.parameters()]:
        raise RuntimeError("Training preparation replaced optimizer parameters")
    images = torch.load(a.images, map_location="cpu", weights_only=True)
    if images.ndim != 4 or images.shape[1:] != (3, 224, 224) or not 0 < len(images) <= 100:
        raise ValueError("Expected preprocessed images ordered one per class, at most 100 classes")
    indices = torch.arange(a.batch) % len(images)
    raw = images[indices].cuda().contiguous(memory_format=torch.channels_last)
    inputs = (0.4 * raw + 0.6 * raw.flip(0)).contiguous(memory_format=torch.channels_last)
    targets = indices.cuda()
    other = targets.flip(0)
    result = {
        "checkpoint_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        "epoch": c["epoch"],
        "global_step": c["global_step"],
        "variant": c["variant"],
        "batch": a.batch,
        "deterministic": a.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "scan_autotune_disabled": os.environ.get("LNET_DISABLE_LAUNCH_AUTOTUNE") == "1",
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch_num_threads": torch.get_num_threads(),
        "mode": a.mode,
        "runtime_builder": "live training harness._build_runtime",
        "spatial": a.spatial,
        "precision": "bf16 autocast",
        "tf32": True,
        "memory_format": "channels_last",
        "input_sha256": hashlib.sha256(inputs.cpu().contiguous().numpy().tobytes()).hexdigest(),
        "parameters": sum(v.numel() for v in model.parameters()),
        "optimizer": "fused AdamW, saved moments and checkpoint LRs during timing",
        "learning_rates": [g["lr"] for g in c["optimizer"]["param_groups"]],
        "validation_learning_rates": [g["lr"] for g in optimizer.param_groups],
        "update_validation_scope": "initial-LR stress update; actual checkpoint LR restored before timing",
        "head": "raw complex correlation 1024 + BatchNorm + Linear100",
        "contract_emulate_precision_casts": recipe.get("emulate_precision_casts"),
        "parameter_names_sha256": hashlib.sha256("\n".join(n for n, _ in model.named_parameters()).encode()).hexdigest(),
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "scope": "fixed real-image MixUp forward + original objective + backward + clip_grad_norm(1.0) + fused AdamW; excludes data loading, augmentation, validation, compilation and profiling",
        "reset_each_step": a.reset_each_step,
        "warmup": a.warmup,
        "steps_per_group": a.steps,
        "repeats": a.repeats,
        "gradient_tolerances": {"relative_l2": 0.02, "max_abs_scale": 0.03, "max_abs_floor": 1e-6},
        "update_tolerances": {"max_abs": 3e-5, "relative_l2": 0.02},
    }
    from torch._inductor import config as inductor_config

    result.update(
        dense_reader=a.dense_reader,
        emulate_precision_casts=inductor_config.emulate_precision_casts,
        gradient_clip_norm=1.0,
        vector=a.vector,
        optimization_info=optimization_info,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        a.output.write_text(json.dumps(result, indent=2) + "\n")

    result["status"] = "initialized"
    save()
    if a.reference:
        baseline = json.loads(a.reference.read_text())
        if baseline.get("scan_autotune_disabled", False) != result["scan_autotune_disabled"]:
            raise ValueError("Baseline scan autotune policy mismatch")
        if baseline.get("reset_each_step", False) != a.reset_each_step:
            raise ValueError("Baseline step reset policy mismatch")
        if baseline.get("deterministic", False) != a.deterministic:
            raise ValueError("Baseline deterministic policy mismatch")
        for key in (
            "checkpoint_sha256",
            "input_sha256",
            "batch",
            "precision",
            "tf32",
            "memory_format",
            "learning_rates",
            "validation_learning_rates",
            "parameter_names_sha256",
            "emulate_precision_casts",
        ):
            if baseline[key] != result[key]:
                raise ValueError(f"Baseline mismatch: {key}")
    if a.mode == "eager":
        runtime_model = model
    else:
        os.environ["LNET_COMPILE_MODE"] = a.mode
        runtime_model = runner.harness._build_runtime(model, recipe)

    def gradients() -> tuple[torch.Tensor, torch.Tensor]:
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = runtime_model(inputs)
            logits, loss, _ = runner.heads._training_objective(model, outputs, targets, other, 0.4)
        loss.backward()
        return logits, loss

    torch.manual_seed(809)
    start = time.perf_counter()
    logits, loss = gradients()
    torch.cuda.synchronize()
    result["first_compile_backward_seconds"] = time.perf_counter() - start
    checks = {
        "logits": logits.detach().float().cpu(),
        "grads": {
            n: v.grad.detach().cpu().clone()
            for n, v in model.named_parameters()
            if v.grad is not None
        },
    }
    result["missing_gradients"] = [
        n for n, v in model.named_parameters() if v.requires_grad and v.grad is None
    ]
    before = {n: v.detach().cpu().clone() for n, v in model.named_parameters()}
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    checks["updates"] = {n: v.detach().cpu() - before[n] for n, v in model.named_parameters()}
    torch.save(checks, a.output.with_suffix(".validation.pt"))
    result["first_loss"] = float(loss.detach())
    result["validation_finite"] = all(
        bool(t.isfinite().all()) for group in ("grads", "updates") for t in checks[group].values()
    ) and bool(checks["logits"].isfinite().all())
    if a.reference:
        ref = torch.load(
            a.reference.with_suffix(".validation.pt"), map_location="cpu", weights_only=True
        )
        result["output_check"] = error_metrics(checks["logits"], ref["logits"])
        details = {}
        for group in ("grads", "updates"):
            details[group] = {}
            if checks[group].keys() != ref[group].keys():
                raise ValueError("Gradient/parameter key mismatch")
            for name, value in checks[group].items():
                target = ref[group][name].float()
                delta = value.float() - target
                absolute = delta.abs().max().item()
                relative = (delta.norm() / target.norm().clamp_min(1e-12)).item()
                limit = 0.03 * target.abs().max().item() + 1e-6 if group == "grads" else 3e-5
                details[group][name] = {
                    "max_abs": absolute,
                    "relative_l2": relative,
                    "max_abs_limit": limit,
                    "passed": absolute <= limit and relative <= 0.02,
                }
        result["gradient_checks"] = details
        save()
        if not result["output_check"]["passed"] or not all(
            v["passed"] for group in details.values() for v in group.values()
        ):
            result["status"] = "failed_numerical_gate"
            save()
            raise RuntimeError("Training numerical gate failed; candidate not timed")
    if result["missing_gradients"] or not result["validation_finite"]:
        save()
        raise RuntimeError("Missing/nonfinite gradients")
    # Restore the identical checkpoint before warmup/timing. Only local tensors change.
    model.load_state_dict(c["model"], strict=True)
    restore_optimizer()
    torch.manual_seed(809)
    del checks, before, logits, loss

    def step() -> torch.Tensor:
        _, current = gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return current

    def restore_step_state() -> None:
        model.load_state_dict(c["model"], strict=True)
        restore_optimizer()
        torch.manual_seed(809)

    if a.reset_each_step:
        result["scope"] += "; checkpoint/optimizer/RNG restored outside each timed step; serialized step latency"
    start = time.perf_counter()
    for _ in range(a.warmup):
        if a.reset_each_step:
            restore_step_state()
        loss = step()
    torch.cuda.synchronize()
    result["warmup_seconds"] = time.perf_counter() - start
    result["setup_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    samples = []
    events = []
    steady_peaks = []
    for _ in range(a.repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if a.reset_each_step:
            group_wall, group_events = [], []
            for _ in range(a.steps):
                restore_step_state()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                begin.record()
                loss = step()
                end.record()
                torch.cuda.synchronize()
                group_wall.append((time.perf_counter() - start) * 1000)
                group_events.append(begin.elapsed_time(end))
                steady_peaks.append(torch.cuda.max_memory_allocated() / 2**20)
            samples.append(statistics.mean(group_wall))
            events.append(statistics.mean(group_events))
        else:
            torch.cuda.synchronize()
            start = time.perf_counter()
            begin.record()
            for _ in range(a.steps):
                loss = step()
            end.record()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000 / a.steps)
            events.append(begin.elapsed_time(end) / a.steps)
    result.update(
        wall_ms=samples,
        event_ms=events,
        median_ms=statistics.median(samples),
        images_s=a.batch * 1000 / statistics.median(samples),
        peak_allocated_mib=max(steady_peaks) if steady_peaks else torch.cuda.max_memory_allocated() / 2**20,
        reserved_mib=torch.cuda.memory_reserved() / 2**20,
        final_loss=float(loss.detach()),
    )
    result["final_finite"] = bool(loss.isfinite()) and all(
        bool(v.isfinite().all()) and (v.grad is None or bool(v.grad.isfinite().all()))
        for v in model.parameters()
    )
    save()
    if not result["final_finite"]:
        raise RuntimeError("Training became nonfinite")
    if a.profile:
        if a.reset_each_step:
            restore_step_state()
            torch.cuda.synchronize()
        result["profile_steps"] = 1 if a.reset_each_step else 3
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(result["profile_steps"]):
                step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(a.output.with_suffix(".trace.json")))
        a.output.with_suffix(".profile.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
        )
    result["status"] = "passed"
    save()
    print(json.dumps({k: v for k, v in result.items() if k != "gradient_checks"}), flush=True)


if __name__ == "__main__":
    main()
