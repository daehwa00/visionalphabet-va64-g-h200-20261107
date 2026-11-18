"""Strict checkpoint-backed Real-memory v3 inference benchmark.

Run each candidate in a fresh process. Profile only after uninstrumented timing.
The legacy benchmark_va128_real_memory.py remains a synthetic training benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import run_va128_real_memory_imagenet100 as runner
import torch
import triton

MAX_ABS = 0.05
RELATIVE_L2 = 0.005


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float().cpu(), expected.float().cpu()
    shape_matches = actual.shape == expected.shape
    finite = bool(actual.isfinite().all() and expected.isfinite().all())
    result = {"shape": list(actual.shape), "shape_matches": shape_matches, "finite": finite}
    if not shape_matches or not finite:
        result["passed"] = False
        return result
    delta = actual - expected
    result.update(
        max_abs=delta.abs().max().item(),
        relative_l2=(delta.norm() / expected.norm().clamp_min(1e-12)).item(),
        top1_agreement=(actual.argmax(-1) == expected.argmax(-1)).float().mean().item(),
    )
    result["passed"] = result["max_abs"] <= MAX_ABS and result["relative_l2"] <= RELATIVE_L2
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=[
            "eager",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default="reduce-overhead",
    )
    parser.add_argument("--prepare", choices=["none", "bake", "reader"], default="none")
    parser.add_argument("--no-spatial-autotune", action="store_true")
    parser.add_argument("--fuse-spatial-rms", action="store_true")
    parser.add_argument("--fuse-projection", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--images", type=Path, help="Fixed preprocessed NCHW FP32 tensor file")
    parser.add_argument("--reference", type=Path, help="JSON from matching baseline process")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.batch, args.warmup, args.steps, args.repeats) < 1:
        raise ValueError("batch/warmup/steps/repeats must be positive")
    if args.prepare != "reader" and (args.fuse_spatial_rms or args.fuse_projection):
        raise ValueError("Fusion candidates require --prepare reader")
    torch.set_num_threads(4)
    torch.manual_seed(809)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint.get("variant"), checkpoint.get("epoch"), checkpoint.get("seed")) != (
        runner.VARIANT,
        100,
        501,
    ):
        raise ValueError("Expected the requested v3 IN100 seed501 epoch100 checkpoint; no fallback")
    model = runner._build(runner.VARIANT, runner.runtime.model_config()).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.classifier.affine.linear.out_features != 100:
        raise ValueError("Expected the original 100-class head")
    learned_parameters = sum(p.numel() for p in model.parameters())
    model = model.cuda().to(memory_format=torch.channels_last)
    inputs = torch.randn(args.batch, 3, 224, 224, device="cuda").contiguous(
        memory_format=torch.channels_last
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(inputs.cpu().contiguous().numpy().tobytes()).hexdigest(),
        "epoch": checkpoint["epoch"],
        "seed": checkpoint["seed"],
        "global_step": checkpoint["global_step"],
        "variant": checkpoint["variant"],
        "parameters": learned_parameters,
        "batch": args.batch,
        "prepare": args.prepare,
        "mode": args.mode,
        "spatial_autotune": not args.no_spatial_autotune,
        "fuse_spatial_rms": args.fuse_spatial_rms,
        "fuse_projection": args.fuse_projection,
        "freezing": os.environ.get("TORCHINDUCTOR_FREEZING", "0") == "1",
        "precision": "bf16 autocast",
        "tf32": False,
        "memory_format": "channels_last",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "warmup": args.warmup,
        "steps_per_group": args.steps,
        "repeats": args.repeats,
        "tolerances": {"max_abs": MAX_ABS, "relative_l2": RELATIVE_L2},
        "timing_excludes": "preparation, compilation, autotune, CUDA Graph capture, profiler, image validation",
        "memory_scope": "steady peak active allocations reset after warmup; setup peak includes reference validation, copy preparation, compile/autotune/capture; reserved includes CUDA Graph pool reservations",
    }
    if args.reference:
        baseline = json.loads(args.reference.read_text())
        for key in [
            "checkpoint_sha256",
            "input_sha256",
            "batch",
            "precision",
            "tf32",
            "memory_format",
        ]:
            if baseline[key] != result[key]:
                raise ValueError(f"Baseline mismatch for {key}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager_reference = model(inputs)[0].clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.prepare == "bake":
            model = copy.deepcopy(model)
            with torch.autocast("cuda", enabled=False):
                model.backbone.prepare_for_inference_()
        elif args.prepare == "reader":
            from lnet.real_memory_inference import prepare_real_memory_inference

            model = prepare_real_memory_inference(
                model,
                autotune_spatial=not args.no_spatial_autotune,
                fuse_spatial_rms=args.fuse_spatial_rms,
                fuse_projection=args.fuse_projection,
            )
        torch.cuda.synchronize()
        result["copy_preparation_seconds"] = time.perf_counter() - start
        forward = (
            torch.compile(model, mode=args.mode, dynamic=False) if args.mode != "eager" else model
        )
        start = time.perf_counter()
        for _ in range(args.warmup):
            torch.compiler.cudagraph_mark_step_begin()
            output = forward(inputs)[0]
        torch.cuda.synchronize()
        result["compile_autotune_capture_warmup_seconds"] = time.perf_counter() - start
        result["eager_check"] = error_metrics(output, eager_reference)
        if args.reference:
            expected = torch.load(
                args.reference.with_suffix(".output.pt"), map_location="cpu", weights_only=True
            )
            result["baseline_check"] = error_metrics(output, expected)
        save()
        if (
            not result["eager_check"]["passed"]
            or not result.get("baseline_check", {"passed": True})["passed"]
        ):
            raise RuntimeError("Numerical criterion failed; candidate not timed")
        result["setup_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
        wall, stream, submit, tail = [], [], [], []
        for _ in range(args.repeats):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            begin.record()
            for _ in range(args.steps):
                torch.compiler.cudagraph_mark_step_begin()
                output = forward(inputs)[0]
            end.record()
            submitted = time.perf_counter()
            torch.cuda.synchronize()
            finished = time.perf_counter()
            wall.append((finished - start) * 1000 / args.steps)
            stream.append(begin.elapsed_time(end) / args.steps)
            submit.append((submitted - start) * 1000 / args.steps)
            tail.append((finished - submitted) * 1000 / args.steps)
        result.update(
            wall_ms=wall,
            event_ms=stream,
            host_submit_ms=submit,
            sync_tail_ms=tail,
            median_ms=statistics.median(wall),
            images_s=args.batch * 1000 / statistics.median(wall),
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            reserved_mib=torch.cuda.memory_reserved() / 2**20,
            host_timing_note="submit includes Python/dispatch and any in-forward synchronization; event interval also includes GPU idle gaps, not kernel-only time",
        )
        serial = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            torch.compiler.cudagraph_mark_step_begin()
            output = forward(inputs)[0]
            torch.cuda.synchronize()
            serial.append((time.perf_counter() - start) * 1000)
        result["synchronized_single_call_ms"] = serial
        result["serial_median_ms"] = statistics.median(serial)
        torch.save(output.clone().cpu(), args.output.with_suffix(".output.pt"))
        save()
        if args.images:
            images = torch.load(args.images, map_location="cpu", weights_only=True)
            if (
                images.ndim != 4
                or images.shape[0] % args.batch
                or images.shape[1:] != (3, 224, 224)
            ):
                raise ValueError("Expected fixed preprocessed image tensor divisible by batch")
            result["images_sha256"] = hashlib.sha256(args.images.read_bytes()).hexdigest()
            outputs = []
            for chunk in images.split(args.batch):
                # Reuse identical input storage to preserve the captured execution path.
                inputs.copy_(chunk)
                torch.compiler.cudagraph_mark_step_begin()
                outputs.append(forward(inputs)[0].clone().cpu())
            actual = torch.cat(outputs)
            torch.save(actual, args.output.with_suffix(".images.pt"))
            result["image_count"] = len(images)
            if args.reference:
                if baseline.get("images_sha256") != result["images_sha256"]:
                    raise ValueError("Baseline image tensor mismatch")
                expected = torch.load(
                    args.reference.with_suffix(".images.pt"), map_location="cpu", weights_only=True
                )
                result["image_check"] = error_metrics(actual, expected)
                save()
                if not result["image_check"]["passed"]:
                    raise RuntimeError("Fixed-image numerical criterion failed")
            save()
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profiler:
                for _ in range(5):
                    torch.compiler.cudagraph_mark_step_begin()
                    forward(inputs)
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
            args.output.with_suffix(".profile.txt").write_text(
                profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
            )
    result["status"] = "passed"
    save()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
