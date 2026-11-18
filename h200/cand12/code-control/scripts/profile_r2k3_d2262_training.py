#!/usr/bin/env python3
"""Profile steady-state compiled BF16 training for the production D2262/P160 model."""

from __future__ import annotations

# Experiment runners intentionally expose private construction hooks.
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any, TypedDict, cast

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_d2262_p160_followup_imagenet100 as runner
import torch
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile


class PreparedStep(TypedDict):
    model: nn.Module
    optimizer: torch.optim.Optimizer
    inputs: Tensor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timing-steps", type=int, default=9)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.warmup_steps,
        args.timing_steps,
        args.profile_steps,
        args.top,
    ) <= 0:
        parser.error("all profile dimensions must be positive")
    return args


def _logits(output: object) -> Tensor:
    if not isinstance(output, tuple) or not output or not isinstance(output[0], Tensor):
        raise TypeError("D2262 model did not return the established classifier tuple")
    return output[0]


def _source_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "scripts/run_a2d_r2k3_d2262_p160_followup_imagenet100.py",
        root / "scripts/run_a2d_r2k3_capacity_insight_overnight_imagenet100.py",
        root / "scripts/run_a2d_r2k3_stage_allocation_screen_imagenet100.py",
        *(sorted((root / "src/lnet").glob("pac_*scan*.py"))),
        root / "src/lnet/pac_factorized_complex_scan_reader.py",
        root / "src/lnet/pac_gated_post_fusion.py",
        root / "src/lnet/complex_scan.py",
    )
    files: dict[str, str] = {}
    bundle = hashlib.sha256()
    for path in dict.fromkeys(path for path in paths if path.is_file()):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = digest
        bundle.update(relative.encode())
        bundle.update(b"\0")
        bundle.update(digest.encode())
        bundle.update(b"\n")
    return {"bundle_sha256": bundle.hexdigest(), "files": files}


def _cuda_processes() -> list[str]:
    command = (
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        return [f"inspection failed: {error}"]
    return completed.stdout.splitlines()


def _prepare(batch_size: int) -> PreparedStep:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, runtime.model_config()).cuda().train()
    model.prepare_for_compiled_training_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    inputs = torch.randn(batch_size, 3, 224, 224, device="cuda")
    compiled = cast(
        "nn.Module",
        torch.compile(model, mode="default", fullgraph=False, dynamic=False),
    )
    return {"model": compiled, "optimizer": optimizer, "inputs": inputs}


def _step(prepared: PreparedStep) -> Tensor:
    prepared["optimizer"].zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = _logits(prepared["model"](prepared["inputs"])).float().square().mean()
    loss.backward()
    prepared["optimizer"].step()
    return loss


def _warmup(prepared: PreparedStep, steps: int) -> None:
    with cast("Any", torch)._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        for _ in range(steps):
            _step(prepared)
    torch.cuda.synchronize()


def _timings(prepared: PreparedStep, steps: int) -> list[float]:
    samples = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _step(prepared)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _device_time(event: object, *, self_time: bool) -> float:
    prefix = "self_" if self_time else ""
    current = getattr(event, f"{prefix}device_time_total", None)
    if current is not None:
        return float(current)
    return float(getattr(event, f"{prefix}cuda_time_total", 0.0))


def _profile_events(
    prepared: PreparedStep,
    *,
    steps: int,
    trace_path: Path,
    top: int,
) -> tuple[list[dict[str, object]], float]:
    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as active:
        for _ in range(steps):
            _step(prepared)
            active.step()
    torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    active.export_chrome_trace(str(trace_path))
    events: list[dict[str, object]] = []
    for event in active.key_averages(group_by_input_shape=False):
        self_device_us = _device_time(event, self_time=True)
        if self_device_us <= 0.0:
            continue
        events.append(
            {
                "name": str(event.key),
                "count": int(event.count),
                "self_device_us": self_device_us,
                "device_us": _device_time(event, self_time=False),
                "self_cpu_us": float(event.self_cpu_time_total),
                "cpu_us": float(event.cpu_time_total),
            }
        )
    events.sort(key=lambda item: cast("float", item["self_device_us"]), reverse=True)
    total_self_device_us = sum(cast("float", item["self_device_us"]) for item in events)
    return events[:top], total_self_device_us


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("D2262 training profile requires CUDA")
    trace_path = args.trace or args.output.with_suffix(".trace.json")
    torch.set_float32_matmul_precision("high")
    torch.compiler.reset()
    prepared = _prepare(args.batch_size)
    _warmup(prepared, args.warmup_steps)
    torch.cuda.reset_peak_memory_stats()
    timing_samples = _timings(prepared, args.timing_steps)
    peak_bytes = int(torch.cuda.max_memory_allocated())
    top_events, total_self_device_us = _profile_events(
        prepared,
        steps=args.profile_steps,
        trace_path=trace_path,
        top=args.top,
    )
    payload: dict[str, object] = {
        "schema": "lnet.r2k3.d2262_training_profile.v1",
        "variant": runner.VARIANT,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "timing_steps": args.timing_steps,
        "profile_steps": args.profile_steps,
        "timing_ms": {
            "samples": timing_samples,
            "median": statistics.median(timing_samples),
            "minimum": min(timing_samples),
            "maximum": max(timing_samples),
        },
        "images_per_second": 1000.0 * args.batch_size / statistics.median(timing_samples),
        "peak_bytes": peak_bytes,
        "top_device_events": top_events,
        "profile_total_self_device_us": total_self_device_us,
        "trace": str(trace_path),
        "source_provenance": _source_provenance(),
        "environment": {
            "hostname": platform.node(),
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "external_cuda_processes": _cuda_processes(),
        },
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
