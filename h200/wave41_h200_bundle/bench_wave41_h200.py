"""Measure a single Wave 41 BF16 training arm."""

import argparse
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from h200.liverms_v2_opt import bench_h200

MODES = ("off", "strict", "strict_ro")
BASE_PRESET = "cuda_graphs,tma_gemm,max_autotune_pointwise"
VARIANT = "17-p1-nr4-1024d-v3"
BUCKETS = (
    "general_vscan", "factorized_reader", "cublas_gemm", "general_hscan",
    "triton", "cudnn_conv", "fused_stem", "other",
)


def profile_step(step: Callable) -> dict:
    import torch  # noqa: PLC0415

    step()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        started = perf_counter()
        step()
        torch.cuda.synchronize()
        wall_ms = (perf_counter() - started) * 1000

    buckets = dict.fromkeys(BUCKETS, 0.0)
    kernels = []
    for event in prof.key_averages():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        bucket = bench_h200.kernel_bucket(event.key)
        elapsed = event.self_device_time_total / 1000
        buckets[bucket if bucket in buckets else "other"] += elapsed
        kernels.append({"name": event.key, "count": event.count, "self_cuda_ms": elapsed})
    if not kernels:
        raise RuntimeError("CUDA profiler returned no device events")
    cuda_ms = sum(buckets.values())
    buckets["wall - CUDA"] = wall_ms - cuda_ms
    return {
        "wall_ms": wall_ms, "cuda_ms": cuda_ms, "self_cuda_ms_by_bucket": buckets,
        "top_kernels": sorted(kernels, key=lambda row: -row["self_cuda_ms"])[:30],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-sgs", choices=MODES, required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(LNET_SCAN_SGS=args.scan_sgs, LNET_MODEL_W36B=args.preset)
    args.variant, args.batch = VARIANT, 1024
    started = perf_counter()
    profile_s = None

    def profiler(step: Callable, _eager_diagnostic: Callable) -> dict:
        nonlocal profile_s
        profile_started = perf_counter()
        try:
            return profile_step(step)
        finally:
            profile_s = perf_counter() - profile_started

    try:
        metrics = bench_h200.train_arm(args, profiler=profiler)
        replays = metrics.get("cudagraph_replays_by_phase", {})
        row = {
            "status": "ok",
            "compile_warmup_s": metrics["compile_warmup_s"],
            "timed_loop_s": metrics["timed_loop_s"],
            "step_ms": {key: metrics[f"{key}_ms"] for key in ("median", "mean", "p90")},
            "img_per_s": metrics["images_per_second"],
            "peak_vram_gib": metrics["peak_allocated_gib"],
            "graph_replays": {
                "fwd": replays.get("forward", 0), "bwd": replays.get("backward", 0),
                "upd": replays.get("update", 0),
            },
            "loss_finite": metrics["finite_losses"],
            "finite_grad_count": metrics["finite_grad_count"],
            "grad_tensor_count": metrics["grad_tensor_count"],
            "profile_s": profile_s,
            **{key: metrics[key] for key in ("profiler", "profiler_error") if key in metrics},
        }
    except Exception:  # noqa: BLE001 - report child errors in JSON before exiting
        row = {"status": "error", "error": traceback.format_exc()}
    bench_h200.write_json(args.out, {
        **row, "total_arm_wall_s": perf_counter() - started,
        "scan_sgs_mode": args.scan_sgs, "preset": args.preset,
    })
    return int(row["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
