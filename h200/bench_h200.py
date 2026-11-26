"""Measure one H200 reference or lock-in arm."""

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from h200 import runner

VARIANT = "17-p1-nr4-1024d-v3"
REFERENCE_ENV = {
    "LNET_SCAN_SGS": "strict_ro",
    "LNET_SCAN_VFWD": "batch4_w2",
    "LNET_SCAN_HSCAN": "bwd_opt",
    "LNET_SCAN_VSCAN_SAVED": "off",
    "LNET_STEM_STAGE0_MODE": "off",
    "VA_READER_OPT_MODE": "off",
    "LNET_MODEL_W36B": "cuda_graphs,tma_gemm,max_autotune_pointwise",
    "LNET_READER_DENSE_NHWC": "off",
    "LNET_READER_EXEC_MODE": "off",
    "LNET_NORM_W50_MODE": "off",
    "LNET_CUDNN_BENCHMARK": "0",
    "LNET_SCAN_W51A": "off",
    "LNET_SCAN_W51B": "off",
    "LNET_SCAN_W52A": "off",
    "LNET_SCAN_W53A": "off",
    "LNET_FUSED_STEM_CONV": "auto"
}
LOCKIN_ENV = {
    **REFERENCE_ENV,
    "LNET_SCAN_HSCAN": "h55_lockin",
    "LNET_STEM_STAGE0_MODE": "triton_v3_m64_w4",
    "LNET_READER_DENSE_NHWC": "dense_nhwc",
    "LNET_READER_EXEC_MODE": "dense_nhwc",
    "LNET_CUDNN_BENCHMARK": "1",
    "LNET_SCAN_W51A": "fp32",
    "LNET_SCAN_W53A": "w57_lockin",
    "LNET_FUSED_STEM_CONV": "on",
    "LNET_MODEL_W56B": "swiglu_vec8_tanh",
    "LNET_MODEL_W57B_RMS": "cuda_sp_vec8_4cta"
}
BUCKETS = (
    "vscan_bwd", "vscan_fwd", "hscan_bwd", "hscan_fwd", "factorized_reader",
    "cudnn_conv", "cublas_gemm", "triton", "fused_stem", "other",
)


def profile_step(step: Callable) -> dict:
    import torch

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
        elapsed = event.self_device_time_total / 1000
        bucket = (
            event.key.removeprefix("va_profile_").split("__", 1)[0]
            if event.key.startswith("va_profile_")
            else runner.kernel_bucket(event.key)
        )
        if bucket not in buckets:
            bucket = "other"
        buckets[bucket] += elapsed
        kernels.append({
            "name": event.key, "count": event.count, "self_cuda_ms": elapsed, "bucket": bucket,
        })
    ranked = sorted(kernels, key=lambda row: -row["self_cuda_ms"])
    if not ranked:
        raise RuntimeError("CUDA profiler returned no device events")
    return {
        "wall_ms": wall_ms,
        "cuda_ms": sum(buckets.values()),
        "self_cuda_ms_by_bucket": buckets,
        "top_kernels": ranked[:30],
        "all_kernels": [row for row in ranked if row["self_cuda_ms"] >= 0.05],
        "kernel_count": len(ranked),
    }


def measure_arm(
    out: Path, *, profile: bool, arm_env_keys: tuple[str, ...] = tuple(REFERENCE_ENV),
) -> int:
    from torch._dynamo.utils import counters

    from lnet import general_cuda_scans as scans

    original_kernel_name = scans._kernel_name  # noqa: SLF001

    def named_scan(key: scans.Kernel) -> str:
        name = original_kernel_name(key)
        if runner.kernel_bucket(name) in (
            "general_scan_shared", "general_vscan", "general_hscan",
        ):
            axis = "vscan" if key.vertical else "hscan"
            phase = "bwd" if key.backward or key.reduction else "fwd"
            return f"va_profile_{axis}_{phase}__{name}"
        return name

    # CUDA graph replay retains kernel symbols but bypasses Python launch hooks.
    # Tag each axis and direction without changing the generated kernel body.
    scans._kernel_name = named_scan  # noqa: SLF001
    result = runner.train_arm(
        argparse.Namespace(variant=VARIANT, batch=1024, profile=profile, out=out),
        profiler=lambda step, _diagnostic: profile_step(step),
    )
    replays = result.get("cudagraph_replays_by_phase", {})
    runner.write_json(out, {
        "status": "ok",
        "compile_warmup_s": result["compile_warmup_s"],
        "timed_loop_s": result["timed_loop_s"],
        "step_ms": {key: result[f"{key}_ms"] for key in ("median", "mean", "p90")},
        "img_per_s": result["images_per_second"],
        "peak_vram_gib": result["peak_allocated_gib"],
        "loss_finite": result["finite_losses"],
        "finite_grad_count": result["finite_grad_count"],
        "grad_tensor_count": result["grad_tensor_count"],
        "graph_replays": {
            "fwd": replays.get("forward", 0), "bwd": replays.get("backward", 0),
            "upd": replays.get("update", 0),
        },
        "cache": {
            key: counters[group].get(key, 0)
            for group, prefix in (("inductor", "fxgraph"), ("aot_autograd", "autograd"))
            for key in (f"{prefix}_cache_hit", f"{prefix}_cache_miss")
        },
        "aot_autograd": {
            key: counters["aot_autograd"].get(f"autograd_cache_{key}", 0)
            for key in ("hit", "miss", "bypass")
        },
        "arm_env": {key: os.environ[key] for key in arm_env_keys},
        "preset": os.environ["LNET_MODEL_W36B"],
        **{key: result[key] for key in ("profiler", "profiler_error") if key in result},
    })
    return 0

def lockin_arm_env(name: str) -> dict[str, str]:
    arm = name.removesuffix("_rev")
    if arm == "w43_ref":
        return REFERENCE_ENV.copy()
    if arm == "w57_lockin":
        return LOCKIN_ENV.copy()
    raise ValueError(f"Unknown arm {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    arms = ("w43_ref", "w57_lockin")
    parser.add_argument("--arm", required=True, choices=(*arms, *(arm + "_rev" for arm in arms)))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    env = lockin_arm_env(args.arm)
    os.environ.update(env)
    return measure_arm(args.out, profile=args.profile, arm_env_keys=tuple(env))


if __name__ == "__main__":
    raise SystemExit(main())
