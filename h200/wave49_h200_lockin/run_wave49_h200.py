"""Run the Wave 49 Whole H200 lock-in and transport its JSON results."""
# ruff: noqa: PLC0415, T201

import argparse
import base64
import hashlib
import io
import os
import sys
import tarfile
import traceback
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from h200.liverms_v2_opt import bench_h200
from h200.wave40_h200_bundle.run_wave40_h200 import BENCH, child, median, successful_tests
from h200.wave41_h200_bundle.bench_wave41_h200 import BASE_PRESET, VARIANT
from h200.wave41_h200_bundle.run_wave41_h200 import arm_passed

REF, WIN, BENCH_TRUE = "w43_ref", "w49_best_combo", "w49_bench_true"
MODES = {
    REF: {
        "LNET_SCAN_SGS": "strict_ro", "LNET_SCAN_VFWD": "batch4", "LNET_SCAN_HSCAN": "bwd_opt",
        "LNET_READER_EXEC_MODE": "off", "VA_READER_OPT_MODE": "off", "LNET_CUDNN_BENCHMARK": "1",
    },
    WIN: {
        "LNET_SCAN_SGS": "strict_ro", "LNET_SCAN_VFWD": "batch4_w2", "LNET_SCAN_HSCAN": "bwd_opt",
        "LNET_READER_EXEC_MODE": "auto_roofline", "VA_READER_OPT_MODE": "proven_floor",
        "LNET_CUDNN_BENCHMARK": "0",
    },
}
MODES[BENCH_TRUE] = {**MODES[WIN], "LNET_CUDNN_BENCHMARK": "1"}
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
            else bench_h200.kernel_bucket(event.key)
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
    out: Path, *, profile: bool, arm_env_keys: tuple[str, ...] = tuple(MODES[REF]),
) -> int:
    from torch._dynamo.utils import counters

    from lnet import general_cuda_scans as scans

    original_kernel_name = scans._kernel_name  # noqa: SLF001

    def named_scan(key: scans.Kernel) -> str:
        name = original_kernel_name(key)
        if bench_h200.kernel_bucket(name) in (
            "general_scan_shared", "general_vscan", "general_hscan",
        ):
            axis = "vscan" if key.vertical else "hscan"
            phase = "bwd" if key.backward or key.reduction else "fwd"
            return f"va_profile_{axis}_{phase}__{name}"
        return name

    # CUDA graph replay retains kernel symbols but bypasses Python launch hooks.
    # Tag each axis and direction without changing the generated kernel body.
    scans._kernel_name = named_scan  # noqa: SLF001
    result = bench_h200.train_arm(
        argparse.Namespace(variant=VARIANT, batch=1024, profile=profile, out=out),
        profiler=lambda step, _diagnostic: profile_step(step),
    )
    replays = result.get("cudagraph_replays_by_phase", {})
    bench_h200.write_json(out, {
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


def run(root: Path, summary: dict) -> None:
    # Keep the device record inside the summary so the transport has eight results.
    device = summary["device"] = bench_h200.probe(argparse.Namespace())
    if not (
        "H200" in device["gpu"] and device["vram_gib"] >= 139
        and device["sm_count"] == 132 and device["capability"] == (9, 0)
        and device["python"] == "3.13.11" and device["torch"] == "2.9.1+cu128"
        and device["triton"] == "3.5.1"
    ):
        raise RuntimeError("Whole H200 or pinned environment verification failed")

    started = perf_counter()
    arms = {}
    order = ((REF, ""), (WIN, ""), (WIN, "_rev"), (REF, "_rev"), (BENCH_TRUE, ""))
    for mode, suffix in order:
        name = mode + suffix
        args = [str(Path(__file__).resolve()), "--arm"]
        if name in (REF, WIN):
            args.append("--profile")
        arms[name] = child(root, name, args, 900, env={
            **MODES[mode], "LNET_MODEL_W36B": BASE_PRESET,
        })
    summary["stage1_s"] = perf_counter() - started

    started = perf_counter()
    win_env = {**MODES[WIN], "LNET_MODEL_W36B": BASE_PRESET}
    fold = child(root, "fold_w49", [
        "-m", BENCH, "infer-fold", "--variant", bench_h200.LOCKIN_FOLD_ALIAS,
        "--batch", "2", "--precision", "highest",
    ], 300, env=win_env)
    highest = fold.get("fp32_head_parity_by_precision", {}).get("highest", {})
    fold["passed"] = (
        fold["exit_code"] == 0 and fold["status"] == "ok" and fold.get("precision") == "highest"
        and fold.get("strict_reload_max_abs_logit_diff") == 0.0
        and fold.get("folded_train_guard") is True and highest.get("passed") is True
        and highest.get("max_abs_logit_diff", float("inf")) < 1e-5
    )
    bench_h200.write_json(root / "fold_w49.json", fold)
    summary["gates"] = {"fold_w49": fold["passed"]}
    for name, files, limit, expected in (
        (
            "pytest_reader",
            ["test_reader_dense_nhwc.py", "test_factorized_reader_kernels.py"],
            300, 335,
        ),
        ("pytest_scan", ["test_general_cuda_scans.py"], 420, 402),
    ):
        tests = child(root, name, [
            "-m", "pytest", "-q", "--tb=short", *(f"tests/{file}" for file in files),
        ], limit, env=win_env, tests=True)
        tests["passed"] = successful_tests(tests, expected)
        summary["gates"][name] = tests["passed"]
        bench_h200.write_json(root / f"{name}.json", tests)
    summary["stage2_s"] = perf_counter() - started
    summary["arm_gates"] = {name: arm_passed(row) for name, row in arms.items()}
    if not all(summary["arm_gates"].values()):
        return
    summary["delta_ms"] = {
        order_name: median(arms[WIN + suffix]) - median(arms[REF + suffix])
        for order_name, suffix in (("forward", ""), ("reverse", "_rev"))
    }
    summary["bench_true_delta_ms"] = median(arms[BENCH_TRUE]) - median(arms[WIN])
    summary["performance_gates"] = {
        name: (
            median(arms[name]) < 170.0
            and arms[name]["img_per_s"] > 6020
            and arms[name]["peak_vram_gib"] < 38.0
        )
        for name in (WIN, WIN + "_rev")
    }
    if (
        all(summary["gates"].values())
        and all(delta < 0 for delta in summary["delta_ms"].values())
        and all(summary["performance_gates"].values())
        and all(arms[name].get("profiler", {}).get("all_kernels") for name in (REF, WIN))
    ):
        summary["verdict"] = "PASS"


def transport(root: Path, *, tag: str = "VA_T173") -> None:
    members = sorted(path for path in root.glob("*.json") if path.name != "manifest.json")
    manifest = root / "manifest.json"
    bench_h200.write_json(manifest, {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in members
    })
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=9) as archive:
        for path in (*members, manifest):
            archive.add(path, arcname=path.name)
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode("ascii")
    print(f"{tag}_SHA256 {hashlib.sha256(raw).hexdigest()}")
    print(f"{tag}_MANIFEST_SHA256 {hashlib.sha256(manifest.read_bytes()).hexdigest()}")
    for path in (*members, manifest):
        print(f"{tag}_JSON_SHA256 {path.name} {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"{tag}_BASE64_CHARS {len(encoded)}")
    print(f"{tag}_BEGIN")
    for start in range(0, len(encoded), 240):
        print(f"{tag}_GZ {encoded[start:start + 240]}")
    print(f"{tag}_END", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--out", dest="arm_out", type=Path)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.arm:
        return measure_arm(args.arm_out, profile=args.profile)
    root = args.out.resolve()
    started = perf_counter()
    bootstrap_s = int(os.environ["VA_BOOTSTRAP_SECONDS"])
    summary = {
        "verdict": "FAIL", "commit": os.environ["VA_BUNDLE_COMMIT"], "bootstrap_s": bootstrap_s,
        "workload": {
            "model": bench_h200.LOCKIN_FOLD_ALIAS, "variant": VARIANT,
            "batch": 1024, "dtype": "bf16", "preset": BASE_PRESET, "warmup": 3, "count": 8,
        },
    }
    try:
        run(root, summary)
    except Exception:  # noqa: BLE001 - transport the failure and any completed stages
        summary["error"] = traceback.format_exc()
    finally:
        summary["runner_wall_s"] = perf_counter() - started
        summary["total_wall_s"] = bootstrap_s + summary["runner_wall_s"]
        bench_h200.write_json(root / "summary.json", summary)
        transport(root)
    return int(summary["verdict"] != "PASS")


if __name__ == "__main__":
    sys.exit(main())
