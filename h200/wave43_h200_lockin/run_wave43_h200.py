"""Run the Wave 43 Whole H200 lock-in and transport its JSON results."""
# ruff: noqa: PLC0415, T201

import argparse
import base64
import hashlib
import io
import os
import sys
import tarfile
import traceback
from pathlib import Path
from time import perf_counter

from h200.liverms_v2_opt import bench_h200
from h200.wave40_h200_bundle.run_wave40_h200 import BENCH, child, median, successful_tests
from h200.wave41_h200_bundle.bench_wave41_h200 import BASE_PRESET, VARIANT, profile_step
from h200.wave41_h200_bundle.run_wave41_h200 import arm_passed

REF, WIN = "w40_ref_combo", "w43_best_combo"
MODES = {
    REF: {"LNET_SCAN_SGS": "strict", "LNET_SCAN_VFWD": "off", "LNET_SCAN_HSCAN": "off"},
    WIN: {"LNET_SCAN_SGS": "strict_ro", "LNET_SCAN_VFWD": "batch4", "LNET_SCAN_HSCAN": "bwd_opt"},
}


def kernel_bucket(name: str) -> str:
    if name == "vscan_sgs":
        return "vscan_sgs"
    if name.startswith("triton") and "tem_mm" in name:
        return "triton_tem_mm"
    return ORIGINAL_BUCKET(name)


ORIGINAL_BUCKET = bench_h200.kernel_bucket


def measure_arm(out: Path, *, profile: bool) -> int:
    from torch._dynamo.utils import counters

    # Use Wave 41's CUDA-only one-step capture with two additional kernel buckets.
    import h200.wave41_h200_bundle.bench_wave41_h200 as wave41

    wave41.BUCKETS += ("vscan_sgs", "triton_tem_mm", "general_scan_shared")
    bench_h200.kernel_bucket = kernel_bucket

    result = bench_h200.train_arm(
        argparse.Namespace(variant=VARIANT, batch=1024, profile=profile),
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
        "graph_replays": {"fwd": replays.get("forward", 0), "bwd": replays.get("backward", 0)},
        "cache": {
            key: counters[group].get(key, 0)
            for group, prefix in (("inductor", "fxgraph"), ("aot_autograd", "autograd"))
            for key in (f"{prefix}_cache_hit", f"{prefix}_cache_miss")
        },
        "scan_env": {key: os.environ[key] for key in MODES[REF]},
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

    scan = child(root, "scan_micro", [
        "scripts/bench_scan_micro.py", "--all-ops", "--modes", "off,strict,strict_ro",
    ], 180)
    summary["stage1_s"] = scan["wall_s"]
    summary["eager_projection"] = scan.get("combined_projection")

    started = perf_counter()
    arms = {}
    for mode, reverse in ((REF, False), (WIN, False), (WIN, True), (REF, True)):
        name = f"{mode}_rev" if reverse else mode
        args = [str(Path(__file__).resolve()), "--arm"]
        if not reverse:
            args.append("--profile")
        arms[name] = child(root, name, args, 900, env={
            **MODES[mode], "LNET_MODEL_W36B": BASE_PRESET,
        })
    summary["stage2_s"] = perf_counter() - started

    started = perf_counter()
    tests = child(root, "pytest_scan", [
        "-m", "pytest", "-q", "--tb=short", "tests/test_general_cuda_scans.py",
        "-k", "sgs or vfwd or hscan",
    ], 180, tests=True)
    fold = child(root, "fold_w43", [
        "-m", BENCH, "infer-fold", "--variant", VARIANT, "--batch", "2", "--precision", "highest",
    ], 300, env={**MODES[WIN], "LNET_MODEL_W36B": BASE_PRESET})
    summary["stage3_s"] = perf_counter() - started
    highest = fold.get("fp32_head_parity_by_precision", {}).get("highest", {})
    fold_passed = (
        fold["exit_code"] == 0 and fold["status"] == "ok" and fold.get("precision") == "highest"
        and fold.get("strict_reload_max_abs_logit_diff") == 0.0
        and fold.get("folded_train_guard") is True
        and highest.get("max_abs_logit_diff", float("inf")) <= 1e-5
    )
    tests["passed"] = successful_tests(tests, 132)
    fold["passed"] = fold_passed
    bench_h200.write_json(root / "pytest_scan.json", tests)
    bench_h200.write_json(root / "fold_w43.json", fold)
    summary["gates"] = {"pytest_scan": tests["passed"], "fold_w43": fold_passed}
    if not all(arm_passed(row) for row in arms.values()):
        return
    deltas = {
        suffix or "forward": median(arms[WIN + suffix]) - median(arms[REF + suffix])
        for suffix in ("", "_rev")
    }
    summary["delta_ms"] = {"forward": deltas["forward"], "reverse": deltas["_rev"]}
    summary["below_180_ms"] = {
        name: median(arms[name]) < 180.0 for name in (WIN, WIN + "_rev")
    }
    if (
        scan["status"] == "ok" and summary["eager_projection"]
        and all(summary["gates"].values()) and all(delta < 0 for delta in deltas.values())
        and all(summary["below_180_ms"].values())
        and all(arms[name]["img_per_s"] > 5690 for name in (WIN, WIN + "_rev"))
        and all(arms[name].get("profiler", {}).get("top_kernels") for name in (REF, WIN))
    ):
        summary["verdict"] = "PASS"


def transport(root: Path) -> None:
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
    print(f"VA_T151_SHA256 {hashlib.sha256(raw).hexdigest()}")
    print(f"VA_T151_MANIFEST_SHA256 {hashlib.sha256(manifest.read_bytes()).hexdigest()}")
    print(f"VA_T151_BASE64_CHARS {len(encoded)}")
    print("VA_T151_BEGIN")
    for start in range(0, len(encoded), 240):
        print(f"VA_T151_GZ {encoded[start:start + 240]}")
    print("VA_T151_END", flush=True)


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
        "workload": {"variant": VARIANT, "batch": 1024, "dtype": "bf16", "preset": BASE_PRESET},
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
