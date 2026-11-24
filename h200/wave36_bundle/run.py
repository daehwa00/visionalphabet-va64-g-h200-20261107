"""Run the isolated arm and verification subprocesses and emit JSON results."""
# ruff: noqa: S603, T201, BLE001

import argparse
import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

from h200.liverms_v2_opt import bench_h200 as bench
from h200.wave36_bundle.metrics import ARMS, BASE, COMBO, OFF, nograph_preset, select

BENCH = "h200.liverms_v2_opt.bench_h200"
VARIANT = "17-p1-nr4-1024d-v3"
VERIFY_PRESET = ("saved_fast_proj,hscan_bwd_hiocc", "stem_cudnn,capture_safe_ortho")
TESTS = {
    "scan": ("tests/test_general_cuda_scans.py",),
    "model": ("tests/test_model_w36b.py",),
    "reader": ("tests/test_factorized_reader_kernels.py", "-k", "not slow"),
}
METRICS = (
    "median_ms", "mean_ms", "min_ms", "max_ms", "p95_ms", "img_per_sec", "compile_time_s",
    "peak_allocated_gib", "peak_reserved_gib", "loss_finite", "grad_finite", "loss_value",
    "total_self_cuda_ms", "wall_minus_cuda_ms", "buckets", "subkernels", "top_kernels", "finite",
)


def run(
    root: Path, name: str, presets: tuple[str, str], args: list[str], limit: int, *,
    pytest_mode: bool = False,
) -> dict:
    target = root / "json" / f"{name}.json"
    log = root / f"{name}.log"
    env = {
        **os.environ,
        "LNET_SCAN_W36A": presets[0],
        "LNET_MODEL_W36B": presets[1],
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor" / name),
        "TRITON_CACHE_DIR": str(root / "triton" / name),
        "CUDA_CACHE_PATH": str(root / "cuda" / name),
    }
    if not pytest_mode:
        args = [*args, "--out", str(target)]
    start = time.monotonic()
    error = {}
    rc = None
    try:
        with log.open("wb") as stream, subprocess.Popen(
            [sys.executable, *args], env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as proc:
            try:
                rc = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                error = {"status": "timeout", "error": f"Exceeded {limit} seconds"}
    except Exception:
        error = {"status": "error", "error": traceback.format_exc()}
    elapsed = time.monotonic() - start
    output = log.read_text(errors="replace") if log.exists() else ""
    try:
        if pytest_mode:
            row = bench.wrap_pytest(argparse.Namespace(log=log, exit_code=rc, elapsed=elapsed))
            for key in ("tail", "scan_plan_tests", "dynamo_tests", "by_file"):
                row.pop(key, None)
            row["wave36_cuda_passes"] = sum(
                line.startswith("tests/") and (" PASSED" in line)
                and any(pattern in line for pattern in (
                    "::test_cuda_wave36a_", "::test_stem_cudnn_capture",
                    "::test_variant_compiled_cudagraph",
                ))
                for line in output.splitlines()
            )
        else:
            row = json.loads(target.read_text()) if target.exists() else {}
    except Exception:
        row = {}
        error.setdefault("error", traceback.format_exc())
    if (rc != 0 or not row or error) and not error.get("status"):
        error.update(status="error", error=error.get("error", f"Child exit code {rc}"))
    row.update(error, presets=presets, duration_seconds=round(elapsed, 3), returncode=rc)
    if row.get("status") != "ok":
        row["log_tail"] = output[-2600:]
    if "cuda_graphs" in presets[1].split(","):
        graph_messages = [line for line in output.splitlines() if "cudagraph" in line.lower()]
        row["cudagraph_messages"] = "\n".join(graph_messages)[-1800:]
    bench.write_json(target, row)
    print(f"{name}: {row.get('status')} ({elapsed:.1f}s)", file=sys.stderr, flush=True)
    return row


def arm(root: Path, name: str, presets: tuple[str, str]) -> dict:
    limit = 600 if "max_autotune_gemm" in presets[1].split(",") else 300
    row = run(root, name, presets, ["-m", "h200.wave36_bundle.arm"], limit)
    losses = row.get("losses", {})
    row.setdefault("loss_value", losses.get("last"))
    row.setdefault("loss_finite", row.get("finite_losses"))
    row.setdefault("grad_finite", row.get("finite_gradients"))
    row.setdefault("img_per_sec", row.get("images_per_second"))
    if row["loss_finite"] is not None and row["grad_finite"] is not None:
        row.setdefault("finite", row["loss_finite"] and row["grad_finite"])
    for field in METRICS:
        row.setdefault(field, None)
    bench.write_json(root / "json" / f"{name}.json", row)
    return row


def fold_passed(row: dict) -> bool:
    return (
        row.get("status") == "ok" and row.get("strict_reload_max_abs_logit_diff") == 0.0
        and row.get("folded_train_guard") is True
    )


def verification(root: Path, presets: tuple[str, str]) -> dict:
    tests = {
        name: run(root, f"pytest_{name}", OFF, ["-m", "pytest", "-q", "-vv", *args],
                  900 if name == "scan" else 480,
                  pytest_mode=True)
        for name, args in TESTS.items()
    }
    dynamo = run(root, "dynamo", presets, ["-m", BENCH, "dynamo"], 240)
    graphs = [count for modes in dynamo.get("graphs", {}).values() for count in modes.values()]
    folds = {}
    for name, spec in (("base", OFF), ("combo", presets)):
        folds[name] = run(
            root, f"fold_{name}", spec,
            ["-m", BENCH, "infer-fold", "--variant", VARIANT, "--batch", "2"], 300,
        )
        if not fold_passed(folds[name]):
            run(
                root, f"fold_{name}_highest", spec,
                ["-m", BENCH, "infer-fold", "--variant", VARIANT, "--batch", "2",
                 "--precision", "highest"], 300,
            )
    return {
        "pytest_scan": tests["scan"].get("status") == "ok"
        and tests["scan"].get("wave36_cuda_passes") == 15,
        "pytest_model": tests["model"].get("status") == "ok"
        and tests["model"].get("wave36_cuda_passes") == 5,
        "pytest_reader": tests["reader"].get("status") == "ok",
        "dynamo": len(graphs) == 18 and all(count == [1, 0] for count in graphs),
        **{
            f"fold_{name}": fold_passed(row)
            for name, row in folds.items()
        },
    }


def benchmark(root: Path) -> None:
    device = run(root, "device", OFF, ["-m", BENCH, "probe"], 60)
    if not (
        device.get("status") == "ok" and "H200" in device["gpu"]
        and device["vram_gib"] >= 139 and device["sm_count"] == 132
        and device["capability"] == [9, 0]
    ):
        bench.write_json(root / "json" / "summary.json",
                         {"status": "error", "error": "Expected a Whole H200"})
        return
    rows = {name: arm(root, name, presets) for name, presets in ARMS.items()}
    selection = select(rows)
    presets = tuple(selection["presets"])
    if selection["winners"]:
        combo = arm(root, COMBO, presets)
        arm(root, f"{COMBO}_rev", presets)
    arm(root, f"{BASE}_rev", OFF)
    if presets[0].startswith("saved_fast_proj") and "cuda_graphs" in presets[1].split(","):
        failed = combo["status"] != "ok" or (combo["peak_allocated_gib"] or 0) > 130
        use_ortho = not failed or "w36b_capture_safe_ortho" in selection["eligible"]
        arm(root, f"{COMBO}_nograph", nograph_preset(presets, use_ortho=use_ortho))
    gates = verification(root, presets if selection["winners"] else VERIFY_PRESET)
    gates["saved_scan_call_counts"] = all(
        row.get("wave36a_11_call_check") and all(row["wave36a_11_call_check"].values())
        for name, row in rows.items() if ARMS[name][0].startswith("saved_fast_proj")
    )
    bench.write_json(root / "json" / "summary.json", {
        "status": "ok" if all(gates.values()) else "verification_failed",
        "selection": selection,
        "verification": gates,
        "commit": env_commit(),
        "compile_time_s_definition": "First training step including lazy compilation and execution",
        "workload": {"model": VARIANT, "batch": 1024, "dtype": "BF16", "warmup": 3, "iters": 8},
    })


def env_commit() -> str:
    return os.environ.get("VA_BUNDLE_COMMIT", "unknown")


def transport(root: Path) -> None:
    archive = root / "task131.tar.xz"
    with tarfile.open(archive, "w:xz", preset=9) as stream:
        for path in sorted((root / "json").glob("*.json")):
            stream.add(path, arcname=path.name)
    raw = archive.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    print(f"VA_T131_SHA256 {hashlib.sha256(raw).hexdigest()}")
    print(f"VA_T131_BASE64_CHARS {len(encoded)}")
    if len(encoded) >= 45_000:
        raise RuntimeError(f"Archive exceeded cap; compact JSON retained in {root}")
    print("VA_T131_BEGIN")
    for start in range(0, len(encoded), 240):
        print(f"VA_T131_XZ {encoded[start:start + 240]}")
    print("VA_T131_END", flush=True)


def main() -> None:
    root = Path(sys.argv[1])
    (root / "json").mkdir()
    try:
        benchmark(root)
    except Exception:
        bench.write_json(root / "json" / "summary.json",
                         {"status": "runner_error", "error": traceback.format_exc()})
    finally:
        transport(root)
    summary = json.loads((root / "json" / "summary.json").read_text())
    if summary["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
