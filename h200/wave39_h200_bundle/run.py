"""Run the isolated arm and verification subprocesses and emit JSON results."""
# ruff: noqa: S603, T201, BLE001

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

from h200.liverms_v2_opt import bench_h200 as bench
from h200.wave39_h200_bundle.metrics import ARMS, BASE, COMBO, OFF, select, valid

BENCH = "h200.liverms_v2_opt.bench_h200"
VARIANT = "17-p1-nr4-1024d-v3"
TESTS = {
    "scan": ("tests/test_general_cuda_scans.py", "-k", "wave37a or wave39a or wave38a_odd_coarse"),
    "model": ("tests/test_model_w36b.py",),
    "reader": ("tests/test_factorized_reader_kernels.py", "-k", "not slow"),
}
METRICS = (
    "median_ms", "mean_ms", "min_ms", "max_ms", "p95_ms", "img_per_sec", "compile_time_s",
    "peak_allocated_gib", "peak_reserved_gib", "loss_finite", "grad_finite", "loss_value",
    "total_self_cuda_ms", "wall_minus_cuda_ms", "buckets", "subkernels", "top_kernels", "finite",
    "error", "cudagraph_replays", "cudagraph_log_notes", "scan_shape_cuda_events",
)


def run(  # noqa: C901, PLR0912, PLR0915 - contain all subprocess and parse failures
    root: Path, name: str, presets: tuple[str, str], args: list[str], limit: int, *,
    pytest_mode: bool = False,
) -> dict:
    target = root / "json" / f"{name}.json"
    log = root / f"{name}.log"
    env = {
        **os.environ,
        "LNET_SCAN_W39A": presets[0],
        "LNET_MODEL_W36B": presets[1],
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor" / name),
        "TRITON_CACHE_DIR": str(root / "triton" / name),
        "CUDA_CACHE_PATH": str(root / "cuda" / name),
    }
    if "cuda_graphs" in presets[1].split(",") or name == "pytest_model":
        env["TORCH_LOGS"] = "cudagraphs"
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
    output = ""
    try:
        output = log.read_text(errors="replace") if log.exists() else ""
        if pytest_mode:
            row = bench.wrap_pytest(argparse.Namespace(log=log, exit_code=rc, elapsed=elapsed))
            for key in ("tail", "scan_plan_tests", "dynamo_tests", "by_file"):
                row.pop(key, None)
            cases = selected_cases(output, name)
            row["selected_passes"] = len(cases)
            if name == "pytest_scan":
                row["cuda_parity_passes"] = {
                    policy: sum(f"-{policy}]" in c for c in cases if "cuda_wave39a" in c)
                    for policy in ("wmma_bwd", "wmma_fwd", "wmma_both", "legacy_sgs")
                }
                row["wave37a_cuda_parity_passes"] = sum("cuda_wave37a_parity" in c for c in cases)
                row["skip_reasons"] = re.findall(r"^SKIPPED \[\d+\] .*", output, re.MULTILINE)
            else:
                row["passed_cases"] = cases
            if name == "pytest_model":
                row["max_abs_logs"] = [
                    line for line in output.splitlines() if " max_abs = " in line
                ]
        else:
            row = json.loads(target.read_text()) if target.exists() else {}
    except Exception:
        row = {}
        error.setdefault("error", traceback.format_exc())
    if (rc != 0 or not row or error) and not error.get("status"):
        error.update(status="error", error=error.get("error", f"Child exit code {rc}"))
    row.update(error, presets=presets, duration_seconds=round(elapsed, 3), returncode=rc)
    if row.get("status") != "ok":
        row["log_tail"] = output[-4000:]
        if pytest_mode:
            marker = re.search(r"^=+\s+FAILURES\s+=+|^FAILED\b", output, re.MULTILINE)
            row["log_head_failures"] = output[marker.start():][:4000] if marker else output[:4000]
            failures = re.findall(r"^(?:FAILED|ERROR) (tests/\S+)", output, re.MULTILINE)
            row["failed_tests"] = sorted(set(row.get("failed_tests", []) + failures))[:20]
    if "TORCH_LOGS" in env:
        notes = []
        for line in output.splitlines():
            lower = line.lower()
            is_graph = "cudagraph" in lower or "cuda graph" in lower
            if is_graph and ("skip" in lower or "replay" in lower):
                note = line.split("] ", 1)[-1].strip()[:480]
                if note not in notes:
                    notes.append(note)
        row["cudagraph_log_notes"] = notes[:8]
    bench.write_json(target, row)
    print(f"{name}: {row.get('status')} ({elapsed:.1f}s)", file=sys.stderr, flush=True)
    return row


def selected_cases(output: str, name: str) -> list[str]:
    patterns = {
        "pytest_reader": ("test_singleton_strict_fp32", "shape0-1-1-1-dtype0",
                          "shape2-64-6-True"),
        "pytest_model": ("replay", "autotune", "parity"),
        "pytest_scan": ("test_cuda_wave37a_parity", "test_cuda_wave39a_parity"),
    }
    cases = re.findall(
        r"^tests/[^:\n]+::([^\s]+)[ \t]+(?:(?!^tests/).)*?\bPASSED\b",
        output, flags=re.MULTILINE | re.DOTALL,
    )
    return [case for case in cases if any(key in case for key in patterns.get(name, ()))]


def arm(root: Path, name: str, presets: tuple[str, str]) -> dict:
    args = ["-m", "h200.wave39_h200_bundle.arm"]
    if name == BASE or name.startswith("w39a_"):
        args.append("--scan-events")
    row = run(root, name, presets, args, 720)
    losses = row.get("losses", {})
    row.setdefault("loss_value", losses.get("last"))
    row.setdefault("loss_finite", row.get("finite_losses"))
    row.setdefault("grad_finite", row.get("finite_gradients"))
    row.setdefault("img_per_sec", row.get("images_per_second"))
    if row["loss_finite"] is not None and row["grad_finite"] is not None:
        row.setdefault("finite", row["loss_finite"] and row["grad_finite"])
    if row.get("cudagraph_replays") is None:
        row["cudagraph_replays"] = row.get("cudagraph_replays_per_forward_backward")
    for field in METRICS:
        row.setdefault(field, None)
    bench.write_json(root / "json" / f"{name}.json", row)
    return row


def fold_passed(row: dict) -> bool:
    return (
        row.get("status") == "ok" and row.get("strict_reload_max_abs_logit_diff") == 0.0
        and row.get("folded_train_guard") is True
        and (row.get("fp32_head_max_abs_logit_diff") or 0) < 1e-5
        and row.get("fp32_head_max_abs_logit_diff") is not None
    )


def verification(root: Path, presets: tuple[str, str]) -> dict:
    tests = {
        name: run(root, f"pytest_{name}", OFF,
                  ["-m", "pytest", "-q", "-vv", "-rs", "--color=no",
                   "--tb=line" if name == "scan" else "--tb=short",
                   *(["-s"] if name == "model" else []), *args],
                  {"scan": 600, "model": 600, "reader": 300}[name],
                  pytest_mode=True)
        for name, args in TESTS.items()
    }
    dynamo = run(root, "dynamo", presets, ["-m", BENCH, "dynamo"], 240)
    graphs = [count for modes in dynamo.get("graphs", {}).values() for count in modes.values()]
    folds = {}
    for name, spec in (("base", OFF), ("combo", presets)):
        folds[name] = run(
            root, f"fold_{name}", spec,
            ["-m", BENCH, "infer-fold", "--variant", VARIANT, "--batch", "2",
             "--precision", "highest"], 240,
        )
    return {
        "pytest_scan": tests["scan"].get("status") == "ok"
        and tests["scan"].get("wave37a_cuda_parity_passes", 0) > 0
        and len(tests["scan"].get("cuda_parity_passes", {})) == 4
        and all(value == 76 for value in tests["scan"]["cuda_parity_passes"].values())
        and all("Compact VScan" in reason for reason in tests["scan"].get("skip_reasons", [])),
        "pytest_model": tests["model"].get("status") == "ok"
        and (tests["model"].get("counts") or {}).get("skipped", 0) == 0
        and all(any(required in case for case in tests["model"].get("passed_cases", []))
                for required in ("compiled_cudagraph_replay", "compiled_update_parity",
                                 "compiled_update_cudagraph_replays",
                                 "ortho_plus_max_autotune_gemm")),
        "pytest_reader": tests["reader"].get("status") == "ok"
        and (tests["reader"].get("counts") or {}).get("passed", 0) == 171,
        "dynamo": len(graphs) == 18 and all(count == [1, 0] for count in graphs),
        **{f"fold_{name}": fold_passed(row) for name, row in folds.items()},
    }


def benchmark(root: Path) -> None:
    device = run(root, "device", OFF, ["-m", BENCH, "probe"], 60)
    if not (
        device.get("status") == "ok" and "H200" in device["gpu"]
        and device["vram_gib"] >= 139 and device["sm_count"] == 132
        and device["capability"] == [9, 0]
    ):
        bench.write_json(root / "json" / "summary.json",
                         {"status": "error", "error": "Expected a Whole H200",
                          "commit": env_commit()})
        return
    rows = {name: arm(root, name, presets) for name, presets in ARMS.items()}
    selection = select(rows)
    presets = tuple(selection["presets"])
    for name, spec in ((COMBO, presets), (f"{COMBO}_rev", presets), (f"{BASE}_rev", OFF)):
        rows[name] = arm(root, name, spec)
    gates = verification(root, presets)
    benchmarks_ok = all(valid(row) for row in rows.values())
    expected_shapes = {
        f"vscan_{phase}_{shape}" for phase in ("fwd", "bwd")
        for shape in ("56_d", "56_c", "28_d", "28_c", "14_d", "14_c", "7_d")
    }
    telemetry = {
        "kineto": all(row.get("profile_cuda_coverage") for row in rows.values()),
        "scan_events": all(expected_shapes <= {event["name"] for event in
                           (row.get("scan_shape_cuda_events") or [])}
                           for name, row in rows.items()
                           if name == BASE or name.startswith("w39a_")),
    }
    bench.write_json(root / "json" / "summary.json", {
        "status": "ok" if all(gates.values()) and benchmarks_ok and all(telemetry.values())
        else "verification_failed",
        "telemetry": telemetry,
        "benchmarks_ok": benchmarks_ok,
        "selection": selection,
        "verification": gates,
        "commit": env_commit(),
        "compile_time_s_definition": "First training step including lazy compilation and execution",
        "workload": {"model": VARIANT, "batch": 1024, "dtype": "BF16", "warmup": 3, "iters": 8},
    })


def env_commit() -> str:
    return os.environ.get("VA_BUNDLE_COMMIT", "unknown")


def transport(root: Path) -> None:
    archive = root / "task140.tar.xz"
    members = sorted((root / "json").glob("*.json"))
    bench.write_json(root / "json" / "manifest.json",
                     {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in members})
    with tarfile.open(archive, "w:xz", preset=9) as stream:
        for path in [*members, root / "json" / "manifest.json"]:
            stream.add(path, arcname=path.name)
    raw = archive.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    print(f"VA_T140_SHA256 {hashlib.sha256(raw).hexdigest()}")
    print(f"VA_T140_BASE64_CHARS {len(encoded)}")
    if len(encoded) >= 45_000:
        raise RuntimeError(f"Archive exceeded cap; compact JSON retained in {root}")
    print("VA_T140_BEGIN")
    for start in range(0, len(encoded), 240):
        print(f"VA_T140_XZ {encoded[start:start + 240]}")
    print("VA_T140_END", flush=True)


def main() -> None:
    root = Path(sys.argv[1])
    (root / "json").mkdir()
    try:
        benchmark(root)
    except Exception:
        bench.write_json(root / "json" / "summary.json",
                         {"status": "runner_error", "error": traceback.format_exc(),
                          "commit": env_commit()})
    finally:
        transport(root)
    summary = json.loads((root / "json" / "summary.json").read_text())
    if summary["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
