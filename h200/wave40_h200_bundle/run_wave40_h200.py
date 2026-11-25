"""Run the lean Wave 40 Whole H200 bundle and publish compact JSON artifacts."""
# ruff: noqa: S603, S607, T201

import argparse
import base64
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import traceback
from math import isfinite
from pathlib import Path
from time import perf_counter

from h200.liverms_v2_opt import bench_h200
from h200.wave40_h200_bundle.bench_wave40_h200 import MODES, PRESET, VARIANT

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path("/tmp/va_w40_torch_cache")  # noqa: S108 - shared within the job
BENCH = "h200.liverms_v2_opt.bench_h200"
ARM = "h200.wave40_h200_bundle.bench_wave40_h200"
BASE = "w33_base"
SGS = "w40a_sgs_win"
COMBO = "w40_best_combo"


def child(
    root: Path, name: str, args: list[str], limit: int, *,
    env: dict[str, str] | None = None, tests: bool = False,
) -> dict:
    target = root / f"{name}.json"
    log = root / f"{name}.log"
    started = perf_counter()
    with log.open("wb") as output, subprocess.Popen(
        [sys.executable, *args, *([] if tests else ["--out", str(target)])],
        cwd=ROOT, env={**os.environ, **(env or {})}, stdout=output,
        stderr=subprocess.STDOUT, start_new_session=True,
    ) as process:
        try:
            code = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    wall_s = perf_counter() - started
    if tests:
        result = bench_h200.wrap_pytest(argparse.Namespace(log=log, exit_code=code, elapsed=wall_s))
        row = {key: result[key] for key in ("counts", "summary", "failed_tests")}
    else:
        row = json.loads(target.read_text()) if target.exists() else {}
    row.update(exit_code=code, wall_s=wall_s)
    if code or row.get("status") == "error" or (not tests and len(row) == 2):
        row.update(status="error", log_tail=log.read_text(errors="replace")[-3000:])
    else:
        row["status"] = "ok"
    bench_h200.write_json(target, row)
    print(
        f"{name} {row['status']} compile_warmup_s={row.get('compile_warmup_s')} "
        f"timed_loop_s={row.get('timed_loop_s')} wall_s={wall_s:.3f}", flush=True,
    )
    return row


def selection(scan: dict) -> dict:
    if scan["status"] != "ok" or scan["batch"] != 1024 or scan["dtype"] != "bf16":
        raise RuntimeError("Missing the BF16 B=1024 scan screen")
    totals = scan["weighted_ms"]
    parity = scan["parity_pass"]
    shapes = scan["shapes"]
    labels = tuple(shapes["off"])
    if len(labels) != 7 or not parity["off"] or not all(
        type(parity[mode]) is bool and isfinite(totals[mode]) and set(shapes[mode]) == set(labels)
        for mode in MODES
    ):
        raise ValueError("Invalid scan measurements or parity evidence")
    best = min(
        (mode for mode in MODES if parity[mode] and totals["off"] - totals[mode] >= 0.15),
        key=totals.__getitem__, default="off",
    )
    if best != scan["best_mode"]:
        raise ValueError("Scan microbench selection disagrees with runner")
    oracle = {}
    for label in labels:
        eligible = [
            mode for mode in MODES if all(
                f"{dtype}/{label}" not in scan["parity_mismatches"][mode]
                for dtype in ("bf16", "fp16")
            )
        ]
        mode = min(eligible, key=lambda entry: shapes[entry][label]["median_ms"])
        oracle[label] = {"mode": mode, **shapes[mode][label]}
    weighted = sum(row["median_ms"] * row["calls"] for row in oracle.values())
    return {
        "best_mode": best, "weighted_ms": totals, "shapes": shapes,
        "parity_pass": parity, "parity_mismatches": scan["parity_mismatches"],
        "oracle": {
            "shapes": oracle, "weighted_ms": weighted,
            "delta_vs_off_ms": weighted - totals["off"],
            "delta_vs_best_ms": weighted - totals[best],
        },
    }


def benchmark(root: Path, mode: str) -> dict:
    arms = [(BASE, "off", "off")]
    if mode != "off":
        arms.append((SGS, mode, "off"))
    arms += [(COMBO, mode, PRESET), (COMBO + "_rev", mode, PRESET), (BASE + "_rev", "off", "off")]
    env = {
        "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "inductor"),
        "TRITON_CACHE_DIR": str(CACHE / "triton"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
    }
    return {
        name: child(root, name, ["-m", ARM, "--scan-sgs", sgs, "--preset", preset], 900, env=env)
        for name, sgs, preset in arms
    }


def successful_tests(row: dict, expected: int) -> bool:
    counts = row.get("counts") or {}
    return row["exit_code"] == 0 and counts.get("passed") == expected and not any(
        counts.get(key) for key in ("failed", "errors", "skipped", "xfailed", "xpassed")
    )


def verification(root: Path, mode: str) -> dict:
    env = {
        "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "stage3_inductor"),
        "TRITON_CACHE_DIR": str(CACHE / "triton"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "LNET_MODEL_W36B": "off", "LNET_SCAN_SGS": "off",
    }
    pytest = ["-m", "pytest", "-q", "--tb=short"]
    scan = child(root, "pytest_scan_sgs", [
        *pytest, "tests/test_general_cuda_scans.py", "-k", "sgs",
    ], 360, tests=True, env=env)
    model = child(root, "pytest_model", [*pytest, "tests/test_model_w36b.py"], 900,
                  tests=True, env=env)
    fold = child(root, "fold_combo", [
        "-m", BENCH, "infer-fold", "--variant", VARIANT, "--batch", "2", "--precision", "highest",
    ], 300, env={**env, "LNET_SCAN_SGS": mode, "LNET_MODEL_W36B": PRESET})
    highest = fold.get("fp32_head_parity_by_precision", {}).get("highest", {})
    fold_ok = (
        fold["exit_code"] == 0 and fold["status"] == "ok" and fold.get("precision") == "highest"
        and fold.get("strict_reload_max_abs_logit_diff") == 0.0
        and fold.get("folded_train_guard") is True and highest.get("passed") is True
    )
    return {
        "pytest_scan_sgs": {**scan, "passed": successful_tests(scan, 26)},
        "pytest_model": {**model, "passed": successful_tests(model, 23)},
        "fold_combo": {**fold, "passed": fold_ok},
    }


def median(row: dict) -> float | None:
    return row.get("step_ms", {}).get("median")


def comparisons(arms: dict) -> dict:
    deltas = {}
    for name, candidate, baseline in (
        ("forward", COMBO, BASE), ("reverse", COMBO + "_rev", BASE + "_rev"),
        ("sgs_only", SGS, BASE),
    ):
        current, reference = median(arms.get(candidate, {})), median(arms[baseline])
        deltas[name] = None if current is None or reference is None else current - reference
    return deltas


def arm_passed(row: dict) -> bool:
    step = median(row)
    return (
        row["status"] == "ok" and row["exit_code"] == 0 and row.get("loss_finite") is True
        and row.get("finite_grad_count") == row.get("grad_tensor_count") == 193
        and step is not None and isfinite(step) and step > 0
    )


def run(root: Path, summary: dict) -> None:
    device = child(root, "device", ["-m", BENCH, "probe"], 60)
    summary["device"] = device
    if not (
        device["status"] == "ok" and "H200" in device["gpu"] and device["vram_gib"] >= 139
        and device["sm_count"] == 132 and device["capability"] == [9, 0]
    ):
        raise RuntimeError("Whole H200 verification failed")
    started = perf_counter()
    scan = child(root, "scan_micro", [
        "scripts/bench_scan_micro.py", "--modes", ",".join(MODES), "--batch", "1024",
        "--dtype", "bf16", "--parity",
    ], 240)
    summary["stage1_s"] = perf_counter() - started
    summary["stage1"] = selection(scan)
    mode = summary["stage1"]["best_mode"]
    started = perf_counter()
    arms = benchmark(root, mode)
    summary.update(stage2_s=perf_counter() - started, stage2=arms, deltas_ms=comparisons(arms))
    started = perf_counter()
    gates = verification(root, mode)
    summary.update(stage3_s=perf_counter() - started, stage3=gates)
    combos = (arms[COMBO], arms[COMBO + "_rev"])
    speed = all(
        (median(row) or float("inf")) < 180 and row.get("img_per_s", 0) > 5690
        for row in combos
    )
    replay = all(row.get("graph_replays", {}).get(phase, 0) >= 1
                 for row in combos for phase in ("fwd", "bwd"))
    wins = all((summary["deltas_ms"][order] or 0) < 0 for order in ("forward", "reverse"))
    summary.update(record_target_met=speed, both_faster=wins, both_replay=replay)
    if speed and replay and wins and all(map(arm_passed, arms.values())) and all(
        gate["passed"] for gate in gates.values()
    ):
        summary["verdict"] = "PASS"


def transport(root: Path) -> None:
    members = sorted(path for path in root.glob("*.json") if path.name != "manifest.json")
    manifest = root / "manifest.json"
    bench_h200.write_json(manifest, {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in members
    })
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz", preset=9) as archive:
        for path in (*members, manifest):
            archive.add(path, arcname=path.name)
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode("ascii")
    print(f"VA_T142_SHA256 {hashlib.sha256(raw).hexdigest()}")
    print(f"VA_T142_BASE64_CHARS {len(encoded)}")
    print("VA_T142_BEGIN")
    for start in range(0, len(encoded), 240):
        print(f"VA_T142_XZ {encoded[start:start + 240]}")
    print("VA_T142_END", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    summary = dict.fromkeys(("stage1_s", "stage2_s", "stage3_s"))
    summary.update(verdict="FAIL", workload={"variant": VARIANT, "batch": 1024, "dtype": "bf16"})
    try:
        summary["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        CACHE.mkdir(parents=True, exist_ok=True)
        for key in ("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
                    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"):
            os.environ.pop(key, None)
        run(root, summary)
    except Exception:  # noqa: BLE001 - always publish any completed artifacts
        summary["error"] = traceback.format_exc()
    finally:
        summary["total_wall_s"] = perf_counter() - started
        bench_h200.write_json(root / "summary.json", summary)
        transport(root)
    return int(summary["verdict"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
