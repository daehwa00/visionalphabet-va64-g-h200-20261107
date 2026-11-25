"""Run the Wave 41 Whole H200 verification and publish JSON artifacts."""
# ruff: noqa: S607, T201

import argparse
import base64
import hashlib
import io
import os
import subprocess
import tarfile
import traceback
from math import isfinite
from pathlib import Path
from time import perf_counter

from h200.liverms_v2_opt import bench_h200
from h200.wave40_h200_bundle.run_wave40_h200 import BENCH, ROOT, child, median
from h200.wave41_h200_bundle.bench_wave41_h200 import BASE_PRESET, MODES, VARIANT

CACHE = Path("/tmp/va_w41_torch_cache")  # noqa: S108
ARM = "h200.wave41_h200_bundle.bench_wave41_h200"
REF = "w40_ref_combo"
REF_REVERSE = REF + "_rev"
WIN_REVERSE = "w41_winner_rev"
HISTORICAL_MS = 186.299
RECORD_MS = 182.459
MODEL_FILTER = "replay or inductor_norms or aggressive_epilogue or multi_kernel or option"
LABELS = {"56d", "56c", "28d", "28c", "14d", "14c", "7d"}


def selection(scan: dict) -> dict:
    if scan["status"] != "ok" or scan["batch"] != 1024 or scan["dtype"] != "bf16":
        raise ValueError("Missing the BF16 B=1024 scan screen")
    totals, parity, shapes = (scan[key] for key in ("weighted_ms", "parity_pass", "shapes"))
    for mode in MODES:
        if set(shapes[mode]) != LABELS or not all(
            isfinite(ms) and ms > 0 for ms in (
                totals[mode], *(row["median_ms"] for row in shapes[mode].values())
            )
        ):
            raise ValueError(f"Invalid scan measurements for {mode}")
    best = "off"
    if parity["strict"] and totals["off"] - totals["strict"] >= 0.15:
        best = "strict"
    if (parity["strict_ro"] and totals["off"] - totals["strict_ro"] >= 0.15
            and totals["strict"] - totals["strict_ro"] >= 0.10):
        best = "strict_ro"

    mismatches = scan["parity_mismatches"]
    return {
        "best_sgs": best, "weighted_ms": totals, "shapes": shapes, "parity_pass": parity,
        "parity_mismatches": mismatches,
        "parity_by_dtype": {
            mode: {dtype: not any(item.startswith(dtype + "/") for item in mismatches[mode])
                   for dtype in ("bf16", "fp16")}
            for mode in MODES
        },
    }


def arm_passed(row: dict) -> bool:
    step_ms = median(row)
    return (
        row["status"] == "ok" and row["exit_code"] == 0 and row.get("loss_finite") is True
        and row.get("finite_grad_count") == row.get("grad_tensor_count") == 193
        and all(row.get("graph_replays", {}).get(phase, 0) >= 1 for phase in ("fwd", "bwd"))
        and step_ms is not None and isfinite(step_ms) and step_ms > 0
    )


def cache_environment(inductor: str) -> dict[str, str]:
    paths = {
        "TORCHINDUCTOR_CACHE_DIR": CACHE / inductor,
        "TRITON_CACHE_DIR": CACHE / "triton",
        "CUDA_CACHE_PATH": CACHE / "cuda",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {
        **{key: str(path) for key, path in paths.items()},
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
    }


def benchmark(root: Path, sgs: str, arms: dict) -> str:
    env = cache_environment("inductor")
    primary = [(REF, "strict", BASE_PRESET)]
    if sgs == "strict_ro":
        primary.append(("w41_scan_only", sgs, BASE_PRESET))
    norms = BASE_PRESET + ",inductor_norms"
    primary.extend([
        ("w41_inductor_norms", sgs, norms),
        ("w41_norms_aggr", sgs, norms + ",aggressive_epilogue"),
    ])

    def measure(name: str, mode: str, preset: str, *, profile: bool) -> None:
        args = ["-m", ARM, "--scan-sgs", mode, "--preset", preset]
        if profile:
            args.append("--profile")
        arms[name] = child(root, name, args, 900, env=env)

    for name, mode, preset in primary:
        measure(name, mode, preset, profile=True)
    winner = min(
        (name for name, _, _ in primary if arm_passed(arms[name])),
        key=lambda name: median(arms[name]), default=None,
    )
    if winner is None:
        raise RuntimeError("No valid primary arm")
    if winner != REF:
        row = arms[winner]
        measure(WIN_REVERSE, row["scan_sgs_mode"], row["preset"], profile=False)
    measure(REF_REVERSE, "strict", BASE_PRESET, profile=False)
    return winner


def pytest_gate(row: dict) -> dict:
    counts = row.get("counts") or {}
    passed = row["exit_code"] == 0 and counts.get("passed", 0) > 0 and not any(
        counts.get(key) for key in ("failed", "errors", "skipped", "xfailed", "xpassed")
    )
    return {**row, "passed": passed}


def verification(root: Path, winner: dict, gates: dict) -> None:
    env = {**cache_environment("stage3_inductor"), "LNET_MODEL_W36B": "off", "LNET_SCAN_SGS": "off"}
    pytest = ["-m", "pytest", "-q", "--tb=short"]
    for name, test, pattern, timeout in (
        ("pytest_scan_sgs", "test_general_cuda_scans.py", "sgs", 180),
        ("pytest_model_targeted", "test_model_w36b.py", MODEL_FILTER, 360),
    ):
        gates[name] = pytest_gate(child(
            root, name, [*pytest, f"tests/{test}", "-k", pattern], timeout, tests=True, env=env,
        ))
    fold_env = {
        **env, "LNET_SCAN_SGS": winner["scan_sgs_mode"], "LNET_MODEL_W36B": winner["preset"],
    }
    fold = child(root, "fold_combo", [
        "-m", BENCH, "infer-fold", "--variant", VARIANT, "--batch", "2", "--precision", "highest",
    ], 300, env=fold_env)
    highest = fold.get("fp32_head_parity_by_precision", {}).get("highest", {})
    passed = (
        fold["exit_code"] == 0 and fold["status"] == "ok" and fold.get("precision") == "highest"
        and fold.get("strict_reload_max_abs_logit_diff") == 0.0
        and fold.get("folded_train_guard") is True
        and highest.get("max_abs_logit_diff", float("inf")) <= 1e-5
    )
    gates["fold_combo"] = {**fold, "passed": passed}


def comparisons(arms: dict, winner: str) -> dict:
    def delta(name: str, baseline: str) -> float | None:
        current, reference = median(arms[name]), median(arms[baseline])
        return current - reference if current is not None and reference is not None else None

    reverse_winner = WIN_REVERSE if winner != REF else REF_REVERSE
    deltas = {}
    for name, row in arms.items():
        step_ms = median(row)
        deltas[name] = {
            "delta_vs_forward_ref_ms": delta(name, REF),
            "delta_vs_w33_base_ms": step_ms - HISTORICAL_MS if step_ms is not None else None,
        }
    return {
        "historical_w33_base_issue_836_ms": HISTORICAL_MS,
        "verified_record_ms": RECORD_MS,
        "verified_record_img_per_s": 5612.2,
        "forward_winner_delta_ms": delta(winner, REF),
        "reverse_winner_delta_ms": delta(reverse_winner, REF_REVERSE),
        "repeatability_delta_ms": {
            REF: delta(REF_REVERSE, REF), winner: delta(reverse_winner, winner),
        },
        "record_confirmed": all(
            (median(arms[name]) or float("inf")) <= RECORD_MS for name in (winner, reverse_winner)
        ),
        "arms": deltas,
    }


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
    ], 180, env=cache_environment("inductor"))
    summary["stage1_s"] = perf_counter() - started
    screen = summary["stage1"] = selection(scan)

    started = perf_counter()
    arms = summary["stage2"] = {}
    winner = summary["winner"] = benchmark(root, screen["best_sgs"], arms)
    summary["stage2_s"] = perf_counter() - started
    summary["comparisons"] = comparisons(arms, winner)
    summary["profiles"] = {}
    for name in dict.fromkeys((REF, winner)):
        row = arms[name]
        if profile := row.get("profiler"):
            summary["profiles"][name] = {**profile, "top_kernels": profile["top_kernels"][:15]}
        else:
            summary["profiles"][name] = {"error": row.get("profiler_error", "Profile unavailable")}

    started = perf_counter()
    gates = summary["stage3"] = {}
    verification(root, arms[winner], gates)
    summary["stage3_s"] = perf_counter() - started
    required_arms = {REF, winner, REF_REVERSE}
    if winner != REF:
        required_arms.add(WIN_REVERSE)
    if (all(screen["parity_pass"][mode] for mode in {"strict", screen["best_sgs"]})
            and all(arm_passed(arms[name]) for name in required_arms)
            and all(gate["passed"] for gate in gates.values())):
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
    print(f"VA_T145_SHA256 {hashlib.sha256(raw).hexdigest()}")
    print(f"VA_T145_MANIFEST_SHA256 {hashlib.sha256(manifest.read_bytes()).hexdigest()}")
    print(f"VA_T145_BASE64_CHARS {len(encoded)}")
    print("VA_T145_BEGIN")
    for start in range(0, len(encoded), 240):
        print(f"VA_T145_GZ {encoded[start:start + 240]}")
    print("VA_T145_END", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    root = parser.parse_args().out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    summary = dict.fromkeys(("stage1_s", "stage2_s", "stage3_s"))
    summary.update(verdict="FAIL", workload={"variant": VARIANT, "batch": 1024, "dtype": "bf16"})
    os.environ.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
    try:
        summary["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        run(root, summary)
    except Exception:  # noqa: BLE001 - publish completed artifacts if a stage exits early
        summary["error"] = traceback.format_exc()
    finally:
        summary["total_wall_s"] = perf_counter() - started
        bench_h200.write_json(root / "summary.json", summary)
        transport(root)
    return int(summary["verdict"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
