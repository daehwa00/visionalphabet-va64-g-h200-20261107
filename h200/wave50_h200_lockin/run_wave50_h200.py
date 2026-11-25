"""Run the Wave 50 Whole H200 full-model lock-in."""

import argparse
import os
import traceback
from pathlib import Path
from time import perf_counter

from h200.liverms_v2_opt import bench_h200
from h200.wave40_h200_bundle.run_wave40_h200 import BENCH, child, median
from h200.wave41_h200_bundle.bench_wave41_h200 import BASE_PRESET, VARIANT
from h200.wave41_h200_bundle.run_wave41_h200 import arm_passed
from h200.wave49_h200_lockin.run_wave49_h200 import measure_arm, transport

REF = "w43_ref"
CANDIDATES = ("w50_50a", "w50_50ac")
COMMON = {
    "LNET_SCAN_SGS": "strict_ro", "LNET_SCAN_VFWD": "batch4_w2", "LNET_SCAN_HSCAN": "bwd_opt",
    "LNET_SCAN_VSCAN_SAVED": "off", "LNET_STEM_STAGE0_MODE": "off", "VA_READER_OPT_MODE": "off",
    "LNET_MODEL_W36B": BASE_PRESET,
}
MODES = {
    REF: ("off", "off", "0"),
    CANDIDATES[0]: ("dense_nhwc", "off", "1"),
    CANDIDATES[1]: ("dense_nhwc", "fast_fwd", "1"),
}
ARM_MODULE = "h200.wave50_h200_lockin.run_wave50_h200"
TEST_FAILURES = ("failed", "errors", "skipped", "xfailed", "xpassed")
SCAN = "tests/test_general_cuda_scans.py"


def arm_env(mode: str) -> dict[str, str]:
    reader, norm, cudnn = MODES[mode]
    return {
        **COMMON,
        # The Wave 50 Reader entry point consumes EXEC_MODE.
        "LNET_READER_DENSE_NHWC": reader, "LNET_READER_EXEC_MODE": reader,
        "LNET_NORM_W50_MODE": norm, "LNET_CUDNN_BENCHMARK": cudnn,
    }


def verify_fold(root: Path, env: dict[str, str]) -> bool:
    fold = child(root, "fold_w50", [
        "-m", BENCH, "infer-fold", "--variant", bench_h200.LOCKIN_FOLD_ALIAS,
        "--batch", "2", "--precision", "highest", "--use-benchmark-env",
    ], 300, env=env)
    highest = fold.get("fp32_head_parity_by_precision", {}).get("highest", {})
    fold["passed"] = (
        fold["exit_code"] == 0 and fold["status"] == "ok" and fold.get("precision") == "highest"
        and fold.get("strict_reload_max_abs_logit_diff") == 0.0
        and fold.get("folded_train_guard") is True and highest.get("passed") is True
        and highest.get("max_abs_logit_diff", float("inf")) <= 1e-5
    )
    bench_h200.write_json(root / "fold_w50.json", fold)
    return fold["passed"]


def verify_tests(root: Path) -> dict[str, bool]:
    test_specs = (
        ("pytest_reader", 300, ["tests/test_factorized_reader_kernels.py",
                                "tests/test_reader_dense_nhwc.py"]),
        ("pytest_rmsnorm", 120, ["tests/test_stem_stage0_cuda_tiled.py", "-k", "cuda_norm"]),
        ("pytest_scan_regular", 300, [SCAN, "-m", "not cuda_compile and not cuda_matrix"]),
        ("pytest_scan_compile", 300, [SCAN, "-m", "cuda_compile"]),
    )
    rows = {}
    for name, limit, args in test_specs:
        row = child(root, name, ["-m", "pytest", "-q", "--tb=short", *args],
                    limit, tests=True)
        counts = row["counts"] or {}
        row["passed"] = (
            row["exit_code"] == 0 and counts.get("passed", 0) > 0
            and not any(counts.get(key, 0) for key in TEST_FAILURES)
        )
        row["timeout_s"] = limit
        bench_h200.write_json(root / f"{name}.json", row)
        rows[name] = row

    regular, compiled = (rows[name] for name in ("pytest_scan_regular", "pytest_scan_compile"))
    scan = {
        "status": "ok" if regular["passed"] and compiled["passed"] else "error",
        "passed": regular["passed"] and compiled["passed"],
        "wall_s": regular["wall_s"] + compiled["wall_s"],
        "counts": (
            {key: regular["counts"][key] + compiled["counts"][key]
             for key in bench_h200.COUNTS if key != "deselected"}
            if regular["counts"] and compiled["counts"] else None
        ),
    }
    bench_h200.write_json(root / "pytest_scan.json", scan)
    return {name: row["passed"] for name, row in {**rows, "pytest_scan": scan}.items()}


def benchmark(root: Path, summary: dict) -> None:
    device = summary["device"] = bench_h200.probe(argparse.Namespace())
    if not (
        "H200" in device["gpu"] and device["vram_gib"] >= 139
        and device["sm_count"] == 132 and device["capability"] == (9, 0)
        and device["python"] == "3.13.11" and device["torch"] == "2.9.1+cu128"
        and device["triton"] == "3.5.1" and device["torch_cuda"] == "12.8"
        and device["cudnn"] == 91002
    ):
        raise RuntimeError("Whole H200 or pinned environment verification failed")

    started = perf_counter()
    arms = {}
    for name in MODES:
        arms[name] = child(root, name, ["-m", ARM_MODULE, "--arm", "--profile"],
                           900, env=arm_env(name))
    eligible = [name for name in CANDIDATES if arm_passed(arms[name])]
    winner = min(eligible, key=lambda name: median(arms[name]), default=None)
    summary["winner"] = winner
    if winner:
        arms["w50_best_rev"] = child(root, "w50_best_rev", ["-m", ARM_MODULE, "--arm"],
                                     900, env=arm_env(winner))
    arms["w43_ref_rev"] = child(root, "w43_ref_rev", ["-m", ARM_MODULE, "--arm"],
                                900, env=arm_env(REF))
    summary["stage1_s"] = perf_counter() - started
    summary["arm_gates"] = {name: arm_passed(row) for name, row in arms.items()}
    summary["profiler_gates"] = {
        name: bool(arms[name].get("profiler", {}).get("all_kernels")) for name in MODES
    }

    started = perf_counter()
    env = arm_env(winner or CANDIDATES[0])
    summary["gates"] = {"fold_w50": verify_fold(root, env)}
    summary["gates"].update(verify_tests(root))
    summary["stage2_s"] = perf_counter() - started
    if not winner:
        return
    summary["performance_gates"] = {
        name: (arm_passed(arms[name]) and median(arms[name]) < 170.0
               and arms[name]["peak_vram_gib"] < 38.0)
        for name in (winner, "w50_best_rev")
    }
    summary["delta_ms"] = {
        name: median(arms[name]) - median(arms[ref])
        for name, ref in ((winner, REF), ("w50_best_rev", "w43_ref_rev"))
        if arm_passed(arms[name]) and arm_passed(arms[ref])
    }
    if all(all(summary[group].values()) for group in (
        "gates", "arm_gates", "performance_gates", "profiler_gates",
    )):
        summary["verdict"] = "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--out", dest="arm_out", type=Path)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.arm:
        return measure_arm(args.arm_out, profile=args.profile, arm_env_keys=tuple(arm_env(REF)))
    root = args.out.resolve()
    started = perf_counter()
    bootstrap_s = int(os.environ["VA_BOOTSTRAP_SECONDS"])
    summary = {
        "verdict": "FAIL", "commit": os.environ["VA_BUNDLE_COMMIT"], "bootstrap_s": bootstrap_s,
        "workload": {
            "model": bench_h200.LOCKIN_FOLD_ALIAS, "variant": VARIANT,
            "batch": 1024, "dtype": "bf16",
            "preset": BASE_PRESET, "warmup": 3, "count": 8,
        },
    }
    try:
        benchmark(root, summary)
    except Exception:  # noqa: BLE001 - transport partial results even on failure
        summary["error"] = traceback.format_exc()
    finally:
        summary["runner_wall_s"] = perf_counter() - started
        summary["total_wall_s"] = bootstrap_s + summary["runner_wall_s"]
        bench_h200.write_json(root / "summary.json", summary)
        transport(root, tag="VA_T179")
    return int(summary["verdict"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
