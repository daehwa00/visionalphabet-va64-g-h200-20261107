"""Verify the Whole H200 device and benchmark result gates."""
# ruff: noqa: T201

import json
import sys
from pathlib import Path

ARMS = ("wave17b", "wave20b", "wave21c", "combined")


def read(out: Path, name: str) -> dict:
    return json.loads((out / f"{name}.json").read_text())


def whole_h200(row: dict) -> bool:
    return (
        row["status"] == "ok"
        and "H200" in row["gpu"]
        and row["sm_count"] == 132
        and row["vram_gib"] > 130
    )


def benchmarks_valid(out: Path) -> bool:
    cases = [(1024, arm) for arm in (*ARMS, "combined_rev", "wave17b_rev")]
    cases.extend((batch, arm) for batch in (128, 256, 512, 2048) for arm in ("wave17b", "combined"))
    valid = True
    for batch, arm in cases:
        row = read(out, f"b{batch}_{arm}")
        if batch == 2048 and row["status"] == "oom":
            continue
        passed = row["status"] == "ok" and row["finite_losses"] and row["finite_gradients"]
        if passed and batch == 1024 and arm in ARMS:
            profile = row["profiler"]
            passed = profile["general_cuda_scans_verified"] and bool(profile["kernels"])
        if not passed:
            print(f"Benchmark gate failed for B={batch} {arm}")
            valid = False
    return valid


def regressions_valid(out: Path) -> bool:
    graphs = read(out, "dynamo")["graphs"]
    dynamo = len(graphs) == 6 and all(
        set(modes) == {"train", "eval", "folded"} and all(v == [1, 0] for v in modes.values())
        for modes in graphs.values()
    )
    fold = read(out, "fold")
    folding = fold["status"] == "ok" and (
        fold["fp32_head_parity_by_precision"]["highest"]["passed"]
        and fold["strict_reload_max_abs_logit_diff"] == 0.0
        and fold["folded_train_guard"]
    )
    pytest = read(out, "pytest")
    cuda_tests = pytest["exit_code"] == 0 and pytest["counts"]["passed"] > 0
    print(f"Dynamo={dynamo} Fold={folding} CUDA pytest={cuda_tests}")
    return dynamo and folding and cuda_tests


def main() -> None:
    command, directory = sys.argv[1:]
    out = Path(directory)
    if command == "device":
        row = read(out, "device")
        print(json.dumps(row))
        if not whole_h200(row):
            raise SystemExit("Expected a whole H200")
    elif command == "gates":
        benchmarks = benchmarks_valid(out)
        regressions = regressions_valid(out)
        raise SystemExit(0 if benchmarks and regressions else 1)
    else:
        raise SystemExit(f"Unknown command {command}")


if __name__ == "__main__":
    main()
