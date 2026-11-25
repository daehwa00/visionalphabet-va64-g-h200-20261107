"""Benchmark Wave 52A fused vertical scans on H200, with graph replay parity."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from bench_wave51_vscan_type1 import (
    MIB,
    ODD_SHAPES,
    describe,
    measure,
    parity,
    parity_reference,
)
from bench_wave51_vscan_type1 import (
    configure as configure_wave51,
)

MODES = (
    "w51a_fp32", "tanh_gain", "tanh_gain_smemw", "l2_floor_g2", "l2_floor_sw",
    "combo_g2", "combo_sw",
)
PARITY_MODES = {"w51a_fp32", "tanh_gain_smemw", "combo_g2", "combo_sw"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--parity-batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--max-rel-l2", type=float, default=0.035)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.parity_batch, args.warmup, args.iters, args.max_rel_l2) <= 0:
        parser.error("Batch sizes, iterations, warmup and tolerance must be positive")
    return args


def configure(mode: str) -> None:
    configure_wave51("51a_fp32")
    os.environ["LNET_SCAN_W52A"] = "off" if mode == "w51a_fp32" else mode


def main() -> int:
    args = parse_args()
    import bench_scan_micro as bench
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required")
    scans = bench.scans
    flush = torch.empty(120 * MIB, device="cuda", dtype=torch.uint8)
    report = {
        "device": torch.cuda.get_device_name(), "batch": args.batch, "flush_mib": 120,
        "warmup": args.warmup, "iterations": args.iters, "parity_batch": args.parity_batch,
        "parity_max_rel_l2_limit": args.max_rel_l2, "modes": {},
    }
    references = {}
    parity_cache = {}
    baseline = 0.0
    for mode in dict.fromkeys(("w51a_fp32", *args.modes)):
        print(f"MODE START {mode}", flush=True)
        configure(mode)
        torch.manual_seed(52)
        rows = {}
        totals = dict.fromkeys(("fwd_ms", "bwd_ms"), 0.0)
        for shape in bench.MODEL_SHAPES:
            label, _, _, modes, coarse, weight = shape
            row = rows[label] = {"calls": weight}
            for direction in ("fwd", "bwd"):
                run, plan = bench._case(
                    f"vscan_{direction}", shape, args.batch, torch.bfloat16,
                    compact=coarse and direction == "bwd",
                )
                launch = describe(scans, plan())
                launch["grid_ctas"] = scans._ceil(launch["workers"], launch["warps"]) * (
                    scans._ceil(modes, 32)
                )
                launch["resident_capacity"] = (
                    scans.device_limits(0)[0] * launch["active_warps_per_sm"] // launch["warps"]
                )
                metrics = measure(torch, run, flush, args.warmup, args.iters)
                row[direction] = {"launch": launch, **metrics}
                totals[f"{direction}_ms"] += weight * metrics["ms"]
                del run, plan
        totals["total_ms"] = totals["fwd_ms"] + totals["bwd_ms"]
        if mode == "w51a_fp32":
            baseline = totals["total_ms"]
        delta = totals["total_ms"] - baseline
        totals["delta_vs_w51a_fp32_ms"] = delta
        totals["projected_full_step_ms"] = 150.53 + delta
        started = time.perf_counter()
        parity_rows = {}
        parity_shapes = (
            (*bench.MODEL_SHAPES, *ODD_SHAPES) if mode in PARITY_MODES
            else (bench.MODEL_SHAPES[0], *ODD_SHAPES)
        )
        for name, dtype in bench.DTYPES.items():
            for shape in parity_shapes:
                key = f"{name}/{shape[0]}"
                if key not in references:
                    torch.manual_seed(52)
                    references[key] = parity_reference(torch, bench, shape, args.parity_batch, dtype)
                parity_rows[key] = parity(
                    torch, bench, shape, references[key], args.max_rel_l2, parity_cache,
                )
        report["modes"][mode] = {
            "weighted": totals, "shapes": rows, "parity": parity_rows,
            "parity_wall_s": time.perf_counter() - started,
            "passed": all(row["passed"] for row in parity_rows.values())
            and all(row[d]["graph_replay_finite"] for row in rows.values() for d in ("fwd", "bwd")),
        }
        if args.output:
            checkpoint = args.output.with_suffix(".tmp")
            checkpoint.write_text(json.dumps(report, indent=2) + "\n")
            checkpoint.replace(args.output)
        print(f"MODE DONE {mode} {totals}", flush=True)
    if not args.output:
        print(json.dumps(report, indent=2))
    return int(not all(mode["passed"] for mode in report["modes"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
