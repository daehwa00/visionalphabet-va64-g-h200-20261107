"""Time Wave 53A scans and verify CUDA graph replay against reference results."""

import argparse
import json
import os
from pathlib import Path

import bench_scan_micro as bench
import torch
from bench_wave51_vscan_type1 import MIB, describe, measure, parity, parity_reference
from bench_wave52_vscan_type1 import configure as configure_w52

from lnet import general_cuda_scans as scans

VERTICAL = ("w51a_fp32", "tanh_gain_smemw", *scans.W53A_MODES[1:])
HORIZONTAL = ("bwd_opt", *scans.H53_MODES)
ODD_SHAPES = (bench.EDGE_SHAPES[0], bench.EDGE_SHAPES[1], bench.EDGE_SHAPES[4])
ODD_WIDTHS = tuple((f"w{w}", 7, w, 65, False, 1) for w in (49, 55, 57))


def configure(mode: str) -> None:
    configure_w52("tanh_gain_smemw" if mode == "tanh_gain_smemw" else "w51a_fp32")
    os.environ["LNET_SCAN_W53A"] = mode if mode in scans.W53A_MODES else "off"
    os.environ["LNET_SCAN_HSCAN"] = mode if mode in HORIZONTAL else "bwd_opt"


def horizontal_parity(
    mode: str, shape: tuple, batch: int, dtype: torch.dtype, limit: float
) -> dict:
    rows = {}
    for direction in ("fwd", "bwd"):
        op = f"hscan_{direction}"
        configure("bwd_opt")
        run, _ = bench._case(op, shape, batch, dtype, compact=shape[4] and direction == "bwd")
        references = run()
        configure(mode)
        eager = run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = run()
        graph.replay()
        for name, got, ref, immediate in zip(
            bench.TENSOR_NAMES[op], outputs, references, eager, strict=True
        ):
            diff = got.float() - ref.float()
            max_abs = diff.abs().amax().item()
            rel = (diff.norm() / ref.float().norm().clamp_min(1e-12)).item()
            rows[name] = {
                "max_abs": max_abs,
                "rel_l2": rel,
                "passed": torch.equal(got, immediate)
                and rel <= limit
                and max_abs <= limit * (1 + ref.abs().amax().item()),
            }
    return {"tensors": rows, "passed": all(row["passed"] for row in rows.values())}


def check_mode(mode: str, batch: int, limit: float) -> dict:
    configure(mode)
    vertical = mode in VERTICAL
    shapes = (*bench.MODEL_SHAPES, *(ODD_SHAPES if vertical else ODD_WIDTHS))
    rows = {}
    for dtype_name, dtype in bench.DTYPES.items():
        for shape in shapes:
            torch.manual_seed(53)
            if vertical:
                ref = parity_reference(torch, bench, shape, batch, dtype)
                result = parity(torch, bench, shape, ref, limit, {})
            else:
                result = horizontal_parity(mode, shape, batch, dtype, limit)
            rows[f"{dtype_name}/{shape[0]}"] = result
    return rows


def benchmark(mode: str, batch: int, warmup: int, iters: int, flush: torch.Tensor) -> dict:
    vertical = mode in VERTICAL
    configure(mode)
    rows = {}
    totals = {"fwd_ms": 0.0, "bwd_ms": 0.0}
    for shape in bench.MODEL_SHAPES:
        name, h, w, m, coarse, weight = shape
        calls = 3 if not vertical and name == "7d" else weight
        row = rows[name] = {"calls": calls}
        for direction in ("fwd", "bwd"):
            backward = direction == "bwd"
            op = ("vscan_" if vertical else "hscan_") + direction
            run, plan = bench._case(op, shape, batch, torch.bfloat16, compact=backward and coarse)
            launch = describe(scans, plan())
            warps = launch["warps"]
            launch.update(
                grid_ctas=scans._ceil(launch["workers"], warps) * scans._ceil(m, 32),
                resident_capacity=scans.device_limits(0)[0]
                * launch["active_warps_per_sm"]
                // warps,
            )
            metrics = measure(torch, run, flush, warmup, iters)
            if not vertical:
                # Logical BF16 tensor traffic per cell (excluding partials and scratch).
                fields = 6 if not backward else 6 + 4 // (1 + coarse)
                metrics["effective_hbm_GB_s"] = (
                    batch * h * w * m * 2 * fields / (1e6 * metrics["ms"])
                )
            row[direction] = {"launch": launch, **metrics}
            totals[direction + "_ms"] += calls * metrics["ms"]
            del run, plan
    return {"shapes": rows, "weighted": totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes", nargs="+", choices=VERTICAL + HORIZONTAL, default=VERTICAL + HORIZONTAL
    )
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--parity-batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--max-rel-l2", type=float, default=0.035)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.parity_batch, args.warmup, args.iters, args.max_rel_l2) <= 0:
        parser.error("Expected positive batch sizes, iteration counts and tolerance")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    report = {
        "device": torch.cuda.get_device_name(),
        "batch": args.batch,
        "warmup": args.warmup,
        "iters": args.iters,
        "parity_batch": args.parity_batch,
        "modes": {},
    }
    flush = None if args.selftest else torch.empty(120 * MIB, device="cuda", dtype=torch.uint8)
    for mode in args.modes:
        row = {} if args.selftest else benchmark(mode, args.batch, args.warmup, args.iters, flush)
        row["parity"] = check_mode(mode, args.parity_batch, args.max_rel_l2)
        row["passed"] = all(r["passed"] for r in row["parity"].values()) and all(
            r[d]["graph_replay_finite"]
            for r in row.get("shapes", {}).values()
            for d in ("fwd", "bwd")
        )
        report["modes"][mode] = row
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.output:
        print(json.dumps(report, indent=2))
    assert all(row["passed"] for row in report["modes"].values()), "Wave 53A parity failed"  # noqa: S101


if __name__ == "__main__":
    main()
