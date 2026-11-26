"""Measure vertical Wave 57A scans against the Wave 55 lock-in."""

import argparse
import json
import os
from pathlib import Path

import bench_scan_micro as bench
import torch
from bench_wave51_vscan_type1 import MIB, describe, measure, parity_reference
from bench_wave55a_scans_type1 import configure as configure_base
from bench_wave56a_vscan_type1 import _scan_outputs

from lnet import general_cuda_scans as scans

ODD_SHAPES = (("15x17x33", 15, 17, 33, False, 1), *bench.EDGE_SHAPES)
VERTICAL = ("w55_lockin", *scans.W57A_MODES[1:5])


def configure(mode: str) -> None:
    configure_base("w55_lockin")
    os.environ["LNET_SCAN_W56A"] = "off"
    os.environ["LNET_SCAN_W57A"] = "off" if mode == "w55_lockin" else mode


def check_mode(mode: str, batch: int, limit: float = 0.035) -> dict:
    configure(mode)
    rows = {}
    for name, dtype in bench.DTYPES.items():
        for shape in (*bench.MODEL_SHAPES, *ODD_SHAPES):
            torch.manual_seed(56)
            inputs, outputs, gradients = parity_reference(torch, bench, shape, batch, dtype)
            _, _, _, m, coarse, _ = shape
            tolerance = min(limit, 0.01) if dtype == torch.float16 else limit
            with torch.no_grad():
                eager = _scan_outputs(inputs, m, coarse=coarse)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay = _scan_outputs(inputs, m, coarse=coarse)
                graph.replay()
                graph_ok = all(torch.equal(a, b) for a, b in zip(eager, replay, strict=True))
                errors = [
                    ((a.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
                    for a, ref in zip(replay, (*outputs, *gradients), strict=True)
                ]
            dx, dw = errors[6], max(errors[7:])
            rows[f"{name}/{shape[0]}"] = {
                "dx_rel_l2": dx,
                "dw_rel_l2": dw,
                "max_dx_dw_rel_l2": max(dx, dw),
                "max_rel_l2": max(errors),
                "graph_matches_eager": graph_ok,
                "passed": graph_ok and all(e <= tolerance for e in errors),
            }
    return rows


def benchmark(mode: str, batch: int, warmup: int, iters: int, flush: torch.Tensor) -> dict:
    configure(mode)
    rows = {}
    weighted_ms = 0.0
    for shape in bench.MODEL_SHAPES:
        name, h, w, m, coarse, calls = shape
        run, plan = bench._case("vscan_bwd", shape, batch, torch.bfloat16, compact=coarse)
        selected = plan()
        launch = describe(scans, selected)
        span = selected.key.chunk or h
        tail = span if selected.key.seg16 else (h - 1) % span + 1
        launch.update(chunk=selected.key.chunk, tail=tail, replayed_rows=h - tail)
        metrics = measure(torch, run, flush, warmup, iters)
        # Logical input/output volume excludes weights, replays and workspace.
        cells = batch * h * w * m
        scanned = batch * (h // (1 + coarse)) * (w // (1 + coarse)) * m
        grad_cells = batch * h * (w // (1 + coarse)) * m
        logical_bytes = 2 * (4 * cells + 2 * scanned + 4 * grad_cells)
        metrics["effective_hbm_GB_s"] = logical_bytes / (1e6 * metrics["ms"])
        rows[name] = {"calls": calls, "launch": launch, **metrics}
        weighted_ms += calls * metrics["ms"]
        del run, plan
    return {"shapes": rows, "weighted_ms": weighted_ms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes", nargs="+", choices=(*VERTICAL, "v57_bwd_seg16_loww"), default=VERTICAL
    )
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--parity-batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.parity_batch, args.warmup, args.iters) < 1:
        parser.error("Batch sizes and iteration counts must be positive")
    modes = dict.fromkeys(("w55_lockin", *args.modes))
    if not torch.cuda.is_available():
        if not args.selftest:
            parser.error("CUDA required")
        for mode in modes:
            configure(mode)
            expected = "off" if mode == "w55_lockin" else mode
            if scans._scan_mode("LNET_SCAN_W57A", scans.W57A_MODES) != expected:
                raise RuntimeError(f"Incorrect dispatch for {mode}")
        report = {"cpu_dispatch_config": "passed", "cuda": "skipped"}
    else:
        rows = {}
        report = {"batch": args.batch, "modes": rows}
        flush = None if args.selftest else torch.empty(120 * MIB, device="cuda", dtype=torch.uint8)
        for mode in modes:
            row = rows[mode] = (
                {} if args.selftest else benchmark(mode, args.batch, args.warmup, args.iters, flush)
            )
            row["parity"] = check_mode(mode, args.parity_batch)
            if not args.selftest:
                baseline = rows["w55_lockin"]
                row["weighted_delta_ms"] = row["weighted_ms"] - baseline["weighted_ms"]
                for name, shape in row["shapes"].items():
                    shape["delta_ms"] = shape["ms"] - baseline["shapes"][name]["ms"]
            row["passed"] = all(r["passed"] for r in row["parity"].values()) and all(
                shape["graph_replay_finite"] for shape in row.get("shapes", {}).values()
            )
    result = json.dumps(report, indent=2)
    print(result)
    if args.out:
        args.out.write_text(result + "\n")
    if not all(row["passed"] for row in report.get("modes", {}).values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
