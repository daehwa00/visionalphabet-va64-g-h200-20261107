"""Measure vertical Wave 56A scans against the Wave 55 lock-in."""

import argparse
import json
import os
from pathlib import Path

import bench_scan_micro as bench
import torch
from bench_wave51_vscan_type1 import MIB, describe, measure, parity_reference
from bench_wave55a_scans_type1 import ODD_SHAPES
from bench_wave55a_scans_type1 import configure as configure_base

from lnet import general_cuda_scans as scans

VERTICAL = ("w55_lockin", *scans.W56A_MODES[1:])


def configure(mode: str) -> None:
    configure_base("w55_lockin")
    os.environ["LNET_SCAN_W56A"] = "off" if mode == "w55_lockin" else mode


def _scan_outputs(inputs: tuple, modes: int, *, coarse: bool) -> tuple:
    pole, states, gains, path, _, _ = inputs
    forward = scans.vertical_forward_packed(pole, states, gains, path, coarse, 1e-6, planar=False)
    return (
        *forward.split(modes, dim=-1),
        *scans.vertical_backward(*inputs, coarse, compact=coarse),
    )


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
            dx, dw = errors[6:8]  # Two outputs, four poles, input state and input weights.
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
    totals = {"fwd_ms": 0.0, "bwd_ms": 0.0}
    for shape in bench.MODEL_SHAPES:
        name, h, w, m, coarse, weight = shape
        calls = weight
        row = rows[name] = {"calls": calls}
        for direction in ("fwd", "bwd"):
            backward = direction == "bwd"
            op = "vscan_" + direction
            run, plan = bench._case(op, shape, batch, torch.bfloat16, compact=backward and coarse)
            selected = plan()
            launch = describe(scans, selected)
            warps = launch["warps"]
            launch.update(
                grid_ctas=scans._ceil(launch["workers"], warps)
                * scans._ceil(m, 32 * (1 + selected.key.horizontal_vec2)),
                resident_capacity=scans.device_limits(0)[0]
                * launch["active_warps_per_sm"]
                // warps,
            )
            metrics = measure(torch, run, flush, warmup, iters)
            # Logical input/output volume excludes weights, replays and workspace.
            cells = batch * h * w * m
            scanned = batch * (h // (1 + coarse)) * (w // (1 + coarse)) * m
            logical_elements = 4 * cells + 2 * scanned
            if backward:
                logical_elements += 4 * batch * h * (w // (1 + coarse)) * m
            metrics["effective_hbm_GB_s"] = 2 * logical_elements / (1e6 * metrics["ms"])
            row[direction] = {"launch": launch, **metrics}
            totals[direction + "_ms"] += calls * metrics["ms"]
            del run, plan
    totals["total_ms"] = sum(totals.values())
    return {"shapes": rows, "weighted": totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=VERTICAL, default=VERTICAL)
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
            if scans._scan_mode("LNET_SCAN_W56A", scans.W56A_MODES) != expected:
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
                row["delta_ms"] = {
                    k: t - baseline["weighted"][k] for k, t in row["weighted"].items()
                }
                for shape, directions in row["shapes"].items():
                    for d in ("fwd", "bwd"):
                        directions[d]["delta_ms"] = (
                            directions[d]["ms"] - baseline["shapes"][shape][d]["ms"]
                        )
            row["passed"] = all(r["passed"] for r in row["parity"].values()) and all(
                r[d]["graph_replay_finite"]
                for r in row.get("shapes", {}).values()
                for d in ("fwd", "bwd")
            )
    result = json.dumps(report, indent=2)
    print(result)
    if args.out:
        args.out.write_text(result + "\n")
    if not all(row["passed"] for row in report.get("modes", {}).values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
