"""Time Wave 55A scans and verify CUDA graph replay against reference results."""

import argparse
import json
import os
from pathlib import Path

import bench_scan_micro as bench
import torch
from bench_wave51_vscan_type1 import MIB, describe, measure, parity, parity_reference
from bench_wave52_vscan_type1 import configure as configure_w52

from lnet import general_cuda_scans as scans

VERTICAL_SCREEN = ("w53_lockin", *scans.W55A_MODES[1:])
HORIZONTAL_SCREEN = ("h53_lockin", *scans.H55_MODES[1:])
VERTICAL = (*VERTICAL_SCREEN, "w55_lockin")
HORIZONTAL = (*HORIZONTAL_SCREEN, "h55_lockin")
ODD_SHAPES = (("15x17x33", 15, 17, 33, False, 1), *bench.EDGE_SHAPES[:2])
ODD_WIDTHS = ODD_SHAPES[:1]


def configure(mode: str) -> None:
    configure_w52("w51a_fp32")
    os.environ.update(
        LNET_SCAN_W53A=mode if mode == "w55_lockin" else "w53_lockin",
        LNET_SCAN_HSCAN=mode if mode == "h55_lockin" else "h53_lockin",
        LNET_SCAN_W54A="off",
        LNET_SCAN_H54="off",
        LNET_SCAN_W55A=mode if mode in scans.W55A_MODES else "off",
        LNET_SCAN_H55=mode if mode in scans.H55_MODES else "off",
    )


def horizontal_parity(
    mode: str, shape: tuple, batch: int, dtype: torch.dtype, limit: float
) -> dict:
    rows = {}
    for direction in ("fwd", "bwd"):
        op = f"hscan_{direction}"
        configure("h53_lockin")
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
            rows[f"{direction}/{name}"] = {
                "max_abs": max_abs,
                "rel_l2": rel,
                "passed": torch.equal(got, immediate)
                and rel <= limit
                and max_abs <= limit * (1 + ref.abs().amax().item()),
            }
    return {"tensors": rows, "passed": all(row["passed"] for row in rows.values())}


def unaligned_parity(mode: str, dtype: torch.dtype) -> bool:
    shape = (1, 15, 17, 32)
    elements = 15 * 17 * 32
    src = [torch.randn(elements + 1, device="cuda", dtype=dtype)[1:].view(shape) for _ in range(2)]
    pole = [torch.full((32,), x, device="cuda") for x in (0.73, 0.13, 0.21, 0.09)]
    configure("h53_lockin")
    states = scans.horizontal_forward(pole, *src)
    grad = torch.randn_like(states)
    ref = scans.horizontal_backward(pole, src, states, grad, compact=False)
    configure(mode)
    actual = scans.horizontal_forward(pole, *src)
    gradients = scans.horizontal_backward(pole, src, states, grad, compact=False)
    return torch.equal(actual, states) and all(
        torch.equal(g, r) for g, r in zip(gradients, ref, strict=True)
    )


def check_mode(mode: str, batch: int, limit: float, *, smoke: bool = False) -> dict:
    configure(mode)
    vertical = mode in VERTICAL
    shapes = (*bench.MODEL_SHAPES, *(ODD_SHAPES if vertical else ODD_WIDTHS))
    if smoke:
        shapes = (bench.MODEL_SHAPES[0], bench.MODEL_SHAPES[1], ODD_SHAPES[0])
    rows = {}
    for dtype_name, dtype in bench.DTYPES.items():
        tolerance = min(limit, 0.01) if dtype == torch.float16 else limit
        if not vertical and "vec2" in mode:
            rows[f"{dtype_name}/unaligned"] = {"passed": unaligned_parity(mode, dtype)}
        for shape in shapes:
            torch.manual_seed(53)
            if vertical:
                ref = parity_reference(torch, bench, shape, batch, dtype)
                result = parity(torch, bench, shape, ref, tolerance, {})
            else:
                result = horizontal_parity(mode, shape, batch, dtype, tolerance)
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
            if vertical:
                logical_elements = 4 * cells + 2 * scanned
                if backward:
                    logical_elements += 4 * batch * h * (w // (1 + coarse)) * m
            else:
                logical_elements = cells * (6 if not backward else 6 + 4 // (1 + coarse))
            metrics["effective_hbm_GB_s"] = 2 * logical_elements / (1e6 * metrics["ms"])
            row[direction] = {"launch": launch, **metrics}
            totals[direction + "_ms"] += calls * metrics["ms"]
            del run, plan
    totals["total_ms"] = sum(totals.values())
    return {"shapes": rows, "weighted": totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vscan-modes", nargs="*", choices=VERTICAL, default=VERTICAL_SCREEN)
    parser.add_argument("--hscan-modes", nargs="*", choices=HORIZONTAL, default=HORIZONTAL_SCREEN)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--parity-batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--max-rel-l2", type=float, default=0.035)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--out", "--output", dest="output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.parity_batch, args.warmup, args.iters, args.max_rel_l2) <= 0:
        parser.error("Expected positive batch sizes, iteration counts and tolerance")
    if not torch.cuda.is_available():
        if args.selftest:
            for mode in (*args.vscan_modes, *args.hscan_modes):
                configure(mode)
                scans._scan_mode("LNET_SCAN_W55A", scans.W55A_MODES)
                scans._scan_mode("LNET_SCAN_H55", scans.H55_MODES)
            result = json.dumps({"cpu_dispatch_config": "passed", "cuda": "skipped"})
            print(result)
            if args.output:
                args.output.write_text(result + "\n")
            return
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
    modes = dict.fromkeys(
        (
            *(("w53_lockin", *args.vscan_modes) if args.vscan_modes else ()),
            *(("h53_lockin", *args.hscan_modes) if args.hscan_modes else ()),
        )
    )
    for mode in modes:
        row = {} if args.selftest else benchmark(mode, args.batch, args.warmup, args.iters, flush)
        row["parity"] = check_mode(mode, args.parity_batch, args.max_rel_l2, smoke=args.selftest)
        row["passed"] = all(r["passed"] for r in row["parity"].values()) and all(
            r[d]["graph_replay_finite"]
            for r in row.get("shapes", {}).values()
            for d in ("fwd", "bwd")
        )
        report["modes"][mode] = row
        if not args.selftest:
            baseline = report["modes"]["w53_lockin" if mode in VERTICAL else "h53_lockin"][
                "weighted"
            ]
            row["delta_ms"] = {k: t - baseline[k] for k, t in row["weighted"].items()}
            print(mode, row["weighted"], row["delta_ms"], flush=True)
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.output:
        print(json.dumps(report, indent=2))
    assert all(row["passed"] for row in report["modes"].values()), "Wave 55A parity failed"  # noqa: S101


if __name__ == "__main__":
    main()
