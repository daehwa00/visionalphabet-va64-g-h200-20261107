"""Type 1 vertical scan screening on a single H200."""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import os
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    from torch import Tensor, dtype

    from lnet.general_cuda_scans import Launch

MODES = {
    "w50_baseline": ("off", "off"),
    "51a_fp32": ("fp32", "off"),
    "51a_l2": ("l2", "off"),
    "51a_fp32_l2": ("fp32_l2", "off"),
    "51a_fp32_hiocc": ("fp32_hiocc", "off"),
    "51b_tiled_dw": ("off", "tiled_dw"),
    "51b_tiled_dw_fp32": ("off", "tiled_dw_fp32"),
}
DEFAULT_MODES = ("w50_baseline", "51b_tiled_dw", "51b_tiled_dw_fp32")
ODD_SHAPES = (("31x23x96c", 31, 23, 96, True, 1), ("17x19x160d", 17, 19, 160, False, 1))
MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=tuple(MODES))
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--parity-batch", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--max-rel-l2", type=float, default=0.035)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if min(args.batch, args.parity_batch, args.iters, args.max_rel_l2) <= 0:
        parser.error("Batch sizes, iteration count, and tolerance must be positive")
    return args


def configure(mode: str) -> None:
    wave_a, wave_b = MODES[mode]
    os.environ.update(
        LNET_SCAN_SGS="strict_ro",
        LNET_SCAN_VFWD="batch4_w2",
        LNET_SCAN_VSCAN_SAVED="off",
        LNET_SCAN_W51A=wave_a,
        LNET_SCAN_W51B=wave_b,
    )


def describe(scans: ModuleType, launch: Launch) -> dict:
    plan, key, workers, partials = launch
    driver, function = scans._function(0, key)
    registers = ct.c_int()
    scans._check_cu(
        driver.cuFuncGetAttribute(ct.byref(registers), 4, function),
        "cuFuncGetAttribute(NUM_REGS)",
    )
    return {
        "warps": plan.warps, "chunk": plan.chunk, "smem": plan.smem,
        "regs": registers.value, "active_warps_per_sm": scans._active_warps(0, launch),
        "workers": workers, "partials": partials,
        "dw1_tile": key.dw1_tile, "dw1_fp32": key.dw1_fp32,
        "fp32_path": getattr(key, "fp32_path", False),
        "l2_single_write": getattr(key, "l2_single_write", False) or bool(key.dw1_tile),
    }


def measure(torch: ModuleType, run: Callable, flush: Tensor, iters: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        flush.zero_()
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / (1024 * MIB)
    return {
        "ms": statistics.median(start.elapsed_time(end) for start, end in events),
        "graph_replay_finite": all(bool(t.isfinite().all()) for t in outputs),
        "peak_vram_gib": peak_gib,
    }


def max_rel_l2(torch: ModuleType, actual: Sequence[Tensor], expected: Sequence[Tensor]) -> float:
    errors = [
        ((a.float() - e).norm() / e.norm().clamp_min(1e-12))
        for a, e in zip(actual, expected, strict=True)
    ]
    return torch.stack(errors).amax().item()


def parity(
    torch: ModuleType, bench: ModuleType, shape: tuple, batch: int, dtype: dtype, limit: float,
) -> dict:
    scans = bench.scans
    _, h, w, m, coarse, _ = shape
    inputs = bench.make_inputs(batch, h, w, m, coarse, dtype)
    pole, states, gains, path, gr, gi = inputs
    fp_pole = [p.float().requires_grad_() for p in pole]
    fp_state = states.float().requires_grad_()
    fp_path = [p.float().requires_grad_() for p in path]
    ref_state, ref_gains = fp_state, gains
    if coarse and (h % 2 or w % 2):
        # Zero padding leaves the recurrence on existing columns and rows intact.
        pad = torch.nn.functional.pad
        ref_state = pad(fp_state, (0, 0, 0, w % 2, 0, h % 2))
        ref_gains = [pad(gains[0], (0, 0, 0, w % 2), value=1),
                     pad(gains[1], (0, 0, 0, h % 2), value=1)]
    fwd = scans._reference_vertical(fp_pole, ref_state, ref_gains, fp_path, coarse, 1e-6)
    fwd = tuple(t[:, :h // (1 + coarse), :w // (1 + coarse)] for t in fwd)
    grads = list(torch.autograd.grad(fwd, [*fp_pole, fp_state, *fp_path], (gr.float(), gi.float())))
    if coarse:
        # The compact result stores active positive and negative columns.
        x = 2 * (w // 2)
        gs = grads[4]
        grads[4] = torch.cat((gs[:2, ..., 1:x:2, :], gs[2:, ..., :x:2, :]))
    with torch.no_grad():
        def run() -> list[Tensor]:
            return scans.vertical_backward(*inputs, coarse, compact=coarse)

        eager = run()
        eager_fwd = scans.vertical_forward_packed(
            pole, states, gains, path, coarse, 1e-6, planar=False,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay = run()
        graph.replay()
        torch.cuda.synchronize()
        graph_ok = all(torch.equal(a, b) for a, b in zip(eager, replay, strict=True))
        bwd_error = max_rel_l2(torch, replay, grads)
        fwd_error = max_rel_l2(torch, eager_fwd.split(m, dim=-1), fwd)
    return {
        "max_rel_l2": max(bwd_error, fwd_error), "fwd_rel_l2": fwd_error,
        "bwd_rel_l2": bwd_error, "graph_matches_eager": graph_ok,
        "passed": graph_ok and bwd_error <= limit and fwd_error <= limit,
    }


def main() -> int:
    args = parse_args()
    # Keep --help usable on machines without Torch.
    import bench_scan_micro as bench
    import torch

    scans = bench.scans
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required")
    modes = tuple(dict.fromkeys(("w50_baseline", *args.modes)))
    missing = [mode for mode in modes if MODES[mode][0] not in getattr(scans, "W51A_MODES", ("off",))]
    if missing:
        raise SystemExit(f"Wave 51A kernels are required for these modes: {', '.join(missing)}")
    flush = torch.empty(120 * MIB, device="cuda", dtype=torch.uint8)
    report = {
        "device": torch.cuda.get_device_name(), "batch": args.batch,
        "parity_batch": args.parity_batch, "flush_mib": 120,
        "iterations": args.iters, "parity_max_rel_l2_limit": args.max_rel_l2, "modes": {},
    }
    baseline = 0.0
    for mode in modes:
        configure(mode)
        torch.manual_seed(51)
        rows = {}
        totals = dict.fromkeys(("fwd_ms", "bwd_ms"), 0.0)
        for shape in bench.MODEL_SHAPES:
            label, _, _, _, coarse, weight = shape
            row = rows[label] = {"calls": weight}
            for direction in ("fwd", "bwd"):
                run, plan = bench._case(
                    f"vscan_{direction}", shape, args.batch, torch.bfloat16,
                    compact=coarse and direction == "bwd",
                )
                launch = describe(scans, plan())
                metrics = measure(torch, run, flush, args.iters)
                row[direction] = {"launch": launch, **metrics}
                totals[f"{direction}_ms"] += weight * metrics["ms"]
                del run, plan
        totals["total_ms"] = totals["fwd_ms"] + totals["bwd_ms"]
        if mode == "w50_baseline":
            baseline = totals["total_ms"]
        totals["delta_vs_baseline_ms"] = totals["total_ms"] - baseline
        parity_rows = {}
        for name, dtype in bench.DTYPES.items():
            for shape in (*bench.MODEL_SHAPES, *ODD_SHAPES):
                torch.manual_seed(51)
                parity_rows[f"{name}/{shape[0]}"] = parity(
                    torch, bench, shape, args.parity_batch, dtype, args.max_rel_l2,
                )
        selected = [label for label, row in rows.items() if row["bwd"]["launch"]["dw1_tile"]]
        report["modes"][mode] = {
            "shapes": rows, "weighted": totals, "parity": parity_rows,
            "tiled_shapes": selected,
            "peak_vram_gib": max(row[d]["peak_vram_gib"] for row in rows.values()
                                 for d in ("fwd", "bwd")),
            "passed": (MODES[mode][1] == "off" or bool(selected))
            and all(row["passed"] for row in parity_rows.values())
            and all(row[d]["graph_replay_finite"] for row in rows.values() for d in ("fwd", "bwd")),
        }
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)
    return int(not all(mode["passed"] for mode in report["modes"].values()))


if __name__ == "__main__":
    sys.exit(main())
