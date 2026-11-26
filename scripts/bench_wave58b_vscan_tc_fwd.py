"""Graph-replay screening for the seven VA-Ablation-17 vertical scans."""

import argparse
import ctypes as ct
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from h200.bench_h200 import LOCKIN_ENV
from scripts.bench_scan_micro import MODEL_SHAPES, _time, make_inputs
from lnet import general_cuda_scans as scans
from lnet import vscan_tc_fwd as wave

BASE = "w57_lockin"


def run_scan(inputs, mode, coarse):
    args = (*inputs[:4], coarse, 1e-6)
    if mode == BASE:
        return scans.vertical_forward_packed(*args, planar=False)
    return wave.vscan_fwd(*args, mode=mode, packed=True)


def resources(mode, states, coarse):
    b, h, w, m = states.shape[1:]
    device = states.device.index
    if mode == BASE:
        launch = scans._plan_launch(
            states,
            (b, h, w, m),
            backward=False,
            vertical=True,
            hidden=8,
            precise=True,
            coarse=coarse,
            compact=False,
            interleaved=(True, True),
        )
        threads = launch.key.warps * 32
        grid = scans._ceil(m, 32) * scans._ceil(launch.workers, launch.key.warps)
        smem = launch.plan.smem
        drv, fn = scans._function(device, launch.key)
    else:
        threads, grid, smem = wave.config(mode, b, h, w, m, coarse)
        drv, fn = wave.function(device, mode, h, w, m, coarse, states.dtype)
    regs, resident = ct.c_int(), ct.c_int()
    scans._check_cu(drv.cuFuncGetAttribute(ct.byref(regs), 4, fn), "register count")
    scans._check_cu(
        drv.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ct.byref(resident), fn, threads, ct.c_size_t(smem)
        ),
        "occupancy",
    )
    return dict(
        regs=regs.value, smem=smem, grid=grid, active_warps_per_sm=resident.value * threads // 32
    )


def parity(shape):
    _, h, w, m, coarse, _ = shape
    results = {mode: {} for mode in (BASE, *wave.MODES)}
    for dtype in (torch.bfloat16, torch.float16):
        inputs = make_inputs(3, h, w, m, coarse, dtype)
        pole, states, aux, path = inputs[:4]
        refs = scans._reference_vertical(
            pole, states.float(), aux, [p.float() for p in path], coarse, 1e-6
        )
        for mode, row in results.items():
            parts = run_scan(inputs, mode, coarse).split(m, dim=-1)
            row[str(dtype)] = {
                name: ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
                for name, out, ref in zip(("real", "imag"), parts, refs, strict=True)
            }
    return results


def screen(shape, batch, iterations):
    name, h, w, m, coarse, count = shape
    errors = parity(shape)
    inputs = make_inputs(batch, h, w, m, coarse, torch.bfloat16)
    rows = []
    for mode in (BASE, *wave.MODES):

        def run():
            return run_scan(inputs, mode, coarse)

        run()
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            output = run()
        graph.replay()
        finite = all(chunk.isfinite().all().item() for chunk in output.split(8, dim=0))
        ms = _time(graph.replay, 3, iterations)
        # Algorithmic useful bytes: selected source planes once and final output.
        useful = batch * (w // (1 + coarse)) * m * (8 * h + 4 * (h // (1 + coarse)))
        rows.append(
            dict(
                shape=name,
                mode=mode,
                ms=ms,
                delta_ms=ms - (rows[0]["ms"] if rows else ms),
                calls=count,
                effective_hbm_GB_s=useful / (1e6 * ms),
                finite=finite,
                relative_l2=errors[mode],
                **resources(mode, inputs.states, coarse),
            )
        )
        del output, graph
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    os.environ.update(LOCKIN_ENV)
    torch.manual_seed(213)
    rows = [row for shape in MODEL_SHAPES for row in screen(shape, args.batch, args.iterations)]
    totals = {
        mode: sum(r["ms"] * r["calls"] for r in rows if r["mode"] == mode)
        for mode in (BASE, *wave.MODES)
    }
    report = dict(
        batch=args.batch,
        rows=rows,
        weighted_ms=totals,
        weighted_delta_ms={m: v - totals[BASE] for m, v in totals.items()},
    )
    data = json.dumps(report, indent=2, allow_nan=False)
    print(data)
    if args.json:
        args.json.write_text(data + "\n")
    if any(not row["finite"] for row in rows):
        raise RuntimeError("Nonfinite CUDA Graph replay")


if __name__ == "__main__":
    main()
