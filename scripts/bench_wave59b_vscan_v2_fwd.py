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
from lnet import vscan_v2_fwd as wave

BASE = "w57_lockin"


def run_scan(inputs, mode, coarse):
    args = (*inputs[:4], coarse, 1e-6)
    if mode == BASE:
        return scans.vertical_forward_packed(*args, planar=False)
    return wave.vscan_fwd(*args, mode=mode, packed=True)


def resources(mode, states, coarse, stage="scan"):
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
        if mode == wave.MODES[3]:
            b = wave.chunk_batch(b, h, w, m, coarse)
        threads, grid, smem = wave.config(mode, b, h, w, m, coarse, stage)
        drv, fn = wave.function(device, mode, h, w, m, coarse, states.dtype, stage)
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
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            output = run()
        peak = max(
            0, torch.cuda.max_memory_allocated() - before - output.numel() * output.element_size()
        )
        graph.replay()
        finite = all(chunk.isfinite().all().item() for chunk in output.split(8, dim=0))
        ms = _time(graph.replay, 3, iterations)
        # Algorithmic useful bytes: selected source planes once and final output.
        useful = batch * (w // (1 + coarse)) * m * (8 * h + 4 * (h // (1 + coarse)))
        rows.append(
            dict(
                shape=name,
                peak_workspace_mb=peak / 1e6,
                mlp_resources=resources(mode, inputs.states, coarse, "mlp")
                if mode in wave.MODES[2:]
                else None,
                mode=mode,
                ms=ms,
                delta_ms=ms - (rows[0]["ms"] if rows else ms),
                calls=count,
                logical_GB_s=useful / (1e6 * ms),
                finite=finite,
                relative_l2=errors[mode],
                **resources(mode, inputs.states, coarse),
            )
        )
        del output, graph
    return rows


def audit_case(case):
    import re
    import subprocess
    import tempfile

    mode, stage, shape, dtype = case
    name, h, w, m, coarse, _ = shape
    with tempfile.TemporaryDirectory() as directory:
        code = wave.source(mode, h, w, m, coarse, dtype, stage)
        cubin, log = wave.compile_cubin(code, directory)
        sass = subprocess.check_output(["cuobjdump", "--dump-sass", cubin], text=True)
    stack, stores, loads = map(
        int,
        re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log
        ).groups(),
    )
    regs = int(re.search(r"Used (\d+) registers", log)[1])
    smem = wave.config(mode, 1024, h, w, m, coarse, stage).smem
    warps = 4 * min(16, 233472 // (1024 + smem), 65536 // (128 * ((regs + 7) // 8 * 8)))
    return dict(
        mode=mode,
        stage=stage,
        shape=name,
        dtype=str(dtype),
        regs=regs,
        stack=stack,
        stores=stores,
        loads=loads,
        smem=smem,
        aw=warps,
        hmma=sass.count("HMMA.16816"),
        hfma2=sass.count("HFMA2"),
        vector_store="STG.E.128" in sass,
    )


def audit_cases():
    return [
        (m, s, sh, d)
        for m in wave.MODES
        for s in (("scan", "mlp") if m in wave.MODES[2:] else ("scan",))
        for sh in MODEL_SHAPES
        for d in (torch.bfloat16, torch.float16)
    ]


def verify(row):
    assert not any(row[k] for k in ("stack", "stores", "loads")), row
    assert bool(row["hmma"]) == (row["stage"] == "mlp" or row["mode"] == wave.MODES[1]), row
    mode, shape = row["mode"], row["shape"]
    if mode == wave.MODES[0]:
        assert row["hfma2"] and row["aw"] >= {"56d": 12, "56c": 24}.get(shape, 16), row
    if mode == wave.MODES[1]:
        assert row["smem"] <= 72 * 1024, row
        if shape in ("56c", "28d", "14d", "7d"):
            assert row["aw"] >= 24, row
        if shape == "56d":
            assert row["smem"] <= 65536 and row["aw"] >= 12, row
    if mode in wave.MODES[2:]:
        if row["stage"] == "scan":
            assert row["vector_store"], row
        else:
            assert row["smem"] <= 4096, row
        assert row["regs"] <= 48, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--offline-audit", action="store_true")
    args = parser.parse_args()
    if args.offline_audit:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(audit_case, audit_cases()))
        data = json.dumps(rows, indent=2)
        if args.json:
            args.json.write_text(data)
        print(data)
        for row in rows:
            verify(row)
        return
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
