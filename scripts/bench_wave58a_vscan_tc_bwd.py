"""Screen four warp-cooperative backward kernels against the Wave 57 lock-in."""

import argparse
import ctypes as ct
import json
import os
from pathlib import Path

import torch

from bench_scan_micro import DTYPES, MODEL_SHAPES, _time, make_inputs
from lnet import general_cuda_scans as scans
from lnet import vscan_tc_bwd as tc

NAMES = ("ddr", "ddi", "dgr", "dgi", "gs", "dwi", "dbi", "dwo", "dbo")
BASE = "w57_lockin"


def configure_baseline(setenv):
    for k, v in {
        "LNET_SCAN_SGS": "strict_ro",
        "LNET_SCAN_W53A": BASE,
        "LNET_SCAN_W51A": "fp32",
        "LNET_GENERAL_SCAN_OPT": "auto",
    }.items():
        setenv(k, v)


def baseline(inputs, coarse, compact):
    return scans._vscan(*inputs[:4], coarse, 1e-6, False, (inputs.gr, inputs.gi), compact=compact)


def parity(inputs, mode, coarse, compact):
    ref_args = [
        [t.float().detach().requires_grad_() for t in inputs.pole],
        inputs.states.float().detach().requires_grad_(),
        inputs.gains,
        [t.float().detach().requires_grad_() for t in inputs.path],
    ]
    with torch.enable_grad():
        output = scans._reference_vertical(*ref_args, coarse, 1e-6)
        fp = list(
            torch.autograd.grad(
                output,
                [*ref_args[0], ref_args[1], *ref_args[3]],
                [inputs.gr.float(), inputs.gi.float()],
            )
        )
    if compact:
        fp[4] = torch.cat([fp[4][:2, :, :, 1::2], fp[4][2:, :, :, ::2]])
    base = baseline(inputs, coarse, compact)
    actual = base if mode == BASE else tc.vscan_tc_bwd(*inputs, coarse, compact, 1e-6, mode)
    return {
        name: {
            key: float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12))
            for key, x, y in zip(NAMES, actual, ref, strict=True)
        }
        for name, ref in (("fp32", fp), (BASE, base))
    }


def resources(inputs, coarse, compact, mode):
    b, h, w, m = inputs.states.shape[1:]
    device = inputs.states.device.index
    if mode == BASE:
        launch = scans._plan_launch(
            inputs.states,
            (b, h, w, m),
            backward=True,
            vertical=True,
            coarse=coarse,
            hidden=8,
            precise=False,
            compact=compact,
            interleaved=(True, True),
        )
        driver, fn = scans._function(device, launch.key)
        threads = 32 * launch.key.warps
        smem = launch.plan.smem
        grid = [scans._ceil(m, 32) * scans._ceil(launch.workers, launch.plan.warps), 1, 1]
    else:
        p = tc.plan(mode, (b, h, w, m), inputs.states.dtype, coarse, compact)
        mod = tc.module(device, p)
        driver, fn = mod.driver, mod.functions["wave58"]
        threads, smem, grid = 256, p.smem, p.grid(b, w, m)
    regs, ctas = ct.c_int(), ct.c_int()
    scans._check_cu(driver.cuFuncGetAttribute(ct.byref(regs), 4, fn), "registers")
    scans._check_cu(
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ct.byref(ctas), fn, threads, ct.c_size_t(smem)
        ),
        "occupancy",
    )
    return dict(
        regs=regs.value, smem=smem, active_warps_per_sm=ctas.value * threads // 32, grid=grid
    )


def benchmark(batch, warmup, iters):
    torch.manual_seed(212)
    configure_baseline(os.environ.__setitem__)
    modes = (BASE, *tc.MODES)
    totals = dict.fromkeys(modes, 0.0)
    rows = []
    for label, h, w, m, coarse, calls in MODEL_SHAPES:
        compact = coarse
        parity_inputs = {
            key: make_inputs(3, h, w, m, coarse, dtype) for key, dtype in DTYPES.items()
        }
        inputs = make_inputs(batch, h, w, m, coarse, torch.bfloat16)
        for mode in modes:

            def run():
                if mode == BASE:
                    return baseline(inputs, coarse, compact)
                return tc.vscan_tc_bwd(*inputs, coarse, compact, 1e-6, mode)

            ms = _time(run, warmup, iters, graph=True)
            if mode == BASE:
                base_ms = ms
            totals[mode] += calls * ms
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gradients = run()
            graph.replay()
            finite = all(bool(t.isfinite().all()) for t in gradients)
            # Effective bandwidth counts logical API input and output bytes.
            logical = sum(
                t.nbytes
                for t in (
                    *inputs.pole,
                    inputs.states,
                    *inputs.gains,
                    *inputs.path,
                    inputs.gr,
                    inputs.gi,
                    *gradients,
                )
            )
            rows.append(
                dict(
                    shape=label,
                    mode=mode,
                    calls=calls,
                    ms=ms,
                    delta_ms=ms - base_ms,
                    effective_hbm_GB_s=logical / ms / 1e6,
                    graph_finite=finite,
                    rel_l2={d: parity(v, mode, coarse, compact) for d, v in parity_inputs.items()},
                    **resources(inputs, coarse, compact, mode),
                )
            )
            del gradients, graph
        del inputs
    return dict(
        batch=batch,
        rows=rows,
        weighted_ms=totals,
        weighted_delta_ms={m: x - totals[BASE] for m, x in totals.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args()
    report = json.dumps(benchmark(args.batch, args.warmup, args.iters), indent=2, allow_nan=False)
    print(report)
    if args.json:
        args.json.write_text(report + "\n")
