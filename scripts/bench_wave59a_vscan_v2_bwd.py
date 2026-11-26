"""Screen four warp-cooperative backward kernels against the Wave 57 lock-in."""

import argparse
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import subprocess
import tempfile
from functools import cache, partial
from pathlib import Path

import torch

from bench_scan_micro import DTYPES, MODEL_SHAPES, _time, make_inputs
from lnet import general_cuda_scans as scans
from lnet import vscan_v2_bwd as tc

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
        _, _, h, w, _ = inputs.states.shape
        if coarse and (h % 2 or w % 2):
            pad = torch.nn.functional.pad
            extended = pad(ref_args[1], (0, 0, 0, w % 2, 0, h % 2))
            gains = [
                pad(t, (0, 0, 0, n % 2), value=1) for t, n in zip(inputs.gains, (w, h), strict=True)
            ]
            dense = scans._reference_vertical(ref_args[0], extended, gains, ref_args[3], True, 1e-6)
            output = tuple(t[:, : h // 2, : w // 2] for t in dense)
        else:
            output = scans._reference_vertical(*ref_args, coarse, 1e-6)
        fp = list(
            torch.autograd.grad(
                output,
                [*ref_args[0], ref_args[1], *ref_args[3]],
                [inputs.gr.float(), inputs.gi.float()],
            )
        )
    if compact:
        fp[4] = torch.cat(
            [fp[4][:2, :, :, 1::2], fp[4][2:, :, :, : 2 * (inputs.states.shape[3] // 2) : 2]]
        )
    base = baseline(inputs, coarse, compact)
    actual = base if mode == BASE else tc.vscan_v2_bwd(*inputs, coarse, compact, 1e-6, mode)
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
        stages = {}
        chunk = tc.chunk_size(mode, b, (h // (1 + coarse)) * (w // (1 + coarse)), m)
        size = (
            tc.padded(chunk * (h // (1 + coarse)) * (w // (1 + coarse))) * 36 * m
            if mode in tc.MODES[:2]
            else 0
        )
        for k in tc.kernels(mode, (b, h, w, m), inputs.states.dtype, coarse, compact):
            mod = tc.module(device, k)
            fn = mod.functions[k.name]
            ctas, regs = ct.c_int(), ct.c_int()
            scans._check_cu(mod.driver.cuFuncGetAttribute(ct.byref(regs), 4, fn), "registers")
            scans._check_cu(
                mod.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    ct.byref(ctas), fn, k.threads, ct.c_size_t(k.smem)
                ),
                "occupancy",
            )
            stages[k.name] = dict(
                regs=regs.value, smem=k.smem, active_warps_per_sm=ctas.value * k.threads // 32
            )
        return dict(stages=stages, chunk_workspace_MiB=size / (1 << 20))
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
            run = (
                partial(baseline, inputs, coarse, compact)
                if mode == BASE
                else partial(tc.vscan_v2_bwd, *inputs, coarse, compact, 1e-6, mode)
            )
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            sample = run()
            peak_workspace = max(
                0, torch.cuda.max_memory_allocated() - before - sum(t.nbytes for t in sample)
            )
            del sample
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
                    peak_workspace_MiB=peak_workspace / (1 << 20),
                    rel_l2={d: parity(v, mode, coarse, compact) for d, v in parity_inputs.items()},
                    **resources(inputs, coarse, compact, mode),
                )
            )
            del gradients, graph, run
        del inputs
    return dict(
        batch=batch,
        rows=rows,
        weighted_ms=totals,
        weighted_delta_ms={m: x - totals[BASE] for m, x in totals.items()},
    )


@cache
def audit(k):
    with tempfile.TemporaryDirectory() as directory:
        cubin, log = tc.compile_cubin(k, directory)
        sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(cubin)], text=True)
    if "for 'sm_90a'" not in log:
        raise RuntimeError("ptxas did not emit a native SM90a entry")
    regs = int(re.search(r"Used (\d+) registers", log)[1])
    mem = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log
    )
    warps = k.threads // 32
    reserved = (k.smem + 1024 + 255) // 256 * 256
    blocks = min(
        32, 64 // warps, 65536 // (warps * ((regs + 7) // 8) * 256), (228 << 10) // reserved
    )
    return dict(
        regs=regs,
        stack=int(mem[1]),
        stores=int(mem[2]),
        loads=int(mem[3]),
        smem=k.smem,
        aw=blocks * warps,
        hmma=sass.count("HMMA.16816"),
        cluster=k.cluster,
        block_barriers=len(re.findall(r"\bBAR\.SYNC", sass)),
        cluster_waits=sass.count("UCGABAR_WAIT"),
        shared_u16_loads=sass.count("LDS.U16"),
        shared_u16_stores=sass.count("STS.U16"),
    )


def passes_audit(stage, info):
    max_regs, min_warps = {
        "pure_vscan_fwd_p": (48, 1),
        "tc_path_mlp_bwd": (64, 48),
        "pure_vscan_bwd_gs": (64, 32),
        "fused": (128, 16),
    }[stage]
    uses_mma = stage in ("tc_path_mlp_bwd", "fused")
    return (
        max(info[key] for key in ("stack", "stores", "loads")) == 0
        and info["regs"] <= max_regs
        and info["aw"] >= min_warps
        and info["smem"] <= 227 * 1024
        and (info["hmma"] >= 3 if uses_mma else info["hmma"] == 0)
        and (stage != "tc_path_mlp_bwd" or (info["block_barriers"] == 0 and info["smem"] <= 4096))
        and (info["cluster"] == 1 or info["cluster_waits"] >= 3)
    )


def offline():
    cases = [
        (label, dtype, mode, k)
        for label, h, w, m, coarse, _ in MODEL_SHAPES
        for dtype in DTYPES.values()
        for mode in tc.MODES
        for k in tc.kernels(mode, (1024, h, w, m), dtype, coarse)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(audit, dict.fromkeys(k for _, _, _, k in cases)))
    return [
        dict(
            shape=label,
            dtype=str(dtype),
            mode=mode,
            stage=k.name,
            passed=passes_audit(k.name, audit(k)),
            **audit(k),
        )
        for label, dtype, mode, k in cases
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--offline-audit", action="store_true")
    args = parser.parse_args()
    result = offline() if args.offline_audit else benchmark(args.batch, args.warmup, args.iters)
    report = json.dumps(
        result,
        indent=2,
        allow_nan=False,
    )
    print(report)
    if args.json:
        args.json.write_text(report + "\n")
    if args.offline_audit and any(not row["passed"] for row in result):
        raise SystemExit(1)
