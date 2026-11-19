"""Replay the same live 56² operands with two- and four-token vertical prefixes."""

from functools import lru_cache
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import FunctionType

import triton
import triton.language as tl

from lnet.r6_segment import _segment_pair
from lnet.triton_product_scan_kernels import _compose_complex_pair


@triton.jit
def _split2(value, REVERSE: tl.constexpr):
    rows: tl.constexpr = value.shape[0] // 2
    modes: tl.constexpr = value.shape[2]
    first, second = tl.split(tl.permute(tl.reshape(value, (rows, 2, modes)), (0, 2, 1)))
    if REVERSE:
        return second, first
    return first, second


@triton.jit
def _join2(first, second, REVERSE: tl.constexpr):
    if REVERSE:
        return tl.reshape(tl.permute(tl.join(second, first), (0, 2, 1)),
                          (first.shape[0] * 2, 1, first.shape[1]))
    return tl.reshape(tl.permute(tl.join(first, second), (0, 2, 1)),
                      (first.shape[0] * 2, 1, first.shape[1]))


@triton.jit
def _prefix2(ar, ai, xr, xi, zr, zi, active, REVERSE: tl.constexpr, DOUBLING: tl.constexpr):
    modes: tl.constexpr = xr.shape[2]
    ar, ai = tl.reshape(ar, (1, modes)), tl.reshape(ai, (1, modes))
    x0, x1 = _split2(tl.where(active, xr, 0), REVERSE)
    y0, y1 = _split2(tl.where(active, xi, 0), REVERSE)
    z0, z1 = _split2(tl.where(active, zr, 0), REVERSE)
    w0, w1 = _split2(tl.where(active, zi, 0), REVERSE)
    xr1, xi1 = ar * x0 - ai * y0 + x1, ar * y0 + ai * x0 + y1
    zr1, zi1 = ar * z0 - ai * w0 + z1, ar * w0 + ai * z0 + w1
    p, q = ar * ar - ai * ai, ar * ai + ai * ar
    _, _, xr1, xi1, zr1, zi1 = tl.associative_scan(
        (tl.broadcast_to(p, xr1.shape), tl.broadcast_to(q, xr1.shape), xr1, xi1, zr1, zi1),
        0, _compose_complex_pair, reverse=REVERSE,
    )
    groups: tl.constexpr = xr1.shape[0]
    row = tl.arange(0, groups)[:, None]
    neighbor = row + 1 if REVERSE else row - 1
    gather = tl.broadcast_to(neighbor & (groups - 1), xr1.shape)
    valid = (neighbor >= 0) & (neighbor < groups)
    xc = tl.where(valid, tl.gather(xr1, gather, 0), 0.0)
    yc = tl.where(valid, tl.gather(xi1, gather, 0), 0.0)
    zc = tl.where(valid, tl.gather(zr1, gather, 0), 0.0)
    wc = tl.where(valid, tl.gather(zi1, gather, 0), 0.0)
    return (
        _join2(ar * xc - ai * yc + x0, xr1, REVERSE),
        _join2(ar * yc + ai * xc + y0, xi1, REVERSE),
        _join2(ar * zc - ai * wc + z0, zr1, REVERSE),
        _join2(ar * wc + ai * zc + w0, zi1, REVERSE),
    )


@triton.jit
def _prefix4(ar, ai, xr, xi, zr, zi, active, REVERSE: tl.constexpr, DOUBLING: tl.constexpr):
    return _segment_pair(
        ar, ai, tl.where(active, xr, 0), tl.where(active, xi, 0),
        tl.where(active, zr, 0), tl.where(active, zi, 0), REVERSE, False,
    )


def _bind(kernel, name, replacement):
    fn = kernel.fn
    shadow = FunctionType(fn.__code__, {**fn.__globals__, name: replacement}, fn.__name__,
                          fn.__defaults__, fn.__closure__)
    shadow.__annotations__ = fn.__annotations__.copy()
    shadow.__kwdefaults__ = fn.__kwdefaults__
    shadow.__module__ = __name__
    return triton.jit(shadow)


@lru_cache(maxsize=16)
def segmented_kernel(kernel, forward, rows):
    prefix = {2: _prefix2, 4: _prefix4}[rows]
    if forward:
        states = _bind(kernel.fn.__globals__["_scan_states"], "_prefix_pair", prefix)
        return _bind(kernel, "_scan_states", states)
    return _bind(kernel, "_prefix_pair", prefix)


def replay(kernel, case, named, grid, geometry, row, checkpoint):
    import torch
    from screen import benchmark, compare_output, compile_proof, output_buffers, outputs

    batch = 1024
    modes = named.get("M", named.get("modes"))
    names, atomics = outputs(case, named)
    source = named.get("SA", named.get("source_real_a", named.get("source_real")))
    if source.dtype != torch.bfloat16 or source.numel() != batch * 56**2 * modes:
        raise ValueError("Expected one whole-batch BF16 source plane")
    if len(grid) < 2 or not 0 < grid[1] <= 65535:
        raise ValueError(f"Invalid replay grid {grid}")
    working = output_buffers(named, names)
    flush = torch.zeros(128 << 20, device="cuda", dtype=torch.uint8)
    row.update(shape=[batch, 56, 56, modes], source_stride=list(source.stride()),
               dtype=str(source.dtype), trials=[])

    def reset(poison=False):
        for name in names:
            if name in atomics:
                working[name].zero_()
            elif poison:
                working[name].fill_(math.nan)

    def launch(fn, args, candidate_grid, warps, stages):
        params = {name: args[name] for name in fn.arg_names if name in args}
        return lambda: fn[candidate_grid](**params, num_warps=warps, num_stages=stages)

    base = launch(kernel, working, grid, geometry.num_warps, geometry.num_stages)

    def run(label, fn, args, launch_grid, warps, stages):
        trial = dict(variant=label, grid=launch_grid, warps=warps, stages=stages)
        row["trials"].append(trial)
        try:
            invoke = launch(fn, args, launch_grid, warps, stages)
            reset(poison=True)
            compiled = invoke()
            torch.cuda.synchronize()
            trial["code"] = compile_proof(compiled)
            trial["numeric"] = {
                n: compare_output(n, working[n], named[n], case, named, batch) for n in names
            }
            trial["numeric_pass"] = all(v["passed"] for v in trial["numeric"].values())
            trial["samples_us"] = benchmark(invoke, reset, flush, reps=7)
            trial["median_us"] = statistics.median(trial["samples_us"])
            if label != "r4":
                ref = benchmark(base, reset, flush, reps=5)
                trial["paired_baseline_us"] = ref
                trial["speedup"] = statistics.median(ref) / trial["median_us"]
        except Exception as exc:
            trial["error"] = f"{type(exc).__name__}: {str(exc)[-700:]}"
        checkpoint()

    run("r4", kernel, working, grid, geometry.num_warps, geometry.num_stages)
    if not row["trials"][0].get("numeric_pass"):
        row["status"] = "baseline_failure"
        return
    if case[0] == "H5":
        from lnet.r8_horizontal import r8_horizontal_forward
        from lnet.r10_horizontal import r10_hfwd56_seg4

        params = dict(working, BLOCK_MODES=32, BLOCK_LINES=1)
        run("r8", r8_horizontal_forward, params, (triton.cdiv(modes, 32), batch * 56), 4, 1)
        run("r10_seg4", r10_hfwd56_seg4, params, (batch * 56, triton.cdiv(modes, 32)), 4, 3)
    else:
        for rows in (2, 4):
            candidate = segmented_kernel(kernel, case[0] == "H3", rows)
            run(f"seg{rows}", candidate, working, grid, geometry.num_warps, geometry.num_stages)
            # Match the wide geometry investigated by the previous screen.
            params = dict(working, BM=8, GROUP=4)
            lines = batch * (56 // 2 if case[2] else 56)
            run(f"seg{rows}_bm8", candidate, params, (modes // 8 * 4, lines // 4), 4, 2)
    row["status"] = "complete" if not any("error" in t for t in row["trials"]) else "partial"


def install(batch):
    import lnet
    import torch
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from run_all import ROOT, save

    here = Path(__file__).resolve().parent
    sys.path.append(str(here.parent / "hopper_r8"))
    from screen import describe, fields

    lnet.__path__.append(str(ROOT / "current/src/lnet"))
    (ROOT / "proof").mkdir(exist_ok=True)
    end = int(os.environ["HOPPER_SCREEN_DEADLINE"])
    path = ROOT / "screen-vertical.json"
    doc = dict(batch=batch, cases={}, capture_errors={})

    def checkpoint():
        save(path, doc)

    original = CachingAutotuner.run

    def wrapped(operator, *args, **kwargs):
        output = original(operator, *args, **kwargs)
        name = operator.fn.__name__
        if (name not in ("_path_backward_kernel", "_path_forward_kernel", "_bidirectional_forward_kernel")
                or kwargs.get("benchmark_run") or time.time() >= end):
            return output
        try:
            named, grid, geometry = fields(operator, args, kwargs)
            case = describe(name, named)
        except Exception as exc:
            doc["capture_errors"][name] = str(exc)[-400:]
            checkpoint()
            return output
        if case is None:
            return output
        tag = "-".join(map(str, case))
        if tag in doc["cases"]:
            return output
        row = doc["cases"][tag] = dict(kernel=name, status="captured")
        checkpoint()
        try:
            with torch.no_grad():
                replay(operator.fn, case, named, grid, geometry, row, checkpoint)
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {str(exc)[-700:]}")
        checkpoint()
        print("SCREEN_CASE", tag, row["status"], flush=True)
        return output

    CachingAutotuner.run = wrapped
    checkpoint()
