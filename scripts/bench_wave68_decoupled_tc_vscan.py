# ruff: noqa: ANN001, ANN201, ANN202, E402, FBT003
"""Hopper benchmark and parity checks for the dense BF16 Wave 68 arm."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import bench_wave66_split_bf16 as base
from bench_scan_micro import _time

from lnet import w68_decoupled_tc_vscan as wave


def run_arm(op, dy):
    p = op.vf()
    out = op.forward(p)
    dp, params = op.backward(p, dy)
    ds, poles = op.vb(p, dp)
    return (out, *poles.unbind(), ds, *params)


def parity(h, scale=1.0):
    os.environ.update(base.lockin_arm_env("w57_lockin"))
    m = 160
    x, a, g, aux, path = base.reference_inputs(h, h, m)
    poles = [v.contiguous() for v in (a.real, a.imag, g.real, g.imag)]
    r, i = x.real.bfloat16(), x.imag.bfloat16()
    weights = [v.bfloat16() for v in path]
    states = base.fp.horizontal(poles, r, i)
    planes = base.unpack(states, m).float()
    vs = [torch.complex(*planes[j:j+2]).requires_grad_() for j in (0, 2)]
    ref = base.vertical(vs, a, g, aux, path, False)
    dy = (torch.randn_like(ref) * (0.05*scale)).bfloat16()
    grad = torch.autograd.grad(ref, (a, g, *vs, *path), dy.float())
    planar = torch.stack(dy.split(m, -1))
    ctrl = base.fp.vertical(poles, r, states, aux, weights, False, planar=True)
    vg = base.fp.vertical(poles, r, states, aux, weights, False, grad=planar.unbind())
    op = wave.Vertical(states, poles, aux, weights)
    actual = run_arm(op, planar)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = run_arm(op, planar)
    graph.replay()
    for v in replay:
        raw = v.view(torch.bfloat16) if v.dtype == torch.uint32 else v
        if not bool(raw.isfinite().all()):
            raise ValueError("nonfinite CUDA graph result")

    def grouped(out, ar, ai, gr, gi, ds, *params: torch.Tensor):
        return [out, torch.stack((ar, ai)), torch.stack((gr, gi)), base.unpack(ds, m), *params]

    expected = [torch.stack(ref.split(m, -1)), *(torch.stack((z.real, z.imag)) for z in grad[:2]),
                base.four(grad[2:4]), *grad[4:]]
    result = {}
    for label, reference in [("w66", grouped(ctrl, *vg)), ("fp32", expected)]:
        stats = {}
        for name, lhs, rhs in zip(("out", "v_da", "v_dg", "d_states", "dwi", "dbi", "dwo", "dbo"),
                                   reference, grouped(*actual), strict=True):
            row = base.metrics(lhs, rhs)
            if not (row["cos"] >= 0.9995 and row["rel_l2"] <= 0.035):
                raise ValueError(f"{label}/{h}/{scale}/{name}: {row}")
            stats[name] = row
        result[label] = stats
    return result


@torch.no_grad()
def benchmark(h, batch, warmup, iters):
    m = 160
    _, a, g, aux, path = base.reference_inputs(h, h, m)
    poles = [v.contiguous() for v in (a.real, a.imag, g.real, g.imag)]
    weights = [v.bfloat16() for v in path]
    shape = (batch, h, h, m)
    r = torch.randn(shape, device="cuda", dtype=torch.bfloat16).mul_(0.1)
    states = base.fp.horizontal(poles, r, r)
    dy = torch.randn((2, *shape), device="cuda", dtype=r.dtype).mul_(0.05)
    op = wave.Vertical(states, poles, aux, weights)

    def baseline_f():
        return base.fp.vertical(poles, r, states, aux, weights, False, planar=True)

    def baseline_b():
        return base.fp.vertical(poles, r, states, aux, weights, False, grad=dy.unbind())

    def baseline():
        return baseline_f(), baseline_b()

    def measure(fn):
        fn()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        value = fn()
        peak = torch.cuda.max_memory_allocated()-before
        del value
        return {"ms": _time(fn, warmup, iters), "peak_MiB": peak/2**20}

    control = {key: measure(fn) for key, fn in [("VF", baseline_f), ("VB", baseline_b), ("VF+VB", baseline)]}
    def forward():
        saved = op.vf()
        return saved, op.forward(saved)

    arm = {"VF+VB": measure(lambda: run_arm(op, dy)),
           "VF_total": measure(forward),
           "VF_pure": measure(op.vf)}
    p = op.vf()

    def backward():
        grad, params = op.backward(p, dy)
        return op.vb(p, grad), params

    arm["VB_total"] = measure(backward)
    arm["PathMLP_fwd"] = measure(lambda: op.forward(p))
    dp = torch.empty_like(p)
    part = p.new_zeros((min(op.stripes, 32), m, 90), dtype=torch.float32)
    arm["PathMLP_bwd"] = measure(lambda: wave.launch(
        "PathMLP_bwd", shape, op.stripes, [p, *op.path, dy, None, dp, part]))
    ds = torch.empty_like(states)
    pg = p.new_empty((op.cols*4, 4, m), dtype=torch.float32)
    arm["VB_pure"] = measure(lambda: wave.launch(
        "VB_pure", shape, op.cols, [*op.scan, p, dp, ds, pg]))
    for key, other in [("VF_total", "VF"), ("VB_total", "VB"), ("VF+VB", "VF+VB")]:
        cur, old = arm[key], control[other]
        cur.update(delta_ms=cur["ms"]-old["ms"], speedup=old["ms"]/cur["ms"],
                   delta_peak_MiB=cur["peak_MiB"]-old["peak_MiB"])
    return {base.fp.ARM: control, wave.ARM: arm}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in [("--batch", 1024), ("--warmup", 5), ("--iters", 15)]:
        parser.add_argument(flag, type=int, default=default)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    os.environ.update(base.lockin_arm_env("w57_lockin"))
    result = {f"{h}d": {"parity": parity(h), "timing": benchmark(h, args.batch, args.warmup, args.iters)}
              for h in (56, 28)}
    result["56d"]["tiny_grad"] = parity(56, 1e-5)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.json:
        args.json.write_text(payload+"\n")
    print(payload)
