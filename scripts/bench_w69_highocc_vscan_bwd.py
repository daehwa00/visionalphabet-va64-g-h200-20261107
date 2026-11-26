# ruff: noqa: ANN001, ANN201
"""Event timing, graph replay and FP32 parity for fused Wave 69 scans."""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import bench_wave66_split_bf16 as ref
import torch

from lnet import w69_highocc_vscan_bwd as wave

BASE = ref.fp.ARM


def inputs(h, coarse, batch):
    m = 160
    x, a, g, aux, path = ref.reference_inputs(h, h, m)
    r = x.real.bfloat16().repeat(batch, 1, 1, 1)
    i = x.imag.bfloat16().repeat(batch, 1, 1, 1)
    pole = [t.contiguous() for t in (a.real, a.imag, g.real, g.imag)]
    weights = [t.bfloat16() for t in path]
    q = ref.fp.horizontal(pole, r, i)
    oh = h // (1 + coarse)
    dy = torch.randn(batch, oh, oh, 2 * m, device=r.device).mul_(0.05).bfloat16()
    return (pole, r, q, aux, weights, coarse), (a, g, path), dy


def operations(args, dy):
    grad = dy.split(160, -1)
    return {
        "w66_vf": partial(ref.fp.vertical, *args),
        BASE: partial(ref.fp.vertical, *args, grad=grad),
        **{arm: partial(wave.backward, *args, grad, arm) for arm in wave.ARMS},
    }


def canonical(out):
    if isinstance(out, torch.Tensor):
        return {"out": out}
    return dict(
        zip(
            ("v_da", "v_dg", "d_states", "dwi", "dbi", "dwo", "dbo"),
            (torch.complex(*out[:2]), torch.complex(*out[2:4]), ref.unpack(out[4], 160), *out[5:]),
            strict=True,
        )
    )


def reference(args, params, dy):
    _, _, q, aux, _, coarse = args
    a, g, path = params
    planes = ref.unpack(q, 160).float()
    states = [torch.complex(planes[i], planes[i + 1]).requires_grad_() for i in (0, 2)]
    out = ref.vertical(states, a, g, aux, path, coarse)
    da, dg, *rest = torch.autograd.grad(out, (a, g, *states, *path), dy.float())
    ds = ref.four(rest[:2])
    if coarse:
        ds = torch.cat((ds[:2, :, :, 1::2], ds[2:, :, :, ::2]))
    return {"out": out}, dict(
        zip(
            ("v_da", "v_dg", "d_states", "dwi", "dbi", "dwo", "dbo"),
            (da, dg, ds, *rest[2:]),
            strict=True,
        )
    )


def compare(expected, values):
    result = {}
    for key, tensor in values.items():
        lhs, rhs = expected[key], tensor
        if lhs.is_complex():
            lhs = torch.view_as_real(lhs.resolve_conj())
            rhs = torch.view_as_real(rhs.resolve_conj())
        a, b = lhs.double().flatten(), rhs.double().flatten()
        an, bn = a.norm(), b.norm()
        cos = float(torch.dot(a, b) / (an * bn)) if an * bn else float(bool(an == bn))
        result[key] = {
            "cos": cos,
            "rel_l2": float((a - b).norm() / an.clamp_min(1e-30)),
            "finite": bool(torch.isfinite(b).all()),
        }
    return {
        "outputs": result,
        "min_cos": min(v["cos"] for v in result.values()),
        "max_rel_l2": max(v["rel_l2"] for v in result.values()),
        "finite": all(v["finite"] for v in result.values()),
    }


def correctness(h, coarse, scale=1.0):
    args, params, dy = inputs(h, coarse, 1)
    dy = (dy.float() * scale).bfloat16()
    refs = reference(args, params, dy)
    ops = operations(args, dy)
    controls = [canonical(ops[k]()) for k in ("w66_vf", BASE)]
    results = {}
    for name, op in ops.items():
        op()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op()
        graph.replay()
        values = canonical(out)
        idx = int(name != "w66_vf")
        results[name] = {
            key: compare(exp, values) for key, exp in (("fp32", refs[idx]), ("w66", controls[idx]))
        }
    return results


def benchmark(h, coarse, batch, warmup, iters):
    args, _, dy = inputs(h, coarse, batch)
    rows = {}
    for name, op in operations(args, dy).items():
        op()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = op()
        peak = torch.cuda.max_memory_allocated() - before
        del output
        rows[name] = {"ms": ref._time(op, warmup, iters), "peak_bytes": peak}
        baseline = rows[BASE]["peak_bytes"] if name in wave.ARMS else peak
        rows[name]["delta_peak_MiB"] = (peak - baseline) / 2**20
    return rows


def rounded(obj):
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    return float(f"{obj:.5g}") if isinstance(obj, float) else obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    opts = parser.parse_args()
    os.environ.update(ref.lockin_arm_env("w57_lockin"))
    report = {}
    passed = True
    for name, h, _, _, coarse in ref.SHAPES[:4]:
        timing = benchmark(h, coarse, opts.batch, opts.warmup, opts.iters)
        checks = {"normal": correctness(h, coarse)}
        if name == "56d":
            checks["tiny"] = correctness(h, coarse, 1e-5)
        report[name] = {"timing": timing, "graph_parity": checks}
        passed &= all(v["delta_peak_MiB"] <= 16 for v in timing.values())
        passed &= all(
            v["finite"] and v["min_cos"] >= 0.999 and v["max_rel_l2"] <= 0.015
            for case in checks.values()
            for arm in case.values()
            for v in arm.values()
        )
    opts.out.write_text(json.dumps(rounded(report), separators=(",", ":")) + "\n")
    if not passed:
        raise SystemExit("Wave 69 validation gate failed")


if __name__ == "__main__":
    main()
