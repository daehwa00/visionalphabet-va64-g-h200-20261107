# ruff: noqa: ANN001, ANN201, ANN202
"""Screen the Wave 73A operators with identical interleaved inputs and graph timings."""

import argparse
import json
import os
from statistics import median

import bench_wave66_split_bf16 as bench
import torch

from lnet import w72_split_fp8_scan as baseline
from lnet import w73a_fp8_core as core

OPS = ("HF", "VF", "VB", "HB")
CONTROL = dict.fromkeys(OPS, "w72")


def operators(inputs, choice, coarse):
    pole, r, i, aux, path, dy = inputs

    def module(op):
        return (baseline, {}) if choice[op] == "w72" else (core, {"arm": choice[op]})

    def hf():
        return baseline.horizontal(pole, r, i)

    def vf(q):
        mod, kw = module("VF")
        return mod.vertical(pole, r, q, aux, path, coarse, **kw)

    def vb(q):
        mod, kw = module("VB")
        return mod.vertical(pole, r, q, aux, path, coarse, grad=dy, **kw)

    def hb(q, v):
        mod, kw = module("HB")
        return mod.horizontal(pole, r, i, q, v[4], coarse, v[-1], **kw)

    def pipeline():
        q = hf()
        out = vf(q)
        v = vb(q)
        h = hb(q, v)
        dx = torch.complex(*(t.float() for t in h[-1].split(r.shape[-1], -1)))
        return [
            out,
            torch.complex(*v[:2]),
            torch.complex(*v[2:4]),
            *v[5:-1],
            dx,
            torch.complex(*h[:2]),
            torch.complex(*h[2:4]),
        ]

    # Keep output formatting out of timed regions.
    def launch_all():
        q = hf()
        out = vf(q)
        v = vb(q)
        return out, v, hb(q, v)

    return hf, vf, vb, hb, pipeline, launch_all


def reference(h, coarse, case):
    x, a, g, aux, path = bench.reference_inputs(h, h, 160)
    hs = bench.horizontal(x, a, g)
    vs = [s.detach().requires_grad_() for s in hs]
    out = bench.vertical(vs, a, g, aux, path, coarse)
    dy = (torch.randn_like(out) * 0.05).bfloat16()
    if case == "tiny":
        dy *= 1e-5
    if case == "zero_first_two":
        dy[:, :2] = 0
    vr = torch.autograd.grad(out, (a, g, *vs, *path), dy.float())
    hb = torch.autograd.grad(hs, (x, a, g), vr[2:4])
    with torch.no_grad():
        r, i = torch.cat((x.real, x.imag), -1).bfloat16().split(160, -1)
        poles = [v.detach().contiguous() for v in (a.real, a.imag, g.real, g.imag)]
        inputs = (poles, r, i, aux, [v.detach().bfloat16() for v in path], dy.split(160, -1))
        ref = [out.detach(), *vr[:2], *vr[4:], *hb]
        control = operators(inputs, CONTROL, coarse)[4]()
    return inputs, ref, control


def parity(data, choice, coarse):
    inputs, ref, control = data
    actual = operators(inputs, choice, coarse)[4]()
    result = {}
    names = ("out", "v_da", "v_dg", "dwi", "dbi", "dwo", "dbo", "dx", "h_da", "h_dg")
    for label, expected in (("w72", control), ("fp32", ref)):
        rows = {}
        for name, expected_tensor, actual_tensor in zip(names, expected, actual, strict=True):
            a, b = expected_tensor, actual_tensor
            if a.is_complex():
                a, b = (torch.view_as_real(t.resolve_conj()) for t in (a, b))
            a, b = (t.detach().double().flatten() for t in (a, b))
            an, bn = a.norm(), b.norm()
            cos = torch.where(an * bn > 0, (a @ b) / (an * bn), (an == bn).double())
            rows[name] = {"cos": float(cos), "rel_l2": float((a - b).norm() / an.clamp_min(1e-30))}
        result[label] = {
            "min_cos": min(v["cos"] for v in rows.values()),
            "max_rel_l2": max(v["rel_l2"] for v in rows.values()),
            "dwi": rows["dwi"],
        }
    result["finite"] = all(bool(t.isfinite().all()) for t in actual)
    if not result["finite"]:
        raise ValueError("nonfinite candidate")
    return result


def graph_time(op, iters):
    op()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op()
    graph.replay()
    times = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    peak = (torch.cuda.max_memory_allocated() - before) / (1 << 20)
    del output
    return {"ms": median(times), "peak_MiB": peak, "graph": "PASS"}


@torch.no_grad()
def large_inputs(h, coarse, batch):
    def rand(*shape: int):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    pole = [torch.full((160,), v, device="cuda") for v in (0.73, 0.13, 0.21, 0.09)]
    aux = [torch.rand(2, h, 160, device="cuda") + 1 for _ in range(2)]
    path = [rand(*s).mul_(0.125) for s in ((160, 8, 8), (160, 8), (160, 2, 8), (160, 2))]
    r, i = rand(batch, h, h, 320).mul_(0.1).split(160, -1)
    dy = rand(batch, h // (1 + coarse), h // (1 + coarse), 320).mul_(0.05).split(160, -1)
    return pole, r, i, aux, path, dy


def screen(label, h, coarse, batch, iters):
    data = reference(h, coarse, "normal")
    inputs = large_inputs(h, coarse, batch)
    hf, _, vb, *_ = operators(inputs, CONTROL, coarse)
    q = hf()
    v = vb(q)
    arms = {
        "VB": [a for a in core.VB if a != core.VB[1] or label == "56c"],
        "VF": [] if label == "28c" else core.VF,
        "HB": core.HB,
    }
    rows = {}
    winners = CONTROL.copy()

    def run(name, choice, targets):
        hf, vf, vb, hb, _, pipeline = operators(inputs, choice, coarse)
        isolated = {"HF": hf, "VF": lambda: vf(q), "VB": lambda: vb(q), "HB": lambda: hb(q, v)}
        row = {"choice": choice, "parity": parity(data, choice, coarse)}
        for target in targets:
            row[target] = graph_time(isolated[target], iters)
        row["pipeline"] = graph_time(pipeline, iters)
        for key in (*targets, "pipeline"):
            row[key]["delta_ms"] = row[key]["ms"] - rows.get("w72", row)[key]["ms"]
        rows[name] = row

    run("w72", CONTROL, OPS)
    for target, variants in arms.items():
        for arm in variants:
            run(arm, CONTROL | {target: arm}, (target,))
        winners[target] = min(("w72", *variants), key=lambda arm: rows[arm][target]["ms"])
    run("best", winners, OPS)
    if label == "56d":
        arm = core.VB[2]
        for case in ("tiny", "zero_first_two"):
            rows[arm][case] = parity(reference(h, coarse, case), CONTROL | {"VB": arm}, coarse)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()
    os.environ.update(bench.lockin_arm_env("w57_lockin"))
    for label, h, _, _, coarse in bench.SHAPES[:4]:
        print(
            json.dumps({label: screen(label, h, coarse, args.batch, args.iters)}, allow_nan=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
