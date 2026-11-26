# ruff: noqa: ANN001, ANN201, ANN202, B023, C408, E402, E731, FBT003
"""Benchmark small-resolution FP8 scans against Wave 57 on Hopper."""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from bench_scan_micro import _time
from bench_wave66_split_bf16 import horizontal, measure, reference_inputs, vertical

from h200.bench_h200 import lockin_arm_env
from lnet import general_cuda_scans as scans
from lnet import w73b_small_fp8_scan as fp

SHAPES = (
    ("14d", 14, 14, 160, False, 5),
    ("14c", 14, 14, 160, True, 1),
    ("7d", 7, 7, 128, False, 1),
)
VARIANTS = ("w57_lockin", "w73b_fp8")


def operations(variant, pole, r, i, aux, path, c, gr, gi):
    if variant == "w57_lockin":
        hf = lambda: scans.horizontal_forward(pole, r, i)
        vf = lambda s: scans.vertical_forward_packed(pole, s, aux, path, c, 1e-6, False)
        vb = lambda s: scans.vertical_backward(pole, s, aux, path, gr, gi, c, 1e-6, c)
        hb = lambda s, v: scans.horizontal_backward(pole, [r, i], s, v[4], c)
    else:
        hf = lambda: fp.horizontal(pole, r, i)
        vf = lambda s: fp.vertical(pole, r, s, aux, path, c)
        vb = lambda s: fp.vertical(pole, r, s, aux, path, c, (gr, gi))
        hb = lambda s, v: fp.horizontal(pole, r, i, s, v[4], c, v[-1])

    return hf, vf, vb, hb


def precision(h, w, m, c, factor=1.0):
    x, a, g, aux, path = reference_inputs(h, w, m)
    hs = horizontal(x, a, g)
    vs = [z.detach().requires_grad_() for z in hs]
    out = vertical(vs, a, g, aux, path, c)
    dy = (torch.randn_like(out) * (0.05 * factor)).bfloat16()
    vr = torch.autograd.grad(out, (a, g, *vs, *path), dy.float())
    hr = torch.autograd.grad(hs, (x, a, g), vr[2:4])
    expected = [out, *vr[:2], *vr[4:], *hr]
    controls = [z.contiguous() for z in (a.real, a.imag, g.real, g.imag)]

    saturation = {}

    def actual(variant):
        hf, vf, vb, hb = operations(
            variant,
            controls,
            x.real.bfloat16(),
            x.imag.bfloat16(),
            aux,
            [p.bfloat16() for p in path],
            c,
            *dy.split(m, -1),
        )
        q = hf()
        v = vb(q)
        if variant == "w73b_fp8":
            magnitude = v[4].view(torch.uint8).bitwise_and(0x7F)
            saturation["dq_max_code_count"] = int((magnitude == 0x7B).sum())
        b = hb(q, v)
        return [
            vf(q),
            torch.complex(*v[:2]),
            torch.complex(*v[2:4]),
            *v[5:9],
            torch.complex(*(z.float() for z in b[-1].split(m, -1))),
            torch.complex(*b[:2]),
            torch.complex(*b[2:4]),
        ]

    control, candidate = map(actual, VARIANTS)

    def compare(reference):
        rows = [measure(a, b) for a, b in zip(reference, candidate, strict=True)]
        finite = all(bool(torch.isfinite(t).all()) for t in [*reference, *candidate])
        return dict(
            min_cos=min(row["cos"] for row in rows),
            max_rel_l2=max(row["rel_l2"] for row in rows),
            finite=finite,
        )

    return dict(w57=compare(control), fp32=compare(expected), **saturation)


def timed(fn, warmup, iters):
    fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    peak = torch.cuda.max_memory_allocated() - base
    del result
    return dict(
        ms=_time(fn, warmup, iters, graph=True), peak_MiB=peak / 2**20, graph_capture_replay="PASS"
    )


def benchmark(batch, warmup, iters, model_shapes=SHAPES):
    os.environ.update(lockin_arm_env("w66_lockin"))
    shapes, totals = {}, {}
    for label, h, w, m, c, count in model_shapes:
        pole = [torch.full((m,), v, device="cuda") for v in (0.73, 0.13, 0.21, 0.09)]
        aux = [1 + torch.rand(2, n, m, device="cuda") for n in (w, h)]
        path = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16).mul_(0.125)
            for shape in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))
        ]
        gr, gi = (
            torch.randn(
                batch, h // (1 + c), w // (1 + c), 2 * m, device="cuda", dtype=torch.bfloat16
            )
            .mul_(0.05)
            .split(m, -1)
        )
        x = torch.randn(batch, h, w, 2 * m, device="cuda", dtype=torch.bfloat16).mul_(0.1)
        r, i = x.split(m, -1)
        rows = {}
        for variant in VARIANTS:
            hf, vf, vb, hb = operations(variant, pole, r, i, aux, path, c, gr, gi)

            def pipeline():
                s = hf()
                out = vf(s)
                v = vb(s)
                return [out, *v[:4], *v[5:9], *hb(s, v)]

            s = hf()
            v = vb(s)
            rows[variant] = {
                op: timed(fn, warmup, iters)
                for op, fn in [
                    ("HF", hf),
                    ("VF", partial(vf, s)),
                    ("VB", partial(vb, s)),
                    ("HB", partial(hb, s, v)),
                ]
            }
            del s, v
            rows[variant]["pipeline"] = timed(pipeline, warmup, iters)
            for op, row in rows[variant].items():
                base = rows["w57_lockin"][op]
                row["delta_ms"] = row["ms"] - base["ms"]
                row["delta_peak_MiB"] = row["peak_MiB"] - base["peak_MiB"]
                totals.setdefault(variant, {}).setdefault(op, 0.0)
                totals[variant][op] += count * row["ms"]
        shapes[label] = dict(calls=count, arms=rows, parity=precision(h, w, m, c))
        if label == "14d":
            shapes[label]["tiny_grad"] = precision(h, w, m, c, 1e-5)
    peak = sum(
        row["calls"] * row["arms"]["w73b_fp8"]["pipeline"]["delta_peak_MiB"]
        for row in shapes.values()
    )
    return dict(
        shapes=shapes,
        weighted_delta_ms=totals["w73b_fp8"]["pipeline"] - totals["w57_lockin"]["pipeline"],
        weighted_pipeline_peak_savings_MiB=-peak,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    result = benchmark(args.batch, args.warmup, args.iters)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.json:
        args.json.write_text(payload + "\n")
    else:
        print(payload)
