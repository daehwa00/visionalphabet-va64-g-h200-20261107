# ruff: noqa: ANN001, ANN201, ANN202, ARG001, B023, C408, E402, E731, FBT003, PLW2901
"""Benchmark split-plane BF16 scans against Wave 57 on Hopper."""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from bench_scan_micro import MODEL_SHAPES, _time

from h200.bench_h200 import lockin_arm_env
from lnet import general_cuda_scans as scans
from lnet import w66_split_bf16_scan as fp

SHAPES = [
    (f"{h}{suffix}", h, h, 128 if h == 7 else 160, suffix == "c")
    for h in (56, 28, 14, 7)
    for suffix in ("d", "c")
    if h != 7 or suffix == "d"
]
OPS = [("HF", False, False), ("VF", True, False), ("VB", True, True), ("HB", False, True)]
VARIANTS = ("w57_lockin", fp.ARM)


def metrics(ref, val):
    a, b = ref.detach().double().flatten(), val.detach().double().flatten()
    return dict(
        cos=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        rel_l2=float((a - b).norm() / a.norm().clamp_min(1e-30)),
    )


def horizontal(x, a, g):
    out = []
    for reverse in (False, True):
        z = torch.zeros_like(x[:, :, 0])
        row = []
        coeff = a.conj() if reverse else a
        gain = g.conj() if reverse else g
        for t in range(x.shape[2]):
            k = x.shape[2] - 1 - t if reverse else t
            z = coeff * z + gain * x[:, :, k]
            row.append(z)
        out.append(torch.stack(row[::-1] if reverse else row, 2))
    return out


def vertical(states, a, g, aux, path, c, reverse_states=None):
    step = 1 + c
    width = states[0].shape[2] // step
    vals = []
    for rev in (False, True):
        p, n = reverse_states if rev and reverse_states is not None else states
        z = torch.stack([p[:, :, c : width * step : step], n[:, :, : width * step : step]], -1)
        acc = torch.zeros_like(z[:, 0])
        seq = []
        for t in range(z.shape[1]):
            y = z.shape[1] - 1 - t if rev else t
            acc = (a.conj() if rev else a)[:, None] * acc + (g.conj() if rev else g)[:, None] * z[
                :, y
            ]
            if y % step == (0 if rev else step - 1) and y // step < z.shape[1] // step:
                vx = torch.stack(
                    (aux[0][0, c : width * step : step], aux[0][1, : width * step : step]), -1
                )
                seq.append(acc * (vx * aux[1][int(rev), y, :, None]).clamp_min(1e-6).rsqrt())
        vals.append(torch.stack(seq[::-1] if rev else seq, 1))
    q = torch.cat(vals, -1)
    features = torch.cat([q.real, q.imag], -1)
    wi, bi, wo, bo = path
    act = torch.nn.functional.silu(torch.einsum("...mk,mjk->...mj", features, wi) + bi)
    both = torch.einsum("...mj,mkj->...mk", act, wo) + bo
    return torch.cat(both.unbind(-1), -1)


def four(states):
    return torch.stack((states[0].real, states[0].imag, states[1].real, states[1].imag))


def measure(a, b):
    if a.is_complex():
        a, b = torch.view_as_real(a.resolve_conj()), torch.view_as_real(b.resolve_conj())
    return metrics(a, b)


def reference_inputs(h, w, m, device="cuda"):
    torch.manual_seed(6224)

    def bf(x):
        return x.bfloat16().float().requires_grad_()

    real, imag = [bf(torch.randn(1, h, w, m, device=device) * 0.1) for _ in range(2)]
    x = torch.complex(real, imag).detach().requires_grad_()
    a = torch.full((m,), 0.73 + 0.13j, device=device, requires_grad=True)
    g = torch.full((m,), 0.21 + 0.09j, device=device, requires_grad=True)
    aux = [torch.rand(2, s, m, device=device) + 1 for s in (w, h)]
    path = [
        bf(torch.randn(s, device=device) * 0.125) for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))
    ]
    return x, a, g, aux, path


def timed(op, warmup, iters, name):
    op()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = op()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op()
    graph.replay()
    values = out if isinstance(out, (list, tuple)) else (out,)
    finite = True
    for t in values:
        if t is None:
            continue
        for chunk in t.reshape(-1).split(1 << 20):
            if chunk.dtype == torch.uint32:
                chunk = chunk.view(torch.bfloat16)
            if chunk.is_floating_point():
                finite = finite and bool(torch.isfinite(chunk).all())
    if not finite:
        raise RuntimeError("nonfinite graph output")
    return dict(ms=_time(op, warmup, iters), peak_bytes=peak, graph_finite=finite)



def unpack(words, m):
    b, t, _, w, h, _ = words.shape
    pairs = words.unsqueeze(-1).view(torch.bfloat16)
    return pairs.permute(2, 6, 0, 4, 3, 1, 5).reshape(4, b, h, w, t*32)[..., :m]

def precision(h, w, m, c, device="cuda", variant=fp.ARM):
    x, a, g, aux, path = reference_inputs(h, w, m, device)
    hs = horizontal(x, a, g)
    vs = [z.detach().requires_grad_() for z in hs]
    output = vertical(vs, a, g, aux, path, c)
    dy = (torch.randn_like(output) * 0.05).bfloat16()
    vr = torch.autograd.grad(output, (a, g, *vs, *path), dy.float())
    hb = torch.autograd.grad(hs, (x, a, g), vr[2:4])
    planes = four(hs)
    step = 1 + c
    dw = w // step

    def compact(raw):
        return torch.cat((raw[:2, :, :, c : dw * step : step], raw[2:, :, :, : dw * step : step]))

    refds = compact(four(vr[2:4]))
    r, i = x.real.bfloat16(), x.imag.bfloat16()
    poles = [z.contiguous() for z in (a.real, a.imag, g.real, g.imag)]
    weights = [v.bfloat16() for v in path]
    if variant == "w57_lockin":
        st = scans.horizontal_forward(poles, r, i)
        out = scans.vertical_forward_packed(poles, st, aux, weights, c, 1e-6, False)
        v = scans.vertical_backward(poles, st, aux, weights, *dy.split(m, -1), c, 1e-6, c)
        hg = scans.horizontal_backward(poles, [r, i], st, v[4], c)
        ds, params = v[4], v[5:]
    else:
        q = fp.horizontal(poles, r, i, variant)
        out = fp.vertical(poles, r, q, aux, weights, c, variant)
        v = fp.vertical(poles, r, q, aux, weights, c, variant, dy.split(m, -1))
        hg = fp.horizontal(poles, r, i, variant, q, v[4], c)
        ds = unpack(v[4], m)
        st = unpack(q, m)
        params = v[5:]
    dx = torch.complex(*(t.float() for t in hg[-1].split(m, -1)))
    if variant == fp.ARM:
        ref_st = scans.horizontal_forward(poles, r, i)
        ref_out = scans.vertical_forward_packed(poles, ref_st, aux, weights, c, 1e-6, False)
        ref_v = scans.vertical_backward(
            poles, ref_st, aux, weights, *dy.split(m, -1), c, 1e-6, c
        )
        ref_h = scans.horizontal_backward(poles, [r, i], ref_st, ref_v[4], c)
        checks = [
            (metrics(lhs, rhs)["rel_l2"], float((lhs.float()-rhs.float()).abs().max()), torch.allclose(lhs.float(), rhs.float(), rtol=2e-3, atol=2e-4))
            for lhs, rhs in zip(
                [ref_st, ref_out, *ref_v, *ref_h],
                [st, out, *v[:4], ds, *params, *hg],
                strict=True,
            )
        ]
    actual = [
        st,
        out,
        torch.complex(*v[:2]),
        torch.complex(*v[2:4]),
        ds,
        *params,
        dx,
        torch.complex(*hg[:2]),
        torch.complex(*hg[2:4]),
    ]
    expected = [planes, output, *vr[:2], refds, *vr[4:], *hb]
    names = ["states", "output", "v_da", "v_dg", "d_states", "dwi", "dbi", "dwo", "dbo", "dx", "h_da", "h_dg"]
    result = {k: measure(e, v) for k, e, v in zip(names, expected, actual, strict=True)}
    result["min_cos"] = min(v["cos"] for v in result.values() if isinstance(v, dict))
    result["max_rel_l2"] = max(v["rel_l2"] for v in result.values() if isinstance(v, dict))
    if variant == fp.ARM:
        result.update(control_rel_l2=max(v[0] for v in checks),control_max_abs=max(v[1] for v in checks),control_close=all(v[2] for v in checks))
    return result


def benchmark(batch, warmup, iters, model_shapes=MODEL_SHAPES, *, hf_variants=False):
    os.environ.update(lockin_arm_env("w57_lockin"))
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
            if variant == "w57_lockin":
                hf = lambda: scans.horizontal_forward(pole, r, i)
                vf = lambda s: scans.vertical_forward_packed(pole, s, aux, path, c, 1e-6, False)
                vb = lambda s: scans.vertical_backward(pole, s, aux, path, gr, gi, c, 1e-6, c)
                hb = lambda s, v: scans.horizontal_backward(pole, [r, i], s, v[4], c)
            else:
                hf = lambda: fp.horizontal(pole, r, i, variant)
                vf = lambda s: fp.vertical(pole, r, s, aux, path, c, variant)
                vb = lambda s: fp.vertical(pole, r, s, aux, path, c, variant, (gr, gi))
                hb = lambda s, v: fp.horizontal(pole, r, i, variant, s, v[4], c)

            def pipeline():
                s = hf()
                out = vf(s)
                v = vb(s)
                return [out, *v[:4], *v[5:], *hb(s, v)]

            s = hf()
            v = vb(s)
            rows[variant] = {
                op: timed(fn, warmup, iters, op)
                for op, fn in [
                    ("HF", hf),
                    ("VF", partial(vf, s)),
                    ("VB", partial(vb, s)),
                    ("HB", partial(hb, s, v)),
                ]
            }
            del s, v
            rows[variant]["pipeline"] = timed(pipeline, warmup, iters, "pipeline")
            for op, row in rows[variant].items():
                base = rows["w57_lockin"][op]
                row["delta_ms"] = row["ms"] - base["ms"]
                row["delta_peak_MiB"] = (row["peak_bytes"] - base["peak_bytes"]) / 2**20
                totals.setdefault(variant, {}).setdefault(op, 0.0)
                totals[variant][op] += count * row["ms"]
            rows[variant]["parity"] = precision(h, w, m, c, "cuda", variant)
        shapes[label] = dict(calls=count, arms=rows)
        if hf_variants and h == 56:
            def streaming():
                return fp.horizontal(pole, r, i, fp.HF_STREAMING)

            streaming()  # Compile before event timing.
            shapes[label]["hf_variant_ms"] = {
                "default": rows[fp.ARM]["HF"]["ms"],
                "stcs": _time(streaming, warmup, iters),
            }
    deltas = {
        v: {op: ms - totals["w57_lockin"][op] for op, ms in values.items()}
        for v, values in totals.items()
    }
    winners = {s: min(d["arms"], key=lambda arm: d["arms"][arm]["pipeline"]["ms"]) for s,d in shapes.items()}
    best = sum(d["calls"]*d["arms"][winners[s]]["pipeline"]["ms"] for s,d in shapes.items())
    hybrid = {s: fp.hybrid_arm(s) for s in shapes}
    hybrid_ms = sum(d["calls"]*d["arms"][hybrid[s]]["pipeline"]["ms"] for s,d in shapes.items())
    return dict(best_ms=best, best_arms=winners, hybrid_ms=hybrid_ms, hybrid_arms=hybrid, shapes=shapes, weighted_ms=totals, weighted_delta_ms=deltas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--hf-variants", action="store_true")
    args = parser.parse_args()
    result = benchmark(args.batch,args.warmup,args.iters,hf_variants=args.hf_variants)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.json:
        args.json.write_text(payload+"\n")
    else:
        print(payload)
