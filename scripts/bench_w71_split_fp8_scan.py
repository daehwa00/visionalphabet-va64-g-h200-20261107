# ruff: noqa: ANN001, ANN201, ANN202, C408
"""Wave 71 per-op graph, memory, latency and independent autograd comparison."""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bench_wave66_split_bf16 as base

from lnet import w66_split_bf16_scan as w66
from lnet import w71_split_fp8_scan as w71

ARMS = (w66.ARM, w71.BF16, w71.ARM)
NAMES = ("out", "v_da", "v_dg", "dwi", "dbi", "dwo", "dbo", "h_da", "h_dg", "dx")


def operations(arm, pole, r, i, aux, weights, c, dy):
    mod = w66 if arm == w66.ARM else w71
    hf = partial(mod.horizontal, pole, r, i, arm)
    vf = partial(mod.vertical, pole, r, aux=aux, path=weights, coarse=c, variant=arm)
    vb = partial(vf, grad=dy)

    def hb(q, v):
        extra = {} if mod is w66 else {"scales": v[-1]}
        return mod.horizontal(pole, r, i, arm, q, v[4], c, **extra)

    args = (pole, r, i, aux, weights, c, dy)
    pipeline = (
        partial(w66.pipeline, *args) if mod is w66 else partial(w71.pipeline, *args, variant=arm)
    )
    return hf, vf, vb, hb, pipeline


def precision(shape, factor):
    _, h, w, m, c = shape
    x, a, g, aux, path = base.reference_inputs(h, w, m)
    hs = base.horizontal(x, a, g)
    vs = [z.detach().requires_grad_() for z in hs]
    out = base.vertical(vs, a, g, aux, path, c)
    dy = (torch.randn_like(out) * 0.05 * factor).bfloat16()
    va, vg, p, n, *params = torch.autograd.grad(out, (a, g, *vs, *path), dy.float())
    dx, ha, hg = torch.autograd.grad(hs, (x, a, g), (p, n))
    refs = {"fp32": [out, va, vg, *params, ha, hg, dx]}
    poles = [z.contiguous() for z in (a.real, a.imag, g.real, g.imag)]
    results = {}
    for arm in ARMS:
        pipeline = operations(
            arm,
            poles,
            x.real.bfloat16(),
            x.imag.bfloat16(),
            aux,
            [t.bfloat16() for t in path],
            c,
            dy.split(m, -1),
        )[-1]
        y, ar, ai, gr, gi, *rest = pipeline()
        ar2, ai2, gr2, gi2, packed = rest[4:]
        actual = [
            y,
            torch.complex(ar, ai),
            torch.complex(gr, gi),
            *rest[:4],
            torch.complex(ar2, ai2),
            torch.complex(gr2, gi2),
            torch.complex(*(t.float() for t in packed.split(m, -1))),
        ]
        if arm == w66.ARM:
            refs[arm] = actual
        results[arm] = {}
        for name, ref in refs.items():
            stats = {k: base.measure(e, v) for k, e, v in zip(NAMES, ref, actual, strict=True)}
            results[arm][name] = dict(
                outputs=stats,
                min_cos=min(v["cos"] for v in stats.values()),
                max_rel_l2=max(v["rel_l2"] for v in stats.values()),
                finite=all(bool(torch.isfinite(t).all()) for t in actual),
            )
    return results


def timed(fn, warmup, iters):
    fn()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    sample = fn()
    peak = torch.cuda.max_memory_allocated() - before
    del sample
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    graph.replay()
    for value in outputs if isinstance(outputs, (tuple, list)) else (outputs,):
        if value is None:
            continue
        for chunk in value.flatten().split(1 << 20):
            decoded = chunk
            if chunk.dtype == torch.uint16:
                decoded = chunk.view(torch.float8_e4m3fn).float()
            elif chunk.dtype == torch.uint32:
                decoded = chunk.view(torch.bfloat16)
            if not torch.isfinite(decoded).all():
                raise RuntimeError("nonfinite graph output")
    del graph, outputs
    return {"ms": base._time(fn, warmup, iters), "peak_bytes": peak, "graph_finite": True}


def benchmark(shape, batch=1024, warmup=5, iters=15):
    _, h, w, m, c = shape
    opts = {"device": "cuda", "dtype": torch.bfloat16}
    r, i = torch.randn(batch, h, w, 2 * m, **opts).mul_(0.1).split(m, -1)
    dy = torch.randn(batch, h // (1 + c), w // (1 + c), 2 * m, **opts).mul_(0.05).split(m, -1)
    poles = [torch.full((m,), a, device="cuda") for a in (0.73, 0.13, 0.21, 0.09)]
    aux = [1 + torch.rand(2, n, m, device="cuda") for n in (w, h)]
    path = [torch.randn(s, **opts).mul_(0.125) for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    rows = {}
    for arm in ARMS:
        hf, vf, vb, hb, pipeline = operations(arm, poles, r, i, aux, path, c, dy)
        q = hf()
        v = vb(q)
        rows[arm] = {}

        def record(name, fn, arm=arm):
            try:
                result = timed(fn, warmup, iters)
            except RuntimeError as exc:
                rows[arm][name] = {"graph_finite": False, "error": str(exc)}
                return
            rows[arm][name] = result
            baseline = rows[w66.ARM][name]
            if "ms" in baseline:
                result.update(
                    delta_ms=result["ms"] - baseline["ms"],
                    speedup=baseline["ms"] / result["ms"],
                    delta_peak_MiB=(result["peak_bytes"] - baseline["peak_bytes"]) / 2**20,
                )

        record("HF", hf)
        record("VF", partial(vf, q))
        record("VB", partial(vb, q))
        record("HB", partial(hb, q, v))
        del q, v
        record("pipeline", pipeline)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    os.environ.update(base.lockin_arm_env("w57_lockin"))
    report = {s[0]: {"timing": benchmark(s), "parity": precision(s, 1)} for s in base.SHAPES[:4]}
    report["56d"]["tiny"] = precision(base.SHAPES[0], 1e-5)
    payload = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(payload + "\n")
    else:
        print(payload)
