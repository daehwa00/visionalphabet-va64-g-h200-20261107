# ruff: noqa: ANN001, ANN201, ANN202
"""Screen the Wave 74 tiles with identical interleaved inputs and graph timings."""

import argparse
import ctypes as ct
import json
import os
from statistics import median

import bench_wave66_split_bf16 as bench
import torch

from lnet import w72_split_fp8_scan as baseline
from lnet import w74_dq_tile_scan as core

OPS = ("HF", "VF", "VB", "HB")
CONTROL = dict.fromkeys(OPS, "w72")


def operators(inputs, choice, coarse):
    pole, r, i, aux, path, dy = inputs

    def hf():
        return baseline.horizontal(pole, r, i)

    def vf(q):
        return baseline.vertical(pole, r, q, aux, path, coarse)

    def vb(q):
        arm = choice["VB"]
        if arm == "w72":
            return baseline.vertical(pole, r, q, aux, path, coarse, grad=dy)
        return core.vertical(pole, r, q, aux, path, coarse, grad=dy, arm=arm)

    def hb(q, v):
        return baseline.horizontal(pole, r, i, q, v[4], coarse, v[-1])

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
            rows[name] = {
                "cos": float(cos),
                "max_abs_diff": float((a - b).abs().max()),
                "rel_l2": float((a - b).norm() / an.clamp_min(1e-30)),
            }
        result[label] = {
            "min_cos": min(v["cos"] for v in rows.values()),
            "max_rel_l2": max(v["rel_l2"] for v in rows.values()),
            "max_abs_diff": max(v["max_abs_diff"] for v in rows.values()),
        }
    result["finite"] = all(bool(t.isfinite().all()) for t in actual)
    if not result["finite"]:
        raise ValueError("nonfinite candidate")
    return result


def graph_time(op, iters, warmup=2):
    for _ in range(warmup):
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


def occupancy(inputs, coarse, arm):
    _, r, _, _, _, (gr, gi) = inputs
    launch = core.plan_for(r, coarse, gr, gi, arm)
    key = launch.key
    device = r.device.index or 0
    _, h, w, m = r.shape
    size = baseline.shared_bytes(launch, h) if arm == "w72" else launch.plan.smem
    args = (device, key, w, h, m, size)
    driver, _, fn, _ = baseline._function(*args) if arm == "w72" else core._function(*args, arm)
    blocks = ct.c_int()
    core.scans._check_cu(
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ct.byref(blocks), fn, key.warps * 32, ct.c_size_t(size)
        ),
        "query VB occupancy",
    )
    return {"blocks_per_sm": blocks.value, "real_AW": blocks.value * key.warps}


def screen(label, h, coarse, batch, iters, warmup):
    data = reference(h, coarse, "normal")
    tiny = reference(h, coarse, "tiny") if label == "56d" else None
    inputs = large_inputs(h, coarse, batch)
    q = operators(inputs, CONTROL, coarse)[0]()
    rows = {}
    for arm in ("w72", *core.VARIANTS[label]):
        choice = CONTROL | {"VB": arm}
        _, _, vb, _, _, pipeline = operators(inputs, choice, coarse)
        row = {"parity": parity(data, choice, coarse), **occupancy(inputs, coarse, arm)}
        if tiny:
            row["tiny"] = parity(tiny, choice, coarse)
        for name, fn in (("VB", lambda vb=vb: vb(q)), ("pipeline", pipeline)):
            row[name] = graph_time(fn, iters, warmup)
            row[name]["delta_ms"] = row[name]["ms"] - rows.get("w72", row)[name]["ms"]
        rows[arm] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    os.environ.update(bench.lockin_arm_env("w57_lockin"))
    for label, h, _, _, coarse in bench.SHAPES[:4]:
        print(
            json.dumps(
                {label: screen(label, h, coarse, args.batch, args.iters, args.warmup)},
                allow_nan=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
