# ruff: noqa: ANN001, ANN201, ANN202
"""Time captured Wave 74/77 vertical forward calls on the seven model layouts."""

import ctypes as ct
import json
import math
import os
from statistics import mean
from time import monotonic
from unittest.mock import patch

import bench_wave66_split_bf16 as bench
import torch

from lnet import w77_vf_reg_scan as vf
from lnet.w77_vf_reg_scan import fp

PICKS = {
    **dict.fromkeys(("56c", "28c", "14d", "14c", "7d"), (vf.ALL,)),
    "56d": (vf.CACHED, vf.ALL),
    "28d": (vf.ALL, vf.HALF),
}


def occupancy(real, launch, width, **options: object):
    key = launch.key
    smem = options.get("smem", fp.shared_bytes(launch, width))
    driver, _, fn, _ = fp._function(
        real.device.index or 0,
        key,
        width,
        width,
        real.shape[-1],
        smem,
        options.get("source_fn", fp.kernel_source),
    )
    blocks = ct.c_int()
    fp.scans._check_cu(
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ct.byref(blocks), fn, key.warps * 32, ct.c_size_t(smem)
        ),
        "VF occupancy",
    )
    regs = ct.c_int()
    fp.scans._check_cu(driver.cuFuncGetAttribute(ct.byref(regs), 4, fn), "VF registers")
    return {"real_AW": blocks.value * key.warps, "regs": regs.value, "dynamic_smem": smem}


def cosine(ref, val):
    sums = torch.zeros(3, device=ref.device, dtype=torch.float64)
    for left, right in zip(ref.flatten().split(1 << 23), val.flatten().split(1 << 23), strict=True):
        x, y = left.float(), right.float()
        sums += torch.stack((x.dot(y), x.dot(x), y.dot(y)))
    xy, xx, yy = sums.tolist()
    if not all(map(math.isfinite, (xy, xx, yy))):
        return float("nan")
    return xy / math.sqrt(xx * yy) if xx and yy else float(xx == yy)


def elapsed(graph, repeats=3):
    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    a.record()
    for _ in range(repeats):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / repeats


def screen(shape, batch=1024):
    label, h, w, m, coarse = shape
    pole = [torch.full((m,), v, device="cuda") for v in (0.73, 0.13, 0.21, 0.09)]
    aux = [torch.rand(2, h, m, device="cuda") + 1 for _ in range(2)]
    path = [
        torch.randn(s, device="cuda", dtype=torch.bfloat16).mul_(0.125)
        for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))
    ]
    r, i = (
        torch.randn(batch, h, w, 2 * m, device="cuda", dtype=torch.bfloat16).mul_(0.1).split(m, -1)
    )
    states = fp.horizontal(pole, r, i)
    captures = {}
    rows = {}
    dispatch = fp._dispatch
    for arm in ("w74", *PICKS[label]):

        def observe(real, launch, width, *args: object, arm=arm, **options: object):
            rows[arm] = occupancy(real, launch, width, **options)
            dispatch(real, launch, width, *args, **options)

        def run(arm=arm):
            return vf.vertical(pole, r, states, aux, path, coarse, arm=arm, separate=h != 7)

        with patch.object(fp, "_dispatch", observe):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = run()
        captures[arm] = graph, outputs

    for graph, _ in captures.values():
        graph.replay()
    control = captures["w74"][0]
    savings = {"w74": 0.0}
    for arm in PICKS[label]:
        ref, cand = captures["w74"][1], captures[arm][1]
        scores = (
            [cosine(x, y) for x, y in zip(ref, cand, strict=True)]
            if h != 7
            else [cosine(ref, cand)]
        )
        rows[arm]["cos"] = min(scores) if all(map(math.isfinite, scores)) else None
        graph = captures[arm][0]
        a, b, c, d = (elapsed(g) for g in (control, graph, graph, control))
        reference, candidate = mean((a, d)), mean((b, c))
        savings[arm] = reference - candidate
        rows[arm].update(
            w74_ms=reference,
            cand_ms=candidate,
            delta_ms=-savings[arm],
            weighted_delta_ms=-savings[arm] * (5 if label == "14d" else 1),
            graph_pass=True,
        )
    valid = ["w74", *(a for a in PICKS[label] if (rows[a]["cos"] or 0) >= 0.9999)]
    best = max(valid, key=savings.__getitem__)
    return {"arms": rows, "best": best, "saving_ms": savings[best]}


def main():
    start = monotonic()
    os.environ.update(bench.lockin_arm_env("w74_fp8_lockin"))
    torch.manual_seed(273)
    shapes = {s[0]: screen(s) for s in bench.SHAPES}
    saving = sum(row["saving_ms"] * (5 if label == "14d" else 1) for label, row in shapes.items())
    duration = monotonic() - start
    print(
        json.dumps(
            {
                "shapes": shapes,
                "weighted_saving_ms": saving,
                "weighted_delta_ms": -saving,
                "go": saving >= 0.30 and duration < 90,
                "elapsed_s": duration,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
