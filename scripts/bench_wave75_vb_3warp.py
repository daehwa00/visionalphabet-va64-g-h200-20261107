# ruff: noqa: ANN001, ANN201, ANN202
"""Time captured Wave 74/75 vertical backward calls on the seven model layouts."""

import ctypes as ct
import json
import math
import os
from statistics import mean
from time import monotonic
from unittest.mock import patch

import bench_wave66_split_bf16 as bench
import torch

from lnet import w72_split_fp8_scan as fp


def occupancy(real, launch, width):
    key = launch.key
    smem = fp.shared_bytes(launch, width)
    driver, _, fn, _ = fp._function(real.device.index or 0, key, width, width, real.shape[-1], smem)
    blocks = ct.c_int()
    fp.scans._check_cu(
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ct.byref(blocks), fn, key.warps * 32, ct.c_size_t(smem)
        ),
        "VB occupancy",
    )
    return blocks.value * key.warps


def cosine(ref, val):
    if ref.dtype == torch.int16:
        ref = ref.view(torch.float8_e5m2)
        val = val.view(torch.float8_e5m2)
    sums = torch.zeros(3, device=ref.device, dtype=torch.float64)
    for left, right in zip(ref.flatten().split(1 << 23), val.flatten().split(1 << 23), strict=True):
        x, y = left.float(), right.float()
        sums += torch.stack((x.dot(y), x.dot(x), y.dot(y)))
    xy, xx, yy = sums.tolist()
    if not all(map(math.isfinite, (xy, xx, yy))):
        raise RuntimeError("nonfinite VB output")
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
    oh = h // (1 + coarse)
    grad = torch.randn(batch, oh, oh, 2 * m, device="cuda", dtype=r.dtype).mul_(0.05).split(m, -1)
    if h != 7:
        grad = tuple(t.contiguous() for t in grad)
    captures = {}
    rows = {}
    dispatch = fp._dispatch
    for arm in ("w74", *fp.VB_ARMS[label]):

        def observe(real, launch, width, *args: object, arm=arm):
            rows[arm] = {"real_AW": occupancy(real, launch, width)}
            dispatch(real, launch, width, *args)

        def run(arm=arm):
            return fp.vertical(pole, r, states, aux, path, coarse, grad, vb_arm=arm)

        with patch.object(fp, "_dispatch", observe):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = run()
        captures[arm] = graph, outputs

    for scale in (1.0, 1e-5):
        if scale != 1.0:
            for t in grad:
                t.mul_(scale)
        for graph, _ in captures.values():
            graph.replay()
        ref = captures["w74"][1]
        for arm, (_, actual) in captures.items():
            if arm == "w74":
                continue
            value = min(cosine(left, right) for left, right in zip(ref, actual, strict=True))
            if value < 0.9999:
                raise RuntimeError(f"{label}/{arm}/{scale} cosine {value}")
            rows[arm][f"cos_{scale:g}"] = value
    for t in grad:
        t.mul_(1e5)
    control = captures["w74"][0]
    savings = {"w74": 0.0}
    for arm in fp.VB_ARMS[label]:
        graph = captures[arm][0]
        a, b, c, d = (elapsed(g) for g in (control, graph, graph, control))
        reference, candidate = mean((a, d)), mean((b, c))
        savings[arm] = reference - candidate
        rows[arm].update(
            control_ms=reference, candidate_ms=candidate, saving_ms=savings[arm], graph=True
        )
    best = max(savings, key=savings.get)
    return {"arms": rows, "best": best, "saving_ms": savings[best]}


def main():
    start = monotonic()
    os.environ.update(bench.lockin_arm_env("w74_fp8_lockin"))
    torch.manual_seed(273)
    shapes = {s[0]: screen(s) for s in bench.SHAPES}
    saving = sum(row["saving_ms"] * (5 if label == "14d" else 1) for label, row in shapes.items())
    print(
        json.dumps(
            {
                "shapes": shapes,
                "weighted_saving_ms": saving,
                "go": saving >= 0.50,
                "elapsed_s": monotonic() - start,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
