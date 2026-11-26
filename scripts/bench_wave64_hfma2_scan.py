"""Audit one SM90a shape offline or benchmark the vertical scan arms on H200."""

# ruff: noqa: E402, S101
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from bench_scan_micro import MODEL_SHAPES, _time

from h200.bench_h200 import lockin_arm_env
from lnet import general_cuda_scans as scans
from lnet import w64_hfma2_scan as wave


def offline_plans() -> Iterator[tuple[str, scans.Launch]]:
    with ExitStack() as stack:
        for name, fn in vars(scans).items():
            if name.startswith(("_plan_", "_launch_for_shape", "_lockin_")) and hasattr(
                fn, "cache_clear"
            ):
                fn.cache_clear()
                stack.callback(fn.cache_clear)
        stack.enter_context(patch.dict(os.environ, lockin_arm_env("w57_lockin")))
        limits = {
            "device_limits": lambda _: (132, 227 << 10, 64),
            "_sgs_device_limits": lambda _: (228 << 10, 65536, 1024),
            "_l2_size": lambda _: 50 << 20,
            "_scan_workspace_budget": lambda _: 128 << 20,
            "_resident": lambda _, k, smem: min(
                64 // k.warps, (228 << 10) // (1024 + smem), 2 if k.vertical and k.backward else 16
            ),
        }
        for name, fn in limits.items():
            stack.enter_context(patch.object(scans, name, fn))
        stack.enter_context(patch.object(torch.cuda, "get_device_capability", lambda _: (9, 0)))
        shape = (1024, 56, 56, 160)
        src = SimpleNamespace(
            dtype=torch.bfloat16,
            is_cuda=True,
            device=torch.device("cuda"),
            numel=lambda: 4 * 1024 * 56 * 56 * 160,
        )
        for op in ("VF", "VB"):
            yield (
                op,
                scans._plan_launch(
                    src,
                    shape,
                    backward=op == "VB",
                    vertical=True,
                    coarse=False,
                    hidden=8,
                    precise=op == "VF",
                    compact=False,
                    interleaved=(True, True),
                ),
            )


def resources(folder: Path, launch: scans.Launch, variant: str, name: str) -> dict:
    cubin = wave.compile_cubin(launch.key, variant, folder / (name + ".cu"))
    usage = subprocess.check_output(["cuobjdump", "-res-usage", cubin], text=True)
    sass = subprocess.check_output(["cuobjdump", "-sass", cubin], text=True)

    def field(name: str) -> int:
        return int(re.search(rf"\b{name}:(\d+)", usage)[1])

    regs = field("REG")
    smem = launch.plan.smem + field("SHARED")
    warps = launch.key.warps
    reg_unit = ((regs + 7) // 8) * 256
    blocks = min(
        32,
        64 // warps,
        (16384 // reg_unit) * 4 // warps,
        (228 << 10) // (((smem + 255) // 256) * 256),
    )
    return dict(  # noqa: C408
        name=name,
        regs=regs,
        dynamic=launch.plan.smem,
        static=field("SHARED"),
        smem=smem,
        AW=warps * blocks,
        stack=field("STACK"),
        LDL=len(re.findall(r"\bLDL\b", sass)),
        STL=len(re.findall(r"\bSTL\b", sass)),
        HFMA2=sass.count("HFMA2"),
    )


def check_56d() -> list[dict]:
    with tempfile.TemporaryDirectory() as folder, concurrent.futures.ThreadPoolExecutor(5) as pool:
        jobs = []
        for op, launch in offline_plans():
            for variant in (
                (wave.BASE, wave.VF) if op == "VF" else (wave.BASE, wave.SAME, wave.HIGH)
            ):
                name = variant if variant != wave.BASE else "w57_" + op.lower()
                jobs.append(
                    pool.submit(
                        resources, Path(folder), wave.plan(launch, 56, variant), variant, name
                    )
                )
        rows = [j.result() for j in jobs]
    for row in rows:
        print(json.dumps(row), flush=True)
        assert row["stack"] == row["LDL"] == row["STL"] == 0, row
        if row["name"].startswith("w64"):
            assert row["HFMA2"] > 0, row
        if row["name"] == wave.HIGH:
            assert row["AW"] >= 12, row
    return rows


def inputs(batch: int, h: int, w: int, m: int, coarse: bool) -> tuple:  # noqa: FBT001
    device = "cuda"
    pole = [torch.full((m,), v, device=device) for v in (0.73, 0.13, 0.21, 0.09)]
    states = torch.randn(4, batch, h, w, m, device=device, dtype=torch.bfloat16).mul_(0.1)
    aux = [torch.rand(2, n, m, device=device) + 1 for n in (w, h)]
    path = [
        torch.randn(s, device=device).mul_(0.125) for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))
    ]
    dy = torch.randn(
        batch, h // (1 + coarse), w // (1 + coarse), 2 * m, device=device, dtype=torch.bfloat16
    )
    return (pole, states, aux, path, coarse), dy


def finite_graph(fn: Callable) -> None:
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    for tensor in out if isinstance(out, list) else (out,):
        for chunk in tensor.reshape(-1).split(1 << 20):
            if not torch.isfinite(chunk).all():
                raise ValueError("nonfinite graph replay")


def parity(h: int, w: int, m: int, c: bool, *, tiny: bool = False, graph: bool = False) -> dict:  # noqa: FBT001
    torch.manual_seed(64229)
    args, dy = inputs(1, h, w, m, c)
    if tiny:
        dy.mul_(1e-5)
    grad = dy.split(m, -1)
    pole, states, aux, path, _ = args
    ref_inputs = [t.detach().float().requires_grad_() for t in [*pole, states, *path]]
    y = scans._reference_vertical(ref_inputs[:4], ref_inputs[4], aux, ref_inputs[5:], c, 1e-6)
    grads = torch.autograd.grad(y, ref_inputs, tuple(t.float() for t in grad))
    if c:
        ds = grads[4]
        grads = (*grads[:4], torch.cat((ds[:2, :, :, 1::2], ds[2:, :, :, ::2])), *grads[5:])
    result = {}
    for variant in (wave.VF, wave.SAME, wave.HIGH):
        fn = partial(
            wave.vertical,
            *args,
            variant,
            grad=None if variant == wave.VF else grad,
            compact=c and variant != wave.VF,
        )
        out = fn()
        if graph:
            finite_graph(fn)
        fields = []
        for ref, val in zip(
            [torch.cat(y, -1)] if variant == wave.VF else grads,
            [out] if variant == wave.VF else out,
            strict=True,
        ):
            a, b = ref.detach().flatten(), val.detach().float().flatten()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-30).item()
            rel = ((a - b).norm() / a.norm().clamp_min(1e-30)).item()
            if not cos >= 0.995 or not rel < (0.10 if tiny else float("inf")):
                raise ValueError((variant, h, c, tiny, len(fields), cos, rel))
            fields.append({"cos": cos, "rel_l2": rel})
        result[variant] = fields
    return result


def benchmark(batch: int, warmup: int, iters: int) -> dict:
    os.environ.update(lockin_arm_env("w57_lockin"))
    result, totals = {}, {}
    for label, h, w, m, c, count in MODEL_SHAPES:
        args, dy = inputs(batch, h, w, m, c)
        row = {}
        for name, variant, bwd in (
            ("w57_vf", wave.BASE, False),
            (wave.VF, wave.VF, False),
            ("w57_vb", wave.BASE, True),
            (wave.SAME, wave.SAME, True),
            (wave.HIGH, wave.HIGH, True),
        ):
            fn = partial(
                wave.vertical,
                *args,
                variant,
                grad=dy.split(m, -1) if bwd else None,
                compact=bwd and c,
            )
            finite_graph(fn)
            ms = _time(fn, warmup, iters)
            row[name] = ms
            totals[name] = totals.get(name, 0.0) + count * ms
        row["parity"] = parity(h, w, m, c)
        if label == "56d":
            row["tiny"] = parity(h, w, m, c, tiny=True)
        result[label] = row
    return {"shapes": result, "weighted_ms": totals}


def rounded(data: Any) -> Any:
    if isinstance(data, float):
        return float(format(data, ".5g"))
    if isinstance(data, dict):
        return {k: rounded(v) for k, v in data.items()}
    if isinstance(data, list):
        return list(map(rounded, data))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-56d", action="store_true")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.check_56d:
        check_56d()
    else:
        report = json.dumps(
            rounded(benchmark(args.batch, args.warmup, args.iters)), separators=(",", ":")
        )
        if args.json:
            args.json.write_text(report + "\n")
        else:
            print(report)
