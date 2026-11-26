# ruff: noqa: ANN201, S607, T201
"""Offline cubin audit and Hopper integration gates."""

import concurrent.futures
import re
import subprocess

import bench_w72_fp8_upgrades as bench
import pytest
import torch
from test_w66_split_bf16 import offline_plans

from lnet import w71_split_fp8_scan as w71
from lnet import w72_fp8_upgrades as w72


def test_sass(tmp_path):
    def compile_one(args):
        op, arm, launch, w, m = args
        mod = w71 if arm == w71.ARM else w72
        if mod is w72:
            launch = w72.configure(launch, 56, arm)
        folder = tmp_path / (op + arm)
        folder.mkdir()
        src = mod.kernel_source(launch.key, arm, w, 56, m)
        path = folder / "scan.cu"
        cubin = path.with_suffix(".cubin")
        path.write_text(src)
        subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-arch=sm_90a",
                *mod.compile_flags(launch.key, arm, w),
                "-cubin",
                path,
                "-o",
                cubin,
            ],
            check=True,
        )
        symbol = w71.profile_name(launch.key, arm)
        sass = subprocess.check_output(["cuobjdump", "-fun", symbol, "-sass", cubin], text=True)
        resource = (
            subprocess.check_output(["cuobjdump", "-res-usage", cubin], text=True)
            .split("Function " + symbol + ":")[1]
            .splitlines()[1]
        )
        data = {
            name: int(re.search(rf"{field}:(\d+)", resource)[1])
            for name, field in (("regs", "REG"), ("stack", "STACK"))
        }
        for name in ("LDL", "STL"):
            data[name] = len(re.findall(rf"\b{name}\b", sass))
        counts = {
            op: len(re.findall(rf"\b{op}\b", sass))
            for op in ("FFMA", "HFMA2", "LDS", "STS", "LDG", "STG", "F2FP", "F2F")
        }
        smem = mod.shared_bytes(launch, 56)
        aw = (
            min(
                64 // launch.key.warps,
                (228 * 1024) // (1024 + smem),
                65536 // (launch.key.warps * 32 * ((data["regs"] + 7) // 8 * 8)),
            )
            * launch.key.warps
        )
        print(op, arm, data["regs"], smem, aw, data["stack"], data["LDL"], data["STL"], counts)
        assert not any(data[k] for k in ("stack", "LDL", "STL")), data
        if arm == w71.ARM:
            assert "E4M3" in sass
        if mod is w72:
            assert smem == {w72.VF: 25600, w72.BF: 39936, w72.FULL: 91648, w72.AW: 75264}[arm]
            assert aw >= {w72.VF: 16, w72.BF: 10, w72.FULL: 8, w72.AW: 12}[arm]
            assert launch.plan.cache_line or launch.key.backward

    jobs = [
        (op, arm, launch, w, m)
        for shape, op, w, m, launch in offline_plans()
        if shape == "56d" and op in ("VF", "VB")
        for arm in ((w71.ARM, w72.FULL, w72.AW) if op == "VB" else (w71.ARM, w72.VF, w72.BF))
    ]
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        list(pool.map(compile_one, jobs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("shape_id", range(4))
def test_hopper(shape_id, monkeypatch):

    for k, v in bench.base.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(k, v)
    shape = bench.base.SHAPES[shape_id]
    for factor in (1, 1e-5) if shape_id == 0 else (1,):
        rows = bench.precision(shape, factor)
        assert all(row["fp32"]["min_cos"] >= 0.99 for row in rows.values())
        assert all(ref["finite"] for arm in rows.values() for ref in arm.values())
    timings = bench.benchmark(shape, 1024, 1, 1)
    assert all(op["graph_finite"] for arm in timings.values() for op in arm.values())


def test_null_workspace(monkeypatch):
    plans = {op: plan for shape, op, _, _, plan in offline_plans() if shape == "56d"}
    monkeypatch.setattr(w71, "_check_source", lambda *_: None)
    monkeypatch.setattr(w72.scans, "_reduce_partials", lambda _, params, __: params)
    r = torch.empty(1, 56, 56, 160, dtype=torch.bfloat16)
    pole = [torch.empty(160)] * 4
    aux = [torch.empty(2, 56, 160)] * 2
    path = [torch.empty(s) for s in ((160, 8, 8), (160, 8), (160, 2, 8), (160, 2))]
    states = w71._words(r, 56, w71.ARM)
    seen = []

    def capture(_, _launch, arm, _width, args, _scale_args):
        assert args[17].value is None
        assert args[18].value is None
        seen.append(arm)

    monkeypatch.setattr(w72, "_dispatch", capture)
    for arm in (w72.VF, w72.BF, w72.FULL, w72.AW):
        backward = arm in (w72.FULL, w72.AW)
        plan = plans["VB" if backward else "VF"]
        monkeypatch.setattr(w72.scans, "_plan_launch", lambda *_, plan=plan, **__: plan)
        w72.vertical(
            pole, r, states, aux, path, coarse=False, variant=arm, grad=(r, r) if backward else None
        )
    assert seen == [w72.VF, w72.BF, w72.FULL, w72.AW]
