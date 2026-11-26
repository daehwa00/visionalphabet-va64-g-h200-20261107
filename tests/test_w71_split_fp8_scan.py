# ruff: noqa: ANN201, S607, T201
"""Offline cubin audit and Hopper integration gates."""

import concurrent.futures
import re
import subprocess

import bench_w71_split_fp8_scan as bench
import pytest
import torch
from test_w66_split_bf16 import offline_plans

from lnet import w66_split_bf16_scan as w66
from lnet import w71_split_fp8_scan as w71


def test_sass(tmp_path):
    def compile_one(args):
        op, arm, launch, w, m = args
        mod = w66 if arm == w66.ARM else w71
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
            capture_output=True,
        )
        symbol = (
            w66.profile_name(launch.key) if arm == w66.ARM else w71.profile_name(launch.key, arm)
        )
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
        if op == "VB":
            assert smem == 112128

    jobs = [
        (op, arm, launch, w, m)
        for shape, op, w, m, launch in offline_plans()
        if shape == "56d"
        for arm in ((w66.ARM, w71.BF16, w71.ARM) if op == "VB" else (w71.ARM,))
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
        assert rows[w71.BF16][w66.ARM]["max_rel_l2"] <= 0.006
        assert rows[w71.ARM]["fp32"]["min_cos"] >= 0.99
        assert all(ref["finite"] for arm in rows.values() for ref in arm.values())
    timings = bench.benchmark(shape, 1024, 1, 1)
    assert all(op["graph_finite"] for arm in timings.values() for op in arm.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
def test_fp8_byte_order(monkeypatch):

    for k, v in bench.base.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(k, v)
    r = torch.ones((1, 56, 56, 160), device="cuda", dtype=torch.bfloat16)
    i = torch.full_like(r, -2)
    one = torch.ones(160, device="cuda")
    zero = torch.zeros_like(one)
    q = w71.horizontal((zero, zero, one, zero), r, i)
    assert torch.all(q.to(torch.int32) == 0xC038)


def test_vb_null_scratch(monkeypatch):
    # Execute both host wrappers with a recorded launch, without a CUDA device.
    launch = next(plan for s, op, _, _, plan in offline_plans() if s == "56d" and op == "VB")
    monkeypatch.setattr(w71, "_check_source", lambda *_: None)
    monkeypatch.setattr(w71.scans, "_plan_launch", lambda *_, **__: launch)
    monkeypatch.setattr(w71.scans, "_reduce_partials", lambda _, params, __: params)

    def capture(_, __, variant, ___, args, scale_args):
        assert args[17].value is None  # scratch follows partials in the vertical ABI
        assert bool(scale_args) == (variant == w71.ARM)

    monkeypatch.setattr(w71, "_dispatch", capture)
    r = torch.empty(1, 56, 56, 160, dtype=torch.bfloat16)
    pole = [torch.empty(160)] * 4
    aux = [torch.empty(2, 56, 160)] * 2
    path = [torch.empty(s) for s in ((160, 8, 8), (160, 8), (160, 2, 8), (160, 2))]
    for arm in (w71.ARM, w71.BF16):
        states = w71._words(r, 56, arm)
        result = w71.vertical(pole, r, states, aux, path, coarse=False, variant=arm, grad=(r, r))
        assert result[4].dtype == states.dtype
