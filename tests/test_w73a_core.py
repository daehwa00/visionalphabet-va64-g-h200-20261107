# ruff: noqa: S607
"""Offline SM90a resource gates and graph replay checks for the standalone arms."""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import bench_wave66_split_bf16 as bench
import bench_wave73a_core as screen
import pytest
import torch
from test_w66_split_bf16 import compile_source, offline_plans

from lnet import w73a_fp8_core as core


def audit(job) -> tuple:
    label, op, width, modes, launch, arm, folder = job
    folder.mkdir()
    key = launch.key
    symbol = core.profile_name(key)
    src = core.kernel_source(key, width, width, modes, arm)
    row = compile_source(src, folder, core.compile_flags(key, width), symbol)
    sass = subprocess.check_output(
        ["cuobjdump", "-fun", symbol, "-sass", *folder.glob("*.cubin")], text=True
    )
    if arm in core.HB:
        assert row["LDG64"]
        assert row["STG64"]
        assert "LDG.E.U16" not in sass
        assert "STG.E.U16" not in sass
    smem = core.shared_bytes(launch, width, arm)
    regs = (row["regs"] + 7) // 8 * 8
    aw = key.warps * min(
        64 // key.warps, (228 << 10) // (1024 + smem), 65536 // (key.warps * 32 * regs)
    )
    ops = [
        len(re.findall(r"\b" + p + r"\b", sass))
        for p in ("FFMA", "HFMA2", "LDG", "STG", r"LDG\.E\.U16", r"STG\.E\.U16")
    ]
    return (label, op, arm, row["regs"], smem, aw, row["stack"], row["LDL"], row["STL"], *ops)


def test_cubins(tmp_path) -> None:
    jobs = []
    for label, op, w, m, original in offline_plans(bench.SHAPES[:4]):
        launch = original
        if op == "VB" and w == 28:
            launch = core.scans._plan_w57(0, launch, (1024, w, w, m), "v57_bwd_seg16_2chunk")
        if launch.key.vertical:
            launch = core.configure(launch, w)
        arms = {"VB": core.VB, "VF": core.VF, "HB": core.HB, "HF": ()}[op]
        for arm in ("w72", *arms):
            if (arm == core.VB[1] and label != "56c") or (
                label == "28c" and op == "VF" and arm != "w72"
            ):
                continue
            jobs.append(
                (
                    label,
                    op,
                    w,
                    m,
                    core.candidate_plan(launch, w, arm),
                    arm,
                    tmp_path / f"{label}_{op}_{arm}",
                )
            )
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(audit, jobs))
    for row in rows:
        print(*row)  # noqa: T201
    for row in rows:
        assert row[6:9] == (0, 0, 0), row
        if row[2] in core.HB:
            assert row[5] >= 16
        if row[2] in core.VB:
            assert row[5] == 8


def test_null_workspace(monkeypatch) -> None:
    plans = {op: plan for shape, op, _, _, plan in offline_plans() if shape == "56d"}
    monkeypatch.setattr(core, "_check_source", lambda *_: None)
    monkeypatch.setattr(core.scans, "_reduce_partials", lambda _, params, *_args: params)
    r = torch.empty(1, 56, 56, 160, dtype=torch.bfloat16)
    pole = [torch.empty(160)] * 4
    aux = [torch.empty(2, 56, 160)] * 2
    path = [torch.empty(s) for s in ((160, 8, 8), (160, 8), (160, 2, 8), (160, 2))]
    seen = []

    def capture(_, launch, _width, args, scale_args=(), *, arm):
        scratch = (args[17], args[18]) if launch.key.vertical else (args[10],)
        assert arm in ("w72", core.VF[1], core.VB[-1], core.HB[0])
        assert all(ptr.value is None for ptr in scratch)
        seen.append(core.profile_name(launch.key))
        assert bool(scale_args) == (launch.key.vertical and launch.key.backward)

    def plan(*_: object, vertical, backward, **__: object):
        return plans[("V" if vertical else "H") + ("B" if backward else "F")]

    monkeypatch.setattr(core, "_dispatch", capture)
    monkeypatch.setattr(core.scans, "_plan_launch", plan)
    states = core.horizontal(pole, r, r)
    core.vertical(pole, r, states, aux, path, coarse=False, arm=core.VF[1])
    grad = core.vertical(pole, r, states, aux, path, coarse=False, grad=(r, r), arm=core.VB[-1])
    assert grad[4].dtype == torch.int16
    core.horizontal(pole, r, r, states=states, adjoints=grad[4], scales=grad[-1], arm=core.HB[0])
    assert len(set(seen)) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("case", ["normal", "tiny", "zero_first_two"])
def test_hopper(case, monkeypatch) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires Hopper")
    for name, value in bench.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(name, value)
    data = screen.reference(56, coarse=False, case=case)
    for arm in (core.VB[2], core.VB[3], *core.VF, *core.HB, core.VB[0]):
        op = arm[:2].upper()
        choice = screen.CONTROL | {op: arm}
        result = screen.parity(data, choice, coarse=False)
        assert result["finite"]
        assert result["w72"]["min_cos"] > 0.99
        assert result["w72"]["max_rel_l2"] < 0.1
        assert result["fp32"]["min_cos"] > 0.9
        fn = screen.operators(data[0], choice, coarse=False)[-1]
        assert screen.graph_time(fn, 1)["graph"] == "PASS"


@pytest.mark.parametrize(("magnitude", "zero_rows"), [(1.0, 0), (1e-5, 0), (1.0, 2)])
def test_host_dwi_scaling(magnitude, zero_rows) -> None:
    # Model one column's fused half accumulation with one rounding per FMA.
    rng = torch.Generator().manual_seed(259)
    dy = (torch.randn(56, 160, 2, generator=rng) * magnitude).bfloat16().float()
    dy[:zero_rows] = 0
    samples = (torch.arange(4096) * 8960 + 4480) // 4096
    peak = dy.reshape(-1, 2)[samples].abs().max()
    scale = 2.0 ** (3 - int(torch.floor(torch.log2(peak))))
    p = torch.randn(56, 8, generator=rng) * 0.1
    dz = dy[:, 0, :] @ torch.randn(2, 8, generator=rng)
    expected = dz.T @ p
    acc = torch.zeros(8, 8, dtype=torch.float16)
    for row in reversed(range(56)):
        d = (dz[row] * scale).half().float()
        acc = (acc.float() + d[:, None] * p[row].half().float()).half()
    actual = acc.float() / scale
    assert torch.isfinite(actual).all()
    assert (expected - actual).norm() / expected.norm() < 0.005
