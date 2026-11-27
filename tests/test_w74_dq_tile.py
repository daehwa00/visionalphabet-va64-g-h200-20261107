# ruff: noqa: S607
"""Compile and inspect each SM90a VB entry without initializing CUDA."""

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import bench_wave66_split_bf16 as bench
import bench_wave74_dq_tile as screen
import pytest
import torch
from test_w66_split_bf16 import compile_source, offline_plans

from lnet import w74_dq_tile_scan as core

SIZES = {
    "56d": (91648, 114688, 115200, 114176, 112640, 120320),
    "56c": (91648, 105984),
    "28d": (62976, 77312, 62976),
    "28c": (62976, 70144),
}


def test_cubins(tmp_path) -> None:
    jobs = []
    for label, op, w, m, initial in offline_plans(bench.SHAPES[:4]):
        if op != "VB":
            continue
        launch = initial
        if w == 28:
            launch = core.scans._plan_w57(0, launch, (1024, w, w, m), "v57_bwd_seg16_2chunk")
        original = core.configure(launch, w)
        for arm, size in zip(("w72", *core.VARIANTS[label]), SIZES[label], strict=True):
            plan = original if arm == "w72" else core.candidate_plan(original, w, arm)
            dynamic = core.base.shared_bytes(plan, w) if arm == "w72" else plan.plan.smem
            assert dynamic == size
            jobs.append((label, arm, w, m, plan.key, size))

    def audit(job: tuple) -> tuple:
        label, arm, w, m, key, size = job
        folder = tmp_path / f"{label}_{arm}"
        folder.mkdir()
        src = (
            core.base.kernel_source(key, w, w, m)
            if arm == "w72"
            else core.kernel_source(key, w, w, m, arm)
        )
        symbol = core.profile_name(key)
        row = compile_source(src, folder, core.compile_flags(key, w), symbol)
        sass = subprocess.check_output(
            ["cuobjdump", "-fun", symbol, "-sass", *folder.glob("*.cubin")], text=True
        )
        total = ((size + row["static"] + 1024 + 255) // 256) * 256
        regs = (row["regs"] + 7) // 8 * 8
        aw = key.warps * min(
            64 // key.warps, (228 << 10) // total, 65536 // (key.warps * 32 * regs)
        )
        return (
            label,
            arm,
            row["regs"],
            size,
            row["static"],
            total,
            aw,
            row["stack"],
            row["LDL"],
            row["STL"],
            row["LDG"],
            row["STG"],
            len(re.findall(r"\bSTG\.E\.U16\b", sass)),
        )

    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(audit, jobs))
    for row in rows:
        print(*row)  # noqa: T201
        assert row[6] == (4 if row[1] in ("dq_tile_hybrid_46", "dq_tile_full_aw4") else 8), row
        assert row[7:10] == (0, 0, 0), row
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    ("label", "w", "coarse"), [(s, h, c) for s, h, _, _, c in bench.SHAPES[:4]]
)
def test_null_workspace(label, w, coarse, monkeypatch) -> None:
    for shape, op, _, _, launch in offline_plans(bench.SHAPES[:4]):
        if shape != label or op != "VB":
            continue
        monkeypatch.setattr(core, "_check_source", lambda *_: None)
        monkeypatch.setattr(core.scans, "_plan_launch", lambda *_, **__: launch)  # noqa: B023
        monkeypatch.setattr(core.scans, "_reduce_partials", lambda _, params, *__: params)

        def capture(_, plan, width, args, scale_args, *, arm):
            assert width == w
            assert arm in core.VARIANTS[label]
            assert args[17].value is None
            assert args[18].value is None
            assert scale_args
            assert not plan.key.chunk

        monkeypatch.setattr(core, "_dispatch", capture)
        r = torch.empty(1, w, w, 160, dtype=torch.bfloat16)
        pole = [torch.empty(160)] * 4
        aux = [torch.empty(2, w, 160)] * 2
        path = [
            torch.empty(s, dtype=torch.bfloat16)
            for s in ((160, 8, 8), (160, 8), (160, 2, 8), (160, 2))
        ]
        grad = r[:, :: 1 + coarse, :: 1 + coarse].contiguous()
        for arm in core.VARIANTS[label]:
            out = core.vertical(
                pole, r, core._words(r, w), aux, path, coarse, (grad, grad), arm=arm
            )
            assert out[4].dtype == torch.int16
        return
    pytest.fail("missing offline launch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize(
    ("label", "w", "coarse"), [(s, h, c) for s, h, _, _, c in bench.SHAPES[:4]]
)
def test_hopper(label, w, coarse, monkeypatch) -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires Hopper")
    for name, value in bench.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(name, value)
    for case in ("normal", "tiny") if label == "56d" else ("normal",):
        data = screen.reference(w, coarse, case)
        for arm in core.VARIANTS[label]:
            choice = screen.CONTROL | {"VB": arm}
            result = screen.parity(data, choice, coarse)
            assert result["finite"]
            assert result["fp32"]["min_cos"] > 0.9
            assert result["w72"]["min_cos"] >= (0.99 if arm.endswith("fp8seg") else 0.999999)
            assert (
                screen.graph_time(screen.operators(data[0], choice, coarse)[-1], 1)["graph"]
                == "PASS"
            )
