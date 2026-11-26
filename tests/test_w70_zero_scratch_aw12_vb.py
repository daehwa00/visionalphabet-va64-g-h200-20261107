# ruff: noqa: S607
"""Compile dense 56- and 28-row cubins locally and gate device checks on SM90."""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import bench_w70_zero_scratch_aw12_vb as bench
import pytest
import test_w66_split_bf16 as baseline
import torch

from lnet import w70_zero_scratch_aw12_vb as wave


def test_dense_cubins(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        baseline, "SHAPES", [row for row in bench.ref.SHAPES if row[0] in ("56d", "28d")]
    )
    plans = [(w, plan) for _, op, w, _, plan in baseline.offline_plans() if op == "VB"]

    def audit(item):
        h, launch, arm = item
        candidate = wave.configure(launch, h, arm) if arm != bench.BASE else launch
        key = candidate.key
        cu = tmp_path / (f"{h}_{arm}.cu")
        cu.write_text(
            wave.kernel_source(key, h, h, 160, arm)
            if arm != bench.BASE
            else wave.fp.kernel_source(key, arm, h, h, 160)
        )
        cubin = cu.with_suffix(".cubin")
        subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-arch=sm_90a",
                *(
                    wave.compile_flags(arm)
                    if arm != bench.BASE
                    else wave.fp.compile_flags(key, arm, h)
                ),
                "--ptxas-options=-v",
                "-cubin",
                str(cu),
                "-o",
                str(cubin),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        sass = subprocess.check_output(["cuobjdump", "-sass", cubin], text=True)
        res = subprocess.check_output(["cuobjdump", "-res-usage", cubin], text=True)
        if arm != bench.BASE:
            assert "MUFU.TANH" in sass
            assert "HFMA2" in sass
        regs = int(re.search(r"REG:(\d+)", res)[1])
        total = candidate.plan.smem + int(re.search(r"SHARED:(\d+)", res)[1])
        blocks = min(65536 // (key.warps * ((regs + 7) // 8) * 256), 232448 // total)
        counts = [
            len(re.findall(r"\b" + op + r"\b", sass))
            for op in ("LDL", "STL", "FFMA", "HFMA2", "LDS", "STS", "LDG", "STG")
        ]
        stack = int(re.search(r"STACK:(\d+)", res)[1])
        scratch = (h // (1 + key.coarse)) * candidate.workers * 160 * 8 if arm == wave.FULL else 0
        if arm == bench.BASE and h == 56:
            scratch = (
                candidate.workers
                * 160
                * 8
                * ((h if not key.sgs or key.l2_single_write else 0) + (0 if key.cache_line else h))
            )
        return (h, arm, regs, total, blocks * key.warps, candidate.workers, scratch, stack, *counts)

    with ThreadPoolExecutor(8) as pool:
        rows = list(
            pool.map(
                audit, [(h, plan, arm) for h, plan in plans for arm in (bench.BASE, *wave.ARMS)]
            )
        )
    for row in rows:
        print(*row)  # noqa: T201
    for row in rows:
        if row[1] != bench.BASE:
            assert row[2] <= 168, row
            assert row[5] >= 1584, row
            assert row[4] == 12, row
            assert row[7:10] == (0, 0, 0), row


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires Hopper",
)
@pytest.mark.parametrize(
    ("height", "coarse", "scale"),
    [(56, False, 1), (56, False, 1e-5), (56, True, 1), (28, False, 1), (28, True, 1)],
)
def test_graph_parity(height, coarse, scale, monkeypatch) -> None:
    for k, v in bench.ref.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(k, v)
    for arm in bench.correctness(height, coarse, scale).values():
        for row in arm.values():
            assert row["finite"]
            assert row["min_cos"] >= (0.9999 if scale != 1 else 0.999), row
            assert row["max_rel_l2"] <= 0.015, row


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires Hopper",
)
@pytest.mark.parametrize(("height", "coarse"), [(56, False), (56, True), (28, False), (28, True)])
def test_large_graph(height, coarse, monkeypatch) -> None:
    for k, v in bench.ref.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(k, v)
    for row in bench.benchmark(height, coarse, 1024, 1, 1).values():
        assert row["graph"] == "PASS"
