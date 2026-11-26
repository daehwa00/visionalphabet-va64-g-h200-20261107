# ruff: noqa: S607
"""Compile the 56d cubins locally and gate device checks on SM90."""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import bench_w69_highocc_vscan_bwd as bench
import pytest
import torch
from test_w66_split_bf16 import offline_plans

from lnet import w69_highocc_vscan_bwd as wave


def test_56d_cubins(tmp_path) -> None:
    launch = next(plan for shape, op, _, _, plan in offline_plans() if (shape, op) == ("56d", "VB"))

    def audit(arm):
        candidate = wave.configure(launch, 56, arm)
        key = candidate.key
        cu = tmp_path / (arm + ".cu")
        cu.write_text(wave.kernel_source(key, 56, 56, 160, arm))
        cubin = cu.with_suffix(".cubin")
        subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-arch=sm_90a",
                "--ptxas-options=-v",
                "-cubin",
                str(cu),
                "-o",
                str(cubin),
            ],
            check=True,
            capture_output=True,
        )
        sass = subprocess.check_output(["cuobjdump", "-sass", cubin], text=True)
        res = subprocess.check_output(["cuobjdump", "-res-usage", cubin], text=True)
        regs = int(re.search(r"REG:(\d+)", res)[1])
        total = candidate.plan.smem + int(re.search(r"SHARED:(\d+)", res)[1])
        blocks = min(65536 // (key.warps * ((regs + 7) // 8) * 256), 232448 // total)
        assert (
            blocks * key.warps >= {wave.FULL: 8, wave.AW12: 12, wave.AW16: 14, wave.PACK: 12}[arm]
        )
        assert arm == wave.FULL or total <= 59 * 1024
        assert "MUFU.TANH" in sass
        assert "STACK:0 " in res
        assert not re.search(r"\b(?:LDL|STL)\b", sass)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(audit, wave.ARMS))


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
            assert row["min_cos"] >= 0.999, row
            assert row["max_rel_l2"] <= 0.015, row
