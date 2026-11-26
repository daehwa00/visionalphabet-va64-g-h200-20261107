# ruff: noqa: ANN201, S607, T201
"""Offline cubin checks and Hopper parity for approximate BF16 state recovery."""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import bench_wave68_decoupled_tc_vscan as bench
import pytest
import torch

from lnet import w68_decoupled_tc_vscan as wave


def test_cubins(tmp_path):
    def inspect(name):
        cubin = wave.compile_kernel(name, 56, tmp_path)
        res = subprocess.check_output(["cuobjdump", "-res-usage", cubin], text=True)
        sass = subprocess.check_output(["cuobjdump", "-sass", cubin], text=True)
        regs, smem, stack = [int(re.search(rf"\b{field}:(\d+)", res)[1])
                             for field in ("REG", "SHARED", "STACK")]
        ldl, stl, hmma = [len(re.findall(rf"\b{op}\b", sass)) for op in ("LDL", "STL", "HMMA")]
        warps = 4 if name.endswith("pure") else 32
        if name.endswith("pure"):
            smem += (56*(2 if name == "VB_pure" else 1)-1)*512
        rounded = ((regs*32+255)//256)*256
        blocks = min(64//warps, 65536//(rounded*warps), (228*1024)//smem)
        ldg, stg, div, rcp = [len(re.findall(rf"\b{op}\b", sass))
                              for op in ("LDG", "STG", "FDIV", r"MUFU\.RCP")]
        print(name, regs, smem, blocks*warps, stack, ldl, stl, hmma, ldg, stg, div, rcp)
        assert (stack, ldl, stl) == (0, 0, 0)
        assert ldg
        assert stg
        if name == "VB_pure":
            assert (div, rcp) == (0, 0)
        if name.startswith("PathMLP"):
            assert hmma
            assert blocks*warps >= 32

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(inspect, wave.NAMES))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
                    reason="requires SM90")
@pytest.mark.parametrize(("height", "scale"), [(56, 1.0), (28, 1.0), (56, 1e-5)])
def test_parity(height, scale):
    bench.parity(height, scale)
