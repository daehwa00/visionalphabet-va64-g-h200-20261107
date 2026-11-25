"""Wave 56A compiler, planner and device checks."""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import bench_scan_micro as micro
import bench_wave55a_scans_type1 as base_bench
import bench_wave56a_vscan_type1 as bench
import pytest
import torch
from test_scan_wave53a import limits
from test_scan_wave55a import occupancy, plan

from lnet import general_cuda_scans as scans

BASE = "923e4f7ab1423321b763bd3d2acaf067822ca740"


@pytest.fixture(autouse=True)
def isolated():
    planners = [
        fn
        for name, fn in vars(scans).items()
        if (name.startswith("_plan_") or name == "_launch_for_shape") and hasattr(fn, "cache_clear")
    ]
    for planner in planners:
        planner.cache_clear()
    with patch.dict(os.environ):
        yield
    for planner in planners:
        planner.cache_clear()


@pytest.fixture
def offline(monkeypatch):
    limits(monkeypatch, scans)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
    path = tmp_path_factory.mktemp("wave56")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        pytest.skip("nvcc unavailable")

    @lru_cache(None)
    def compile_source(source, arch="sm_90a", ptx=False, revision="working"):
        src = path / "scan.cu"
        src.write_text(source)
        out = src.with_suffix(".ptx" if ptx else ".cubin")
        proc = subprocess.run(
            [
                nvcc,
                "-arch=" + arch,
                "-std=c++17",
                "--ptx" if ptx else "--cubin",
                "--ptxas-options=-v",
                str(src),
                "-o",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        if ptx:
            return hashlib.sha256(out.read_bytes()).hexdigest()
        regs = int(re.search(r"Used (\d+) registers", proc.stderr)[1])
        mem = tuple(
            map(
                int,
                re.search(
                    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
                    proc.stderr,
                ).groups(),
            )
        )
        sass = subprocess.check_output(["cuobjdump", "-sass", str(out)], text=True)
        ops = [
            len(re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?" + op + r"\b", sass))
            for op in (
                "HFMA2",
                "FFMA",
                r"MUFU\.TANH",
                r"MUFU\.EX2",
                r"MUFU\.RCP",
                r"LDS\.128",
                "(?:F2FP|F2F|I2F|F2I|I2I)",
            )
        ]
        return regs, mem, ops

    return compile_source


def test_baseline_ptx(monkeypatch, offline, compiler):
    root = Path(scans.__file__).resolve().parents[2]
    if subprocess.run(["git", "cat-file", "-e", BASE], cwd=root).returncode:
        pytest.skip("Baseline commit unavailable")
    code = subprocess.check_output(
        ["git", "show", f"{BASE}:src/lnet/general_cuda_scans.py"], cwd=root
    )
    previous = ModuleType("lnet._wave56_base")
    previous.__package__ = "lnet"
    with patch.dict(sys.modules, {previous.__name__: previous}):
        exec(
            compile(code.split(b"# Custom ops and public API", 1)[0], "<base>", "exec"),
            vars(previous),
        )
    limits(monkeypatch, previous)
    for mode in ("off", "w53_lockin", "w55_lockin", "h53_lockin", "h55_lockin"):
        base_bench.configure(mode)
        if mode == "off":
            for name in tuple(os.environ):
                if name.startswith("LNET_SCAN_"):
                    monkeypatch.delenv(name)
        monkeypatch.setenv("LNET_SCAN_W56A", "off")
        for dtype in (torch.bfloat16, torch.float16):
            for shape in micro.MODEL_SHAPES:
                for backward in (True, False):
                    for vertical in (True, False) if mode == "off" else (mode.startswith("w"),):
                        hashes = []
                        for module in (previous, scans):
                            launch = plan(module, shape, vertical, backward, dtype)
                            hashes.append(
                                compiler(
                                    module._kernel_source(launch.key),
                                    ptx=True,
                                    revision=module.__name__,
                                )
                            )
                        assert hashes[0] == hashes[1], (mode, shape, dtype, backward)


def test_resources(offline, compiler, monkeypatch):
    monkeypatch.setattr(scans, "_scan_workspace_budget", lambda _: 114 << 20)

    def active_warps(regs, smem, warps):
        # cuda_occupancy.h allocates registers in 256-register warp units
        # independently on each of the four SM90 register-file partitions.
        regs_per_warp = ((regs * 32 + 255) // 256) * 256
        reg_warps = 4 * (16384 // regs_per_warp)
        return min(occupancy(regs, smem, warps), (reg_warps // warps) * warps)

    def resident(device, key, smem):
        regs, mem, _ = compiler(scans._kernel_source(key))
        return 0 if any(mem) else active_warps(regs, smem, key.warps) // key.warps

    monkeypatch.setattr(scans, "_resident", resident)
    failures = []
    for mode in bench.VERTICAL:
        bench.configure(mode)
        for dtype in (torch.bfloat16, torch.float16):
            for shape in micro.MODEL_SHAPES:
                for backward in (False, True):
                    launch = plan(scans, shape, True, backward, dtype)
                    key = launch.key
                    regs, mem, ops = compiler(scans._kernel_source(key))
                    aw = active_warps(regs, launch.plan.smem, key.warps)
                    print(
                        "W56RESOURCE",
                        mode,
                        str(dtype).split(".")[-1],
                        shape[0],
                        "bwd" if backward else "fwd",
                        regs,
                        launch.plan.smem,
                        aw,
                        launch.workers,
                        key.chunk,
                        *mem,
                        *ops,
                        flush=True,
                    )
                    if mode == "v56_bwd_quad_hfma2" and backward:
                        assert key.vertical_w56 == mode
                        source = scans._kernel_source(key)
                        quad = source.split("for (; end >= 4 * step", 1)[1].split(
                            "for (; end >= 2 * step", 1
                        )[0]
                        assert {int(k) for k in re.findall(r"p\[i\]\[(\d)\] =", quad)} == set(
                            range(8)
                        )
                    if mode != "w55_lockin":
                        assert ops[0] >= 64 and ops[2] > 0
                    if any(mem):
                        failures.append((mode, shape[0], dtype, backward, mem))
                    if backward and mode == "v56_bwd_hfma2_dw_l2" and dtype == torch.bfloat16:
                        assert key.cache_source and key.vertical_w56 == mode
                        assert key.chunk == (
                            22 if shape[0] == "56d" else 20 if shape[0] == "56c" else 0
                        )
                        assert launch.partials == launch.workers
                        assert launch.partials * 94 * shape[3] * 4 <= 44 << 20
                    if mode == "v56_fwd_hiworker" and shape[0] == "56d" and not backward:
                        assert launch.workers > 528
                        assert launch.workers * 5 >= 132 * aw
    assert not failures, failures


@pytest.mark.parametrize("arch", ["sm_75", "sm_86"])
def test_fallback(offline, monkeypatch, compiler, arch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: tuple(map(int, arch[-2:])))
    bench.configure("v56_bwd_hfma2_dw_l2")
    for dtype, shape in (
        (torch.float32, micro.MODEL_SHAPES[0]),
        (torch.float16, bench.ODD_SHAPES[0]),
    ):
        for backward in (False, True):
            launch = plan(scans, shape, True, backward, dtype)
            if dtype == torch.float32 or arch == "sm_75":
                assert launch.key.vertical_w56 == "off"
            compiler(scans._kernel_source(launch.key), arch)


def test_standalone_and_fp32(offline, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("LNET_SCAN_"):
            monkeypatch.delenv(name)
    shape = micro.MODEL_SHAPES[0]
    reference = plan(scans, shape, True, True, torch.float32)
    for mode in scans.W56A_MODES[1:]:
        monkeypatch.setenv("LNET_SCAN_W56A", mode)
        assert plan(scans, shape, True, True, torch.float32) == reference
        assert plan(scans, shape, True, True).key.vertical_w56 != "off"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", scans.W56A_MODES[1:])
def test_cuda(mode):
    assert all(row["passed"] for row in bench.check_mode(mode, 1).values())
    for backward in (False, True):
        run, _ = micro._case(
            "vscan_bwd" if backward else "vscan_fwd",
            micro.MODEL_SHAPES[1],
            1,
            torch.bfloat16,
            compact=backward,
        )
        torch.testing.assert_close(torch.compile(run, backend="aot_eager", fullgraph=True)(), run())
