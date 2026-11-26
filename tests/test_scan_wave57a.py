"""Wave 57A SM90a resource, dispatch and gradient regression checks."""

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import bench_scan_micro as micro
import bench_wave55a_scans_type1 as base_bench
import bench_wave57a_vscan_type1 as bench
import pytest
import torch
from test_scan_wave53a import limits
from test_scan_wave55a import occupancy, plan
from test_scan_wave56a import compiler, isolated, offline  # noqa: F401

from lnet import general_cuda_scans as scans

BASE = "12c83fc12c9645c8185a8e1f47c1ca0cc891f435"


def test_baseline_ptx(monkeypatch, offline, compiler):
    root = Path(scans.__file__).resolve().parents[2]
    if subprocess.run(["git", "cat-file", "-e", BASE], cwd=root).returncode:
        pytest.skip("Baseline commit unavailable")
    code = subprocess.check_output(
        ["git", "show", f"{BASE}:src/lnet/general_cuda_scans.py"], cwd=root
    )
    previous = ModuleType("lnet._wave57_base")
    previous.__package__ = "lnet"
    with patch.dict(sys.modules, {previous.__name__: previous}):
        exec(
            compile(code.split(b"# Custom ops and public API", 1)[0], "<base>", "exec"),
            vars(previous),
        )
    limits(monkeypatch, previous)
    for mode in (
        "off",
        "w53_lockin",
        "w55_lockin",
        "h53_lockin",
        "h55_lockin",
        *scans.W56A_MODES[1:],
    ):
        base_bench.configure("w55_lockin" if mode.startswith("v56") else mode)
        if mode == "off":
            for name in tuple(os.environ):
                if name.startswith("LNET_SCAN_"):
                    monkeypatch.delenv(name)
        monkeypatch.setenv("LNET_SCAN_W56A", mode if mode.startswith("v56") else "off")
        monkeypatch.setenv("LNET_SCAN_W57A", "off")
        for dtype in (torch.bfloat16, torch.float16):
            for shape in micro.MODEL_SHAPES:
                for backward in (True, False):
                    for vertical in (
                        (True, False) if mode == "off" else (mode.startswith(("w", "v")),)
                    ):
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
    # Match the Type 1 workspace budget and use compiled register counts.
    monkeypatch.setattr(scans, "_scan_workspace_budget", lambda _: 128 << 20)

    def active_warps(regs, smem, warps):
        # SM90 allocates registers independently in four warp-scheduler partitions.
        warp_regs = ((regs * 32 + 255) // 256) * 256
        register_warps = 4 * (16384 // warp_regs)
        return min(occupancy(regs, smem, warps), register_warps // warps * warps)

    def resident(device, key, smem):
        regs, mem, _ = compiler(scans._kernel_source(key))
        return 0 if any(mem) else active_warps(regs, smem, key.warps) // key.warps

    monkeypatch.setattr(scans, "_resident", resident)
    for dtype in (torch.bfloat16, torch.float16):
        for shape in micro.MODEL_SHAPES:
            name, h, _, _, coarse, _ = shape
            bench.configure("w55_lockin")
            base = plan(scans, shape, True, True, dtype)
            for mode in bench.VERTICAL[1:]:
                bench.configure(mode)
                launch = plan(scans, shape, True, True, dtype)
                key = launch.key
                regs, spills, ops = compiler(scans._kernel_source(key))
                aw = active_warps(regs, launch.plan.smem, key.warps)
                print(
                    "W57RESOURCE",
                    mode,
                    dtype,
                    name,
                    regs,
                    launch.plan.smem,
                    aw,
                    launch.workers,
                    key.chunk,
                    *spills,
                    flush=True,
                )
                assert spills == (0, 0, 0)
                low = mode == "v57_bwd_loww_only"
                assert key.seg16 == (not low)
                if low:
                    assert (key, launch.plan) == (base.key, base.plan)
                else:
                    assert key.chunk == (48 if h == 56 else 0)
                    assert key.cache_source and not key.shared_acc and not key.global_checkpoints
                    assert ops[5] > 0
                    cached_line = h // (1 + coarse) if key.cache_line else 0
                    rows = h + (key.chunk or h) + cached_line + launch.plan.sgs_rows
                    assert launch.plan.smem == 5632 + 1024 * rows
                if h == 56:
                    assert aw == 8 and launch.plan.smem <= 114688
                    span = key.chunk
                    tail = (h - 1) % span + 1 if low else span
                    assert h - tail == ((40 if coarse else 44) if low else 8)
                cap = 208 if mode.endswith("1wave") else 420
                capped = name in ("56d", "28c") and mode != "v57_bwd_seg16_2chunk"
                assert launch.workers == (min(base.workers, cap) if capped else base.workers)


def test_alias_and_forward(offline):
    for shape in micro.MODEL_SHAPES:
        bench.configure("w55_lockin")
        forward = plan(scans, shape, True, False)
        for mode in bench.VERTICAL[1:]:
            bench.configure(mode)
            assert plan(scans, shape, True, False) == forward
        bench.configure("v57_bwd_seg16_2wave")
        two_waves = plan(scans, shape, True, True)
        bench.configure("v57_bwd_seg16_loww")
        assert plan(scans, shape, True, True) == two_waves


@pytest.mark.parametrize("arch", ["sm_75", "sm_86"])
def test_fallback(arch, monkeypatch, offline, compiler):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: tuple(map(int, arch[-2:])))
    for shape, dtype in [(s, torch.float16) for s in bench.ODD_SHAPES] + [
        (micro.MODEL_SHAPES[0], torch.float32)
    ]:
        bench.configure("w55_lockin")
        base = plan(scans, shape, True, True, dtype)
        for mode in bench.VERTICAL[1:]:
            bench.configure(mode)
            assert plan(scans, shape, True, True, dtype) == base
        compiler(scans._kernel_source(base.key), arch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", (*bench.VERTICAL[1:], "w57_lockin"))
def test_cuda(mode):
    assert all(row["passed"] for row in bench.check_mode(mode, 2).values())
    run, _ = micro._case("vscan_bwd", micro.MODEL_SHAPES[1], 2, torch.bfloat16, compact=True)
    torch.testing.assert_close(torch.compile(run, backend="aot_eager", fullgraph=True)(), run())


DELEGATES = {
    "56d": ("w53_lockin", "v57_bwd_seg16_2chunk"),
    "56c": ("v55_bf16x2_bf16w", "v57_bwd_seg16_2chunk"),
    "28d": ("v55_bf16x2_bf16w", "v55_bf16x2_bf16w"),
    "28c": ("v56_noround", "v57_bwd_loww_only"),
    "14d": ("v56_noround", "v55_bf16x2_bf16w"),
    "14c": ("v56_noround", "v57_bwd_seg16_2chunk"),
    "7d": ("v56_noround", "v55_bf16x2_predp"),
}


def configure_delegate(mode):
    bench.configure("w55_lockin")
    if mode.startswith("v56"):
        os.environ["LNET_SCAN_W56A"] = mode
    elif mode.startswith("v57"):
        os.environ["LNET_SCAN_W57A"] = mode
    else:
        base_bench.configure(mode)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_lockin_delegates(offline, compiler, dtype):
    for shape in micro.MODEL_SHAPES:
        for backward, mode in enumerate(DELEGATES[shape[0]]):
            bench.configure("w57_lockin")
            actual = plan(scans, shape, True, bool(backward), dtype)
            configure_delegate(mode)
            expected = plan(scans, shape, True, bool(backward), dtype)
            assert actual == expected
            assert compiler(scans._kernel_source(actual.key), ptx=True) == compiler(
                scans._kernel_source(expected.key), ptx=True
            )


@pytest.mark.parametrize("arch", ["sm_75", "sm_86"])
def test_lockin_fallback(offline, monkeypatch, compiler, arch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: tuple(map(int, arch[-2:])))
    for shape, dtype in [(s, torch.float16) for s in bench.ODD_SHAPES] + [
        (micro.MODEL_SHAPES[0], torch.float32), (micro.MODEL_SHAPES[4], torch.float16)
    ]:
        for backward in (False, True):
            bench.configure("w57_lockin")
            actual = plan(scans, shape, True, backward, dtype)
            _, h, _, _, coarse, _ = shape
            mode = "w55_lockin"
            if arch != "sm_75" and dtype != torch.float32:
                if backward and (h >= 48 or (coarse and h <= 16)):
                    mode = "v57_bwd_seg16_2chunk"
                elif backward and coarse:
                    mode = "v57_bwd_loww_only"
                elif not backward and h < (48 if coarse else 24):
                    mode = "v56_noround"
            configure_delegate(mode)
            assert actual == plan(scans, shape, True, backward, dtype)
            compiler(scans._kernel_source(actual.key), arch)
