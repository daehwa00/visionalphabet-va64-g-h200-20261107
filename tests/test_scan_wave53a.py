"""Wave 53A planner, PTX compatibility, and standalone CUDA parity coverage."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import bench_scan_micro as micro
import bench_wave53a_scans_type1 as bench
import pytest
import torch

from lnet import general_cuda_scans as scans

BASE = "dbcdbf2ff420e709c87a04d9ff22da8656c13388"


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    with patch.dict(os.environ):
        yield


@pytest.fixture(scope="module")
def previous() -> ModuleType:
    module = ModuleType("lnet._wave53_base")
    module.__package__ = "lnet"
    root = Path(scans.__file__).resolve().parents[2]
    src = subprocess.check_output(
        ["git", "show", f"{BASE}:src/lnet/general_cuda_scans.py"], cwd=root
    )
    sys.modules[module.__name__] = module
    try:
        src = src.split(b"# Custom ops and public API", 1)[0]
        exec(compile(src, "<wave52>", "exec"), vars(module))
    finally:
        del sys.modules[module.__name__]
    return module


def limits(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "device_limits", lambda _: (132, 227 * 1024, 64))
    monkeypatch.setattr(module, "_sgs_device_limits", lambda _: (228 * 1024, 65536, 1024))
    monkeypatch.setattr(module, "_l2_size", lambda _: 50 << 20)
    monkeypatch.setattr(module, "_scan_workspace_budget", lambda _: 64 << 20)
    monkeypatch.setattr(
        module,
        "_resident",
        lambda _, k, smem: min(
            64 // k.warps, (228 << 10) // (1024 + smem), 2 if k.vertical and k.backward else 16
        ),
    )


def test_modes(monkeypatch) -> None:
    for env, modes in (
        ("LNET_SCAN_HSCAN", scans.HSCAN_MODES),
        ("LNET_SCAN_W53A", scans.W53A_MODES),
    ):
        monkeypatch.delenv(env, raising=False)
        assert scans._scan_mode(env, modes) == "off"
        for mode in modes:
            monkeypatch.setenv(env, mode)
            assert scans._scan_mode(env, modes) == mode
        monkeypatch.setenv(env, "invalid")
        with pytest.raises(ValueError, match=env):
            scans._scan_mode(env, modes)


@pytest.mark.parametrize("horizontal", [None, "off", "bwd_opt"])
def test_ptx_digest(monkeypatch, previous, horizontal) -> None:
    bench.configure("w51a_fp32")
    if horizontal:
        monkeypatch.setenv("LNET_SCAN_HSCAN", horizontal)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    for module in (scans, previous):
        limits(monkeypatch, module)
    digests = {}
    for name, h, w, m, coarse, _ in micro.MODEL_SHAPES:
        shape = (1024, h, w, m)
        source = SimpleNamespace(
            is_cuda=True,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            numel=lambda size=(1 if horizontal else 4) * 1024 * h * w * m: size,
        )
        for backward in (False, True):
            pair = []
            for module in (previous, scans):
                launch = module._plan_launch(
                    source,
                    shape,
                    vertical=not horizontal,
                    backward=backward,
                    coarse=coarse and not horizontal,
                    hidden=0 if horizontal else 8,
                    precise=not horizontal and not backward,
                    compact=coarse and backward,
                    interleaved=(True, True),
                )
                key = launch.key
                if (module, key) not in digests:
                    ptx = module._compile_ptx(key, (9, 0)).value
                    digests[module, key] = hashlib.sha256(ptx).hexdigest()
                pair.append((launch.plan, launch.workers, launch.partials, digests[module, key]))
            assert pair[0] == pair[1], (name, backward)
            print(name, backward, horizontal, pair[0][-1])


@pytest.mark.parametrize("mode", scans.H53_MODES)
@pytest.mark.parametrize("width", [49, 55, 56, 57])
def test_h53(monkeypatch, mode, width) -> None:
    limits(monkeypatch, scans)
    key = scans.Kernel(torch.bfloat16, 4, True, cache_line=True)
    plan = scans.ScanPlan(4, width * 128 * 8, cache_line=True)
    launch = scans._plan_h53(0, scans.Launch(plan, key, 32000, 8000), (1024, 56, width, 160), mode)
    key = launch.key
    assert key.horizontal_readonly and key.horizontal_unroll == 8
    assert launch.partials == scans._ceil(launch.workers, key.warps)
    assert launch.workers < 32000
    code = scans._kernel_source(key)
    if mode == "h53_l2_scratch":
        assert (
            not key.cache_line and key.warps == 2 and width * launch.workers * 160 * 8 <= 32 << 20
        )
    elif mode == "h53_bf16_buf":
        assert key.horizontal_bf16_buf and "Pair<__nv_bfloat16>::pack(right)" in code
    else:
        assert key.horizontal_half_replay and "descend(half, 0)" in code
        assert launch.plan.smem == (width - width // 2) * 128 * 8


def test_unroll_fallback(monkeypatch) -> None:
    limits(monkeypatch, scans)
    monkeypatch.setattr(scans, "_resident", lambda _, k, smem: 0 if k.horizontal_unroll == 8 else 4)
    key = scans.Kernel(torch.bfloat16, 4, backward=True, cache_line=True)
    base = scans.Launch(scans.ScanPlan(4, 57344, cache_line=True), key, 2048, 512)
    out = scans._plan_h53(0, base, (100, 7, 56, 160), "h53_ro_all")
    assert out.key.horizontal_unroll == 4
    assert out.key.horizontal_half_replay


def test_compact_source() -> None:
    key = scans.Kernel(
        torch.float16,
        3,
        True,
        compact=True,
        horizontal_readonly=True,
        horizontal_unroll=8,
        horizontal_compact2=True,
    )
    source = scans._kernel_source(key)
    assert "consume(2*k+1, a*even)" in source and "consume(2*k, even)" in source


@pytest.mark.parametrize("mode", scans.W53A_MODES[1:])
def test_winners(monkeypatch, mode) -> None:
    limits(monkeypatch, scans)
    for height, coarse, floor in ((56, False, False), (56, True, True), (28, False, True)):
        key = scans.Kernel(
            torch.bfloat16,
            4,
            True,
            vertical=True,
            hidden8=True,
            coarse=coarse,
            vector_weights=True,
            backward_batch2=True,
        )
        launch = scans.Launch(scans.ScanPlan(4, 8192), key, 16384, 4096)
        out = scans._plan_w53a(0, launch, (1024, height, height, 160), mode)
        assert out.key.fp32_path and out.key.tanh_gain
        assert (out.workers < launch.workers) == floor
        assert out.key.path_bwd_pipe == scans.W53A_MODES.index(mode) - 1
        if out.key.path_bwd_pipe:
            source = scans._kernel_source(out.key)
            assert "preactivate(j+1" in source if mode.endswith("swpipe") else "jb += 4" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", bench.HORIZONTAL + bench.VERTICAL)
def test_cuda(mode) -> None:
    rows = bench.check_mode(mode, 2, 0.035)
    assert all(r["passed"] for r in rows.values()), rows
