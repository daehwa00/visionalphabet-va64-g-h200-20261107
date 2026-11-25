"""Wave 52A planning, offline code generation and CUDA parity checks."""

import hashlib
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import bench_scan_micro as bench
import pytest
import torch
from bench_wave51_vscan_type1 import ODD_SHAPES, parity, parity_reference

from lnet import general_cuda_scans as scans

_BASE = "bae6cf85e7ecf029777280761af90a6e6cef9a8d"


@pytest.fixture(scope="module")
def baseline() -> ModuleType:
    git = shutil.which("git")
    root = Path(scans.__file__).resolve().parents[2]
    if git is None:
        pytest.skip("Baseline comparison requires Git")
    result = subprocess.run(
        [git, "show", f"{_BASE}:src/lnet/general_cuda_scans.py"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        pytest.skip(f"Baseline checkout object {_BASE} is unavailable")
    module = ModuleType("lnet._wave52a_baseline")
    module.__package__ = "lnet"
    sys.modules[module.__name__] = module
    try:
        exec(compile(result.stdout, f"<{_BASE}>", "exec"), vars(module))  # noqa: S102
    finally:
        del sys.modules[module.__name__]
    return module


@pytest.fixture(scope="module")
def key56fwd() -> scans.Kernel:
    return scans.Kernel(
        torch.bfloat16, 4, backward=False, vertical=True, hidden8=True, chunk=8,
        cache_weights=True, vector_weights=True, batch_path=True, forward_readonly=True,
        real_interleaved=True, imag_interleaved=True,
    )


@pytest.fixture(scope="module")
def key56bwd() -> scans.Kernel:
    return scans.Kernel(
        torch.bfloat16, 4, backward=True, vertical=True, hidden8=True, chunk=20,
        cache_weights=True, vector_weights=True, cache_source=True, backward_batch2=True,
        backward_readonly=True, backward_reduce_warps=2,
        real_interleaved=True, imag_interleaved=True,
    )


@pytest.mark.parametrize("mode", scans.W52A_MODES)
def test_w52a_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.delenv("LNET_SCAN_W52A", raising=False)
    assert scans._scan_mode("LNET_SCAN_W52A", scans.W52A_MODES) == "off"
    monkeypatch.setenv("LNET_SCAN_W52A", mode)
    assert scans._scan_mode("LNET_SCAN_W52A", scans.W52A_MODES) == mode
    monkeypatch.setenv("LNET_SCAN_W52A", "invalid")
    with pytest.raises(ValueError, match="LNET_SCAN_W52A"):
        scans._scan_mode("LNET_SCAN_W52A", scans.W52A_MODES)


@pytest.mark.parametrize(("resident", "workers", "grid"), [(2, 208, 260), (5, 528, 660)])
def test_persistent_workers_floor(resident: int, workers: int, grid: int) -> None:
    args = (1024 * 56, 56, 56, 160, 2, 4, 132, resident)
    assert scans._persistent_workers_floor(*args, 45 * 2**20) == (workers, resident == 2)
    assert scans._ceil(workers, 4) * 5 == grid
    assert grid <= 132 * resident
    assert scans._persistent_workers_floor(*args, 56 * workers * 160 * 8 - 1) == (0, False)
    assert scans._persistent_workers_floor(*args, 56 * workers * 160 * 8) == (workers, False)


@pytest.mark.parametrize("single_residency", [1, 2])
def test_floor_sw_residency(
    monkeypatch: pytest.MonkeyPatch, key56bwd: scans.Kernel, single_residency: int,
) -> None:
    monkeypatch.setattr(scans, "device_limits", lambda _: (132, 227 * 1024, 64))
    monkeypatch.setattr(scans, "_l2_size", lambda _: 60 * 1024**2)
    base = scans.Launch(scans.ScanPlan(4, 113664, 20), key56bwd, 1688, 844)
    shape = (1024, 56, 56, 160)
    monkeypatch.setattr(
        scans, "_resident", lambda _, key, __: single_residency if key.l2_single_write else 2,
    )
    try:
        g2 = scans._plan_w52a(0, base, shape, "l2_floor_g2")
        sw = scans._plan_w52a(0, base, shape, "l2_floor_sw")
        assert g2.workers == sw.workers == 208
        assert g2.partials == sw.partials == 104
        assert not g2.key.l2_single_write
        assert sw.key.l2_single_write == (single_residency == 2)
    finally:
        scans._plan_w52a.cache_clear()


@pytest.mark.parametrize("w51a", ["off", "fp32"])
def test_off_ptx_sha(
    monkeypatch: pytest.MonkeyPatch, baseline: ModuleType, w51a: str,
    key56fwd: scans.Kernel, key56bwd: scans.Kernel,
) -> None:
    monkeypatch.setenv("LNET_SCAN_W51A", w51a)
    monkeypatch.setenv("LNET_SCAN_W52A", "off")
    try:
        scans._nvrtc()
    except OSError as exc:
        pytest.skip(str(exc))
    for seed in (key56fwd, key56bwd, replace(key56bwd, coarse=True, compact=True)):
        key = replace(seed, fp32_path=w51a == "fp32")
        previous = baseline.Kernel(**{k: v for k, v in vars(key).items() if k != "tanh_gain"})
        assert scans._kernel_source(key) == baseline._kernel_source(previous)
        old = baseline._compile_ptx(previous, (9, 0)).value
        new = scans._compile_ptx(key, (9, 0)).value
        assert hashlib.sha256(old).hexdigest() == hashlib.sha256(new).hexdigest()


@pytest.mark.parametrize("arch", ["90a", "86", "89"])
@pytest.mark.parametrize("fast_math", [False, True])
def test_nvcc_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arch: str, fast_math: bool,
    key56fwd: scans.Kernel, key56bwd: scans.Kernel,
) -> None:
    nvcc, cuobjdump = shutil.which("nvcc"), shutil.which("cuobjdump")
    if nvcc is None or cuobjdump is None:
        pytest.skip("Offline SASS check requires nvcc and cuobjdump")
    monkeypatch.setattr(scans, "_resident", lambda *_: 1)
    register_key = replace(
        key56fwd, register_path=True, cache_weights=False, vector_weights=False, cache_line=True,
    )
    plan28 = scans.ScanPlan(4, 28672, 8, cache_line=True)
    launch28 = scans.Launch(plan28, register_key, 4116, 4116)
    try:
        key28 = scans._plan_w52a(0, launch28, (1024, 28, 28, 160), "tanh_gain_smemw").key
    finally:
        scans._plan_w52a.cache_clear()
    assert key28.cache_weights
    assert key28.vector_weights
    assert not key28.register_path
    for name, key, max_sfu in (
        ("56d_fwd", key56fwd, 55), ("56d_bwd", key56bwd, 25), ("28d_fwd", key28, 55),
    ):
        source, cubin = tmp_path / f"{name}.cu", tmp_path / f"{name}.cubin"
        source.write_text(scans._kernel_source(replace(key, fp32_path=True, tanh_gain=True)))
        flags = ["--use_fast_math"] if fast_math else []
        result = subprocess.run(
            [nvcc, "-std=c++17", "-cubin", f"-arch=sm_{arch}", "-Xptxas=-v",
             *flags, str(source), "-o", str(cubin)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads" in result.stderr
        if arch == "90a":
            sass = subprocess.check_output([cuobjdump, "--dump-sass", str(cubin)], text=True)
            assert "MUFU.TANH" in sass
            if not fast_math:
                assert len(re.findall(r"\bMUFU\.", sass)) <= max_sfu
            if name == "28d_fwd":
                assert int(re.search(r"Used (\d+) registers", result.stderr)[1]) <= 104


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [bench.MODEL_SHAPES[0], *ODD_SHAPES], ids=lambda s: s[0])
def test_cuda_parity(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, shape: tuple,
) -> None:
    for name, value in {
        "SGS": "strict_ro", "VFWD": "batch4_w2", "VSCAN_SAVED": "off", "W51B": "off",
    }.items():
        monkeypatch.setenv(f"LNET_SCAN_{name}", value)
    monkeypatch.setenv("LNET_VSCAN_DECOUPLED", "off")
    monkeypatch.setenv("LNET_SCAN_W51A", "off")
    torch.manual_seed(52)
    batch = 8 if shape[0] == "56d" else 2
    reference = parity_reference(torch, bench, shape, batch, dtype)
    cache = {}
    for mode in scans.W52A_MODES[1:]:
        monkeypatch.setenv("LNET_SCAN_W52A", mode)
        result = parity(torch, bench, shape, reference, 0.035, cache)
        assert result["passed"], (mode, result)
