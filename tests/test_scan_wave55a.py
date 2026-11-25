"""Wave 55A offline SM90 resource checks and opt-in device integration tests."""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import bench_scan_micro as micro
import bench_wave55a_scans_type1 as bench
import pytest
import torch
from test_scan_wave53a import limits

from lnet import general_cuda_scans as scans

BASE = "33d9c5435f8b4e052c5e55073bd96ad767cbbc05"


@pytest.fixture(autouse=True)
def isolate():
    with patch.dict(os.environ):
        yield


def plan(module, shape, vertical, backward, dtype=torch.bfloat16, aligned=True):
    _, h, w, m, coarse, _ = shape
    src = SimpleNamespace(
        is_cuda=True,
        dtype=dtype,
        device=torch.device("cuda"),
        numel=lambda: 1024 * h * w * m * (4 if vertical else 1),
    )
    return module._plan_launch(
        src,
        (1024, h, w, m),
        backward=backward,
        vertical=vertical,
        coarse=coarse and vertical,
        hidden=8 if vertical else 0,
        precise=vertical and not backward,
        compact=coarse and backward,
        interleaved=(True, True),
        **({"aligned": aligned} if module is scans else {}),
    )


@pytest.mark.parametrize("lockin", [False, True])
def test_lockin_ptx(monkeypatch, lockin):
    root = Path(scans.__file__).resolve().parents[2]
    code = subprocess.check_output(
        ["git", "show", f"{BASE}:src/lnet/general_cuda_scans.py"], cwd=root
    )
    previous = ModuleType("lnet._wave55_base")
    previous.__package__ = "lnet"
    with patch.dict(sys.modules, {previous.__name__: previous}):
        exec(
            compile(code.split(b"# Custom ops and public API", 1)[0], "<base>", "exec"),
            vars(previous),
        )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    for module in (scans, previous):
        limits(monkeypatch, module)
    cache = {}
    for vertical in (False, True):
        bench.configure("w53_lockin" if vertical else "h53_lockin")
        if not lockin:
            for name in tuple(os.environ):
                if name.startswith("LNET_SCAN_"):
                    monkeypatch.delenv(name)
        for dtype in (torch.float16, torch.bfloat16):
            for shape in micro.MODEL_SHAPES:
                for backward in (False, True):
                    pair = []
                    for module in (scans, previous):
                        launch = plan(module, shape, vertical, backward, dtype)
                        sig = module, launch.key
                        if sig not in cache:
                            cache[sig] = hashlib.sha256(
                                module._compile_ptx(launch.key, (9, 0)).value
                            ).digest()
                        pair.append((launch.plan, launch.workers, launch.partials, cache[sig]))
                    assert pair[0] == pair[1], (shape, dtype, vertical, backward)


@pytest.fixture(scope="module")
def resources(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    if not nvcc:
        pytest.skip("nvcc unavailable")
    path = tmp_path_factory.mktemp("sm90")
    cache = {}

    def compile_key(key):
        if key not in cache:
            src = path / f"kernel{len(cache)}.cu"
            src.write_text(scans._kernel_source(key))
            proc = subprocess.run(
                [
                    nvcc,
                    "-arch=sm_90a",
                    "-std=c++17",
                    "--cubin",
                    "--ptxas-options=-v",
                    str(src),
                    "-o",
                    str(src.with_suffix(".cubin")),
                ],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, proc.stderr
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
            sass = subprocess.check_output(
                ["cuobjdump", "-sass", str(src.with_suffix(".cubin"))], text=True
            )
            if key.vertical_w55 != "off":
                opcode = (
                    r"\bHFMA2(?:\.MMA)?\.BF16_V2\b" if key.dtype == torch.bfloat16 else r"\bHFMA2\b"
                )
                # The 8-by-8 packed product contributes at least 64 instructions.
                assert len(re.findall(opcode, sass)) >= 64
            cache[key] = regs, mem, sass.count(" HFMA2"), sass.count(" FFMA")
        return cache[key]

    return compile_key


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sm90(monkeypatch, resources, dtype):
    limits(monkeypatch, scans)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    failures = []
    for mode in (*bench.VERTICAL, *bench.HORIZONTAL):
        vertical = mode in bench.VERTICAL
        bench.configure(mode)
        for shape in micro.MODEL_SHAPES:
            for backward in (False, True):
                launch = plan(scans, shape, vertical, backward, dtype)
                regs, mem, hfma, ffma = resources(launch.key)
                smem = launch.plan.smem
                print(
                    "RESOURCE",
                    str(dtype),
                    mode,
                    shape[0],
                    "bwd" if backward else "fwd",
                    regs,
                    smem,
                    occupancy(regs, smem, launch.key.warps),
                    *mem,
                    hfma,
                    ffma,
                    flush=True,
                )
                if any(mem) or (vertical and mode.startswith("v55") and not hfma):
                    failures.append((mode, shape[0], backward, mem, hfma))
    assert not failures, failures


def occupancy(regs, smem, warps):
    warp_regs = ((regs * 32 + 255) // 256) * 256
    return warps * min(
        32,
        64 // warps,
        65536 // (warp_regs * warps),
        (228 << 10) // (1024 + ((smem + 255) // 256) * 256),
    )


def test_alignment_and_modes(monkeypatch):
    for env, modes in (("LNET_SCAN_W55A", scans.W55A_MODES), ("LNET_SCAN_H55", scans.H55_MODES)):
        monkeypatch.delenv(env, raising=False)
        assert scans._scan_mode(env, modes) == "off"
        monkeypatch.setenv(env, "invalid")
        with pytest.raises(ValueError, match=env):
            scans._scan_mode(env, modes)
    x = torch.zeros(4, 36, dtype=torch.float16)
    assert scans._vec2_aligned(x)
    assert not scans._vec2_aligned(x[:, 1:])
    assert not scans._vec2_aligned(x.as_strided((4, 30), (35, 1)))
    limits(monkeypatch, scans)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    bench.configure("h55_bwd_vec2_nocs")
    for backward in (False, True):
        for m, aligned in ((33, True), (32, False)):
            k = plan(scans, ("edge", 15, 17, m, False, 1), False, backward, aligned=aligned).key
            assert not k.horizontal_vec2


def test_planner_precision_and_arch(monkeypatch):
    limits(monkeypatch, scans)
    shape = micro.MODEL_SHAPES[0]
    bench.configure("v55_bf16x2_chunkacc")
    for arch in ((7, 5), (8, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _, arch=arch: arch)
        for dtype in (torch.float16, torch.bfloat16):
            for backward in (False, True):
                key = plan(scans, shape, True, backward, dtype).key
                if arch < (8, 0):
                    assert key.vertical_w55 == "off"
                elif dtype == torch.float16:
                    assert key.vertical_w55 == "v55_bf16x2_bf16w"
                else:
                    assert key.vertical_w55 == "v55_bf16x2_chunkacc"
                    assert key.partial_shared_acc == backward


def test_ablation17_cpu_bf16():
    from run_va_eight_ablations import build_variant

    bench.configure("v55_bf16x2_chunkacc")
    os.environ["LNET_SCAN_H55"] = "h55_bwd_vec2_nocs"
    model = build_variant("17-p1-nr4-1024d-v3").train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(torch.randn(1, 3, 224, 224))[0]
    grads = torch.autograd.grad(logits.float().square().mean(), tuple(model.parameters()))
    assert len(grads) == 193 and all(g.isfinite().all() for g in grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", (*scans.W55A_MODES[1:], *scans.H55_MODES[1:]))
def test_cuda(mode):
    rows = bench.check_mode(mode, 1, 0.035)
    assert all(row["passed"] for row in rows.values()), rows
    for direction in ("fwd", "bwd"):
        bench.configure(mode)
        op = ("vscan_" if mode in scans.W55A_MODES else "hscan_") + direction
        run, _ = micro._case(
            op, micro.MODEL_SHAPES[1], 1, torch.bfloat16, compact=direction == "bwd"
        )
        expected = run()
        got = torch.compile(run, backend="aot_eager", fullgraph=True)()
        torch.testing.assert_close(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_misaligned_cuda():
    m = 32
    shape = (1, 15, 17, m)
    src = [
        torch.randn(1 + 15 * 17 * m, device="cuda", dtype=torch.bfloat16)[1:].view(shape)
        for _ in range(2)
    ]
    assert not scans._vec2_aligned(*src)
    pole = [torch.full((m,), x, device="cuda") for x in (0.73, 0.13, 0.21, 0.09)]
    bench.configure("h53_lockin")
    expected = scans.horizontal_forward(pole, *src)
    bench.configure("h55_fwd_vec2_cached")
    torch.testing.assert_close(scans.horizontal_forward(pole, *src), expected, atol=0, rtol=0)
    grad = torch.randn_like(expected)
    bench.configure("h53_lockin")
    reference = scans.horizontal_backward(pole, src, expected, grad, compact=False)
    bench.configure("h55_bwd_vec2_nocs")
    torch.testing.assert_close(
        scans.horizontal_backward(pole, src, expected, grad, compact=False),
        reference,
        atol=0,
        rtol=0,
    )
