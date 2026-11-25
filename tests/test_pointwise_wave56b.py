"""Wave 56B dispatch, offline code generation and device contracts."""

import re
import shutil
import subprocess

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from bench_wave56b_pointwise_type1 import MODES, RMS, SWIGLU, case, errors
from lnet import model_w29c as ops


def test_modes(monkeypatch):
    monkeypatch.delenv("LNET_MODEL_W56B", raising=False)
    assert ops._wave56b_mode() == "off"
    for mode in MODES:
        monkeypatch.setenv("LNET_MODEL_W56B", mode)
        assert ops._wave56b_mode() == mode
        assert not ops.model_w29c_enabled(torch.device("cpu"), "rmsnorm_bwd")
    monkeypatch.setenv("LNET_MODEL_W56B", "typo")
    with pytest.raises(ValueError, match="LNET_MODEL_W56B"):
        ops._wave56b_mode()


def test_fake_outputs():
    with FakeTensorMode():
        x = torch.empty(3, 64, dtype=torch.float16)
        w, inv = torch.empty(64), torch.empty(3, 1)
        for dx, dw in ((True, True), (True, False), (False, True)):
            a, b = ops._vec_rms(x, w, inv, x, dx, dw, False)
            assert a.shape == (x.shape if dx else (0,))
            assert b.shape == (w.shape if dw else (0,))
        assert ops._vec_swiglu(x, x, False, True).shape == (3, 32)
        assert ops._vec_swiglu(x, x, True, False).shape == x.shape


@pytest.mark.parametrize("arch", ["sm_75", "sm_86", "sm_90a"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_offline_compiler(tmp_path, arch, dtype):
    if not shutil.which("nvcc") or not shutil.which("cuobjdump"):
        pytest.skip("requires CUDA compiler and binary utilities")
    configs = [
        (s[1], b, t, False, False) for s in SWIGLU for b in (False, True) for t in (False, True)
    ]
    configs += [
        (s[1], False, False, dx, dw)
        for s in RMS
        for dx, dw in ((True, True), (False, True), (True, False))
    ]
    if arch != "sm_90a":
        configs = [configs[3], configs[-3]]
    # Group independent specializations into a single compiler invocation.
    units = []
    names = []
    for i, args in enumerate(configs):
        _, _, tanh, dx, dw = args
        source = ops._vec_source(dtype, *args)
        name = f"wave56b_{i}"
        # Give each specialization private C++ types and a distinct C entry.
        source = source.replace("using T = SCALAR;", f"namespace ns{i} {{\nusing T = SCALAR;")
        source = source.replace("swiglu_vec8_kernel", name).replace("rmsnorm_bwd_vec8_kernel", name)
        units.append(
            source
            + "\n}\n"
            + "\n".join(
                "#undef " + k
                for k in (
                    "SCALAR",
                    "WIDTH",
                    "BACKWARD",
                    "TANH_MODE",
                    "DX",
                    "DW",
                    "RMS_KERNEL",
                    "RMS_BOUNDS",
                )
            )
        )
        names.append((name, tanh, args))
    cu, cubin = tmp_path / "wave.cu", tmp_path / "wave.cubin"
    cu.write_text("\n".join(units))
    compiler = subprocess.run(
        [
            "nvcc",
            "-arch=" + arch,
            "-std=c++17",
            "--cubin",
            "--ptxas-options=-v",
            str(cu),
            "-o",
            str(cubin),
        ],
        capture_output=True,
        text=True,
    )
    assert compiler.returncode == 0, compiler.stderr
    assert len(
        re.findall(
            r"0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads", compiler.stderr
        )
    ) == len(names)
    sass = subprocess.check_output(["cuobjdump", "-sass", str(cubin)], text=True)
    for name, tanh, args in names:
        section = sass.split("Function : " + name + "\n")[1].split("Function :")[0]
        assert re.search(r"LDG\.E(?:\.[A-Z]+)*\.128", section)
        assert re.search(r"STG\.E(?:\.[A-Z]+)*\.128", section)
        if tanh and arch != "sm_75":
            assert "MUFU.TANH" in section
        stats = compiler.stderr.split("Function properties for " + name + "\n")[1]
        regs = int(re.search(r"Used (\d+) registers", stats)[1])
        if arch == "sm_90a":
            assert regs <= 40
            assert "BAR.SYNC" not in section
        shared = re.search(r"(\d+) bytes smem", stats.split("Compiling entry")[0])
        print("AUDIT", arch, dtype, args, regs, int(shared[1]) if shared else 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 0.003), (torch.bfloat16, 0.01)])
def test_device(mode, dtype, tol, monkeypatch):
    for kind in ("fwd", "bwd", "rms"):
        for width in (64, 65, 512, 1024):
            pitch = width + 8 if kind == "rms" else 2 * width
            run, ref, _ = case(kind, (17, width, pitch, width, 1), 1, dtype, "cuda")
            monkeypatch.setenv("LNET_MODEL_W56B", "off")
            baseline = run()
            monkeypatch.setenv("LNET_MODEL_W56B", mode)
            eager = run()
            assert max(errors(eager, ref()) + errors(eager, baseline)) <= tol
            torch.compiler.reset()
            assert max(errors(torch.compile(run, fullgraph=True)(), eager)) <= tol
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                outputs = run()
            g.replay()
            assert max(errors(outputs, eager)) <= tol


def test_off_launch_geometry(monkeypatch):
    monkeypatch.delenv("LNET_MODEL_W56B", raising=False)
    launches = []

    class Probe:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(ops, "wrap_triton", lambda kernel: Probe())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: type("SM", (), {"multi_processor_count": 132}),
    )
    x = torch.empty(17000, 128, dtype=torch.bfloat16)
    ops._launch_swiglu(x, x, x[:, :64], backward=False)
    ops._launch_rms_bwd(x, x[0], x[:, 0], x, True, True, False)
    assert launches[0][0] == (2125, 1)
    assert launches[0][1][-3:] == (8, 64, False)
    assert launches[0][2] == dict(hiocc=False, divisor_tiled=False, enable_fp_fusion=False)
    assert launches[1][0] == (264,)
    assert launches[1][1][-4:] == (32, 128, True, True)
    assert launches[1][2] == dict(reduce_in_loop=False, enable_fp_fusion=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 0.003), (torch.bfloat16, 0.01)])
def test_grad_masks_and_offsets(monkeypatch, mode, dtype, tol):
    monkeypatch.setenv("LNET_MODEL_W56B", mode)
    for offset in (0, 1):

        def pitched(rows, width):
            return torch.randn(rows, width + 8, device="cuda", dtype=dtype)[
                :, offset : offset + width
            ]

        x, dy = pitched(19, 96), pitched(19, 96)
        w = torch.randn(96, device="cuda", dtype=dtype)
        inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        n = x.float() * inv
        scaled = dy.float() * w.float()
        for dx, dw in ((True, False), (False, True), (True, True)):
            a, b = ops._channel_rmsnorm_backward(x, w, inv, dy, need_dx=dx, need_dw=dw)
            if dx:
                assert (
                    max(errors([a], [(scaled - n / 96 * (scaled * n).sum(-1, keepdim=True)) * inv]))
                    < tol
                )
            else:
                assert a.numel() == 0
            if dw:
                assert max(errors([b], [(dy.float() * n).sum(0)])) < tol
            else:
                assert b.numel() == 0
        joint = torch.randn(19 * 192 + 8, device="cuda", dtype=dtype)[
            offset : offset + 19 * 192
        ].view(19, 192)
        joint.requires_grad_()
        out = ops.packed_swiglu_vec2d(joint)
        (grad,) = torch.autograd.grad(out, joint, dy.contiguous())
        a, b = joint.detach().float().chunk(2, -1)
        sig = b.sigmoid()
        assert (
            max(errors([grad], [torch.cat((dy * b * sig, dy * a * sig * (1 + b * (1 - sig))), -1)]))
            < tol
        )
