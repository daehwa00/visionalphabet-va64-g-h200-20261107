import re
import shutil
import subprocess

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from bench_wave56b_pointwise_type1 import errors
from lnet import model_w29c as ops


FLAGS = ((True, True), (True, False), (False, True))


def test_dispatch(monkeypatch):
    monkeypatch.delenv("LNET_MODEL_W57B_RMS", raising=False)
    monkeypatch.setenv("LNET_MODEL_W56B", "off")
    assert ops._wave57b_mode() == "off"
    monkeypatch.setattr(ops, "_off_rms_bwd", lambda *args, **kwargs: "baseline")
    assert (
        ops._channel_rmsnorm_backward(None, None, None, None, need_dx=True, need_dw=True)
        == "baseline"
    )
    for mode in ops._W57_MODES:
        monkeypatch.setenv("LNET_MODEL_W57B_RMS", mode)
        assert ops._wave57b_mode() == mode
    monkeypatch.setenv("LNET_MODEL_W57B_RMS", "bad")
    with pytest.raises(ValueError, match="LNET_MODEL_W57B_RMS"):
        ops._wave57b_mode()


def test_fake():
    with FakeTensorMode():
        x = torch.empty(3, 64, dtype=torch.bfloat16)
        w, inv = torch.empty(64), torch.empty(3, 1)
        for dx, dw in FLAGS:
            a, b = ops._sp_rms(x, w, inv, x, dx, dw, "cuda_sp_vec8_4cta")
            assert a.shape == (x.shape if dx else (0,)) and a.dtype == x.dtype
            assert b.shape == (w.shape if dw else (0,)) and b.dtype == w.dtype


@pytest.mark.parametrize("arch", ["sm_90a", "sm_75", "sm_86"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_offline(tmp_path, arch, dtype):
    if not all(shutil.which(t) for t in ("nvcc", "cuobjdump")):
        pytest.skip("requires nvcc and cuobjdump")
    configs = [(w, dx, dw) for w in (64, 96, 128, 256) for dx, dw in FLAGS]
    if arch != "sm_90a":
        configs = configs[:1]
    units = []
    for i, (width, dx, dw) in enumerate(configs):
        source = ops._sp_source(dtype, width, dx, dw).replace(
            "using T =", f"namespace n{i} {{\nusing T ="
        )
        units.append(
            source.replace("void rms_sp(", f"void sp{i}(")
            + "\n}\n"
            + "\n".join("#undef " + k for k in ("SCALAR", "G", "WIDTH", "DX", "DW"))
        )
    cu, cubin = tmp_path / "audit.cu", tmp_path / "audit.cubin"
    cu.write_text("\n".join(units))
    p = subprocess.run(
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
    assert p.returncode == 0, p.stderr
    sass = subprocess.check_output(["cuobjdump", "-sass", str(cubin)], text=True)
    for i, (width, dx, dw) in enumerate(configs):
        stats = p.stderr.split(f"Function properties for sp{i}\n")[1].split("Compiling entry")[0]
        assert "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads" in stats
        code = sass.split(f"Function : sp{i}\n")[1].split("Function :")[0]
        if arch != "sm_90a":
            continue
        ins = [(int(a, 16), op) for a, op in re.findall(r"/\*([0-9a-f]+)\*/\s+([^/\n]+?;)", code)]
        loops = [
            (int(m[1], 16), a)
            for a, op in ins
            if (m := re.search(r"BRA\s+0x([0-9a-f]+)", op)) and int(m[1], 16) < a
        ]
        body = [op for a, op in ins if any(lo <= a <= hi for lo, hi in loops)]
        loads = sum("LDG.E.128" in op for op in body)
        assert loads == 2 and all("BAR.SYNC" not in op for op in body), (loops, body)
        assert "STG.E.128" in code
        assert ("SHFL" in code) == dx
        regs = int(re.search(r"Used (\d+) registers", stats)[1])
        smem = re.search(r"(\d+) bytes smem", stats)
        warps = min(8, 65536 // (256 * ((regs + 7) // 8 * 8))) * 8
        print(
            "AUDIT", dtype, width, int(dx), int(dw), regs, int(smem[1]) if smem else 0, warps, loads
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ops._W57_MODES[1:])
@pytest.mark.parametrize("dtype,tol", [(torch.bfloat16, 0.01), (torch.float16, 0.003)])
def test_device(monkeypatch, mode, dtype, tol):
    monkeypatch.setenv("LNET_MODEL_W56B", "off")
    for width, offset in [(64, 0), (96, 0), (128, 0), (256, 0), (65, 0), (64, 1)]:

        def sample():
            return torch.randn(37, width + 8, device="cuda", dtype=dtype)[
                :, offset : offset + width
            ]

        x, dy = sample(), sample()
        w = torch.randn(width, device="cuda", dtype=dtype)
        inv = (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        norm, scaled = x.float() * inv, dy.float() * w.float()
        ref = (
            (scaled - norm * (scaled * norm).mean(-1, keepdim=True)) * inv,
            (dy.float() * norm).sum(0),
        )
        for dx, dw in FLAGS:

            def run():
                return ops._channel_rmsnorm_backward(x, w, inv, dy, need_dx=dx, need_dw=dw)

            monkeypatch.setenv("LNET_MODEL_W57B_RMS", "off")
            baseline = run()
            monkeypatch.setenv("LNET_MODEL_W57B_RMS", mode)
            actual = run()
            target = tuple(v if enabled else v.new_empty(0) for v, enabled in zip(ref, (dx, dw)))
            assert max(errors(actual, target) + errors(actual, baseline)) <= tol
            if width == 96 and dx and dw:
                torch.compiler.reset()
                assert max(errors(torch.compile(run, fullgraph=True)(), actual)) <= tol
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                graph.replay()
                assert max(errors(captured, actual)) <= tol


def test_swiglu_best(monkeypatch):
    monkeypatch.setenv("LNET_MODEL_W56B", "swiglu_best")
    monkeypatch.delenv("LNET_MODEL_W57B_RMS", raising=False)
    monkeypatch.setattr(ops, "_off_swiglu", lambda x: "off")
    monkeypatch.setattr(ops, "_tuned_swiglu", lambda x, mode: mode)
    monkeypatch.setattr(ops, "_tuned_swiglu_backward", lambda x, grad, mode: mode)
    monkeypatch.setattr(ops, "_off_rms_bwd", lambda *args, **kwargs: "off")
    for width in (96, 144, 192, 384):
        joint = torch.empty(1, 2 * width)
        assert ops.packed_swiglu_vec2d(joint) == ("off" if width <= 96 else "cuda_vec8_tanh")
        assert ops._swiglu_vec2d_backward(joint, joint) == "cuda_vec8_tanh"
    assert ops._channel_rmsnorm_backward(None, None, None, None, need_dx=True, need_dw=True) == "off"
