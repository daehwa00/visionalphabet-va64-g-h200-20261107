import re
import shutil
import subprocess

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from alphabet import tiled_swiglu_rmsnorm as ops

FLAGS = ((True, True), (True, False), (False, True))


def errors(actual, reference) -> list[float]:
    return [
        ((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)).item()
        for x, y in zip(actual, reference, strict=True)
    ]



def test_dispatch(monkeypatch) -> None:
    monkeypatch.delenv("ALPHABET_RMSNORM_KERNEL", raising=False)
    monkeypatch.setenv("ALPHABET_SWIGLU_KERNEL", "off")
    assert ops._vector_rmsnorm_mode() == "off"
    monkeypatch.setattr(ops, "_off_rms_bwd", lambda *_args, **_kwargs: "baseline")
    assert (
        ops._channel_rmsnorm_backward(None, None, None, None, need_dx=True, need_dw=True)
        == "baseline"
    )
    for mode in ops._RMSNORM_KERNEL_MODES:
        monkeypatch.setenv("ALPHABET_RMSNORM_KERNEL", mode)
        assert ops._vector_rmsnorm_mode() == mode
    monkeypatch.setenv("ALPHABET_RMSNORM_KERNEL", "bad")
    with pytest.raises(ValueError, match="ALPHABET_RMSNORM_KERNEL"):
        ops._vector_rmsnorm_mode()


def test_fake() -> None:
    with FakeTensorMode():
        x = torch.empty(3, 64, dtype=torch.bfloat16)
        w, inv = torch.empty(64), torch.empty(3, 1)
        for dx, dw in FLAGS:
            a, b = ops._sp_rms(x, w, inv, x, dx, dw, "cuda_spatial_vec8_4cta")
            assert a.shape == (x.shape if dx else (0,))
            assert a.dtype == x.dtype
            assert b.shape == (w.shape if dw else (0,))
            assert b.dtype == w.dtype


@pytest.mark.parametrize("arch", ["sm_90a", "sm_75", "sm_86"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_offline(tmp_path, arch, dtype) -> None:
    nvcc, cuobjdump = shutil.which("nvcc"), shutil.which("cuobjdump")
    if not nvcc or not cuobjdump:
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
            nvcc,
            "-arch=" + arch,
            "-std=c++17",
            "--cubin",
            "--ptxas-options=-v",
            str(cu),
            "-o",
            str(cubin),
        ],
        check=False, capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    sass = subprocess.check_output([cuobjdump, "-sass", str(cubin)], text=True)
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
        assert loads == 2, (loops, body)
        assert all("BAR.SYNC" not in op for op in body), (loops, body)
        assert "STG.E.128" in code
        assert ("SHFL" in code) == dx
        regs = int(re.search(r"Used (\d+) registers", stats)[1])
        smem = re.search(r"(\d+) bytes smem", stats)
        warps = min(8, 65536 // (256 * ((regs + 7) // 8 * 8))) * 8
        print(  # noqa: T201 -- offline compiler resource report
            "AUDIT", dtype, width, int(dx), int(dw), regs, int(smem[1]) if smem else 0, warps, loads
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ops._RMSNORM_KERNEL_MODES[1:])
@pytest.mark.parametrize(("dtype", "tol"), [(torch.bfloat16, 0.01), (torch.float16, 0.003)])
def test_device(monkeypatch, mode, dtype, tol) -> None:
    monkeypatch.setenv("ALPHABET_SWIGLU_KERNEL", "off")
    for width, offset in [(64, 0), (96, 0), (128, 0), (256, 0), (65, 0), (64, 1)]:

        def sample(width=width, offset=offset):
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

            def run(x=x, w=w, inv=inv, dy=dy, dx=dx, dw=dw):
                return ops._channel_rmsnorm_backward(x, w, inv, dy, need_dx=dx, need_dw=dw)

            monkeypatch.setenv("ALPHABET_RMSNORM_KERNEL", "off")
            baseline = run()
            monkeypatch.setenv("ALPHABET_RMSNORM_KERNEL", mode)
            actual = run()
            target = tuple(
                v if enabled else v.new_empty(0) for v, enabled in zip(ref, (dx, dw), strict=True)
            )
            assert max(errors(actual, target) + errors(actual, baseline)) <= tol
            if width == 96 and dx and dw:
                torch.compiler.reset()
                assert max(errors(torch.compile(run, fullgraph=True)(), actual)) <= tol
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                graph.replay()
                assert max(errors(captured, actual)) <= tol


def test_lockin_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("ALPHABET_SWIGLU_KERNEL", "vec8_tanh")
    monkeypatch.setattr(ops, "_tuned_swiglu", lambda _joint: "forward")
    monkeypatch.setattr(ops, "_vec_swiglu", lambda _joint, _grad, backward: backward)
    monkeypatch.setattr(ops, "_off_rms_bwd", lambda *_args, **_kwargs: "off")
    monkeypatch.setattr(ops, "_sp_rms", lambda *args: args[-1])
    for rms in ("off", "cuda_spatial_vec8_4cta"):
        monkeypatch.setenv("ALPHABET_RMSNORM_KERNEL", rms)
        assert ops.packed_swiglu_vec2d(None) == "forward"
        assert ops._swiglu_vec2d_backward(None, None) is True
        assert (
            ops._channel_rmsnorm_backward(None, None, None, None, need_dx=True, need_dw=True) == rms
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("dtype", "tol"), [(torch.bfloat16, 0.035), (torch.float16, 0.01)])
def test_lockin_device(monkeypatch, dtype, tol) -> None:
    joint = torch.randn(64, 192, device="cuda", dtype=dtype)
    grad = torch.randn_like(joint[:, :96]).contiguous()
    weight = torch.randn(96, device="cuda", dtype=dtype)

    def run():
        x = ops.packed_swiglu_vec2d(joint)
        inv = (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        dx, dw = ops._channel_rmsnorm_backward(x, weight, inv, grad, need_dx=True, need_dw=True)
        return x, ops._swiglu_vec2d_backward(joint, dx), dw

    monkeypatch.setenv("ALPHABET_SWIGLU_KERNEL", "off")
    monkeypatch.setenv("ALPHABET_RMSNORM_KERNEL", "off")
    reference = run()
    monkeypatch.setenv("ALPHABET_SWIGLU_KERNEL", "vec8_tanh")
    monkeypatch.setenv("ALPHABET_RMSNORM_KERNEL", "cuda_spatial_vec8_4cta")
    eager = run()
    assert max(errors(eager, reference)) <= tol
    assert max(errors(torch.compile(run, fullgraph=True)(), eager)) <= tol
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    assert max(errors(captured, eager)) <= tol
