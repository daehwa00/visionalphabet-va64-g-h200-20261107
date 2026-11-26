"""CPU contracts exercise the production kernels through Triton's interpreter."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_wave60b_hopper_fp8 as bench
from lnet import hopper_fp8_ops as ops


def test_production_shapes():
    torch.set_num_threads(1)
    assert bench.production_shapes() == [bench.CONV, bench.GEMM]


@pytest.mark.parametrize("dtype", [ops.E4, ops.E5])
def test_scale(dtype):
    x = torch.randn(1024, generator=torch.Generator().manual_seed(219))
    q, s = ops.quantize(x, dtype)
    assert F.cosine_similarity(q.float() * s, x, dim=0) >= 0.99
    zeros, s = ops.quantize(torch.zeros(16), dtype)
    assert zeros.float().eq(0).all() and s > 0


def test_fake():
    with FakeTensorMode():
        x = torch.empty(3, 7, 7, 16, device="cuda")
        t = ops.patches(x, x.new_empty(()), True)
        assert t.shape == (144, 160) and t.dtype == ops.E4


def test_unaligned():
    x, w = torch.randn(32, 15), torch.randn(17, 15)
    with patch.object(ops, "hopper", return_value=True):
        torch.testing.assert_close(ops.fp8_linear(x, w), F.linear(x, w), atol=0, rtol=0)


def test_patch_preserves_checkpoint_and_cpu_outputs():
    model = bench.build_variant(bench.VARIANT).eval()
    state = {name: id(param) for name, param in model.named_parameters()}
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad(), patch.dict(os.environ, LNET_READER_EXEC_MODE="dense_nhwc"):
        expected = model(x)
        keys = tuple(model.state_dict())
        ops.apply_fp8_module_patch(model)
        actual = model(x)
    assert {name: id(param) for name, param in model.named_parameters()} == state
    assert tuple(model.state_dict()) == keys
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_kernel_interpreter():
    subprocess.run(
        [sys.executable, __file__],
        check=True,
        cwd=ROOT,
        env={**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""},
    )


def interpret():
    torch.manual_seed(219)
    x = torch.randint(-4, 5, (2, 3, 5, 16)).float()
    col = F.unfold(x.permute(0, 3, 1, 2), 3, padding=1).reshape(2, 16, 9, 15)
    col = col.permute(0, 3, 2, 1).reshape(30, 144)
    padded = F.pad(col, (0, 0, 0, 2))
    for t in (False, True):
        actual = ops.patches(x, torch.tensor(1.0), t)
        torch.testing.assert_close(actual.float(), padded.T if t else padded, atol=0, rtol=0)
    grad = torch.randn_like(padded).round()
    dx = torch.empty_like(x)
    ops._fold[(2,)](grad, dx, 3, 5, 16, x.numel(), 256)
    ref = F.fold(
        grad[:30].reshape(2, 15, 9, 16).permute(0, 3, 2, 1).reshape(2, 144, 15),
        (3, 5),
        3,
        padding=1,
    )
    torch.testing.assert_close(dx, ref.permute(0, 2, 3, 1), atol=0, rtol=0)
    for dtype in (ops.E4, ops.E5):
        q, t, s = ops.pair(x.reshape(30, 16), dtype, True)
        expected, _ = ops.quantize(x.reshape(30, 16), dtype)
        assert F.cosine_similarity(q.float().flatten(), expected.float().flatten(), dim=0) >= 0.99
        torch.testing.assert_close(t.float(), q.T.float(), rtol=0, atol=0)

    # Emulate only the Hopper GEMM instruction. Real layout transforms, scales,
    # padding, multi-chunk assembly, and autograd formulas execute on the CPU.
    def software_mm(a, bt, sa, sb, dtype, fast):
        assert a.is_contiguous() and bt.is_contiguous()
        assert a.dtype in (ops.E4, ops.E5) and bt.dtype == ops.E4
        return ((a.float() @ bt.float().T) * sa * sb).to(dtype)

    for kind in ("linear", "conv"):
        if kind == "linear":
            x, w = torch.randn(64, 32), torch.randn(48, 32)
            fn = lambda a, b: ops._Linear.apply(a, b, True, False, False)
            base = F.linear
        else:
            x, w = torch.randn(3, 5, 7, 16), torch.randn(32, 16, 3, 3)
            fn = ops._Conv.apply
            base = lambda a, b: F.conv2d(a.permute(0, 3, 1, 2), b, padding=1).permute(0, 2, 3, 1)
        x.requires_grad_()
        w.requires_grad_()
        with patch.object(ops, "mm", software_mm), patch.object(ops, "chunk_size", return_value=2):
            y, ref = fn(x, w), base(x, w)
            g = torch.randn_like(ref)
            actual = (y, *torch.autograd.grad(y, (x, w), g))
            expected = (ref, *torch.autograd.grad(ref, (x, w), g))
            assert all(bench.metrics(a, b)["cosine"] >= 0.99 for a, b in zip(actual, expected))


hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires Hopper",
)


@hopper
@pytest.mark.parametrize("kind", ["conv", "linear"])
def test_hopper(kind):
    if kind == "conv":
        fn = ops.fp8_dense_nhwc_conv2d
        x, w = bench.sample((3, 9, 9, 32)), bench.sample((64, 32, 3, 3))
        base = lambda a, b: F.conv2d(a.permute(0, 3, 1, 2), b, padding=1).permute(0, 2, 3, 1)
    else:
        fn = ops.fp8_linear
        x, w = bench.sample((128, 64)), bench.sample((32, 64))
        base = F.linear
    a, b = (v.detach().float().requires_grad_() for v in (x, w))
    y = base(a, b)
    g = torch.randn_like(y, dtype=x.dtype)
    refs = (y, *torch.autograd.grad(y, (a, b), g.float()))
    record = bench.evaluate(torch.compile(fn, fullgraph=True), x, w, g, refs, 2)
    assert record["graph_finite"]
    assert all(e["cosine"] >= 0.99 for e in record["errors"].values())


@hopper
def test_training():
    result = bench.train_gate(16, 5)
    assert result["max_relative_drift"] < 0.1
    assert all(i["gradient_cosine"] > 0 for i in result["steps"])


if __name__ == "__main__":
    interpret()
