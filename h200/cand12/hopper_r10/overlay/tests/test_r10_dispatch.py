"""CPU/FakeTensor tests for opt-in 56² Hopper scan and RMS kernels."""

import pytest
import torch
from torch._inductor import config
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Graph

from lnet.r10_rms import RMS56ScaleGradientPass, VALUE_STRIDE, rms56_partial_rows
from lnet.r6_dispatch import associative_launch, r6_graph_pass
from lnet.r7_dispatch import direct_backward_launch, forward_launch
from lnet.r8_dispatch import horizontal_forward_launch


def test_horizontal_56_precedence_and_fallback(monkeypatch):
    with FakeTensorMode():
        source = torch.empty((1024, 56, 56, 160), device="cuda", dtype=torch.bfloat16)
        large = torch.empty((1070, 56, 56, 160), device="cuda", dtype=torch.bfloat16)
        other = torch.empty((1024, 28, 28, 160), device="cuda", dtype=torch.bfloat16)
        assert horizontal_forward_launch(source) is None
        monkeypatch.setenv("LNET_R8_HFWD", "1")
        assert horizontal_forward_launch(source)[1] == (5, 57344)
        monkeypatch.setenv("LNET_R10_HFWD56_SEG4", "1")
        kernel, grid = horizontal_forward_launch(source)
        assert kernel.fn.__name__ == "r10_hfwd56_seg4"
        assert grid == (57344, 5)
        cfg = kernel.configs[0]
        assert (cfg.num_warps, cfg.num_stages, cfg.kwargs) == (4, 3, {"BLOCK_MODES": 32})
        assert horizontal_forward_launch(other)[1] == (5, 28672)
        assert horizontal_forward_launch(large) is None


@pytest.mark.parametrize("batch", [4, 128, 512, 1024])
def test_tuned_geometries(batch, monkeypatch):
    monkeypatch.setenv("LNET_R10_TUNED_LAUNCH", "1")
    kernel, bh, group = direct_backward_launch(14, 14, batch, 160)
    cfg = kernel.configs[0]
    assert (bh, group, cfg.num_warps, cfg.num_stages) == (16, 4, 4, 2)
    assert cfg.kwargs == {"BLOCK_MODES": 4, "BLOCK_LINES": 2}
    kernel, grid, _, _, _ = associative_launch(28, 28, batch, 160, coarse=True)
    cfg = kernel.configs[0]
    assert grid == (20, batch * 14)
    assert (cfg.num_warps, cfg.num_stages) == (4, 2)
    for h, bh, lines in ((14, 16, 4), (28, 32, 1), (56, 64, 1)):
        kernel, block, options = forward_launch(h, h, batch, 160, coarse=True)
        cfg = kernel.configs[0]
        assert (block, cfg.num_warps, cfg.num_stages) == (bh, 4, 2)
        assert cfg.kwargs == {"BLOCK_MODES": 8, "BLOCK_LINES": lines}
        assert "SEGMENT" not in options
    monkeypatch.setenv("LNET_R10_SEG4PACK2", "1")
    kernel, _, options = forward_launch(56, 56, batch, 160, coarse=True)
    assert kernel.configs[0].num_stages == 1
    assert options["SEGMENT"] is True


@pytest.mark.parametrize("batch", [1, 37, 1024, 10699, 10700])
@pytest.mark.parametrize("gradient_stride", ["dense", "source"])
def test_rms56_fx_match_and_int32_bounds(batch, gradient_stride):
    shape = (batch, 56, 56, 64)
    with FakeTensorMode():
        graph = Graph()
        grad = graph.placeholder("grad")
        x = graph.placeholder("x")
        inv = graph.placeholder("inverse")
        grad.meta["val"] = torch.empty_strided(
            shape, VALUE_STRIDE if gradient_stride == "source" else (200704, 3584, 64, 1),
            device="cuda", dtype=torch.bfloat16,
        )
        x.meta["val"] = torch.empty_strided(shape, VALUE_STRIDE, device="cuda", dtype=torch.bfloat16)
        inv.meta["val"] = torch.empty((batch, 56, 56, 1), device="cuda")
        cast = graph.call_function(torch.ops.prims.convert_element_type.default, (inv, torch.bfloat16))
        unit = graph.call_function(torch.ops.aten.mul.Tensor, (x, cast))
        value = graph.call_function(torch.ops.aten.mul.Tensor, (unit, grad))
        reduced = graph.call_function(torch.ops.aten.sum.dim_IntList, (value, [0, 1, 2], True),
                                      {"dtype": torch.float32})
        output = graph.output(reduced)
    with config.patch(emulate_precision_casts=True):
        RMS56ScaleGradientPass(None)(graph)
    assert output.args[0] is reduced
    with config.patch(emulate_precision_casts=False):
        RMS56ScaleGradientPass(None)(graph)
    graph.lint()
    rows = rms56_partial_rows(batch)
    if batch * 56**2 * 64 < 2**31:
        assert output.args[0].target is torch.ops.lnet.r10_rms56_scale_gradient.default
        assert output.args[0].args == (grad, x, inv)
        assert 0 < rows <= 65535
    else:
        assert output.args[0] is reduced
        assert rows is None


def test_rms56_whole_workload_and_compile_scope(monkeypatch):
    assert rms56_partial_rows(1024) == 6272
    assert rms56_partial_rows(0) is None
    assert r6_graph_pass(None) is None
    monkeypatch.setenv("LNET_R10_RMS_DW56", "1")
    rms = r6_graph_pass(None)
    assert isinstance(rms.previous, RMS56ScaleGradientPass)
    monkeypatch.setenv("LNET_R10_HFWD56_SEG4", "1")
    assert r6_graph_pass(None).uuid() != rms.uuid()
