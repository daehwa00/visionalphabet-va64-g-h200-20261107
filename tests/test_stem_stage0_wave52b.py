"""Wave 52B Stage 0 dispatch, fake tensors, and CUDA parity."""

from __future__ import annotations

import pytest
import torch
from bench_wave52b_stem_type1 import (
    DTYPES,
    LAYOUTS,
    MODES,
    graph_finite,
    parity_errors,
    reference_gradients,
    run_step,
    stem_case,
    stem_forward,
    stem_parameters,
)
from torch._subclasses.fake_tensor import FakeTensorMode

from lnet.fused_stem_ops import (
    _conv_backward,
    _conv_forward,
    _stage0_cudnn_conv,
    _stage0_cudnn_grad,
    resolve_stage0_candidate,
    stem_stage0_mode,
    tiled_stem_enabled,
)
from lnet.image_layers import LayerNorm2d, _fused_stem_block

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_cpu_stem_results_have_no_grad_fn() -> None:
    x = torch.randn(1, 3, 5, 7, requires_grad=True)
    conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
    norm = LayerNorm2d(32)
    dy = torch.randn(1, 32, 3, 4)

    def forward(inputs):
        return _fused_stem_block(inputs, conv, norm)

    assert all(t.grad_fn is None for t in reference_gradients(x, conv, norm, dy))
    assert all(t.grad_fn is None for t in run_step(forward, x, stem_parameters(x, conv, norm), dy))


def test_default_and_invalid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LNET_STEM_STAGE0_MODE", raising=False)
    assert stem_stage0_mode() == "off"
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", "invalid")
    with pytest.raises(ValueError, match="LNET_STEM_STAGE0_MODE"):
        stem_stage0_mode()


@pytest.mark.parametrize("mode", [*MODES, "auto", "cuda_tiled"])
def test_parse_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", mode)
    assert stem_stage0_mode() == mode
    assert tiled_stem_enabled() == (mode == "cuda_tiled")


@pytest.mark.parametrize("mode", MODES)
def test_fallback(mode: str) -> None:
    direct = resolve_stage0_candidate(mode, needs_dx=False, width=224)
    dx = resolve_stage0_candidate(mode, needs_dx=True, width=224)
    assert direct == mode
    assert dx == ("cudnn_fused" if mode.startswith("triton_") else mode)
    wide = resolve_stage0_candidate(mode, needs_dx=False, width=512)
    assert wide == ("triton_tuned_v2" if mode == "triton_row_coalesced" else mode)


@pytest.mark.parametrize("mode", MODES)
def test_cpu_fullgraph_off_preservation(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    torch._dynamo.reset()
    x = torch.randn(1, 3, 17, 19, requires_grad=True)
    conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
    norm = LayerNorm2d(32)
    params = stem_parameters(x, conv, norm)
    dy = torch.randn(1, 32, 9, 10)

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        return _fused_stem_block(inputs, conv, norm)

    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", "off")
    expected = run_step(forward, x, params, dy)
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", mode)
    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(run_step(compiled, x, params, dy), expected, rtol=0, atol=0)


def test_cuda_op_fake_registrations() -> None:
    with FakeTensorMode():
        x = torch.empty((4, 3, 33, 35), dtype=torch.bfloat16, device="cuda")
        w = torch.empty((32, 3, 3, 3), dtype=x.dtype, device=x.device,
                        memory_format=torch.channels_last)
        packed = w.new_empty((27, 32))
        g = x.new_empty((32,), dtype=torch.float32)
        cb = x.new_empty((32,))
        for mode in ("off", "triton_row_coalesced", "triton_tuned_v2"):
            y, stats = _conv_forward(x, packed, g, g, g, 1e-5, mode)
            assert y.shape == (4, 17, 18, 32)
            assert stats.shape == (4, 17, 18, 2)
            assert [t.shape for t in _conv_backward(x, packed, g, g, g, y, stats, 1e-5, mode)] == [
                packed.shape, g.shape, g.shape, g.shape,
            ]
        y = _stage0_cudnn_conv(x, w, cb)
        assert y.shape == (4, 32, 17, 18)
        assert y.is_contiguous(memory_format=torch.channels_last)
        for needs_dx in (False, True):
            dx, dw, dc = _stage0_cudnn_grad(y, x, w, needs_dx, True, True)  # noqa: FBT003
            assert dx.shape == (x.shape if needs_dx else (0,))
            assert dw.shape == w.shape
            assert dc.shape == cb.shape


@CUDA
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("needs_dx", [False, True])
@pytest.mark.parametrize("layout", LAYOUTS.values())
def test_cuda_stage0_parity_compile_and_graph(
    monkeypatch: pytest.MonkeyPatch, mode: str, dtype: torch.dtype,
    needs_dx: bool, layout: torch.memory_format,
) -> None:
    torch._dynamo.reset()
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", mode)
    monkeypatch.setenv("LNET_FUSED_STEM_CONV", "on")
    x, conv, norm, dy = stem_case((4, 3, 33, 63), layout, dtype, needs_dx=needs_dx)
    forward = stem_forward(conv, norm, dtype)
    params = stem_parameters(x, conv, norm)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        expected = reference_gradients(x, conv, norm, dy)
        actual = run_step(forward, x, params, dy)
        errors = parity_errors(actual, expected)
        limit = 0.04 if dtype == torch.bfloat16 else 0.008
        assert all(value < limit for value in errors.values()), errors
        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(run_step(compiled, x, params, dy), actual)
        assert graph_finite(lambda: run_step(forward, x, params, dy))
