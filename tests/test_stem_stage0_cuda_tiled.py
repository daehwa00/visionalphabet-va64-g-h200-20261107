"""Wave 50C stem and RMSNorm parity through the model entry points."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional

from lnet.fused_stem_ops import tiled_stem_enabled
from lnet.image_layers import LayerNorm2d, ModeScaledTwoConvStem, _fused_stem_block
from lnet.memory_backbone import ChannelRMSNorm
from lnet.model_w29c import _fast_rmsnorm

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_auto_stem_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", "auto")
    assert not tiled_stem_enabled()


def _run(forward, inputs: torch.Tensor, parameters: tuple, dy: torch.Tensor) -> tuple:
    y = forward(inputs)
    return (y, *torch.autograd.grad(y, parameters, dy.to(y.dtype)))


@pytest.mark.parametrize("input_grad", [False, True])
def test_wave50_cpu_stem_fullgraph(monkeypatch: pytest.MonkeyPatch, input_grad: bool) -> None:
    torch.manual_seed(176)
    stem = ModeScaledTwoConvStem(7, strides=(2, 3))
    x = torch.randn(2, 3, 33, 35, requires_grad=input_grad)
    params = (*stem.parameters(), *((x,) if input_grad else ()))
    dy = torch.randn(2, 6, 6, 14)
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", "off")
    expected = _run(stem, x, params, dy)
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", "cuda_tiled")
    compiled = torch.compile(stem, backend="aot_eager", fullgraph=True)
    actual = _run(compiled, x, params, dy)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wave50_ablation17_fullgraph_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from run_va_eight_ablations import build_variant  # noqa: PLC0415

    torch.manual_seed(176)
    model = build_variant("17-p1-nr4-1024d-v3").train()
    inputs = torch.randn(1, 3, 224, 224)
    candidate = deepcopy(model)
    results = []
    for stem, norm, module in (
        ("off", "off", model), ("cuda_tiled", "fast_fwd", candidate),
    ):
        monkeypatch.setenv("LNET_STEM_STAGE0_MODE", stem)
        monkeypatch.setenv("LNET_NORM_W50_MODE", norm)
        compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
        logits = compiled(inputs)[0]
        grads = torch.autograd.grad(logits.square().mean(), tuple(module.parameters()))
        assert len(grads) == 193
        assert all(torch.isfinite(grad).all() for grad in grads)
        results.append((logits, *grads))
    torch.testing.assert_close(*results, rtol=0, atol=0)


@pytest.fixture
def reset_dynamo() -> None:
    torch._dynamo.reset()


@pytest.mark.parametrize("width", range(1, 11))
def test_wave51a_dynamo_reset_cpu(width: int, reset_dynamo: None) -> None:  # noqa: ARG001
    with torch._dynamo.config.patch(recompile_limit=1):
        compiled = torch.compile(lambda x: x + 1, backend="eager", fullgraph=True, dynamic=False)
        torch.testing.assert_close(compiled(torch.zeros(width)), torch.ones(width))


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize(("stride", "padding"), [(2, 1), (3, 2)])
def test_wave50_cuda_stem_fullgraph(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, input_grad: bool,
    stride: int, padding: int, reset_dynamo: None,  # noqa: ARG001
) -> None:
    torch.manual_seed(176)
    monkeypatch.setenv("LNET_FUSED_STEM_CONV", "on")
    x = torch.randn(2, 3, 33, 35, device="cuda").contiguous(memory_format=torch.channels_last)
    x.requires_grad_(input_grad)
    conv = nn.Conv2d(3, 32, 3, stride, padding, device="cuda")
    norm = LayerNorm2d(32).cuda()
    params = (*conv.parameters(), *norm.parameters(), *((x,) if input_grad else ()))

    def forward(source: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype):
            return _fused_stem_block(source, conv, norm)

    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", "off")
    dy = torch.randn_like(forward(x))
    expected = _run(forward, x, params, dy)
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", "cuda_tiled")
    actual = _run(torch.compile(forward, backend="aot_eager", fullgraph=True), x, params, dy)
    limit = 0.035 if dtype == torch.bfloat16 else 0.006
    assert all(
        (got.float() - ref.float()).norm() <= limit * ref.float().norm().clamp_min(1e-8)
        for got, ref in zip(actual, expected, strict=True)
    )


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [63, 64, 96, 128, 256, 272])
def test_wave50_cuda_norm_fullgraph(
    monkeypatch: pytest.MonkeyPatch, dtype, width: int, reset_dynamo: None,  # noqa: ARG001
) -> None:
    torch.manual_seed(176)
    monkeypatch.setenv("LNET_NORM_W50_MODE", "fast_fwd")
    monkeypatch.setenv("LNET_MODEL_W36B", "off")
    norm = ChannelRMSNorm(width).cuda().to(dtype)
    x = torch.randn(2, 7, 9, width, device="cuda", dtype=dtype, requires_grad=True)

    def native(source: torch.Tensor) -> torch.Tensor:
        return functional.rms_norm(source, (width,), norm.weight, norm.epsilon)

    def forward(source: torch.Tensor) -> torch.Tensor:
        if dtype == torch.float32:
            return _fast_rmsnorm(source, norm.weight, norm.epsilon, True)[0]  # noqa: FBT003
        return norm(source)

    dy = torch.randn_like(x)
    params = x, norm.weight
    expected = _run(native, x, params, dy)
    actual = _run(torch.compile(forward, backend="aot_eager", fullgraph=True), x, params, dy)
    limit = {torch.bfloat16: 0.035, torch.float16: 0.006, torch.float32: 1e-5}[dtype]
    assert all(
        (got.float() - ref.float()).norm() <= limit * ref.float().norm().clamp_min(1e-8)
        for got, ref in zip(actual, expected, strict=True)
    )
