"""Wave 25C Stage 0 accuracy, layout, and input-gradient fallback."""

# ruff: noqa: T201
import pytest
import torch
from torch import nn
from torch.nn import functional

from lnet.image_layers import LayerNorm2d, ModeScaledTwoConvStem, _fused_stem_block


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 11, 13, 32), (1, 4, 8, 6, 64), (1, 5, 9, 7, 35)])
@pytest.mark.parametrize("input_grad", [False, True])
def test_stage0_split_cpu(monkeypatch, dtype, shape, input_grad) -> None:
    torch.manual_seed(160)
    batch, ci, height, width, co = shape
    stem = ModeScaledTwoConvStem(4, input_channels=ci, hidden_width=co)
    conv, norm = stem.layers[:2]
    x = torch.randn(batch, ci, height, width).requires_grad_(input_grad)
    params = (*conv.parameters(), *norm.parameters(), *((x,) if input_grad else ()))
    with torch.autocast("cpu", dtype=dtype, enabled=dtype != torch.float32):
        # The default CPU model uses separate modules, while the
        # existing split block provides the CUDA rounding contract on CPU.
        monkeypatch.setenv("LNET_STEM_STAGE0_OPT", "off")
        expected = _fused_stem_block(x, conv, norm)
        dy = torch.randn_like(expected)
        targets = (expected, *torch.autograd.grad(expected, params, dy))
        for mode in ("cudnn_split", "cudnn_pad4"):
            monkeypatch.setenv("LNET_STEM_STAGE0_OPT", mode)
            actual = _fused_stem_block(x, conv, norm)
            assert actual.dtype == dtype
            assert actual.is_contiguous(memory_format=torch.channels_last)
            grads = torch.autograd.grad(actual, params, dy)
            errors = [((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)).item()
                      for a, b in zip((actual, *grads), targets, strict=True)]
            bound = {torch.float32: 1e-5, torch.float16: 0.006, torch.bfloat16: 0.035}[dtype]
            assert max(errors) < bound, errors
            features = stem(x)
            expected_features = stem.layers[5](
                stem.layers[4].norm(stem.layers[3](actual).permute(0, 2, 3, 1))
            )
            torch.testing.assert_close(features, expected_features, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hw", [(224, 224), (113, 127)])
@pytest.mark.parametrize("ci", [3, 4])
@pytest.mark.parametrize("co", [32, 64])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("channels_last", [False, True])
def test_conv_stem(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    hw: tuple[int, int],
    ci: int,
    co: int,
    bias: bool,
    input_grad: bool,
    channels_last: bool,
) -> None:
    torch.manual_seed(91)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    conv = nn.Conv2d(ci, co, 3, 2, 1, bias=bias, device="cuda")
    norm = LayerNorm2d(co).cuda()
    x = torch.randn((2, ci, *hw), device="cuda")
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    x.requires_grad_(input_grad)
    params = (*conv.parameters(), *norm.parameters(), *((x,) if input_grad else ()))
    dy = torch.randn((2, co, (hw[0] + 1) // 2, (hw[1] + 1) // 2), device="cuda")
    results = {}
    for mode in ("off", "on", "fp32"):
        monkeypatch.setenv("LNET_FUSED_STEM_CONV", "off" if mode == "off" else "on")
        if mode == "fp32":
            y = functional.gelu(norm(conv(x)))
        else:
            with torch.autocast("cuda", dtype=dtype):
                y = _fused_stem_block(x, conv, norm)
        assert y.is_contiguous(memory_format=torch.channels_last)
        grads = torch.autograd.grad(y, params, dy.to(y.dtype))
        results[mode] = (y, *grads)
    for ref in ("off", "fp32"):
        errors = []
        for got, expected in zip(results["on"], results[ref], strict=True):
            delta = got.float() - expected.float()
            rel = delta.norm() / expected.float().norm().clamp_min(1e-8)
            errors.append((delta.abs().max().item(), rel.item()))
            bound = 0.035 if dtype == torch.bfloat16 else 0.006
            assert rel < bound, errors
        print(dtype, hw, ci, co, bias, input_grad, channels_last, ref, errors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_conv_stem_fullgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_FUSED_STEM_CONV", "on")
    conv = nn.Conv2d(3, 32, 3, 2, 1, device="cuda")
    norm = LayerNorm2d(32).cuda()
    x = torch.randn(2, 3, 33, 35, device="cuda")

    def forward(source: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return _fused_stem_block(source, conv, norm)

    explained = torch._dynamo.explain(forward)(x)
    assert (explained.graph_count, explained.graph_break_count) == (1, 0)
    assert "va_stem.conv_forward" in explained.graphs[0].code
    expected = forward(x)
    compiled = torch.compile(forward, fullgraph=True)(x)
    torch.testing.assert_close(compiled, expected)
    params = (*conv.parameters(), *norm.parameters())
    dy = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(compiled, params, dy)
    expected_grads = torch.autograd.grad(expected, params, dy)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference)
