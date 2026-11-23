"""CUDA convolution parity and fallback coverage."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from lnet.fused_conv_stem_ops import conv3x3s2
from lnet.image_layers import CifarConvStem

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("mode", ["auto", "on", "off"])
def test_cpu_fallback(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("LNET_DIRECT_CONV_STEM", mode)
    stem = CifarConvStem(strides=(2, 2), bias=True)
    x = torch.randn(2, 3, 17, 23).contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(stem(x), stem.layers(x).permute(0, 2, 3, 1), atol=0, rtol=0)


@_CUDA
@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 1, 1), (2, 3, 37, 53, 32), (2, 4, 22, 19, 64), (1, 8, 9, 6, 19)]
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("grad_input", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_convolution_parity(dtype, grad_input, bias, shape, monkeypatch) -> None:
    monkeypatch.setenv("LNET_DIRECT_CONV_STEM", "on")
    torch.manual_seed(921)
    b, ci, h, w, co = shape
    x = (
        torch.randn(b, ci, h, w, device="cuda", dtype=dtype)
        .contiguous(
            memory_format=torch.channels_last,
        )
        .requires_grad_(grad_input)
    )
    weight = torch.randn(co, ci, 3, 3, device="cuda", dtype=dtype)
    if ci % 2 == 0:
        weight = weight.contiguous(memory_format=torch.channels_last)
    weight.requires_grad_()
    offset = torch.randn(co, device="cuda", dtype=dtype, requires_grad=True) if bias else None
    result = conv3x3s2(x, weight, offset)
    assert result.is_contiguous(memory_format=torch.channels_last)
    ref = functional.conv2d(
        x.float(),
        weight.float(),
        offset.float() if offset is not None else None,
        stride=2,
        padding=1,
    )
    grad = torch.randn_like(result)
    differentiated = tuple(t for t in (x, weight, offset) if t is not None and t.requires_grad)
    actual = (result, *torch.autograd.grad(result, differentiated, grad))
    expected = (ref, *torch.autograd.grad(ref, differentiated, grad.float()))
    tol = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}[dtype]
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right.to(dtype), atol=tol, rtol=tol)


@_CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compile_autocast(dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_DIRECT_CONV_STEM", "on")
    x = torch.randn(2, 3, 37, 53, device="cuda").contiguous(memory_format=torch.channels_last)
    w = torch.randn(32, 3, 3, 3, device="cuda", requires_grad=True)
    compiled = torch.compile(conv3x3s2, fullgraph=True)
    with torch.autocast("cuda", dtype=dtype):
        eager, result = conv3x3s2(x, w), compiled(x, w)
    assert result.dtype == dtype
    torch.testing.assert_close(result, eager, atol=0, rtol=0)
    grad = torch.randn_like(result)
    (actual,) = torch.autograd.grad(result, w, grad)
    (expected,) = torch.autograd.grad(eager, w, grad)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@_CUDA
def test_cifar_hook_has_one_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_DIRECT_CONV_STEM", "on")
    stem = CifarConvStem(strides=(2, 2), bias=True).cuda().half()
    x = torch.randn(2, 3, 17, 23, device="cuda", dtype=torch.float16).contiguous(
        memory_format=torch.channels_last,
    )
    result = torch._dynamo.explain(stem)(x)
    assert (result.graph_count, result.graph_break_count) == (1, 0)
    assert any(
        "va_direct_conv.forward" in str(node.target) for node in result.graphs[0].graph.nodes
    )


@_CUDA
def test_frozen_weight_and_bias_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_DIRECT_CONV_STEM", "on")
    x = (
        torch.randn(1, 3, 19, 26, device="cuda", dtype=torch.float16)
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_()
    )
    w = torch.randn(32, 3, 3, 3, device="cuda", dtype=x.dtype)
    b = torch.rand(32, device="cuda", dtype=x.dtype, requires_grad=True)
    y = conv3x3s2(x, w, b)
    ref = functional.conv2d(x.float(), w.float(), b.float(), stride=2, padding=1)
    grad = torch.randn_like(y)
    actual = torch.autograd.grad(y, (x, b), grad)
    expected = torch.autograd.grad(ref, (x, b), grad.float())
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=2e-3, rtol=2e-3)
