"""Wave 36B compiler policy, stem layout, and CUDA Graph regression checks."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from run_va_eight_ablations import build_variant
from torch import nn
from torch.nn import functional

from lnet.image_layers import ModeScaledTwoConvStem
from lnet.model_w36b import (
    compile_model_w36b,
    mark_model_w36b_step_begin,
    model_w36b_features,
    prepare_model_w36b_,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_compile_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(4, 4)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: calls.append(kwargs) or module)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: calls.append("step"))
    monkeypatch.setenv("LNET_MODEL_W36B", "off")
    assert compile_model_w36b(model) is model
    assert not calls
    monkeypatch.setenv("LNET_MODEL_W36B", "cuda_graphs")
    assert compile_model_w36b(model) is model
    assert calls.pop()["mode"] == "reduce-overhead"
    monkeypatch.setenv("LNET_MODEL_W36B", "cuda_graphs, max_autotune_gemm")
    assert compile_model_w36b(model) is model
    assert calls.pop()["options"] == {
        "triton.cudagraphs": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
        "aggressive_fusion": False,
    }
    mark_model_w36b_step_begin()
    assert calls == ["step"]
    monkeypatch.setenv("LNET_MODEL_W36B", "off,cuda_graphs")
    with pytest.raises(ValueError, match="LNET_MODEL_W36B"):
        model_w36b_features()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
def test_stem_cudnn_capture(
    dtype: torch.dtype, layout: torch.memory_format, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16")
    monkeypatch.setenv("LNET_MODEL_W36B", "stem_cudnn")
    torch.manual_seed(11)
    stem = ModeScaledTwoConvStem(32).cuda()
    x = torch.randn(2, 3, 32, 36, device="cuda").contiguous(memory_format=layout)
    first, norm1, _, second, norm2, _ = stem.layers

    def reference(hidden: torch.Tensor, conv: nn.Module, norm: nn.Module) -> torch.Tensor:
        weight = conv.weight.contiguous(memory_format=torch.channels_last)
        hidden = functional.conv2d(hidden, weight, None, conv.stride, conv.padding)
        pixels = hidden.permute(0, 2, 3, 1).float() + conv.bias
        return functional.gelu(norm.norm(pixels)).to(dtype)

    def forward() -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype):
            return stem(x)

    with torch.autocast("cuda", dtype=dtype):
        middle = reference(x, first, norm1).permute(0, 3, 1, 2)
        expected = reference(middle, second, norm2)
    params = tuple(stem.parameters())
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, params, grad)
    layouts = []

    original_conv = functional.conv2d

    def observe_conv(inputs: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        layouts.append(inputs.is_contiguous(memory_format=torch.channels_last))
        return original_conv(inputs, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(functional, "conv2d", observe_conv)
        forward()
    assert layouts[-1] is True

    current = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        actual = forward()
        torch.autograd.grad(actual, params, grad)
    current.wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        actual = forward()
        actual_grads = torch.autograd.grad(actual, params, grad)
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.03)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want, atol=0.08, rtol=0.04)


@requires_cuda
def test_variant_compiled_cudagraph(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16")
    monkeypatch.setenv("LNET_MODEL_W36B", "stem_cudnn,cuda_graphs")
    torch.manual_seed(501)
    eager = prepare_model_w36b_(build_variant("17-p1-nr4-1024d-v3").cuda())
    model = deepcopy(eager)
    compiled = compile_model_w36b(model)
    replays = []
    replay = torch.cuda.CUDAGraph.replay

    def count_replay(graph: torch.cuda.CUDAGraph) -> None:
        replays.append(graph)
        replay(graph)

    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", count_replay)
    sample = torch.randn(2, 3, 224, 224, device="cuda").contiguous(
        memory_format=torch.channels_last,
    )
    for _ in range(4):
        model.zero_grad(set_to_none=True)
        eager.zero_grad(set_to_none=True)
        mark_model_w36b_step_begin()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = compiled(sample)[0]
            expected = eager(sample)[0]
            loss = actual.float().square().mean()
            reference = expected.float().square().mean()
        loss.backward()
        reference.backward()
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
        for got, want in zip(model.parameters(), eager.parameters(), strict=True):
            if want.grad is None:
                assert got.grad is None
            else:
                torch.testing.assert_close(got.grad, want.grad, atol=0.005, rtol=0.04)
    assert len(set(replays)) >= 2, "expected forward and backward CUDA Graph replay"
    for name, buffer in model.named_buffers():
        if name.endswith(("running_ms", "ema_weight")):
            torch.testing.assert_close(buffer, dict(eager.named_buffers())[name])
