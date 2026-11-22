import pytest
import torch
from torch.nn import functional

from lnet.memory_backbone import ChannelRMSNorm

_FAST_FLAGS = ("LNET_FUSED_FACTORIZED_READER", "LNET_GENERAL_SCANS")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fast_flag", [None, "off", *_FAST_FLAGS])
def test_channel_rms_norm_forward_backward_fullgraph(device, dtype, fast_flag, monkeypatch) -> None:
    torch.compiler.reset()
    for flag in _FAST_FLAGS:
        if flag == fast_flag:
            monkeypatch.setenv(flag, "1")
        elif fast_flag is None:
            monkeypatch.delenv(flag, raising=False)
        else:
            monkeypatch.setenv(flag, "0")

    generator = torch.Generator().manual_seed(71)
    x = torch.randn((2, 3, 70), generator=generator).to(device=device, dtype=dtype)
    x = x[..., ::2].requires_grad_()
    norm = ChannelRMSNorm(35, epsilon=1e-4).to(device=device)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 35))

    weight = norm.weight.to(dtype)
    if device == "cuda" and fast_flag != "off":
        expected = functional.rms_norm(x, (35,), weight, norm.epsilon)
    else:
        inverse = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.epsilon).to(dtype)
        expected = x * inverse * weight

    grad = torch.randn(expected.shape, generator=generator).to(expected)
    inputs = (x, norm.weight)
    expected_grad = torch.autograd.grad(expected, inputs, grad)

    eager = norm(x)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    eager_grad = torch.autograd.grad(eager, inputs, grad)
    for actual, reference in zip(eager_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    compiled = torch.compile(
        norm, fullgraph=True, backend="inductor" if device == "cuda" else "aot_eager"
    )
    result = compiled(x)
    result_grad = torch.autograd.grad(result, inputs, grad)
    tol = {torch.float32: 2e-5, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(result, expected, rtol=tol, atol=tol)
    for actual, reference in zip(result_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference, rtol=tol, atol=tol)
