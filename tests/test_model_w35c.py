"""CPU compilation and CUDA parity tests for the Wave 35C experiment arms."""

# ruff: noqa: PLC0415

import pytest
import torch
from torch import Tensor, nn

from lnet.memory_backbone import ChannelRMSNorm, SwiGLU
from lnet.model_w29c import packed_swiglu_vec2d
from lnet.model_w35c import apply_inductor_tune

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
MODES = (
    "off", "auto", "stem_v2", "swiglu_v3", "rmsnorm_bwd_inductor",
    "inductor_tune", "full_w35c",
)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("residual", [False, True])
def test_cpu_aot_autograd(mode: str, residual: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W35C", mode)
    norm, ffn = ChannelRMSNorm(16), SwiGLU(16, 24)
    x = torch.randn(2, 3, 16, requires_grad=True)
    skip = torch.randn_like(x, requires_grad=True)

    def run(x: Tensor, skip: Tensor) -> Tensor:
        return ffn(norm(x + skip if residual else x))

    params = (x, skip, *norm.parameters(), *ffn.parameters())
    expected = run(x, skip)
    gradients = torch.autograd.grad(expected.square().sum(), params, allow_unused=True)
    with apply_inductor_tune():
        actual = torch.compile(run, backend="aot_eager", fullgraph=True)(x, skip)
        actual_grad = torch.autograd.grad(actual.square().sum(), params, allow_unused=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for got, want in zip(actual_grad, gradients, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=2e-6)


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [64, 96, 128, 160, 256, 320, 512, 1024])
def test_swiglu_exact_cuda(width: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    joint = torch.randn(39, width * 2, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(39, width, device="cuda", dtype=dtype)
    pairs = []
    for mode in ("off", "swiglu_v3"):
        monkeypatch.setenv("LNET_MODEL_W35C", mode)
        y = packed_swiglu_vec2d(joint)
        (dx,) = torch.autograd.grad(y, joint, dy)
        pairs.append((y, dx))
    torch.testing.assert_close(*pairs, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", [torch.contiguous_format, torch.channels_last])
@pytest.mark.parametrize("channels", [32, 64])
def test_stem_cuda(
    channels: int, dtype: torch.dtype, layout: torch.memory_format,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lnet.image_layers import LayerNorm2d, _fused_stem_block

    monkeypatch.setenv("LNET_FUSED_STEM_CONV", "on")
    conv = nn.Conv2d(3, channels, 3, 2, 1, device="cuda")
    norm = LayerNorm2d(channels).cuda()
    x = torch.randn(2, 3, 35, 133, device="cuda").contiguous(memory_format=layout)
    dy = torch.randn(2, channels, 18, 67, device="cuda", dtype=dtype)
    pairs = []
    for mode in ("off", "stem_v2"):
        monkeypatch.setenv("LNET_MODEL_W35C", mode)
        with torch.autocast("cuda", dtype=dtype):
            y = _fused_stem_block(x, conv, norm)
        pairs.append((y, *torch.autograd.grad(y, (*conv.parameters(), *norm.parameters()), dy)))
    for expected, actual in zip(*pairs, strict=True):
        relative = (expected.float() - actual.float()).norm() / expected.float().norm()
        assert relative < (0.035 if dtype == torch.bfloat16 else 0.006)


@CUDA
@pytest.mark.parametrize("residual", [False, True])
def test_rmsnorm_cuda_aot_eager(residual: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W29C", "rmsnorm_bwd")
    x = torch.randn(21, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    skip = torch.randn_like(x, requires_grad=True)
    norm = ChannelRMSNorm(64).cuda()

    def run(x: Tensor, skip: Tensor) -> Tensor:
        return norm(x + skip if residual else x)

    dy = torch.randn_like(x)
    pairs = []
    for mode in ("off", "rmsnorm_bwd_inductor"):
        monkeypatch.setenv("LNET_MODEL_W35C", mode)
        y = torch.compile(run, backend="aot_eager", fullgraph=True)(x, skip)
        pairs.append((y, *torch.autograd.grad(y, (x, skip, norm.weight), dy, allow_unused=True)))
    torch.testing.assert_close(*pairs, rtol=0.02, atol=0.015)


def test_fusion_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    from torch._inductor import config

    monkeypatch.setenv("LNET_MODEL_W35C", "inductor_tune")
    before = config.epilogue_fusion, config.aggressive_fusion
    with apply_inductor_tune():
        assert config.epilogue_fusion
        assert config.aggressive_fusion
    assert before == (config.epilogue_fusion, config.aggressive_fusion)
