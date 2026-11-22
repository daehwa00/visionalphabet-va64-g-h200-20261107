"""Check the fused stem contract against native PyTorch."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.nn import functional

from lnet.fused_stem_ops import fused_nhwc_layernorm_gelu
from lnet.image_layers import ModeScaledTwoConvStem, _fused_stem_enabled, _fused_stem_large_enough


def test_fused_stem_default_and_opt_out(monkeypatch) -> None:
    flags = "LNET_FUSED_FACTORIZED_READER", "LNET_GENERAL_SCANS"
    for flag in flags:
        monkeypatch.delenv(flag, raising=False)
    cuda_input = cast("torch.Tensor", SimpleNamespace(is_cuda=True))
    assert _fused_stem_enabled(cuda_input)
    assert not _fused_stem_enabled(torch.empty(()))
    monkeypatch.setenv(flags[0], "0")
    assert _fused_stem_enabled(cuda_input)
    monkeypatch.setenv(flags[1], "0")
    assert not _fused_stem_enabled(cuda_input)


@pytest.mark.parametrize(
    ("batch", "height", "width", "modes", "expected"),
    [
        pytest.param(4, 512, 768, 32, False, id="dense"),
        pytest.param(128, 224, 224, 64, True, id="va17"),
    ],
)
def test_fused_stem_size_gate(batch, height, width, modes, expected) -> None:
    stem = ModeScaledTwoConvStem(modes)
    inputs = torch.empty(()).expand(batch, 3, height, width)
    assert _fused_stem_large_enough(inputs, stem.layers[0], stem.layers[3]) is expected


def _check(
    device, dtype, channels, *, nchw, mixed, compiled=False, unaligned=False, autocast=False
):
    generator = torch.Generator(device=device).manual_seed(123)
    shape = (2, 3, 5, channels)
    flat = torch.randn(2 * 3 * 5 * channels + 1, generator=generator, device=device, dtype=dtype)
    pixels = flat[int(unaligned) : int(unaligned) + 2 * 3 * 5 * channels].view(shape)
    x = (pixels.permute(0, 3, 1, 2) if nchw else pixels).requires_grad_()
    affine_dtype = torch.float32 if mixed else dtype
    w = torch.randn(channels, generator=generator, device=device, dtype=affine_dtype)
    b = torch.randn(channels, generator=generator, device=device, dtype=affine_dtype)
    w.requires_grad_()
    b.requires_grad_()
    # Rebuild reference from x so the gradient is checked in the public layout.
    ref_input = x.permute(0, 2, 3, 1) if nchw else x
    if device == "cuda" and mixed:
        normalized = functional.layer_norm(ref_input.float(), (channels,), w, b, 2e-5)
        expected = functional.gelu(normalized if autocast else normalized.to(dtype))
    else:
        expected = functional.gelu(functional.layer_norm(ref_input, (channels,), w, b, 2e-5))
    if nchw:
        expected = expected.permute(0, 3, 1, 2)
    if compiled:
        torch.compiler.reset()
    op = (
        torch.compile(fused_nhwc_layernorm_gelu, fullgraph=True)
        if compiled
        else (fused_nhwc_layernorm_gelu)
    )
    with torch.autocast(device, dtype=dtype, enabled=autocast):
        actual = op(x, w, b, 2e-5)
    assert actual.stride() == expected.stride()
    dy = torch.randn(actual.shape, generator=generator, device=device, dtype=expected.dtype)
    inputs = (x, w, b)
    tol = {
        torch.float32: (7e-5, 7e-5),
        torch.float16: (3e-3, 3e-3),
        torch.bfloat16: (3e-2, 4e-2),
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol[0], atol=tol[1])
    for got, want in zip(
        torch.autograd.grad(actual, inputs, dy),
        torch.autograd.grad(expected, inputs, dy),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=tol[0], atol=tol[1])


@pytest.mark.parametrize("nchw", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu(dtype, nchw) -> None:
    _check("cpu", dtype, 13, nchw=nchw, mixed=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("channels", [1, 7, 32, 64, 80, 128, 160, 257, 1024, 1031])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("nchw", [False, True])
def test_cuda(dtype, channels, nchw) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16-capable CUDA")
    _check("cuda", dtype, channels, nchw=nchw, mixed=nchw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("nchw", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_compile_cuda(dtype, nchw) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16-capable CUDA")
    _check("cuda", dtype, 32, nchw=nchw, mixed=True, compiled=True, autocast=dtype != torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_unaligned_cuda() -> None:
    _check("cuda", torch.float32, 64, nchw=False, mixed=True, unaligned=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_empty_cuda() -> None:
    x = torch.empty(0, 4, 3, 32, device="cuda", requires_grad=True)
    w = torch.ones(32, device="cuda", requires_grad=True)
    b = torch.zeros_like(w, requires_grad=True)
    y = fused_nhwc_layernorm_gelu(x, w, b)
    dx, dw, db = torch.autograd.grad(y.sum(), (x, w, b))
    assert dx.shape == x.shape
    torch.testing.assert_close(dw, torch.zeros_like(w))
    torch.testing.assert_close(db, torch.zeros_like(b))


@pytest.mark.parametrize(
    ("device", "dtype", "channels", "compiled", "nhwc_output"),
    [
        ("cpu", torch.bfloat16, 13, False, True),
        *(
            pytest.param(
                "cuda",
                dtype,
                channels,
                compiled,
                nhwc_output,
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            )
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
            for channels, compiled, nhwc_output in (
                (7, False, False),
                (32, True, True),
                (80, False, True),
                (1031, False, True),
            )
        ),
    ],
)
def test_conv_bias(device, dtype, channels, compiled, nhwc_output) -> None:
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16-capable CUDA")
    generator = torch.Generator(device=device).manual_seed(24)
    x = torch.randn((2, 3, 5, channels), device=device, dtype=dtype, generator=generator)
    x = x.permute(0, 3, 1, 2).requires_grad_()
    w, b, cb = (
        torch.randn(channels, device=device, generator=generator, requires_grad=True)
        for _ in range(3)
    )
    values = x.permute(0, 2, 3, 1).float() + cb
    expected = functional.gelu(functional.layer_norm(values, (channels,), w, b)).to(dtype)
    if not nhwc_output:
        expected = expected.permute(0, 3, 1, 2)
    if compiled:
        torch.compiler.reset()
    op = (
        torch.compile(fused_nhwc_layernorm_gelu, fullgraph=True)
        if compiled
        else fused_nhwc_layernorm_gelu
    )
    with torch.autocast(device, dtype=dtype, enabled=compiled and dtype != torch.float32):
        actual = op(x, w, b, nchw=True, conv_bias=cb, output_dtype=dtype, nhwc_output=nhwc_output)
    assert actual.stride() == expected.stride()
    assert actual.dtype == dtype
    dy = torch.randn(actual.shape, device=device, dtype=dtype, generator=generator)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.008, atol=2e-5)
    for index, (got, want) in enumerate(
        zip(
            torch.autograd.grad(actual, (x, w, b, cb), dy),
            torch.autograd.grad(expected, (x, w, b, cb), dy),
            strict=True,
        )
    ):
        rtol = 0.008 if index == 0 else 8e-5
        torch.testing.assert_close(got.float(), want.float(), rtol=rtol, atol=8e-5)
