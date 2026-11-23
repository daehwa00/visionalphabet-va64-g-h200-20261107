"""Numerical checks for the CUDA RMSNorm experimental path."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from lnet.fused_rmsnorm_ops import fused_residual_rmsnorm, fused_rmsnorm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(("rows", "channels", "offset"), [
    (13, 47, 0), (19, 64, 1), (63, 160, 0), (129, 320, 0),
    (55, 640, 0), (17, 1024, 0), (7, 1025, 0),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reference(dtype: torch.dtype, rows: int, channels: int, offset: int) -> None:
    rng = torch.Generator(device="cuda").manual_seed(935)

    def random(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, generator=rng, device="cuda")

    x = random((rows * channels + offset,))[offset:].view(rows, channels).requires_grad_()
    w = random((channels,)).requires_grad_()
    dy = random((rows, channels))
    actual = fused_rmsnorm(x, w, 1e-6)
    ax, aw = torch.autograd.grad(actual, (x, w), dy)
    xr, wr = x.detach().float().requires_grad_(), w.detach().float().requires_grad_()
    expected = functional.rms_norm(xr, (channels,), wr, 1e-6)
    ex, ew = torch.autograd.grad(expected, (xr, wr), dy.float())

    for got, ref in ((actual, expected), (ax, ex), (aw, ew)):
        delta = got.float() - ref.to(dtype).float()
        if dtype == torch.float32:
            assert delta.abs().max() <= 1e-5
        else:
            assert delta.norm() / ref.norm().clamp_min(1e-12) <= 1e-3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_residual(dtype: torch.dtype) -> None:
    rng = torch.Generator(device="cuda").manual_seed(391)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, generator=rng, device="cuda")

    left, right = (random(3, 7, 64).requires_grad_() for _ in range(2))
    w = random(64).requires_grad_()
    residual, actual = fused_residual_rmsnorm(left, right, w, 1e-6)
    summed = left + right
    expected = functional.rms_norm(summed.float(), (64,), w.float(), 1e-6).to(dtype)
    dy, dr = random(*actual.shape), random(*residual.shape)
    actual_grads = torch.autograd.grad((residual, actual), (left, right, w), (dr, dy))
    expected_grads = torch.autograd.grad((summed, expected), (left, right, w), (dr, dy))
    torch.testing.assert_close(residual, summed, rtol=0, atol=0)
    for got, ref in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        delta = got.float() - ref.float()
        if dtype == torch.float32:
            assert delta.abs().max() <= 1e-5
        else:
            assert delta.norm() / ref.float().norm().clamp_min(1e-12) <= 1e-3
