from __future__ import annotations

import pytest
import torch

from lnet.pac_product_scan_reference import bidirectional_product_scan_reference
from lnet.pac_triton_bidirectional_product_scan import (
    pac_triton_bidirectional_product_scan,
)


def _inputs(device: torch.device, *, batch: int = 2, width: int = 8, modes: int = 16):
    generator = torch.Generator(device=device).manual_seed(721)
    pole = tuple(
        (0.03 * torch.randn((1, 1, 1, modes), generator=generator, device=device)).requires_grad_()
        for _ in range(4)
    )
    with torch.no_grad():
        pole[0].fill_(0.82)
        pole[2].fill_(0.18)
    source = tuple(
        torch.randn(
            (batch, width, width, modes),
            generator=generator,
            device=device,
        ).requires_grad_()
        for _ in range(2)
    )
    return pole, source


def _reference(pole, source):
    return bidirectional_product_scan_reference(pole, source)


def test_cpu_forward_and_backward_match_two_direction_reference() -> None:
    pole, source = _inputs(torch.device("cpu"))
    leaves = (*pole, *source)
    expected = _reference(pole, source)
    actual = pac_triton_bidirectional_product_scan(pole, source)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0.0, atol=0.0)

    sum(value.square().mean() for value in expected).backward(retain_graph=True)
    expected_gradients = tuple(value.grad.detach().clone() for value in leaves)
    for value in leaves:
        value.grad = None
    sum(value.square().mean() for value in actual).backward()
    for value, reference in zip(leaves, expected_gradients, strict=True):
        torch.testing.assert_close(value.grad, reference, rtol=5.0e-6, atol=5.0e-6)


def _relative_l2_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    difference = value.float() - reference.float()
    denominator = reference.float().norm().clamp_min(1.0e-8)
    return float((difference.norm() / denominator).detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [8, 56])
def test_cuda_associative_scan_matches_fp32_recurrence_forward_and_backward(
    dtype: torch.dtype,
    width: int,
) -> None:
    pole, float_source = _inputs(torch.device("cuda"), width=width, modes=64)
    source = tuple(value.detach().to(dtype).requires_grad_() for value in float_source)
    reference_source = tuple(value.detach().float().requires_grad_() for value in source)
    expected = _reference(pole, reference_source)
    actual = pac_triton_bidirectional_product_scan(pole, source)

    assert all(value.dtype == dtype for value in actual)
    forward_tolerance = 5.0e-6 if dtype == torch.float32 else 2.0e-2
    for value, reference in zip(actual, expected, strict=True):
        assert _relative_l2_error(value, reference) <= forward_tolerance

    actual_grad_outputs = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(
        expected,
        (*pole, *reference_source),
        tuple(value.float() for value in actual_grad_outputs),
    )
    actual_gradients = torch.autograd.grad(
        actual,
        (*pole, *source),
        actual_grad_outputs,
    )
    gradient_tolerance = 5.0e-5 if dtype == torch.float32 else 5.0e-2
    for value, reference in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(value).all()
        assert _relative_l2_error(value, reference) <= gradient_tolerance
