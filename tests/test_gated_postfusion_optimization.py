from __future__ import annotations

# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING

import pytest
import torch

from scripts import benchmark_gated_postfusion_complex_linear as benchmark

if TYPE_CHECKING:
    from lnet.pac_gated_post_fusion import _GatedPostFusionBase


@pytest.mark.parametrize("family", sorted(benchmark.FAMILIES))
def test_candidate_matches_split_forward_and_backward(family: str) -> None:
    reference_type, candidate_type = benchmark.FAMILIES[family]
    torch.manual_seed(7)
    reference: _GatedPostFusionBase = reference_type(5, 7).double()
    candidate: _GatedPostFusionBase = candidate_type(5, 7).double()
    if isinstance(candidate, benchmark._NativePackedWidelyPostFusion):
        candidate.load_reference(reference)
    else:
        candidate.load_state_dict(reference.state_dict(), strict=True)
    real = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    candidate_real = real.detach().clone().requires_grad_()
    candidate_imag = imag.detach().clone().requires_grad_()
    expected = reference(real, imag)
    actual = candidate(candidate_real, candidate_imag)
    torch.testing.assert_close(actual[0], expected[0], rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(actual[1], expected[1], rtol=1.0e-12, atol=1.0e-12)
    (expected[0].square().mean() + expected[1].square().mean()).backward()
    (actual[0].square().mean() + actual[1].square().mean()).backward()
    assert candidate_real.grad is not None
    assert candidate_imag.grad is not None
    assert real.grad is not None
    assert imag.grad is not None
    torch.testing.assert_close(candidate_real.grad, real.grad, rtol=1.0e-11, atol=1.0e-11)
    torch.testing.assert_close(candidate_imag.grad, imag.grad, rtol=1.0e-11, atol=1.0e-11)
    candidate_gradients = (
        candidate.reference_gradients()
        if isinstance(candidate, benchmark._NativePackedWidelyPostFusion)
        else {name: parameter.grad for name, parameter in candidate.named_parameters()}
    )
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None
        candidate_gradient = candidate_gradients[name]
        assert candidate_gradient is not None
        torch.testing.assert_close(
            candidate_gradient,
            parameter.grad,
            rtol=1.0e-11,
            atol=1.0e-11,
        )


def test_benchmark_weights_match_all_resolution_postfusion_sites() -> None:
    assert benchmark.ROWS == (401_408, 100_352, 25_088, 6_272)
    assert benchmark.MULTIPLICITY == (1, 2, 2, 2)
    assert sum(benchmark.MULTIPLICITY) == 7


def test_benchmark_covers_deployed_and_native_storage_families() -> None:
    assert set(benchmark.FAMILIES) == {"strict", "widely", "native_widely"}
    for family in ("strict", "widely"):
        reference_type, candidate_type = benchmark.FAMILIES[family]
        assert issubclass(reference_type, candidate_type)
        assert reference_type.forward is not candidate_type.forward
    reference_type, candidate_type = benchmark.FAMILIES["native_widely"]
    assert reference_type is benchmark.GatedComplexPostFusion
    assert candidate_type is benchmark._NativePackedWidelyPostFusion
