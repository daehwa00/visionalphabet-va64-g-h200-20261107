from __future__ import annotations

import torch

from lnet.pac_complex_layers import (
    ComplexLinear,
    ScaleStableComplexLinear,
    semi_orthogonal_complex_linear_,
)


def test_scale_stable_projection_preserves_source_function_and_rng() -> None:
    torch.manual_seed(41)
    source = ComplexLinear(7, 11).double()
    semi_orthogonal_complex_linear_(source)
    real = torch.randn(2, 3, 7, dtype=torch.double)
    imag = torch.randn_like(real)
    expected = source(real, imag)
    rng_state = torch.random.get_rng_state()

    projection = ScaleStableComplexLinear.from_projection(source)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    actual = projection(real, imag)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_scale_stable_projection_keeps_reference_energy_while_learning_direction() -> None:
    torch.manual_seed(43)
    projection = ScaleStableComplexLinear(6, 9).double()
    reference = projection.reference_row_norm.detach().clone()
    with torch.no_grad():
        projection.weight_real.mul_(3.0)
        projection.weight_imag.add_(torch.randn_like(projection.weight_imag))

    weight_real, weight_imag = projection.normalized_weight()
    row_norm = (weight_real.square() + weight_imag.square()).sum(-1, keepdim=True).sqrt()
    torch.testing.assert_close(row_norm, reference, rtol=1.0e-12, atol=1.0e-12)

    real = torch.randn(2, 4, 6, dtype=torch.double, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    output = projection(real, imag)
    (output[0].square().mean() + output[1].square().mean()).backward()
    assert projection.weight_real.grad is not None
    assert projection.weight_imag.grad is not None
    assert torch.isfinite(projection.weight_real.grad).all()
    assert torch.isfinite(projection.weight_imag.grad).all()
