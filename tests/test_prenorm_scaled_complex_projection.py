from __future__ import annotations

import math

import torch

from lnet.pac_complex_layers import ComplexLinear, semi_orthogonal_complex_linear_
from lnet.pac_complex_projection_adapter import PreNormScaledComplexProjection


def test_prenorm_adapter_preserves_projection_weights_and_rng() -> None:
    torch.manual_seed(53)
    source = ComplexLinear(7, 11).double()
    semi_orthogonal_complex_linear_(source)
    rng_state = torch.random.get_rng_state()

    adapter = PreNormScaledComplexProjection.from_projection(
        source,
        initial_scale=1.0 / math.sqrt(7),
    )

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    torch.testing.assert_close(adapter.projection.weight_real, source.weight_real)
    torch.testing.assert_close(adapter.projection.weight_imag, source.weight_imag)
    torch.testing.assert_close(
        adapter.scale.detach(),
        adapter.scale.new_tensor(1.0 / math.sqrt(7)),
    )


def test_prenorm_adapter_controls_input_scale_and_keeps_gradients_active() -> None:
    torch.manual_seed(59)
    adapter = PreNormScaledComplexProjection(6, 9, initial_scale=0.25).double()
    real = torch.randn(2, 4, 6, dtype=torch.double, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    expected = adapter(real, imag)
    actual = adapter(7.0 * real, 7.0 * imag)
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=3.0e-7)
    (actual[0].square().mean() + actual[1].square().mean()).backward()
    assert real.grad is not None
    assert imag.grad is not None
    assert adapter.scale.grad is not None
    assert adapter.projection.weight_real.grad is not None
    assert adapter.projection.weight_imag.grad is not None
