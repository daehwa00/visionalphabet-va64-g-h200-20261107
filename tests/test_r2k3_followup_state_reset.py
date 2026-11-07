from __future__ import annotations

# pyright: reportPrivateUsage=false
import torch

from lnet.pac_product_scan_reference import (
    _product_full4_reference,
    product_scan_full4_reference,
)
from scripts.run_r2k3_followup_state_reset import (
    _axis_scan,
    segmented_product_scans,
)


def _pole(
    modes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    decay_radius = 0.25 + 0.65 * torch.rand(modes, generator=generator)
    phase = torch.rand(modes, generator=generator) - 0.5
    gamma = 0.3 + 0.4 * torch.rand(modes, generator=generator)
    return (
        (decay_radius * phase.cos()).reshape(1, 1, 1, modes),
        (decay_radius * phase.sin()).reshape(1, 1, 1, modes),
        gamma.reshape(1, 1, 1, modes),
        torch.zeros(1, 1, 1, modes),
    )


def test_unsegmented_dynamic_recurrence_matches_product_scan_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    source = (
        torch.randn(2, 4, 6, 3, generator=generator),
        torch.randn(2, 4, 6, 3, generator=generator),
    )
    pole_x, pole_y = _pole(3, 11), _pole(3, 13)

    expected = product_scan_full4_reference(pole_x, pole_y, source)
    scans = segmented_product_scans(pole_x, pole_y, source, interval=0)
    actual = _product_full4_reference(
        scans,
        epsilon=1.0e-6,
        gain_normalization="pointwise",
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1.0e-5, atol=1.0e-5)


def test_interval_one_state_contains_only_current_input() -> None:
    source_real = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    source_imag = torch.zeros_like(source_real)
    source_variance = torch.ones_like(source_real)
    pole = (
        torch.tensor([[[[0.8]]]]),
        torch.tensor([[[[0.0]]]]),
        torch.tensor([[[[0.5]]]]),
        torch.tensor([[[[0.0]]]]),
    )

    state_real, state_imag, variance = _axis_scan(
        pole,
        (source_real, source_imag),
        source_variance,
        interval=1,
        reverse=False,
    )

    torch.testing.assert_close(state_real, 0.5 * source_real)
    torch.testing.assert_close(state_imag, torch.zeros_like(state_imag))
    torch.testing.assert_close(variance, torch.full_like(variance, 0.25))
