from __future__ import annotations

import torch

from scripts.analyze_direct_path_collapse_distillation import (
    DirectPathCollapse,
    RidgeMoments,
)


def test_direct_path_collapse_represents_an_arbitrary_real_affine_map() -> None:
    torch.manual_seed(79)
    modes = 5
    weight = torch.randn(modes, 2, 8, dtype=torch.double)
    bias = torch.randn(modes, 2, dtype=torch.double)
    module = DirectPathCollapse(modes).double()
    module.load_real_affine(weight, bias)
    source_real = torch.randn(2, 3, 2, 4, modes, dtype=torch.double)
    source_imag = torch.randn_like(source_real)

    features = torch.cat(
        (
            source_real.permute(0, 1, 2, 4, 3),
            source_imag.permute(0, 1, 2, 4, 3),
        ),
        dim=-1,
    )
    expected = torch.einsum("...mi,moi->...mo", features, weight) + bias
    actual_real, actual_imag = module.forward_packed(source_real, source_imag)
    actual = torch.stack((actual_real[..., 0, :], actual_imag[..., 0, :]), dim=-1)

    torch.testing.assert_close(actual, expected)


def test_ridge_moments_recovers_a_direct_path_map() -> None:
    torch.manual_seed(83)
    modes = 4
    source_real = torch.randn(3, 5, 2, 4, modes)
    source_imag = torch.randn_like(source_real)
    weight = torch.randn(modes, 2, 8)
    bias = torch.randn(modes, 2)
    features = torch.cat(
        (
            source_real.permute(0, 1, 2, 4, 3),
            source_imag.permute(0, 1, 2, 4, 3),
        ),
        dim=-1,
    )
    target = torch.einsum("...mi,moi->...mo", features, weight) + bias
    moments = RidgeMoments.create(modes)
    moments.add(
        source_real,
        source_imag,
        target[..., 0],
        target[..., 1],
        retained_rows=source_real.numel(),
    )

    fitted_weight, fitted_bias, metrics = moments.solve(ridge=0.0)

    torch.testing.assert_close(fitted_weight, weight, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(fitted_bias, bias, rtol=2.0e-5, atol=2.0e-5)
    assert metrics["centered_r2"] > 0.999999
