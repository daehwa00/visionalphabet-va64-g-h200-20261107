"""Tests for terminal imaginary correlation normalization behavior."""

from __future__ import annotations

import pytest
import torch
from va_model_builder import (
    PARAMETERS,
    VisionAlphabetIN100,
    build_dw_terminal_rmsnorm,
    model_config,
)

from lnet.terminal_imaginary_correlation import _vertical_correlations


def test_normalized_correlation_value_and_detached_gradient() -> None:
    torch.manual_seed(3)
    pole = tuple((torch.rand(3) * 0.2).requires_grad_() for _ in range(4))
    source = tuple(torch.randn(2, 3, 4, 3, requires_grad=True) for _ in range(2))
    rawpairs = _vertical_correlations(pole, source)
    actual = _vertical_correlations(pole, source, normalize=True)
    expected = []
    r, i = source
    for reverse in (False, True):
        gr = pole[2]
        gi = (-1 if reverse else 1) * pole[3]
        ur, ui = gr * r - gi * i, gr * i + gi * r
        denominator = (ur**2 + ui**2).mean((1, 2)).clamp_min(1e-6).detach()
        expected.extend(x / denominator for x in rawpairs[int(reverse)])
    flat = [x for pair in actual for x in pair]
    for a, b in zip(flat, expected, strict=True):
        torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(sum(x.sum() for x in flat), source + pole, retain_graph=True)
    gb = torch.autograd.grad(sum(x.sum() for x in expected), source + pole)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize(("factorized", "parameters"), [(False, PARAMETERS), (True, 2_988_260)])
def test_direct_builder_preserves_parameter_counts_and_accepts_output_dim(
    factorized: bool,
    parameters: int,
) -> None:
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        for output_dim in (100, 7):
            model = build_dw_terminal_rmsnorm(
                model_config(output_dim=output_dim),
                factorized_reader=factorized,
            )
            assert isinstance(model, VisionAlphabetIN100)
            assert (
                sum(p.numel() for p in model.parameters()) == parameters + (output_dim - 100) * 1025
            )
            assert model.classifier.affine.linear.out_features == output_dim
            model.eval()
            with torch.no_grad():
                output = model(torch.zeros(2, 3, 32, 32))
            assert output[0].shape == (2, output_dim)
            for block in model.backbone.same_resolution_blocks.values():
                post = block.post_fusion
                assert post.hidden == round(1.5 * post.width)
    finally:
        torch.set_num_threads(original_threads)
