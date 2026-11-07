# pyright: reportMissingImports=false
from __future__ import annotations

import copy

import pytest
import torch

from lnet.pac_complex_ffn import complex_ffn, packed_cartesian_cffn
from lnet.pac_complex_layers import WidelyLinear


def _layers(
    input_modes: int,
    hidden_modes: int,
    output_modes: int,
) -> tuple[WidelyLinear, WidelyLinear]:
    return (
        WidelyLinear(input_modes, hidden_modes),
        WidelyLinear(hidden_modes, output_modes),
    )


def _reference(
    real: torch.Tensor,
    imag: torch.Tensor,
    first: WidelyLinear,
    second: WidelyLinear,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_real, hidden_imag = first(real, imag)
    return second(
        torch.nn.functional.silu(hidden_real),
        torch.nn.functional.silu(hidden_imag),
    )


@pytest.mark.parametrize(
    ("input_modes", "hidden_modes", "output_modes"),
    [(3, 5, 4), (8, 16, 8), (12, 24, 7), (48, 96, 48)],
)
def test_complex_ffn_matches_arbitrary_shape_reference(
    input_modes: int,
    hidden_modes: int,
    output_modes: int,
) -> None:
    torch.manual_seed(501)
    reference_first, reference_second = _layers(
        input_modes,
        hidden_modes,
        output_modes,
    )
    active_first = copy.deepcopy(reference_first)
    active_second = copy.deepcopy(reference_second)
    reference_real = torch.randn(2, 3, input_modes, requires_grad=True)
    reference_imag = torch.randn_like(reference_real, requires_grad=True)
    active_real = reference_real.detach().clone().requires_grad_()
    active_imag = reference_imag.detach().clone().requires_grad_()

    expected = _reference(
        reference_real,
        reference_imag,
        reference_first,
        reference_second,
    )
    actual = complex_ffn(
        active_real,
        active_imag,
        input_projection=active_first,
        output_projection=active_second,
    )
    torch.testing.assert_close(actual, expected)

    cotangent = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(
        expected,
        (
            reference_real,
            reference_imag,
            *reference_first.parameters(),
            *reference_second.parameters(),
        ),
        cotangent,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        (
            active_real,
            active_imag,
            *active_first.parameters(),
            *active_second.parameters(),
        ),
        cotangent,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_complex_ffn_supports_residual_and_synthesis() -> None:
    torch.manual_seed(509)
    modes, paths, output_paths = 6, 5, 3
    first, second = _layers(paths, 9, paths)
    real = torch.randn(2, 4, modes, paths)
    imag = torch.randn_like(real)
    scale = torch.randn(paths) * 0.01
    synthesis_real = torch.randn(modes, output_paths, paths)
    synthesis_imag = torch.randn_like(synthesis_real)

    update_real, update_imag = _reference(real, imag, first, second)
    mixed_real = real + scale * update_real
    mixed_imag = imag + scale * update_imag
    expected = (
        torch.einsum("...mp,mqp->...qm", mixed_real, synthesis_real)
        - torch.einsum("...mp,mqp->...qm", mixed_imag, synthesis_imag),
        torch.einsum("...mp,mqp->...qm", mixed_real, synthesis_imag)
        + torch.einsum("...mp,mqp->...qm", mixed_imag, synthesis_real),
    )
    actual = complex_ffn(
        real,
        imag,
        input_projection=first,
        output_projection=second,
        residual_scale=scale,
        synthesis_real=synthesis_real,
        synthesis_imag=synthesis_imag,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("modes", "paths", "hidden_modes", "output_paths"),
    [(4, 3, 5, 2), (6, 5, 9, 3), (7, 8, 16, 5)],
)
def test_packed_cartesian_cffn_supports_arbitrary_synthesis_widths_and_gradients(
    modes: int,
    paths: int,
    hidden_modes: int,
    output_paths: int,
) -> None:
    torch.manual_seed(523)
    reference_first, reference_second = _layers(paths, hidden_modes, paths)
    active_first = copy.deepcopy(reference_first)
    active_second = copy.deepcopy(reference_second)
    reference_real = torch.randn(2, 3, modes, paths, requires_grad=True)
    reference_imag = torch.randn_like(reference_real, requires_grad=True)
    active_real = reference_real.detach().clone().requires_grad_()
    active_imag = reference_imag.detach().clone().requires_grad_()
    reference_scale = torch.nn.Parameter(torch.randn(paths) * 0.01)
    active_scale = torch.nn.Parameter(reference_scale.detach().clone())
    reference_synthesis_real = torch.nn.Parameter(torch.randn(modes, output_paths, paths) * 0.1)
    reference_synthesis_imag = torch.nn.Parameter(torch.randn(modes, output_paths, paths) * 0.1)
    active_synthesis_real = torch.nn.Parameter(reference_synthesis_real.detach().clone())
    active_synthesis_imag = torch.nn.Parameter(reference_synthesis_imag.detach().clone())

    update_real, update_imag = _reference(
        reference_real,
        reference_imag,
        reference_first,
        reference_second,
    )
    mixed_real = reference_real + reference_scale * update_real
    mixed_imag = reference_imag + reference_scale * update_imag
    expected = (
        torch.einsum(
            "...mp,mqp->...qm",
            mixed_real,
            reference_synthesis_real,
        )
        - torch.einsum(
            "...mp,mqp->...qm",
            mixed_imag,
            reference_synthesis_imag,
        ),
        torch.einsum(
            "...mp,mqp->...qm",
            mixed_real,
            reference_synthesis_imag,
        )
        + torch.einsum(
            "...mp,mqp->...qm",
            mixed_imag,
            reference_synthesis_real,
        ),
    )
    packed = packed_cartesian_cffn(
        torch.cat((active_real, active_imag), dim=-1),
        input_projection=active_first,
        output_projection=active_second,
        residual_scale=active_scale,
        synthesis_real=active_synthesis_real,
        synthesis_imag=active_synthesis_imag,
    )
    packed_real, packed_imag = packed.split(output_paths, dim=-1)
    actual = packed_real.transpose(-2, -1), packed_imag.transpose(-2, -1)
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

    cotangent = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(
        expected,
        (
            reference_real,
            reference_imag,
            *reference_first.parameters(),
            *reference_second.parameters(),
            reference_scale,
            reference_synthesis_real,
            reference_synthesis_imag,
        ),
        cotangent,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        (
            active_real,
            active_imag,
            *active_first.parameters(),
            *active_second.parameters(),
            active_scale,
            active_synthesis_real,
            active_synthesis_imag,
        ),
        cotangent,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2.0e-5,
            atol=2.0e-6,
        )


@pytest.mark.parametrize(
    ("anchor_shape", "scale_shape"),
    [((1, 3, 4), (4,)), ((2, 3, 4), (1,))],
)
def test_residual_contract_rejects_incompatible_shapes(
    anchor_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
) -> None:
    first, second = _layers(4, 7, 4)
    real = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="residual requires matching"):
        complex_ffn(
            real,
            torch.randn_like(real),
            input_projection=first,
            output_projection=second,
            residual_scale=torch.ones(scale_shape),
            residual_source=(torch.randn(anchor_shape), torch.randn(anchor_shape)),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_widely_linear_preserves_bfloat16_autocast_storage() -> None:
    layer = WidelyLinear(16, 24).cuda().train()
    real = torch.randn(2, 3, 5, 16, device="cuda", requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = layer(real, imag)
    assert output_real.dtype == torch.bfloat16
    assert output_imag.dtype == torch.bfloat16
    (output_real.float().square().mean() + output_imag.float().square().mean()).backward()
    assert layer.bias_real is not None
    assert layer.bias_real.grad is not None
    assert layer.bias_real.grad.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_bfloat16_fusion_matches_eager_outputs_and_gradients() -> None:
    torch.manual_seed(509)
    reference_first, reference_second = (layer.cuda() for layer in _layers(64, 128, 64))
    active_first = copy.deepcopy(reference_first)
    active_second = copy.deepcopy(reference_second)
    reference_scale = torch.nn.Parameter(torch.full((64,), 0.01, device="cuda"))
    active_scale = torch.nn.Parameter(reference_scale.detach().clone())

    def run(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return complex_ffn(
            real,
            imag,
            input_projection=active_first,
            output_projection=active_second,
            residual_scale=active_scale,
        )

    compiled = torch.compile(run, mode="default", fullgraph=True, dynamic=False)
    reference_real = torch.randn(2, 7, 7, 64, device="cuda", requires_grad=True)
    reference_imag = torch.randn_like(reference_real, requires_grad=True)
    active_real = reference_real.detach().clone().requires_grad_()
    active_imag = reference_imag.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = complex_ffn(
            reference_real,
            reference_imag,
            input_projection=reference_first,
            output_projection=reference_second,
            residual_scale=reference_scale,
        )
        actual = compiled(active_real, active_imag)
    torch.testing.assert_close(actual, expected, atol=2.0e-2, rtol=2.0e-2)

    cotangent = tuple(torch.randn_like(value) for value in actual)
    reference_tensors = (
        reference_real,
        reference_imag,
        *reference_first.parameters(),
        *reference_second.parameters(),
        reference_scale,
    )
    active_tensors = (
        active_real,
        active_imag,
        *active_first.parameters(),
        *active_second.parameters(),
        active_scale,
    )
    expected_gradients = torch.autograd.grad(expected, reference_tensors, cotangent)
    actual_gradients = torch.autograd.grad(actual, active_tensors, cotangent)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        difference = actual_gradient.float() - expected_gradient.float()
        denominator = expected_gradient.float().square().sum().sqrt().clamp_min(1.0e-12)
        relative_l2 = difference.square().sum().sqrt() / denominator
        assert float(relative_l2) <= 1.0e-2
        assert bool(torch.isfinite(actual_gradient).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_compiled_synthesis_fusion_matches_eager_outputs_and_gradients(
    precision: str,
) -> None:
    """Path CFFNs retain exact synthesis math on the common packed path."""
    torch.manual_seed(521)
    modes, paths, output_paths = 7, 4, 4
    reference_first, reference_second = (layer.cuda() for layer in _layers(paths, 8, paths))
    active_first = copy.deepcopy(reference_first)
    active_second = copy.deepcopy(reference_second)
    reference_scale = torch.nn.Parameter(torch.full((paths,), 0.01, device="cuda"))
    active_scale = torch.nn.Parameter(reference_scale.detach().clone())
    synthesis_real = torch.randn(modes, output_paths, paths, device="cuda")
    synthesis_imag = torch.randn_like(synthesis_real)

    def run(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return complex_ffn(
            real,
            imag,
            input_projection=active_first,
            output_projection=active_second,
            residual_scale=active_scale,
            synthesis_real=synthesis_real,
            synthesis_imag=synthesis_imag,
        )

    compiled = torch.compile(run, mode="default", fullgraph=True, dynamic=False)
    reference_real = torch.randn(2, 3, 5, modes, paths, device="cuda", requires_grad=True)
    reference_imag = torch.randn_like(reference_real, requires_grad=True)
    active_real = reference_real.detach().clone().requires_grad_()
    active_imag = reference_imag.detach().clone().requires_grad_()
    use_autocast = precision == "bfloat16"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
        expected = complex_ffn(
            reference_real,
            reference_imag,
            input_projection=reference_first,
            output_projection=reference_second,
            residual_scale=reference_scale,
            synthesis_real=synthesis_real,
            synthesis_imag=synthesis_imag,
        )
        actual = compiled(active_real, active_imag)
    tolerance = 2.0e-2 if use_autocast else 2.0e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert actual[0].is_contiguous()
    assert actual[1].is_contiguous()

    cotangent = tuple(torch.randn_like(value) for value in actual)
    reference_tensors = (
        reference_real,
        reference_imag,
        *reference_first.parameters(),
        *reference_second.parameters(),
        reference_scale,
    )
    active_tensors = (
        active_real,
        active_imag,
        *active_first.parameters(),
        *active_second.parameters(),
        active_scale,
    )
    expected_gradients = torch.autograd.grad(expected, reference_tensors, cotangent)
    actual_gradients = torch.autograd.grad(actual, active_tensors, cotangent)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        if use_autocast:
            difference = actual_gradient.float() - expected_gradient.float()
            denominator = expected_gradient.float().square().sum().sqrt().clamp_min(1.0e-12)
            relative_l2 = difference.square().sum().sqrt() / denominator
            assert float(relative_l2) <= 1.0e-2
        else:
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=3.0e-4,
                rtol=3.0e-4,
            )
