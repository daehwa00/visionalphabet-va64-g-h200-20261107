# pyright: reportMissingImports=false
from __future__ import annotations

import inspect
import math

import pytest
import torch
from torch import Tensor
from torch.nn import functional

from lnet.pac_complex_layers import (
    packed_complex_linear_weight,
    unit_row_complex_linear_weight,
)
from lnet.pac_phase_gated_cffn import (
    PhaseGatedComplexFFN,
    phase_gated_complex_ffn_reference,
)
from lnet.pac_reduction_tiling import workload_parameter_reduction_rows
from lnet.pac_triton_complex_rmsnorm import packed_complex_rms_norm_reference
from lnet.pac_triton_hardware import single_warp_dot_tiles_supported
from lnet.pac_triton_phase_gate import phase_gate_reference


def _manual_forward(
    module: PhaseGatedComplexFFN,
    real: Tensor,
    imag: Tensor,
) -> tuple[Tensor, Tensor]:
    energy = (real.float().square() + imag.float().square()).mean(
        dim=-1,
        keepdim=True,
    )
    inverse_rms = torch.rsqrt(energy + module.norm.epsilon).to(dtype=real.dtype)
    weight = module.norm.weight.to(dtype=real.dtype)
    unit_real = real * inverse_rms * weight
    unit_imag = imag * inverse_rms * weight
    projected_real, projected_imag = module.input_projection(unit_real, unit_imag)
    if module.self_gated:
        value_real, value_imag = projected_real, projected_imag
        gate_real, gate_imag = value_real, value_imag
    else:
        value_real, gate_real = projected_real.split(module.hidden_modes, dim=-1)
        value_imag, gate_imag = projected_imag.split(module.hidden_modes, dim=-1)
    magnitude = torch.log1p(gate_real.float().square() + gate_imag.float().square())
    centered = magnitude - magnitude.mean(dim=-1, keepdim=True)
    relative = 1.0 + module.gate_redistribution * torch.tanh(module.alpha.float() * centered)
    gate = (relative / relative.mean(dim=-1, keepdim=True)).to(dtype=real.dtype)
    delta_real, delta_imag = module.output_projection(
        value_real * gate,
        value_imag * gate,
    )
    update_real = module.gamma * delta_real
    update_imag = module.gamma * delta_imag
    return real + update_real, imag + update_imag


def _gradients(
    outputs: tuple[Tensor, Tensor],
    inputs: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    real, imag = outputs
    loss = real.square().mean() + 0.7 * imag.square().mean()
    return torch.autograd.grad(loss, inputs)


def test_packed_complex_linear_weight_matches_complex_equation() -> None:
    generator = torch.Generator().manual_seed(17)
    real = torch.randn(13, 7, generator=generator, dtype=torch.float64)
    imag = torch.randn(13, 7, generator=generator, dtype=torch.float64)
    weight_real = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    weight_imag = torch.randn(11, 7, generator=generator, dtype=torch.float64)

    packed_output = (
        torch.cat((real, imag), dim=-1)
        @ packed_complex_linear_weight(
            weight_real,
            weight_imag,
        ).T
    )
    output_real, output_imag = packed_output.split(11, dim=-1)

    torch.testing.assert_close(
        output_real,
        real @ weight_real.T - imag @ weight_imag.T,
    )
    torch.testing.assert_close(
        output_imag,
        real @ weight_imag.T + imag @ weight_real.T,
    )


def test_unit_row_complex_linear_weight_normalizes_each_complex_output() -> None:
    generator = torch.Generator().manual_seed(19)
    weight_real = torch.randn(11, 7, generator=generator, dtype=torch.float64)
    weight_imag = torch.randn(11, 7, generator=generator, dtype=torch.float64)

    effective_real, effective_imag = unit_row_complex_linear_weight(
        weight_real,
        weight_imag,
    )
    row_energy = effective_real.square().add(effective_imag.square()).sum(dim=-1)

    torch.testing.assert_close(
        row_energy,
        torch.ones_like(row_energy),
        rtol=2.0e-6,
        atol=2.0e-6,
    )


def test_reduction_grouping_comes_from_runtime_hardware_capacity() -> None:
    rows = 1_000_000
    partial_rows = workload_parameter_reduction_rows(
        rows,
        multiprocessors=128,
        warp_size=32,
    )

    assert partial_rows == 512
    assert partial_rows & (partial_rows - 1) == 0


def test_reduction_grouping_rejects_invalid_runtime_properties() -> None:
    with pytest.raises(ValueError, match="positive"):
        workload_parameter_reduction_rows(
            0,
            multiprocessors=128,
            warp_size=32,
        )


def test_single_warp_fusion_capacity_uses_reported_warp_width() -> None:
    assert single_warp_dot_tiles_supported((8, 16, 32), warp_size=32)
    assert not single_warp_dot_tiles_supported((8, 16, 33), warp_size=32)
    assert single_warp_dot_tiles_supported((8, 16, 33), warp_size=64)


@pytest.mark.parametrize("self_gated", [False, True])
def test_phase_gated_complex_ffn_matches_reference(self_gated: bool) -> None:
    torch.manual_seed(23)
    module = PhaseGatedComplexFFN(7, 11, self_gated=self_gated).double()
    real = torch.randn(2, 3, 4, 7, dtype=torch.float64, requires_grad=True)
    imag = torch.randn(2, 3, 4, 7, dtype=torch.float64, requires_grad=True)
    parameters = tuple(module.parameters())

    expected = _manual_forward(module, real, imag)
    expected_gradients = _gradients(expected, (real, imag, *parameters))
    actual = module(real, imag)
    actual_gradients = _gradients(actual, (real, imag, *parameters))

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_phase_gated_complex_ffn_supports_an_exact_unit_residual() -> None:
    torch.manual_seed(27)
    module = PhaseGatedComplexFFN(
        7,
        11,
        residual_scale_init=1.0,
        learnable_residual_scale=False,
    ).double()
    real = torch.randn(2, 3, 4, 7, dtype=torch.float64, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    actual = module(real, imag)
    expected = phase_gated_complex_ffn_reference(module, real, imag)
    loss = actual[0].square().mean() + actual[1].square().mean()
    loss.backward()

    assert module.gamma is None
    assert "gamma" not in module.state_dict()
    assert module.effective_residual_scale().item() == 1.0
    assert module.diagnostic_metrics()["fixed_unit_residual_scale"] == 1.0
    assert "gamma_grad_norm" not in module.gradient_metrics()
    assert module.gradient_metrics()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("self_gated", [False, True])
def test_packed_phase_gated_pipeline_matches_original_equation(
    self_gated: bool,
) -> None:
    torch.manual_seed(29)
    module = PhaseGatedComplexFFN(7, 11, self_gated=self_gated).double()
    real = torch.randn(2, 3, 4, 7, dtype=torch.float64, requires_grad=True)
    imag = torch.randn(2, 3, 4, 7, dtype=torch.float64, requires_grad=True)
    parameters = tuple(module.parameters())
    expected = phase_gated_complex_ffn_reference(module, real, imag)
    expected_gradients = _gradients(expected, (real, imag, *parameters))

    normalized = packed_complex_rms_norm_reference(
        real,
        imag,
        module.norm.weight,
        module.norm.epsilon,
    )
    projected = functional.linear(
        normalized,
        packed_complex_linear_weight(
            module.input_projection.weight_real,
            module.input_projection.weight_imag,
        ),
    )
    hidden = phase_gate_reference(
        projected,
        module.alpha,
        redistribution=module.gate_redistribution,
        self_gated=self_gated,
    )
    update = functional.linear(
        hidden,
        packed_complex_linear_weight(
            module.output_projection.weight_real,
            module.output_projection.weight_imag,
        ),
    )
    gamma = module.gamma
    assert gamma is not None
    update_real, update_imag = (gamma * update).split(module.modes, dim=-1)
    actual = real + update_real, imag + update_imag
    actual_gradients = _gradients(actual, (real, imag, *parameters))

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_phase_gated_complex_ffn_is_globally_phase_equivariant() -> None:
    torch.manual_seed(31)
    module = PhaseGatedComplexFFN(6, 9).double().eval()
    real = torch.randn(2, 5, 4, 6, dtype=torch.float64)
    imag = torch.randn(2, 5, 4, 6, dtype=torch.float64)
    angle = 0.37
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated_real = cosine * real - sine * imag
    rotated_imag = sine * real + cosine * imag

    output_real, output_imag = module(real, imag)
    actual_real, actual_imag = module(rotated_real, rotated_imag)

    torch.testing.assert_close(actual_real, cosine * output_real - sine * output_imag)
    torch.testing.assert_close(actual_imag, sine * output_real + cosine * output_imag)


def test_phase_gated_complex_ffn_preserves_checkpoint_contract() -> None:
    module = PhaseGatedComplexFFN(96, 144)
    assert "fused" not in inspect.signature(PhaseGatedComplexFFN).parameters
    assert "fused" not in inspect.signature(module.forward).parameters
    assert set(module.state_dict()) == {
        "alpha",
        "gamma",
        "norm.weight",
        "input_projection.weight_real",
        "input_projection.weight_imag",
        "output_projection.weight_real",
        "output_projection.weight_imag",
    }
    assert module.input_projection.weight_real.shape == (288, 96)
    assert module.output_projection.weight_real.shape == (96, 144)


def test_phase_gated_complex_ffn_uses_the_control_initialization() -> None:
    torch.manual_seed(37)
    module = PhaseGatedComplexFFN(96, 192)

    torch.testing.assert_close(module.alpha, torch.full_like(module.alpha, 0.075))
    gamma = module.gamma
    assert gamma is not None
    torch.testing.assert_close(gamma, gamma.new_tensor(0.01))
    assert module.gate_redistribution == 0.5
    assert module.output_projection.weight_real.std() > 0.02
    assert module.output_projection.weight_imag.std() > 0.02
    assert module.input_projection.weight_real.shape == (384, 96)
    assert sum(parameter.numel() for parameter in module.parameters()) == 110_881


def test_unit_row_phase_gated_contract_removes_projection_scale_freedom() -> None:
    torch.manual_seed(38)
    module = PhaseGatedComplexFFN(
        7,
        11,
        unit_row_projections=True,
        residual_scale_max=0.5,
    ).double()
    real = torch.randn(2, 3, 4, 7, dtype=torch.float64)
    imag = torch.randn_like(real)
    expected = module(real, imag)

    for weight_real, weight_imag in module.effective_projection_weights():
        row_energy = weight_real.square().add(weight_imag.square()).sum(dim=-1)
        torch.testing.assert_close(
            row_energy,
            torch.ones_like(row_energy),
            rtol=2.0e-6,
            atol=2.0e-6,
        )
    gamma = module.gamma
    assert gamma is not None
    torch.testing.assert_close(module.effective_residual_scale(), gamma.new_tensor(0.01))

    with torch.no_grad():
        for projection in (module.input_projection, module.output_projection):
            row_scale = torch.linspace(
                0.25,
                4.0,
                projection.output_modes,
                dtype=projection.weight_real.dtype,
            ).unsqueeze(-1)
            projection.weight_real.mul_(row_scale)
            projection.weight_imag.mul_(row_scale)
    actual = module(real, imag)

    torch.testing.assert_close(actual, expected, rtol=1.0e-7, atol=2.0e-9)


def test_unit_row_phase_gated_contract_is_differentiable_and_bounds_gamma() -> None:
    torch.manual_seed(40)
    module = PhaseGatedComplexFFN(
        5,
        8,
        unit_row_projections=True,
        residual_scale_max=0.5,
    ).double()
    real = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    output = module(real, imag)
    loss = output[0].square().mean() + output[1].square().mean()
    loss.backward()

    assert real.grad is not None
    assert torch.isfinite(real.grad).all()
    assert imag.grad is not None
    assert torch.isfinite(imag.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )
    gamma = module.gamma
    assert gamma is not None
    with torch.no_grad():
        gamma.fill_(100.0)
    positive_scale = float(module.effective_residual_scale().detach())
    assert 0.49 < positive_scale <= 0.5
    with torch.no_grad():
        gamma.fill_(-100.0)
    negative_scale = float(module.effective_residual_scale().detach())
    assert -0.5 <= negative_scale < -0.49


def test_direction_output_gain_contract_has_selective_bounded_mode_scales() -> None:
    torch.manual_seed(41)
    module = PhaseGatedComplexFFN(5, 8).double()
    module.enable_direction_output_gain_contract_(output_gain_max=0.5)
    assert module.gamma is None
    assert module.output_gain_logits is not None

    gains = module.effective_output_gains()
    assert gains is not None
    torch.testing.assert_close(gains, gains.new_full((5,), 0.01))
    with torch.no_grad():
        module.output_gain_logits.copy_(
            module.output_gain_logits.new_tensor((-100.0, -1.0, 0.0, 1.0, 100.0))
        )
    gains = module.effective_output_gains()
    assert gains is not None
    detached_gains = gains.detach()
    assert -0.5 <= float(detached_gains[0]) < -0.49
    assert float(detached_gains[2]) == 0.0
    assert 0.49 < float(detached_gains[-1]) <= 0.5

    real = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    output_real, output_imag = module(real, imag)
    (output_real.square().mean() + output_imag.square().mean()).backward()
    assert module.output_gain_logits.grad is not None
    assert torch.isfinite(module.output_gain_logits.grad).all()


def test_direction_output_gain_projection_retracts_raw_rows() -> None:
    module = PhaseGatedComplexFFN(5, 8).double()
    module.enable_direction_output_gain_contract_(output_gain_max=0.5)
    with torch.no_grad():
        for projection in (module.input_projection, module.output_projection):
            projection.weight_real.mul_(3.0)
            projection.weight_imag.mul_(3.0)
    module.project_direction_rows_()

    for projection in (module.input_projection, module.output_projection):
        row_energy = (
            projection.weight_real.square().add(projection.weight_imag.square()).sum(dim=-1)
        )
        torch.testing.assert_close(
            row_energy,
            torch.ones_like(row_energy),
            rtol=3.0e-6,
            atol=3.0e-6,
        )


def test_phase_gate_is_exactly_mean_one_and_cannot_close_globally() -> None:
    torch.manual_seed(38)
    module = PhaseGatedComplexFFN(5, 8)
    gate = module._gate(  # pyright: ignore[reportPrivateUsage]
        torch.randn(3, 4, 8),
        torch.randn(3, 4, 8),
    )

    torch.testing.assert_close(
        gate.float().mean(dim=-1),
        torch.ones_like(gate[..., 0], dtype=torch.float32),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert gate.min() > 0.5
    assert gate.max() < 1.5


def test_phase_gated_gradient_metrics_separate_value_and_gate_branches() -> None:
    torch.manual_seed(39)
    module = PhaseGatedComplexFFN(5, 8)
    real = torch.randn(2, 3, 4, 5, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    _loss_real, _loss_imag = module(real, imag)
    (_loss_real.square().mean() + _loss_imag.square().mean()).backward()
    metrics = module.gradient_metrics()

    assert metrics.keys() == {
        "alpha_grad_norm",
        "gamma_grad_norm",
        "input_gate_to_value_grad_ratio",
        "input_gate_weight_grad_norm",
        "input_value_weight_grad_norm",
        "output_weight_grad_norm",
    }
    assert all(math.isfinite(value) and value >= 0.0 for value in metrics.values())
    assert metrics["input_value_weight_grad_norm"] > 0.0
    assert metrics["input_gate_weight_grad_norm"] > 0.0
    assert metrics["output_weight_grad_norm"] > 0.0
    assert metrics["gamma_grad_norm"] > 0.0


def test_phase_gated_complex_ffn_updates_bounded_diagnostics() -> None:
    torch.manual_seed(41)
    module = PhaseGatedComplexFFN(5, 8)
    real = torch.randn(2, 3, 4, 5)
    imag = torch.randn(2, 3, 4, 5)

    module(real, imag)
    metrics = module.diagnostic_metrics()

    assert module.diagnostic_updates.item() == 1
    assert metrics["gate_mean"] == pytest.approx(1.0, abs=1.0e-6)
    assert metrics["gate_std"] >= 0.0
    assert 0.5 < metrics["gate_min"] <= metrics["gate_max"] < 1.5
    assert metrics["update_ratio"] >= 0.0
    assert metrics["energy_ratio"] >= 0.0


def test_phase_gated_complex_ffn_handles_empty_batches_without_nan_diagnostics() -> None:
    module = PhaseGatedComplexFFN(5, 8)
    real = torch.empty(0, 3, 5)
    imag = torch.empty_like(real)

    output_real, output_imag = module(real, imag)

    assert output_real.shape == output_imag.shape == real.shape
    assert module.diagnostic_updates.item() == 0
    assert all(math.isfinite(value) for value in module.diagnostic_metrics().values())


def test_phase_gated_complex_ffn_rejects_coordinate_dtype_mismatch() -> None:
    module = PhaseGatedComplexFFN(4, 7)
    with pytest.raises(ValueError, match="device and dtype"):
        module(torch.randn(2, 4), torch.randn(2, 4, dtype=torch.float64))


@pytest.mark.parametrize(
    ("real_shape", "imag_shape"),
    [((2, 4), (2, 5)), ((2, 3), (2, 3))],
)
def test_phase_gated_complex_ffn_rejects_incompatible_inputs(
    real_shape: tuple[int, ...],
    imag_shape: tuple[int, ...],
) -> None:
    module = PhaseGatedComplexFFN(4, 7)
    with pytest.raises(ValueError, match="incompatible shapes"):
        module(torch.randn(real_shape), torch.randn(imag_shape))
