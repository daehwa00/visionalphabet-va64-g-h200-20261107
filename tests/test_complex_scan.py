"""Tests for the continuous complex scan backbone."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import pytest
import torch

from lnet import complex_scan_stage
from lnet.complex_scan import (
    AugmentedComplexTransition,
    CifarConvOnlyStem,
    CifarNormalizedLinearStem,
    ComplexInteractionTransition,
    ComplexLinear,
    ComplexModulatedTransition,
    ComplexPixelStem,
    ComplexRMSNorm,
    ComplexScanBackbone,
    ComplexScanConfig,
    ComplexScanStage,
    LocalComplexFourierStem,
    ModalFusionHead,
    WidelyLinear,
    complex_carry_coordinates,
    pole_aligned_complex_downsample,
)
from lnet.image_layers import LayerNorm2d
from lnet.pac_directional import direction_aligned_endpoints


def _rotate(
    real: torch.Tensor,
    imag: torch.Tensor,
    angle: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = torch.cos(torch.tensor(angle))
    sine = torch.sin(torch.tensor(angle))
    return cosine * real - sine * imag, sine * real + cosine * imag


def test_complex_linear_and_rmsnorm_are_global_phase_equivariant() -> None:
    linear = ComplexLinear(8, 4)
    norm = ComplexRMSNorm(4)
    real = torch.randn(2, 3, 5, 8)
    imag = torch.randn(2, 3, 5, 8)
    output = norm(*linear(real, imag))
    rotated_output = norm(*linear(*_rotate(real, imag, 0.67)))
    expected = _rotate(output[0], output[1], 0.67)
    assert torch.allclose(rotated_output[0], expected[0], atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(rotated_output[1], expected[1], atol=2.0e-6, rtol=2.0e-6)


def test_complex_rmsnorm_uses_direct_affine_weight() -> None:
    norm = ComplexRMSNorm(4).double()
    assert set(norm.state_dict()) == {"weight"}
    torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, -0.25], dtype=torch.float64))
    real = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    output_real, output_imag = norm(real, imag)
    inverse_rms = torch.rsqrt(
        (real.square() + imag.square()).mean(dim=-1, keepdim=True) + norm.epsilon
    )
    expected = real * inverse_rms * norm.weight, imag * inverse_rms * norm.weight
    torch.testing.assert_close((output_real, output_imag), expected)
    torch.autograd.grad(output_real.sum() + output_imag.sum(), (real, imag, norm.weight))


def test_pole_aligned_downsample_preserves_carrier_and_global_phase() -> None:
    width = 8
    phase_x = torch.tensor([math.pi / 2.0])
    phase_y = torch.zeros(1)
    coordinate_x = torch.arange(width).view(1, 1, width, 1)
    real = torch.cos(phase_x.view(1, 1, 1, 1) * coordinate_x).expand(1, width, -1, -1)
    imag = torch.sin(phase_x.view(1, 1, 1, 1) * coordinate_x).expand(1, width, -1, -1)
    output = pole_aligned_complex_downsample(real, imag, phase_x, phase_y)
    expected_phase = 2.0 * phase_x.view(1, 1, 1, 1) * torch.arange(width // 2).view(1, 1, -1, 1)
    expected_real = torch.cos(expected_phase).expand(1, width // 2, -1, -1)
    expected_imag = torch.sin(expected_phase).expand(1, width // 2, -1, -1)
    assert torch.allclose(output[0][:, 1:-1, 1:-1], expected_real[:, 1:-1, 1:-1], atol=1e-6)
    assert torch.allclose(output[1][:, 1:-1, 1:-1], expected_imag[:, 1:-1, 1:-1], atol=1e-6)

    rotated = pole_aligned_complex_downsample(*_rotate(real, imag, 0.37), phase_x, phase_y)
    expected_rotated = _rotate(output[0], output[1], 0.37)
    assert torch.allclose(rotated[0], expected_rotated[0], atol=2.0e-6, rtol=2.0e-6)
    assert torch.allclose(rotated[1], expected_rotated[1], atol=2.0e-6, rtol=2.0e-6)


def test_wide_complex_transition_is_phase_equivariant_and_identity_gated() -> None:
    transition = ComplexModulatedTransition(16, 16, 6)
    real = torch.randn(2, 4, 4, 16)
    imag = torch.randn(2, 4, 4, 16)
    gain, phase = transition.modulation(real, imag)
    assert torch.equal(gain, torch.ones_like(gain))
    assert torch.equal(phase, torch.zeros_like(phase))
    assert torch.equal(
        transition.layer_scale,
        torch.full_like(transition.layer_scale, 1.0e-2),
    )
    output = transition(real, imag)
    rotated_output = transition(*_rotate(real, imag, 0.53))
    expected = _rotate(output[0], output[1], 0.53)
    assert torch.allclose(rotated_output[0], expected[0], atol=3.0e-6, rtol=3.0e-6)
    assert torch.allclose(rotated_output[1], expected[1], atol=3.0e-6, rtol=3.0e-6)


def test_complex_interaction_uses_invariant_coherence_and_is_equivariant() -> None:
    transition = ComplexInteractionTransition(16, 16, 6, coherence_rank=4)
    real = torch.randn(2, 4, 4, 16)
    imag = torch.randn(2, 4, 4, 16)
    hidden = transition.input_norm(*transition.input_projection(real, imag))
    context = transition.coherence_context(*hidden)
    rotated_hidden = _rotate(hidden[0], hidden[1], 0.71)
    rotated_context = transition.coherence_context(*rotated_hidden)
    assert torch.allclose(rotated_context, context, atol=3.0e-6, rtol=3.0e-6)
    assert torch.equal(
        2.0 * torch.sigmoid(transition.gate(context)),
        torch.ones_like(hidden[0]),
    )
    output = transition(real, imag)
    rotated_output = transition(*_rotate(real, imag, 0.71))
    expected = _rotate(output[0], output[1], 0.71)
    assert torch.allclose(rotated_output[0], expected[0], atol=4.0e-6, rtol=4.0e-6)
    assert torch.allclose(rotated_output[1], expected[1], atol=4.0e-6, rtol=4.0e-6)


def test_complex_linear_matches_native_complex_reference() -> None:
    linear = ComplexLinear(5, 3)
    real = torch.randn(2, 4, 5)
    imag = torch.randn(2, 4, 5)
    output_real, output_imag = linear(real, imag)
    weight = torch.complex(linear.weight_real, linear.weight_imag)
    reference = torch.matmul(torch.complex(real, imag), weight.transpose(0, 1))
    assert torch.allclose(output_real, reference.real, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(output_imag, reference.imag, atol=1.0e-6, rtol=1.0e-6)


def test_widely_linear_exactly_matches_arbitrary_real_affine_map() -> None:
    layer = WidelyLinear(5, 3, bias=True)
    real_weight = torch.randn(6, 10)
    real_bias = torch.randn(6)
    layer.load_real_affine(real_weight, real_bias)
    real = torch.randn(2, 4, 5)
    imag = torch.randn(2, 4, 5)
    output_real, output_imag = layer(real, imag)
    reference = torch.nn.functional.linear(
        torch.cat((real, imag), dim=-1),
        real_weight,
        real_bias,
    )
    assert torch.allclose(output_real, reference[..., :3], atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(output_imag, reference[..., 3:], atol=1.0e-6, rtol=1.0e-6)


def test_combined_weight_widely_linear_matches_decomposed_forward_and_backward() -> None:
    torch.manual_seed(20260804)
    baseline = WidelyLinear(7, 5, bias=True)
    candidate = WidelyLinear(7, 5, bias=True)
    candidate.load_state_dict(baseline.state_dict())

    base_real = torch.randn(3, 11, 7, requires_grad=True)
    base_imag = torch.randn(3, 11, 7, requires_grad=True)
    candidate_real = base_real.detach().clone().requires_grad_()
    candidate_imag = base_imag.detach().clone().requires_grad_()
    cotangent_real = torch.randn(3, 11, 5)
    cotangent_imag = torch.randn(3, 11, 5)

    baseline_output = baseline(base_real, base_imag)
    candidate_output = candidate(candidate_real, candidate_imag)
    for actual, expected in zip(candidate_output, baseline_output, strict=True):
        assert torch.allclose(actual, expected, atol=2.0e-6, rtol=2.0e-6)

    baseline_loss = (
        baseline_output[0] * cotangent_real + baseline_output[1] * cotangent_imag
    ).sum()
    candidate_loss = (
        candidate_output[0] * cotangent_real + candidate_output[1] * cotangent_imag
    ).sum()
    baseline_gradients = torch.autograd.grad(
        baseline_loss,
        (base_real, base_imag, *baseline.parameters()),
    )
    candidate_gradients = torch.autograd.grad(
        candidate_loss,
        (candidate_real, candidate_imag, *candidate.parameters()),
    )
    for actual, expected in zip(candidate_gradients, baseline_gradients, strict=True):
        assert torch.allclose(actual, expected, atol=3.0e-5, rtol=3.0e-5)


def test_augmented_complex_transition_shapes_and_backward() -> None:
    transition = AugmentedComplexTransition(16, 12, 6)
    real = torch.randn(2, 4, 4, 16, requires_grad=True)
    imag = torch.randn(2, 4, 4, 16, requires_grad=True)
    output_real, output_imag = transition(real, imag)
    assert output_real.shape == (2, 4, 4, 6)
    assert output_imag.shape == (2, 4, 4, 6)
    output_real.square().mean().add(output_imag.square().mean()).backward()
    assert real.grad is not None
    assert imag.grad is not None
    assert torch.isfinite(real.grad).all()
    assert torch.isfinite(imag.grad).all()


def test_complex_s2d_carry_is_energy_preserving() -> None:
    real = torch.randn(2, 8, 6, 4)
    imag = torch.randn(2, 8, 6, 4)
    reference_energy = real.square().sum() + imag.square().sum()
    carried_real, carried_imag = complex_carry_coordinates(real, imag, "s2d")
    assert carried_real.shape == (2, 4, 3, 16)
    assert carried_imag.shape == (2, 4, 3, 16)
    carried_energy = carried_real.square().sum() + carried_imag.square().sum()
    assert torch.allclose(carried_energy, reference_energy, atol=2.0e-5, rtol=2.0e-6)


def test_zero_scaled_pole_main_carry_exactly_matches_augmented_transition() -> None:
    torch.manual_seed(79)
    baseline = AugmentedComplexTransition(16, 12, 6)
    torch.manual_seed(79)
    carried = AugmentedComplexTransition(
        16,
        12,
        6,
        carry_input_modes=16,
        carry_scale_initial=0.0,
    )
    baseline_state = baseline.state_dict()
    carried_state = carried.state_dict()
    common_keys = baseline_state.keys() & carried_state.keys()
    assert all(torch.equal(baseline_state[key], carried_state[key]) for key in common_keys)
    real = torch.randn(2, 4, 4, 16)
    imag = torch.randn(2, 4, 4, 16)
    carry_real = torch.randn(2, 4, 4, 16)
    carry_imag = torch.randn(2, 4, 4, 16)
    reference = baseline(real, imag)
    output = carried(real, imag, carry_real, carry_imag)
    assert torch.equal(output[0], reference[0])
    assert torch.equal(output[1], reference[1])


def test_direction_aligned_endpoints_use_the_locked_quadrant_indices() -> None:
    real = torch.arange(16.0).reshape(1, 4, 4, 1)
    imag = -real
    expected = {
        (1, 1): real[:, 1::2, 1::2],
        (-1, 1): real[:, 1::2, 0::2],
        (1, -1): real[:, 0::2, 1::2],
        (-1, -1): real[:, 0::2, 0::2],
    }
    for direction, expected_real in expected.items():
        output_real, output_imag = direction_aligned_endpoints(
            real,
            imag,
            direction_x=direction[0],
            direction_y=direction[1],
        )
        assert torch.equal(output_real, expected_real)
        assert torch.equal(output_imag, -expected_real)


def test_conv_only_stem_contains_only_linear_convolutions() -> None:
    stem = CifarConvOnlyStem(64)
    assert [type(module) for module in stem.layers] == [
        torch.nn.Conv2d,
        torch.nn.Conv2d,
    ]
    output = stem(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 32, 32, 64)


def test_normalized_linear_stem_omits_activation_functions() -> None:
    stem = CifarNormalizedLinearStem(64)
    assert [type(module) for module in stem.layers] == [
        torch.nn.Conv2d,
        LayerNorm2d,
        torch.nn.Conv2d,
        LayerNorm2d,
    ]
    assert not any(isinstance(module, torch.nn.GELU) for module in stem.modules())
    output = stem(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 32, 32, 64)


def test_complex_product_pole_stage_preserves_phase_and_downsamples() -> None:
    stage = ComplexScanStage(
        4,
        maximum_phase=torch.pi * 0.75,
        output_modes=8,
    )
    real = torch.randn(1, 8, 8, 4)
    imag = torch.randn(1, 8, 8, 4)
    output, descriptor = stage(real, imag)
    assert output is not None
    assert output[0].shape == (1, 4, 4, 8)
    assert output[1].shape == (1, 4, 4, 8)
    assert descriptor.shape == (1, 16)
    rotated_output, rotated_descriptor = stage(*_rotate(real, imag, 0.41))
    assert rotated_output is not None
    expected = _rotate(output[0], output[1], 0.41)
    assert torch.allclose(rotated_output[0], expected[0], atol=3.0e-5, rtol=3.0e-5)
    assert torch.allclose(rotated_output[1], expected[1], atol=3.0e-5, rtol=3.0e-5)
    assert torch.allclose(rotated_descriptor, descriptor, atol=3.0e-5, rtol=3.0e-5)


def test_product_only_stage_automatically_uses_fused_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InheritedProductPoleBank(ComplexScanStage):
        pass

    stage = InheritedProductPoleBank(
        4,
        maximum_phase=torch.pi * 0.75,
        output_modes=4,
        quadrant_path_mode_cffn_width=8,
        quadrant_path_cffn_width=4,
    )
    calls = 0
    fused_scan = complex_scan_stage.run_product_scan_pipeline

    def observed_fused_scan(
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return fused_scan(*args, **kwargs)

    monkeypatch.setattr(
        complex_scan_stage,
        "run_product_scan_pipeline",
        observed_fused_scan,
    )
    source_real = torch.randn(1, 4, 4, 4)
    source_imag = torch.randn_like(source_real)

    output, descriptor = stage(source_real, source_imag)

    assert calls == 1
    assert output is not None
    assert output[0].shape == (1, 2, 2, 4)
    assert descriptor.shape == (1, 16)


def test_plain_stage_automatically_uses_associative_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = ComplexScanStage(
        4,
        maximum_phase=torch.pi * 0.75,
        output_modes=4,
    )
    fused_scan = complex_scan_stage.run_product_scan_pipeline
    calls: list[str] = []

    def observed_fused_scan(
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(str(kwargs["epilogue"]))
        return fused_scan(*args, **kwargs)

    monkeypatch.setattr(
        complex_scan_stage,
        "run_product_scan_pipeline",
        observed_fused_scan,
    )
    assert not hasattr(stage, "_run_recurrence")

    output, descriptor = stage(torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4))

    assert calls == ["coarse"]
    assert output is not None
    assert output[0].shape == (1, 2, 2, 4)
    assert descriptor.shape == (1, 16)


def test_complex_backbone_supports_rectangular_inputs_and_backward() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=16,
            modes=(4, 4, 4),
            quadratic_rank=3,
        )
    )
    inputs = torch.randn(2, 3, 24, 32, requires_grad=True)

    state2, state3 = model.complex_features(inputs)
    logits = model(inputs)

    assert state2[0].shape == (2, 12, 16, 4)
    assert state3[0].shape == (2, 6, 8, 4)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_product_scan_recomputation_matches_retained_forward_and_gradients() -> None:
    def make_stage(policy: Literal["retain", "recompute"]) -> ComplexScanStage:
        return ComplexScanStage(
            4,
            maximum_phase=torch.pi * 0.75,
            output_modes=4,
            quadrant_path_mode_cffn_width=8,
            quadrant_path_cffn_width=4,
            scan_memory_policy=policy,
        )

    retained = make_stage("retain")
    recomputed = make_stage("recompute")
    recomputed.load_state_dict(retained.state_dict())
    retained_inputs = tuple(
        value.requires_grad_() for value in (torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4))
    )
    recomputed_inputs = tuple(value.detach().clone().requires_grad_() for value in retained_inputs)

    def evaluate(
        stage: ComplexScanStage,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        output, descriptor = stage(*inputs)
        assert output is not None
        values = (*output, descriptor)
        parameters = tuple(stage.parameters())
        gradients = torch.autograd.grad(
            sum(value.float().square().mean() for value in values),
            (*inputs, *parameters),
        )
        return values, gradients

    retained_values, retained_gradients = evaluate(retained, retained_inputs)
    recomputed_values, recomputed_gradients = evaluate(recomputed, recomputed_inputs)
    for actual, expected in zip(recomputed_values, retained_values, strict=True):
        assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)
    for actual, expected in zip(recomputed_gradients, retained_gradients, strict=True):
        assert torch.allclose(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_stage_uses_one_associative_product_pipeline_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = ComplexScanStage(
        4,
        maximum_phase=torch.pi * 0.75,
        output_modes=4,
        quadrant_path_mode_cffn_width=8,
        quadrant_path_cffn_width=4,
    )
    calls = 0
    fused_scan = complex_scan_stage.run_product_scan_pipeline

    def observed_fused_scan(
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return fused_scan(*args, **kwargs)

    monkeypatch.setattr(
        complex_scan_stage,
        "run_product_scan_pipeline",
        observed_fused_scan,
    )

    output, descriptor = stage(torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4))

    assert calls == 1
    assert output is not None
    assert descriptor.shape == (1, 16)


def test_complex_backbone_shapes_and_backward_are_finite() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=16,
            modes=(4, 4, 4),
            quadratic_rank=3,
        )
    )
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    state2, state3 = model.complex_features(inputs)
    assert state2[0].shape == (2, 16, 16, 4)
    assert state3[0].shape == (2, 8, 8, 4)
    descriptor = model.raw_descriptor(inputs)
    assert descriptor.shape == (2, 48)
    logits = model(inputs)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_complex_backbone_supports_global_modal_fusion_head() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=16,
            modes=(4, 4, 4),
            fusion_width=7,
        )
    )
    assert isinstance(model.classifier, ModalFusionHead)
    assert model.classifier.fusion.weight.shape == (7, 48)
    assert model.classifier.classifier.weight.shape == (10, 7)
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    logits = model(inputs)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_imagenet_stem_strides_reduce_before_pole_transport() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=16,
            stem_strides=(2, 2),
            modes=(4, 4, 4),
            quadratic_rank=3,
        )
    )
    inputs = torch.randn(2, 3, 64, 64)
    assert model.stem.layers[0].bias is not None
    assert model.stem.layers[3].bias is not None
    stem = model.stem(inputs)
    state2, state3 = model.complex_features(inputs)
    assert stem.shape == (2, 16, 16, 16)
    assert state2[0].shape == (2, 8, 8, 4)
    assert state3[0].shape == (2, 4, 4, 4)


def test_local_fourier_stem_integrates_rgb_and_lifts_complex_modes() -> None:
    stem = LocalComplexFourierStem(
        8,
        color_stride=1,
        fourier_hop=4,
        window_size=8,
    )
    inputs = torch.randn(2, 3, 64, 64, requires_grad=True)
    real, imag = stem(inputs)
    assert real.shape == (2, 16, 16, 8)
    assert imag.shape == real.shape
    assert torch.isfinite(real).all()
    assert torch.isfinite(imag).all()
    real.square().add(imag.square()).mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert stem.color[0].weight.grad is not None
    assert stem.phase_x.grad is not None
    assert stem.phase_y.grad is not None


def test_local_fourier_stem_backbone_shapes_and_backward() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=4,
            stem="local_fourier",
            stem_strides=(1, 4),
            modes=(4, 4, 4),
            augmented_widths=(6, 8),
            use_pole_aligned_shortcuts=True,
            quadratic_rank=3,
        )
    )
    inputs = torch.randn(2, 3, 64, 64, requires_grad=True)
    assert isinstance(model.stem, LocalComplexFourierStem)
    excitation = model.stem(inputs)
    assert excitation[0].shape == (2, 16, 16, 4)
    state2, state3 = model.complex_features(inputs)
    assert state2[0].shape == (2, 8, 8, 4)
    assert state3[0].shape == (2, 4, 4, 4)
    logits = model(inputs)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_complex_pixel_encoding_and_s2d_packing_are_lossless() -> None:
    stem = ComplexPixelStem(32, reductions=(2, 2))
    inputs = torch.randn(2, 3, 32, 32)
    encoded = stem.encode_pixels(inputs)
    reconstructed = stem.decode_pixels(*encoded)
    assert torch.allclose(reconstructed, inputs, atol=1.0e-6, rtol=1.0e-6)
    packed = stem.pack_pixels(*encoded)
    unpacked = stem.unpack_pixels(*packed)
    assert torch.equal(unpacked[0], encoded[0])
    assert torch.equal(unpacked[1], encoded[1])


def test_complex_pixel_stem_shapes_and_complex_gradients() -> None:
    stem = ComplexPixelStem(32, reductions=(2, 2))
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    real, imag = stem(inputs)
    assert real.shape == (2, 8, 8, 32)
    assert imag.shape == real.shape
    real.square().add(imag.square()).mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert stem.first_block.depthwise.weight_real.grad is not None
    assert stem.second_block.mixing.conjugate_gate.grad is not None
    assert torch.isfinite(stem.second_block.mixing.conjugate_gate.grad).all()


def test_complex_pixel_stem_backbone_shapes_and_backward() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=4,
            stem="complex_pixel",
            stem_strides=(2, 2),
            modes=(32, 32, 32),
            augmented_widths=(64, 64),
            use_pole_aligned_shortcuts=True,
            quadratic_rank=3,
        )
    )
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    assert isinstance(model.stem, ComplexPixelStem)
    excitation = model.stem(inputs)
    assert excitation[0].shape == (2, 8, 8, 32)
    state2, state3 = model.complex_features(inputs)
    assert state2[0].shape == (2, 4, 4, 32)
    assert state3[0].shape == (2, 2, 2, 32)
    logits = model(inputs)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_amp_phase_wide_backbone_contract_and_backward() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=16,
            modes=(4, 4, 4),
            transition_widths=(8, 8),
            quadratic_rank=3,
        )
    )
    assert isinstance(model.stage1.transition, ComplexModulatedTransition)
    assert isinstance(model.stage2.transition, ComplexModulatedTransition)
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    descriptor = model.raw_descriptor(inputs)
    logits = model(inputs)
    assert descriptor.shape == (2, 48)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.stage1.transition.phase_gate.weight.grad is not None
    assert torch.isfinite(model.stage1.transition.phase_gate.weight.grad).all()


def test_complex_interaction_backbone_contract_and_backward() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=16,
            modes=(4, 4, 4),
            interaction_ranks=(2, 2),
            quadratic_rank=3,
        )
    )
    assert isinstance(model.stage1.interaction, ComplexInteractionTransition)
    assert isinstance(model.stage2.interaction, ComplexInteractionTransition)
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    descriptor = model.raw_descriptor(inputs)
    logits = model(inputs)
    assert descriptor.shape == (2, 48)
    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.stage1.interaction.coherence_left.weight_real.grad is not None
    assert torch.isfinite(model.stage1.interaction.coherence_left.weight_real.grad).all()


def test_augmented_and_widely_linear_backbone_contracts() -> None:
    common = ComplexScanConfig(
        output_dim=10,
        stem_width=16,
        modes=(4, 4, 4),
        quadratic_rank=3,
    )
    augmented = ComplexScanBackbone(replace(common, augmented_widths=(6, 8)))
    widely = ComplexScanBackbone(replace(common, widely_linear_bridges=True))
    assert isinstance(augmented.stage1.augmented, AugmentedComplexTransition)
    assert isinstance(augmented.stage2.augmented, AugmentedComplexTransition)
    assert isinstance(widely.stage1.widely_bridge, WidelyLinear)
    assert isinstance(widely.stage2.widely_bridge, WidelyLinear)
    inputs = torch.randn(2, 3, 32, 32)
    for model in (augmented, widely):
        logits = model(inputs)
        assert logits.shape == (2, 10)
        assert torch.isfinite(logits).all()


def test_complex_stage_carry_backbone_contract_and_backward() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=16,
            modes=(4, 4, 4),
            augmented_widths=(6, 8),
            carry_bases=("s2d", "s2d"),
            carry_merge="pole_main",
            carry_scale_initial=1.0e-2,
            quadratic_rank=3,
        )
    )
    assert model.stage1.augmented is not None
    assert model.stage2.augmented is not None
    assert model.stage1.augmented.carry_mixer is not None
    assert model.stage2.augmented.carry_mixer is not None
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    logits = model(inputs)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    carry_weight = model.stage1.augmented.carry_mixer.weight_real
    carry_scale = model.stage1.augmented.carry_scale
    assert carry_weight.grad is not None
    assert carry_scale is not None
    assert carry_scale.grad is not None
    assert torch.isfinite(carry_weight.grad).all()
    assert torch.isfinite(carry_scale.grad).all()


def test_pole_aligned_shortcut_backbone_contract_and_backward() -> None:
    model = ComplexScanBackbone(
        ComplexScanConfig(
            output_dim=10,
            stem_width=16,
            modes=(4, 4, 4),
            augmented_widths=(6, 8),
            use_pole_aligned_shortcuts=True,
            stage_residual_scale_initial=0.1,
            quadratic_rank=3,
        )
    )
    assert model.stage1.stage_residual_scale is not None
    assert model.stage2.stage_residual_scale is not None
    assert torch.equal(
        model.stage1.stage_residual_scale,
        torch.full_like(model.stage1.stage_residual_scale, 0.1),
    )
    inputs = torch.randn(2, 3, 32, 32, requires_grad=True)
    state2, state3 = model.complex_features(inputs)
    assert state2[0].shape == (2, 16, 16, 4)
    assert state3[0].shape == (2, 8, 8, 4)
    logits = model(inputs)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.stage1.stage_residual_scale.grad is not None
    assert torch.isfinite(model.stage1.stage_residual_scale.grad).all()
    assert model.stage1.phase_x.grad is not None
    assert torch.isfinite(model.stage1.phase_x.grad).all()


def test_pole_aligned_shortcut_rejects_mode_changes_and_parallel_carry() -> None:
    with pytest.raises(ValueError, match="equal mode counts"):
        ComplexScanConfig(
            modes=(4, 8, 8),
            use_pole_aligned_shortcuts=True,
        ).validate()
    with pytest.raises(ValueError, match="replace S2D"):
        ComplexScanConfig(
            modes=(4, 4, 4),
            augmented_widths=(6, 8),
            carry_bases=("s2d", "s2d"),
            use_pole_aligned_shortcuts=True,
        ).validate()


def test_zero_scaled_complex_stage_carry_matches_existing_backbone() -> None:
    common = ComplexScanConfig(
        output_dim=10,
        stem_width=16,
        modes=(4, 4, 4),
        augmented_widths=(6, 8),
        quadratic_rank=3,
    )
    torch.manual_seed(83)
    baseline = ComplexScanBackbone(common)
    torch.manual_seed(83)
    carried = ComplexScanBackbone(
        replace(
            common,
            carry_bases=("s2d", "s2d"),
            carry_scale_initial=0.0,
        )
    )
    baseline_state = baseline.state_dict()
    carried_state = carried.state_dict()
    common_keys = baseline_state.keys() & carried_state.keys()
    assert all(torch.equal(baseline_state[key], carried_state[key]) for key in common_keys)
    inputs = torch.randn(2, 3, 32, 32)
    baseline.eval()
    carried.eval()
    assert torch.equal(carried(inputs), baseline(inputs))
