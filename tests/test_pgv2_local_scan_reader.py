from __future__ import annotations

# pyright: reportMissingImports=false, reportPrivateUsage=false
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_imagenet100 as control
import run_a2d_deep4_calibrated_uniform_p96_phase_gated_h96_local_reader_imagenet100 as candidate

from lnet.complex_scan import ComplexScanConfig
from lnet.pac_complex_scan_reader import PackedComplexConv2dReader
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN
from lnet.pac_phase_gated_transition import (
    AveragePoolMagnitudeGateTransition,
    AveragePoolPhaseGatedRefinementTransition,
    MemoryOnlyMagnitudeGateTransition,
    PathOnlyCollapse,
    PathPhaseGatedCollapse,
    PhaseGatedModeResidualPathCollapse,
)


def _config() -> ComplexScanConfig:
    return ComplexScanConfig(output_dim=100, stem_strides=(2, 2))


def test_local_reader_validates_shape_and_kernel() -> None:
    with pytest.raises(ValueError, match="positive odd"):
        PackedComplexConv2dReader(4, 4, kernel_size=2)
    with pytest.raises(ValueError, match="variance epsilon"):
        PackedComplexConv2dReader(4, 4, variance_epsilon=0.0)
    reader = PackedComplexConv2dReader(4, 6)
    with pytest.raises(ValueError, match="matching NHWM"):
        reader(torch.randn(2, 3, 4), torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="mode dimension"):
        reader(torch.randn(2, 3, 3, 5), torch.randn(2, 3, 3, 5))


def test_local_reader_is_exact_point_identity_after_identity_initialization() -> None:
    reader = PackedComplexConv2dReader(8, 8, kernel_size=5)
    reader.initialize_identity_()
    real = torch.randn(2, 7, 9, 8)
    imag = torch.randn_like(real)
    output_real, output_imag = reader(real, imag)
    assert output_real.is_contiguous()
    assert output_imag.is_contiguous()
    torch.testing.assert_close(output_real, real, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(output_imag, imag, rtol=1.0e-6, atol=1.0e-6)

    restored = PackedComplexConv2dReader(8, 8, kernel_size=5)
    restored.load_state_dict(reader.state_dict())
    restored_real, restored_imag = restored(real, imag)
    torch.testing.assert_close(restored_real, output_real, rtol=0, atol=0)
    torch.testing.assert_close(restored_imag, output_imag, rtol=0, atol=0)


def test_local_reader_is_global_phase_equivariant() -> None:
    torch.manual_seed(41)
    reader = PackedComplexConv2dReader(
        5,
        7,
        kernel_size=3,
        match_input_rms=True,
    )
    with torch.no_grad():
        reader.weight_real.normal_(std=0.08)
        reader.weight_imag.normal_(std=0.08)
    real = torch.randn(2, 6, 8, 5)
    imag = torch.randn_like(real)
    angle = 0.73
    cosine, sine = math.cos(angle), math.sin(angle)
    expected_real, expected_imag = reader(real, imag)
    rotated_real = cosine * real - sine * imag
    rotated_imag = sine * real + cosine * imag
    actual_real, actual_imag = reader(rotated_real, rotated_imag)
    torch.testing.assert_close(
        actual_real,
        cosine * expected_real - sine * expected_imag,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        actual_imag,
        sine * expected_real + cosine * expected_imag,
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_local_reader_preserves_unit_drive_variance_after_weight_updates() -> None:
    torch.manual_seed(47)
    reader = PackedComplexConv2dReader(5, 7, kernel_size=3)
    with torch.no_grad():
        reader.weight_real.normal_(mean=0.2, std=1.7)
        reader.weight_imag.normal_(mean=-0.1, std=0.9)
    weight_real, weight_imag = reader.normalized_weight()
    row_energy = weight_real.square().add(weight_imag.square()).sum((1, 2, 3))
    torch.testing.assert_close(row_energy, torch.ones_like(row_energy), rtol=2.0e-6, atol=2.0e-6)


def test_local_reader_backward_reaches_full_complex_kernel_weights() -> None:
    reader = PackedComplexConv2dReader(
        4,
        6,
        kernel_size=3,
        match_input_rms=True,
    )
    real = torch.randn(2, 5, 7, 4, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    output = reader(real, imag)
    loss = output[0].square().mean() + output[1].square().mean()
    loss.backward()
    assert real.grad is not None
    assert imag.grad is not None
    for parameter in reader.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_local_reader_can_match_complex_rms_per_spatial_token() -> None:
    torch.manual_seed(53)
    reader = PackedComplexConv2dReader(
        5,
        7,
        kernel_size=3,
        match_input_rms=True,
    )
    with torch.no_grad():
        reader.weight_real.normal_(std=0.2)
        reader.weight_imag.normal_(std=0.2)
    real = torch.randn(2, 6, 8, 5)
    imag = torch.randn_like(real)
    output_real, output_imag = reader(real, imag)
    input_energy = real.square().add(imag.square()).mean(dim=-1)
    output_energy = output_real.square().add(output_imag.square()).mean(dim=-1)
    torch.testing.assert_close(
        output_energy,
        input_energy,
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_local_reader_is_one_full_spatial_mode_mixing_convolution() -> None:
    for input_modes, output_modes, kernel_size in ((16, 32, 3), (96, 96, 3), (128, 64, 5)):
        reader = PackedComplexConv2dReader(
            input_modes,
            output_modes,
            kernel_size=kernel_size,
        )
        actual = sum(parameter.numel() for parameter in reader.parameters())
        expected = 2 * input_modes * output_modes * kernel_size**2
        assert actual == expected
        assert reader.weight_real.shape == (
            output_modes,
            input_modes,
            kernel_size,
            kernel_size,
        )


def test_model_changes_only_four_local_scan_readers() -> None:
    torch.manual_seed(501)
    baseline = control._build(control.VARIANT, _config())
    torch.manual_seed(501)
    model = candidate._build(candidate.VARIANT, _config())

    baseline_state = baseline.state_dict()
    candidate_state = model.state_dict()
    added = {key for key in candidate_state if ".pole_input_projection." in key}
    assert added == {
        f"{stage}.pole_input_projection.{coordinate}"
        for stage in ("stage1", "stage2", "stage3", "terminal")
        for coordinate in (
            "weight_real",
            "weight_imag",
        )
    }
    assert set(baseline_state) == set(candidate_state) - added
    for key, value in baseline_state.items():
        torch.testing.assert_close(candidate_state[key], value, rtol=0, atol=0)


def test_model_preserves_control_output_and_rng_at_initialization() -> None:
    torch.manual_seed(501)
    baseline = control._build(control.VARIANT, _config()).eval()
    expected_rng = torch.rand(32)
    torch.manual_seed(501)
    model = candidate._build(candidate.VARIANT, _config()).eval()
    actual_rng = torch.rand(32)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = baseline(inputs)
        actual = model(inputs)
    assert isinstance(expected, tuple)
    assert isinstance(actual, tuple)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(right, left, rtol=3.0e-5, atol=3.0e-6)

    torch.manual_seed(501)
    rms_matched = candidate._build(candidate.RMS_MATCH_VARIANT, _config()).eval()
    with torch.no_grad():
        matched = rms_matched(inputs)
    assert isinstance(matched, tuple)
    for left, right in zip(expected, matched, strict=True):
        torch.testing.assert_close(right, left, rtol=3.0e-5, atol=3.0e-6)


def test_rms_matched_variant_enables_only_shared_token_gain_control() -> None:
    regular = candidate._build(candidate.VARIANT, _config())
    matched = candidate._build(candidate.RMS_MATCH_VARIANT, _config())
    assert set(regular.state_dict()) == set(matched.state_dict())
    assert sum(parameter.numel() for parameter in regular.parameters()) == sum(
        parameter.numel() for parameter in matched.parameters()
    )
    for name in ("stage1", "stage2", "stage3", "terminal"):
        assert not getattr(regular, name).pole_input_projection.match_input_rms
        assert getattr(matched, name).pole_input_projection.match_input_rms


def test_unit_row_variant_constrains_only_phase_gated_scale_contract() -> None:
    matched = candidate._build(candidate.RMS_MATCH_VARIANT, _config())
    unit_row = candidate._build(candidate.UNIT_ROW_VARIANT, _config())

    assert set(matched.state_dict()) == set(unit_row.state_dict())
    assert sum(parameter.numel() for parameter in matched.parameters()) == sum(
        parameter.numel() for parameter in unit_row.parameters()
    )
    for name in ("stage1", "stage2", "stage3"):
        mode = getattr(unit_row, name).quadrant_path_mode_combiner.mode
        assert mode.unit_row_projections
        assert mode.residual_scale_max == candidate.PG_RESIDUAL_SCALE_MAX
        torch.testing.assert_close(
            mode.effective_residual_scale(),
            mode.gamma.new_tensor(0.01),
        )
        for weight_real, weight_imag in mode.effective_projection_weights():
            row_energy = weight_real.square().add(weight_imag.square()).sum(dim=-1)
            torch.testing.assert_close(
                row_energy,
                torch.ones_like(row_energy),
                rtol=2.0e-6,
                atol=2.0e-6,
            )
    for name in ("stage1", "stage2", "stage3", "terminal"):
        assert getattr(unit_row, name).pole_input_projection.match_input_rms


def test_mode_gain_variant_uses_retracted_directions_and_per_mode_gains() -> None:
    matched = candidate._build(candidate.RMS_MATCH_VARIANT, _config())
    mode_gain = candidate._build(candidate.MODE_GAIN_VARIANT, _config())

    matched_state = set(matched.state_dict())
    mode_gain_state = set(mode_gain.state_dict())
    removed = matched_state - mode_gain_state
    added = mode_gain_state - matched_state
    assert removed == {
        f"{stage}.quadrant_path_mode_combiner.mode.gamma"
        for stage in ("stage1", "stage2", "stage3")
    }
    assert added == {
        f"{stage}.quadrant_path_mode_combiner.mode.output_gain_logits"
        for stage in ("stage1", "stage2", "stage3")
    }
    for name in ("stage1", "stage2", "stage3"):
        mode = getattr(mode_gain, name).quadrant_path_mode_combiner.mode
        assert mode.projected_direction_rows
        assert mode.gamma is None
        assert mode.output_gain_max == candidate.PG_OUTPUT_GAIN_MAX
        gains = mode.effective_output_gains()
        assert gains is not None
        torch.testing.assert_close(gains, gains.new_full((candidate.P,), 0.01))
        for projection in (mode.input_projection, mode.output_projection):
            row_energy = (
                projection.weight_real.square().add(projection.weight_imag.square()).sum(dim=-1)
            )
            torch.testing.assert_close(
                row_energy,
                torch.ones_like(row_energy),
                rtol=2.0e-6,
                atol=2.0e-6,
            )
    for name in ("stage1", "stage2", "stage3", "terminal"):
        assert getattr(mode_gain, name).pole_input_projection.match_input_rms


@pytest.mark.parametrize(
    ("variant", "removed_stages"),
    [
        (candidate.NO_PG_STAGE3_VARIANT, frozenset(("stage3",))),
        (
            candidate.NO_PG_ALL_VARIANT,
            frozenset(("stage1", "stage2", "stage3")),
        ),
    ],
)
def test_no_pg_variants_remove_only_selected_mode_residuals(
    variant: str,
    removed_stages: frozenset[str],
) -> None:
    torch.manual_seed(501)
    matched = candidate._build(candidate.RMS_MATCH_VARIANT, _config())
    expected_rng = torch.rand(32)
    torch.manual_seed(501)
    ablated = candidate._build(variant, _config())
    actual_rng = torch.rand(32)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    matched_state = matched.state_dict()
    ablated_state = ablated.state_dict()
    expected_removed = {
        key
        for key in matched_state
        if any(
            key.startswith(f"{stage}.quadrant_path_mode_combiner.mode.") for stage in removed_stages
        )
    }
    assert set(matched_state) - set(ablated_state) == expected_removed
    assert not (set(ablated_state) - set(matched_state))
    for key, value in ablated_state.items():
        torch.testing.assert_close(value, matched_state[key], rtol=0, atol=0)

    for name in ("stage1", "stage2", "stage3"):
        mixer = getattr(ablated, name).quadrant_path_mode_combiner
        if name in removed_stages:
            assert isinstance(mixer, PathOnlyCollapse)
        else:
            assert isinstance(mixer, PhaseGatedModeResidualPathCollapse)
    for name in ("stage1", "stage2", "stage3", "terminal"):
        assert getattr(ablated, name).pole_input_projection.match_input_rms


@pytest.mark.parametrize(
    ("variant", "activation"),
    [
        (candidate.PATH_PG_VARIANT, "silu"),
        (candidate.PATH_PG_NO_SILU_VARIANT, "identity"),
    ],
)
def test_path_pg_variant_moves_the_same_shared_pg_between_path_projections(
    variant: str,
    activation: str,
) -> None:
    apply_cartesian_silu = activation == "silu"
    torch.manual_seed(501)
    matched = candidate._build(candidate.RMS_MATCH_VARIANT, _config())
    expected_rng = torch.rand(32)
    torch.manual_seed(501)
    path_pg = candidate._build(variant, _config())
    actual_rng = torch.rand(32)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    matched_state = matched.state_dict()
    path_pg_state = path_pg.state_dict()
    assert set(path_pg_state) == set(matched_state)
    for key, value in matched_state.items():
        torch.testing.assert_close(path_pg_state[key], value, rtol=0, atol=0)

    for name in ("stage1", "stage2", "stage3"):
        mixer = getattr(path_pg, name).quadrant_path_mode_combiner
        assert isinstance(mixer, PathPhaseGatedCollapse)
        assert mixer.path_input.input_paths == 4
        assert mixer.path_input.output_paths == candidate.control.PATH_HIDDEN
        assert mixer.mode.modes == candidate.P
        assert mixer.mode.hidden_modes == candidate.control.MODE_HIDDEN
        assert mixer.path_output.input_paths == candidate.control.PATH_HIDDEN
        assert mixer.path_output.output_paths == 1
        assert mixer.apply_cartesian_silu is apply_cartesian_silu
        assert sum(isinstance(child, PhaseGatedComplexFFN) for child in mixer.modules()) == 1
    for name in ("stage1", "stage2", "stage3", "terminal"):
        assert getattr(path_pg, name).pole_input_projection.match_input_rms


def test_average_reexcitation_changes_only_the_declared_post_scan_transition() -> None:
    torch.manual_seed(501)
    no_silu = candidate._build(candidate.PATH_PG_NO_SILU_VARIANT, _config())
    expected_rng = torch.rand(32)
    torch.manual_seed(501)
    model = candidate._build(candidate.AVG_REEXCITE_VARIANT, _config())
    actual_rng = torch.rand(32)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    control_state = no_silu.state_dict()
    model_state = model.state_dict()
    unchanged_control = {
        key: value for key, value in control_state.items() if ".augmented." not in key
    }
    unchanged_model = {key: value for key, value in model_state.items() if ".augmented." not in key}
    assert set(unchanged_model) == set(unchanged_control)
    for key, value in unchanged_control.items():
        torch.testing.assert_close(unchanged_model[key], value, rtol=0, atol=0)

    input_weights = []
    for name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        assert isinstance(mixer, PathPhaseGatedCollapse)
        assert mixer.apply_cartesian_silu is False
        assert isinstance(transition, AveragePoolPhaseGatedRefinementTransition)
        assert transition.carry_input_modes == 4 * candidate.P
        assert transition.refine.modes == candidate.P
        assert transition.refine.hidden_modes == candidate.control.MODE_HIDDEN
        assert not transition.refine.learnable_residual_scale
        assert transition.refine.gamma is None
        assert stage.product_gain_normalization == "pointwise"
        input_weights.append(transition.refine.input_projection.weight_real)
    assert not torch.equal(input_weights[0], input_weights[1])
    assert not torch.equal(input_weights[1], input_weights[2])
    assert model.terminal.output_modes is None
    assert model.terminal.product_gain_normalization == "pointwise"


def test_average_pure_gate_removes_only_the_reexcitation_projections() -> None:
    torch.manual_seed(501)
    no_silu = candidate._build(candidate.PATH_PG_NO_SILU_VARIANT, _config())
    expected_rng = torch.rand(32)
    torch.manual_seed(501)
    model = candidate._build(candidate.AVG_PURE_GATE_VARIANT, _config())
    actual_rng = torch.rand(32)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    control_state = no_silu.state_dict()
    model_state = model.state_dict()
    unchanged_control = {
        key: value for key, value in control_state.items() if ".augmented." not in key
    }
    unchanged_model = {key: value for key, value in model_state.items() if ".augmented." not in key}
    assert set(unchanged_model) == set(unchanged_control)
    for key, value in unchanged_control.items():
        torch.testing.assert_close(unchanged_model[key], value, rtol=0, atol=0)

    for name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, name)
        mixer = stage.quadrant_path_mode_combiner
        transition = stage.augmented
        assert isinstance(mixer, PathPhaseGatedCollapse)
        assert mixer.apply_cartesian_silu is False
        assert isinstance(transition, AveragePoolMagnitudeGateTransition)
        assert set(dict(transition.named_parameters())) == {"norm.weight", "gate.alpha"}
        assert not any(isinstance(child, PhaseGatedComplexFFN) for child in transition.modules())
        assert stage.product_gain_normalization == "pointwise"
    assert sum(isinstance(child, PhaseGatedComplexFFN) for child in model.modules()) == 3
    assert model.terminal.output_modes is None
    assert model.terminal.product_gain_normalization == "pointwise"


def test_memory_only_pure_gate_removes_only_the_s2d_carry() -> None:
    torch.manual_seed(501)
    average_carry = candidate._build(candidate.AVG_PURE_GATE_VARIANT, _config())
    expected_rng = torch.rand(32)
    torch.manual_seed(501)
    memory_only = candidate._build(
        candidate.MEMORY_ONLY_PURE_GATE_VARIANT,
        _config(),
    )
    actual_rng = torch.rand(32)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    average_state = average_carry.state_dict()
    memory_state = memory_only.state_dict()
    assert set(memory_state) == set(average_state)
    for key, value in memory_state.items():
        torch.testing.assert_close(value, average_state[key], rtol=0, atol=0)

    config = candidate._variant_config(candidate.MEMORY_ONLY_PURE_GATE_VARIANT)
    transition_config = config["backbone"]["stage_transition"]
    assert transition_config["local_shortcut"]["operator"] == "none"
    assert transition_config["post_fusion"]["equation"] == ("E_next = M * g = M + M * (g - 1)")
    assert isinstance(memory_only.input_norm, nn.RMSNorm)
    assert sum(parameter.numel() for parameter in memory_only.parameters()) == sum(
        parameter.numel() for parameter in average_carry.parameters()
    )
    for name in ("stage1", "stage2", "stage3"):
        transition = getattr(memory_only, name).augmented
        assert type(transition) is MemoryOnlyMagnitudeGateTransition
        assert not hasattr(transition, "carry_input_modes")
        assert set(dict(transition.named_parameters())) == {"norm.weight", "gate.alpha"}


def test_no_stem_rms_variant_removes_only_the_pre_analysis_norm() -> None:
    torch.manual_seed(501)
    pure_gate = candidate._build(candidate.AVG_PURE_GATE_VARIANT, _config())
    expected_rng = torch.rand(32)
    torch.manual_seed(501)
    no_stem_rms = candidate._build(
        candidate.AVG_PURE_GATE_NO_STEM_RMS_VARIANT,
        _config(),
    )
    actual_rng = torch.rand(32)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

    pure_config = candidate._variant_config(candidate.AVG_PURE_GATE_VARIANT)
    no_norm_config = candidate._variant_config(
        candidate.AVG_PURE_GATE_NO_STEM_RMS_VARIANT,
    )
    assert pure_config["backbone"]["stem"]["interface_norm"] == "RMSNorm192"
    assert no_norm_config["backbone"]["stem"]["interface_norm"] == "identity; RMSNorm192 removed"

    pure_state = pure_gate.state_dict()
    no_norm_state = no_stem_rms.state_dict()
    assert set(pure_state) - set(no_norm_state) == {"input_norm.weight"}
    assert not (set(no_norm_state) - set(pure_state))
    for key, value in no_norm_state.items():
        torch.testing.assert_close(value, pure_state[key], rtol=0, atol=0)

    assert isinstance(pure_gate.input_norm, nn.RMSNorm)
    assert type(no_stem_rms.input_norm) is nn.Identity
    assert (
        sum(parameter.numel() for parameter in pure_gate.parameters())
        - sum(parameter.numel() for parameter in no_stem_rms.parameters())
        == 2 * candidate.P
    )

    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        transformed = no_stem_rms.precomplex_fc(no_stem_rms.stem(inputs))
        expected = no_stem_rms.analysis(transformed).chunk(2, dim=-1)
        actual = no_stem_rms._initial_excitation(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_model_reader_parameters_use_standard_base_lr_decay_group() -> None:
    model = candidate._build(candidate.VARIANT, _config())
    recipe = {
        "learning_rate": 3.0e-3,
        "modal_learning_rate_multiplier": 1.0 / 3.0,
        "pole_geometry_learning_rate_multiplier": 0.1,
        "weight_decay": 0.05,
        "fused_optimizer": False,
    }
    optimizer_source = (
        control.control.control.stemres.uniform.base.backbone.a2d_base.residuals.optimizer_source
    )
    optimizer = optimizer_source._build_optimizer(model, recipe)
    owner = {
        id(parameter): group for group in optimizer.param_groups for parameter in group["params"]
    }
    for name in ("stage1", "stage2", "stage3", "terminal"):
        reader = getattr(model, name).pole_input_projection
        for parameter in reader.parameters():
            group = owner[id(parameter)]
            assert group["lr"] == recipe["learning_rate"]
            assert group["weight_decay"] == recipe["weight_decay"]
