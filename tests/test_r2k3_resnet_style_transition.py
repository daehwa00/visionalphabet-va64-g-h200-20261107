from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import torch

from lnet.complex_scan_transitions import complex_rms_unit
from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    NormalizedShortcutGatedTransition,
    ResNetStyleGatedTransition,
    resnet_style_transition,
)
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_resnet_style_transition_d2262_imagenet100 as runner


def test_projected_transition_matches_folded_normalized_function() -> None:
    torch.manual_seed(23)
    source = NormalizedShortcutGatedTransition(
        7,
        5,
        9,
        post_hidden=12,
        memory_scale_initial=1.0,
    ).double()
    with torch.no_grad():
        source.shortcut_norm.weight.uniform_(0.5, 1.5)
        source.memory_norm.weight.uniform_(0.5, 1.5)
        source.memory_scale.uniform_(0.5, 1.5)
    candidate = resnet_style_transition(source)
    assert candidate.carry_gain is not None
    with torch.no_grad():
        candidate.carry_gain.copy_(source.shortcut_norm.weight)
        candidate.memory_gain.copy_(source.memory_norm.weight * source.memory_scale)
    memory = torch.randn(2, 3, 7, dtype=torch.double)
    carry = torch.randn(2, 3, 20, dtype=torch.double)
    torch.manual_seed(41)
    memory_imag = torch.randn_like(memory)
    carry_imag = torch.randn_like(carry)
    expected = source(memory, memory_imag, carry, carry_imag)
    actual = candidate(memory, memory_imag, carry, carry_imag)

    torch.testing.assert_close(actual, expected)


def test_resnet_style_replacement_preserves_seeded_operators() -> None:
    source = GatedPoleExcitationS2DTransition(7, 5, 9, post_hidden=12).double()
    source.eval()
    replacement = resnet_style_transition(source)

    assert type(replacement) is ResNetStyleGatedTransition
    assert replacement.training is source.training
    for owner in ("memory_projection", "carry_projection", "post_fusion"):
        target_state = getattr(replacement, owner).state_dict()
        source_state = getattr(source, owner).state_dict()
        assert target_state.keys() == source_state.keys()
        for name in target_state:
            torch.testing.assert_close(target_state[name], source_state[name])


def test_equal_width_transition_preserves_the_raw_identity_shortcut() -> None:
    source = GatedPoleExcitationS2DTransition(7, 5, 5, post_hidden=8).double()
    replacement = resnet_style_transition(source)

    assert type(replacement) is ResNetStyleGatedTransition
    assert replacement.carry_projection is None
    assert replacement.carry_gain is None
    memory = torch.randn(2, 3, 7, dtype=torch.double)
    carry = torch.randn(2, 3, 20, dtype=torch.double)
    memory_imag = torch.randn_like(memory)
    carry_imag = torch.randn_like(carry)
    projected_memory = replacement._project(
        replacement.memory_projection,
        (memory, memory_imag),
    )
    memory_unit = complex_rms_unit(
        *projected_memory,
        epsilon=replacement.normalization_epsilon,
    )
    identity = replacement._carry(carry, carry_imag)
    gain = replacement.memory_gain
    expected = replacement.post_fusion(
        identity[0] + gain * memory_unit[0],
        identity[1] + gain * memory_unit[1],
    )
    actual = replacement(memory, memory_imag, carry, carry_imag)

    torch.testing.assert_close(actual, expected)


def test_campaign_partitions_two_exact_controls() -> None:
    assert runner.SPECS[runner.WIDENED].excitation_modes == (32, 64, 96, 128)
    assert runner.SPECS[runner.COMPACT].excitation_modes == (32, 64, 64, 128)
    assert runner.JOBS_BY_GPU == {0: (runner.WIDENED,), 1: (runner.COMPACT,)}


def test_campaign_uses_projected_identity_shortcuts() -> None:
    expected_parameters = {
        runner.WIDENED: 2_487_620,
        runner.COMPACT: 1_858_244,
    }
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        runner._assert_model(model, variant)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == expected_parameters[variant]
        )
        schedule = runner.SPECS[variant].excitation_modes
        for name, input_modes, output_modes in zip(
            runner.STAGE_NAMES[:3],
            schedule[:-1],
            schedule[1:],
            strict=True,
        ):
            transition = getattr(model, name).augmented
            assert type(transition) is ResNetStyleGatedTransition
            assert (transition.carry_projection is None) == (input_modes == output_modes)
            assert (transition.carry_gain is None) == (input_modes == output_modes)


def test_width_comparison_pairs_every_common_state_tensor() -> None:
    models = {}
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        models[variant] = runner._build(variant, runtime.model_config())
    widened_state = models[runner.WIDENED].state_dict()
    compact_state = models[runner.COMPACT].state_dict()
    common = {
        name
        for name in widened_state.keys() & compact_state.keys()
        if widened_state[name].shape == compact_state[name].shape
    }

    assert common
    assert all(torch.equal(widened_state[name], compact_state[name]) for name in common)
    assert any(
        widened_state[name].shape != compact_state[name].shape
        for name in widened_state.keys() & compact_state.keys()
    )
