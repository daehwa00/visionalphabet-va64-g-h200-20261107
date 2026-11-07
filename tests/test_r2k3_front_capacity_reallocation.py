from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
from typing import cast

import torch

from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    NormalizedShortcutGatedTransition,
)
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_front_capacity_reallocation_d2262_imagenet100 as runner
from scripts import run_a2d_r2k3_normalized_shortcut_k160_d2262_imagenet100 as normalized_runner

EXPECTED_SCHEDULES = {
    runner.K128_P128: ((128, 128, 160, 160), (128, 160, 160, 160)),
    runner.K96_P128: ((96, 128, 160, 160), (128, 160, 160, 160)),
    runner.K128_P96: ((128, 128, 160, 160), (96, 160, 160, 160)),
    runner.K96_P96: ((96, 128, 160, 160), (96, 160, 160, 160)),
    runner.MID_POLE_REALLOCATION: (
        (96, 128, 160, 160),
        (96, 192, 192, 160),
    ),
    runner.MID_EXCITATION_REALLOCATION: (
        (96, 128, 192, 160),
        (96, 160, 192, 160),
    ),
}
EXPECTED_PARAMETERS = {
    runner.K128_P128: 5_574_564,
    runner.K96_P128: 5_378_756,
    runner.K128_P96: 5_582_628,
    runner.K96_P96: 5_345_860,
    runner.MID_POLE_REALLOCATION: 5_973_060,
    runner.MID_EXCITATION_REALLOCATION: 6_848_516,
}


def test_front_capacity_factorial_and_lanes_are_exact() -> None:
    assert len(runner.VARIANTS) == 6
    assert set(runner.VARIANTS) == set(EXPECTED_SCHEDULES)
    assert runner.JOBS_BY_GPU == {
        0: (
            runner.K128_P128,
            runner.K128_P96,
            runner.MID_POLE_REALLOCATION,
        ),
        1: (
            runner.K96_P128,
            runner.K96_P96,
            runner.MID_EXCITATION_REALLOCATION,
        ),
    }
    for variant, (excitation, poles) in EXPECTED_SCHEDULES.items():
        spec = runner.SPECS[variant]
        assert spec.excitation_modes == excitation
        assert spec.pole_modes == poles
        assert spec.depth == (2, 2, 6, 2)
        assert spec.q4_dim == 640
        assert spec.descriptor_dim == 4 * sum(poles)


def test_all_front_capacity_models_use_only_required_projections() -> None:
    config = runtime.model_config()
    for variant, (excitation, _) in EXPECTED_SCHEDULES.items():
        torch.manual_seed(501)
        model = runner._build(variant, config)
        runner._assert_model(model, variant)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == (EXPECTED_PARAMETERS[variant])
        )
        transitions = tuple(
            cast(
                "GatedPoleExcitationS2DTransition",
                getattr(model, name).augmented,
            )
            for name in runner.STAGE_NAMES[:3]
        )
        for index, transition in enumerate(transitions):
            changes_width = excitation[index] != excitation[index + 1]
            assert isinstance(transition, NormalizedShortcutGatedTransition) == changes_width
            if changes_width:
                normalized_transition = cast(
                    "NormalizedShortcutGatedTransition",
                    transition,
                )
                assert torch.equal(
                    normalized_transition.memory_scale,
                    torch.full_like(
                        normalized_transition.memory_scale,
                        runner.MEMORY_SCALE_INITIAL,
                    ),
                )
        for owner in normalized_runner._memory_owners(model):
            if isinstance(owner, GatedPoleExcitationS2DTransition):
                input_modes = owner.input_modes
                output_modes = owner.output_modes
            else:
                input_modes = owner.pole_modes
                output_modes = owner.modes
            assert (owner.memory_projection is None) == (input_modes == output_modes)


def test_representative_front_capacity_build_is_seed_deterministic() -> None:
    config = runtime.model_config()
    torch.manual_seed(501)
    first = runner._build(runner.MID_EXCITATION_REALLOCATION, config)
    torch.manual_seed(501)
    second = runner._build(runner.MID_EXCITATION_REALLOCATION, config)
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for key, value in first_state.items():
        torch.testing.assert_close(value, second_state[key], rtol=0.0, atol=0.0)
