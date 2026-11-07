from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
from typing import cast

import torch
from torch import nn

from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    NormalizedShortcutGatedTransition,
    normalized_shortcut_transition,
)
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_normalized_shortcut_k160_d2262_imagenet100 as runner

EXPECTED_PARAMETERS = 5_697_572


def test_normalized_shortcut_transition_anchors_the_projected_carry() -> None:
    torch.manual_seed(17)
    module = NormalizedShortcutGatedTransition(
        8,
        6,
        8,
        post_hidden=12,
        memory_scale_initial=0.0,
    ).double()
    memory_real = torch.randn(2, 3, 8, dtype=torch.double)
    memory_imag = torch.randn_like(memory_real)
    carry_real = torch.randn(2, 3, 24, dtype=torch.double)
    carry_imag = torch.randn_like(carry_real)
    carry = module._project(
        module.carry_projection,
        module._carry(carry_real, carry_imag),
    )
    shortcut = module.shortcut_norm(*carry)
    expected = module.post_fusion(*shortcut)
    actual = module(memory_real, memory_imag, carry_real, carry_imag)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_normalized_shortcut_runner_changes_only_the_stage_boundary_contract() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, runtime.model_config())
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    assert runner.SPEC.excitation_modes == (128, 128, 160, 160)
    assert runner.SPEC.pole_modes == (160,) * 4
    transition = model.stage2.augmented
    assert type(transition) is NormalizedShortcutGatedTransition
    assert transition.memory_projection is None
    assert transition.carry_projection is not None
    assert transition.shortcut_norm.modes == 160
    assert transition.memory_norm.modes == 160
    assert torch.equal(
        transition.memory_scale,
        torch.full_like(transition.memory_scale, runner.MEMORY_SCALE_INITIAL),
    )
    assert model.stage3.augmented.memory_projection is None
    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    assert blocks["14"].memory_projection is None
    assert blocks["7"].memory_projection is None
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    extra14 = cast("nn.ModuleList", extras["14"])
    assert all(block.memory_projection is None for block in extra14)


def test_full_memory_residual_differs_only_by_initial_scale() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.FULL_MEMORY_VARIANT, runtime.model_config())
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    transition = model.stage2.augmented
    assert type(transition) is NormalizedShortcutGatedTransition
    assert torch.equal(
        transition.memory_scale,
        torch.ones_like(transition.memory_scale),
    )


def test_normalized_shortcut_preserves_rectangular_projection_initialization() -> None:
    torch.manual_seed(23)
    source = GatedPoleExcitationS2DTransition(
        6,
        4,
        8,
        post_hidden=12,
    ).double()
    replacement = normalized_shortcut_transition(
        source,
        memory_scale_initial=0.25,
    )
    assert source.memory_projection is not None
    assert replacement.memory_projection is not None
    for key, value in source.memory_projection.state_dict().items():
        torch.testing.assert_close(
            replacement.memory_projection.state_dict()[key],
            value,
            rtol=0.0,
            atol=0.0,
        )
    assert source.carry_projection is not None
    assert replacement.carry_projection is not None
    for key, value in source.carry_projection.state_dict().items():
        torch.testing.assert_close(
            replacement.carry_projection.state_dict()[key],
            value,
            rtol=0.0,
            atol=0.0,
        )
