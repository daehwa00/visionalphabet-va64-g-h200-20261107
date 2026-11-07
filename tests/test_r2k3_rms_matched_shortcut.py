from __future__ import annotations

import torch

from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    RMSMatchedShortcutGatedTransition,
    rms_matched_shortcut_transition,
)
from scripts import run_a2d_r2k3_k32_64_96_128_p128_rms_shortcut_d2262_imagenet100 as runner


def _complex_rms(field: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt((field[0].square() + field[1].square()).mean(dim=-1))


def test_rms_matched_shortcut_preserves_token_amplitude() -> None:
    module = RMSMatchedShortcutGatedTransition(7, 5, 9, post_hidden=12).double()
    source = torch.randn(2, 3, 5, dtype=torch.double), torch.randn(
        2, 3, 5, dtype=torch.double
    )
    projected = module._project(module.carry_projection, source)
    matched = module._match_carry_rms(source, projected)

    torch.testing.assert_close(_complex_rms(matched), _complex_rms(source))


def test_rms_matched_replacement_preserves_projection_and_postfusion_state() -> None:
    source = GatedPoleExcitationS2DTransition(7, 5, 9, post_hidden=12).double()
    replacement = rms_matched_shortcut_transition(source)

    assert type(replacement) is RMSMatchedShortcutGatedTransition
    assert replacement.training is source.training
    for name in ("carry_weight",):
        torch.testing.assert_close(getattr(replacement, name), getattr(source, name))
    for owner in ("memory_projection", "carry_projection", "post_fusion"):
        target_state = getattr(replacement, owner).state_dict()
        source_state = getattr(source, owner).state_dict()
        assert target_state.keys() == source_state.keys()
        for name in target_state:
            torch.testing.assert_close(target_state[name], source_state[name])


def test_k96_runner_uses_only_rms_matched_width_shortcuts() -> None:
    model = runner._build(runner.VARIANT, runner.runtime.model_config())
    runner._assert_model(model, runner.VARIANT)

    for name in runner.STAGE_NAMES[:3]:
        assert type(getattr(model, name).augmented) is RMSMatchedShortcutGatedTransition
