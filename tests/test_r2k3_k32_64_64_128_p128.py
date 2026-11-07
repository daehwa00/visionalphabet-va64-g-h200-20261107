from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportPrivateUsage=false
from typing import cast

import torch

from lnet.pac_gated_post_fusion import NormalizedShortcutGatedTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k32_64_64_128_p128_d2262_imagenet100 as runner

EXPECTED_PARAMETERS = 1_858_372


def test_k32_p128_schedule_and_lane_contract() -> None:
    assert runner.SPEC.excitation_modes == (32, 64, 64, 128)
    assert runner.SPEC.pole_modes == (128, 128, 128, 128)
    assert runner.SPEC.depth == (2, 2, 6, 2)
    assert runner.SPEC.q4_dim == 512
    assert runner.JOBS_BY_GPU == {0: (runner.VARIANT,)}
    assert runner.MEMORY_SCALE_INITIAL == 1.0


def test_k32_p128_model_uses_only_width_changing_shortcuts() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, runtime.model_config())
    runner._assert_model(model, runner.VARIANT)
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    assert isinstance(model.stage1.augmented, NormalizedShortcutGatedTransition)
    assert not isinstance(model.stage2.augmented, NormalizedShortcutGatedTransition)
    assert isinstance(model.stage3.augmented, NormalizedShortcutGatedTransition)
    for stage in (model.stage1, model.stage3):
        transition = cast("NormalizedShortcutGatedTransition", stage.augmented)
        assert torch.equal(
            transition.memory_scale,
            torch.ones_like(transition.memory_scale),
        )
