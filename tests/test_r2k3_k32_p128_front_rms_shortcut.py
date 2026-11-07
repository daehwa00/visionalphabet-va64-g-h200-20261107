from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import torch

from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    RMSMatchedShortcutGatedTransition,
)
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k32_p128_front_rms_shortcut_d2262_imagenet100 as runner


def test_k32_p128_rms_shortcut_contract_matches_completed_control() -> None:
    assert runner.SPEC.excitation_modes == (32, 64, 64, 128)
    assert runner.SPEC.pole_modes == (128, 128, 128, 128)
    assert runner.SPEC.depth == (2, 2, 6, 2)
    assert runner.JOBS_BY_GPU == {1: (runner.VARIANT,)}


def test_k32_p128_replaces_only_width_changing_shortcuts() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, runtime.model_config())
    runner._assert_model(model, runner.VARIANT)

    assert type(model.stage1.augmented) is RMSMatchedShortcutGatedTransition
    assert type(model.stage2.augmented) is GatedPoleExcitationS2DTransition
    assert type(model.stage3.augmented) is RMSMatchedShortcutGatedTransition
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_857_796
