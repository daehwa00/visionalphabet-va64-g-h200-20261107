from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import torch

from lnet.pac_gated_post_fusion import NormalizedShortcutGatedTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k32_p128_front_d2262_imagenet100 as runner


def test_k32_p128_front_contract_is_exact() -> None:
    assert runner.SPEC.excitation_modes == (32, 64, 64, 128)
    assert runner.SPEC.pole_modes == (128, 128, 128, 128)
    assert runner.SPEC.depth == (2, 2, 6, 2)
    assert runner.SPEC.q4_dim == 512
    assert runner.JOBS_BY_GPU == {1: (runner.VARIANT,)}


def test_k32_p128_front_builds_with_expected_capacity() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, runtime.model_config())
    runner._assert_model(model, runner.VARIANT)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_858_372
    assert isinstance(model.stage1.augmented, NormalizedShortcutGatedTransition)
    assert isinstance(model.stage3.augmented, NormalizedShortcutGatedTransition)
    assert model.stage1.augmented.memory_projection is not None
    assert model.stage3.augmented.memory_projection is None
    assert torch.equal(
        model.stage1.augmented.memory_scale,
        torch.ones_like(model.stage1.augmented.memory_scale),
    )
    assert torch.equal(
        model.stage3.augmented.memory_scale,
        torch.ones_like(model.stage3.augmented.memory_scale),
    )
