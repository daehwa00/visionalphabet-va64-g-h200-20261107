from __future__ import annotations

# pyright: reportPrivateUsage=false
import pytest
import torch

from lnet.pac_gated_post_fusion import GatedComplexPostFusion
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_same_resolution_factorial_imagenet100 as base
from scripts import run_a2d_r2k3_wl_budget_match_imagenet100 as runner


@pytest.mark.parametrize(
    ("variant", "hidden", "parameters"),
    [
        (runner.WL_H115, 115, 2_082_596),
        (runner.WL_H154, 154, 2_432_036),
    ],
)
def test_wl_budget_candidate_changes_only_seven_postfusions(
    variant: str,
    hidden: int,
    parameters: int,
) -> None:
    torch.manual_seed(501)
    reference = base._build(runner.BASE_VARIANT, runtime.model_config())
    torch.manual_seed(501)
    candidate = runner._build(variant, runtime.model_config())
    runner._assert_model(candidate, variant)
    assert sum(parameter.numel() for parameter in candidate.parameters()) == parameters
    assert all(
        type(owner.post_fusion) is GatedComplexPostFusion
        and owner.post_fusion.hidden_modes == hidden
        for owner in runner._postfusions(candidate)
    )

    prefixes = (
        "stage1.augmented.post_fusion.",
        "stage2.augmented.post_fusion.",
        "stage3.augmented.post_fusion.",
        "same_resolution_blocks.56.post_fusion.",
        "same_resolution_blocks.28.post_fusion.",
        "same_resolution_blocks.14.post_fusion.",
        "same_resolution_blocks.7.post_fusion.",
    )
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    for key in set(reference_state).intersection(candidate_state):
        if not key.startswith(prefixes):
            torch.testing.assert_close(candidate_state[key], reference_state[key], rtol=0, atol=0)


def test_wl_budget_queue_covers_each_candidate_once() -> None:
    jobs = runner.JOBS_BY_GPU[0]
    assert jobs == runner.VARIANTS
    assert len(jobs) == len(set(jobs))

