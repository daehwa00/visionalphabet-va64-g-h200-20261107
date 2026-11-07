from __future__ import annotations

# pyright: reportPrivateUsage=false
import torch

from lnet.pac_gated_post_fusion import GatedComplexPostFusion
from lnet.pac_phase_gated_transition import PathOnlyCollapse
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as runner
from scripts import run_a2d_r2k3_same_resolution_factorial_imagenet100 as base


def test_wl192_pathh4_changes_only_the_seven_path_collapses() -> None:
    torch.manual_seed(501)
    reference = base._build(runner.BASE_VARIANT, runtime.model_config())
    torch.manual_seed(501)
    candidate = runner._build(runner.VARIANT, runtime.model_config())
    runner._assert_model(candidate)
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 2_693_668
    assert all(
        type(collapse) is PathOnlyCollapse
        and collapse.path_input.output_paths == 4
        and collapse.path_output.input_paths == 4
        for collapse in runner._path_collapses(candidate)
    )
    assert all(
        type(owner.post_fusion) is GatedComplexPostFusion
        and owner.post_fusion.hidden_modes == 192
        for owner in runner._postfusions(candidate)
    )

    prefixes = (
        "stage1.quadrant_path_mode_combiner.",
        "stage2.quadrant_path_mode_combiner.",
        "stage3.quadrant_path_mode_combiner.",
        "same_resolution_blocks.56.path_collapse.",
        "same_resolution_blocks.28.path_collapse.",
        "same_resolution_blocks.14.path_collapse.",
        "same_resolution_blocks.7.path_collapse.",
    )
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    for key in set(reference_state).intersection(candidate_state):
        if not key.startswith(prefixes):
            torch.testing.assert_close(candidate_state[key], reference_state[key], rtol=0, atol=0)


def test_wl192_pathh4_declares_one_gpu0_job() -> None:
    assert runner.JOBS_BY_GPU == {0: (runner.VARIANT,)}

