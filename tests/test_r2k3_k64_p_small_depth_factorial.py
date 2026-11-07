from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import torch

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k64_p96_128_128_96_depth_factorial_imagenet100 as runner


def test_depth_factorial_changes_one_resolution_before_interaction() -> None:
    assert runner.EXTRA_BLOCKS == {
        runner.D2242: (0, 0, 2, 0),
        runner.D2222: (0, 0, 0, 0),
        runner.D2282: (0, 0, 6, 0),
        runner.D2362: (0, 1, 4, 0),
        runner.D3262: (1, 0, 4, 0),
        runner.D3362: (1, 1, 4, 0),
    }
    assert runner.JOBS_BY_GPU == {
        0: (runner.D2242, runner.D2282, runner.D3262),
        1: (runner.D2222, runner.D2362, runner.D3362),
    }


def test_every_depth_candidate_preserves_k_p_and_raw_identity_carry() -> None:
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        runner._assert_model(model, variant)
        spec = runner.SPECS[variant]
        assert spec.excitation_modes == runner.EXCITATION_MODES
        assert spec.pole_modes == runner.POLE_MODES
        for name in runner.STAGE_NAMES[:3]:
            transition = getattr(model, name).augmented
            assert type(transition) is GatedPoleExcitationS2DTransition
            assert transition.carry_projection is None


def test_depth_candidates_pair_common_state_with_d2262_reference() -> None:
    torch.manual_seed(501)
    reference = runner._build_spec(runner.REFERENCE_SPEC, runtime.model_config()).state_dict()
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        candidate = runner._build(variant, runtime.model_config()).state_dict()
        common = {
            name
            for name in reference.keys() & candidate.keys()
            if reference[name].shape == candidate[name].shape
        }
        assert common
        assert all(torch.equal(reference[name], candidate[name]) for name in common)
