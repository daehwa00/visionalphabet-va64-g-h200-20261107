from __future__ import annotations

# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from lnet.pac_complex_layers import PackedComplexLinear
from lnet.pac_gated_post_fusion import (
    GatedComplexPostFusion,
    GatedStrictComplexMagnitudePostFusion,
    GatedStrictComplexPostFusion,
)
from scripts import a2d_r2k3_runtime as runtime
from scripts import queue_r2k3_campaign as queue
from scripts import run_a2d_r2k3_k128_p256_pf2k_all_imagenet100 as p256_runner
from scripts import run_a2d_r2k3_path_collapse_overnight_imagenet100 as overnight_runner
from scripts import run_a2d_r2k3_same_resolution_all_cl256_imagenet100 as cl256_runner
from scripts import run_a2d_r2k3_same_resolution_all_cl320_imagenet100 as cl320_runner
from scripts import run_a2d_r2k3_same_resolution_all_clmaggate_imagenet100 as mag_runner
from scripts import (
    run_a2d_r2k3_same_resolution_all_clpost_directpath_imagenet100 as directpath_runner,
)
from scripts import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as runner
from scripts import run_a2d_r2k3_same_resolution_all_clpost_pathh4_imagenet100 as pathh4_runner
from scripts import (
    run_a2d_r2k3_same_resolution_all_clpost_pathswiglu_imagenet100 as pathswiglu_runner,
)
from scripts import run_a2d_r2k3_same_resolution_all_clpost_rank4_imagenet100 as rank4_runner

if TYPE_CHECKING:
    from pathlib import Path


def test_queue_rejects_corrupt_status_instead_of_restarting(tmp_path: Path) -> None:
    status = tmp_path / "queue.json"
    status.write_text("not-json")

    with pytest.raises(RuntimeError, match="invalid queue state"):
        queue._previous_state(status, runner.VARIANT)


@pytest.mark.parametrize("contents", ["not-json", "[]", "null"])
def test_queue_rejects_corrupt_completed_result(tmp_path: Path, contents: str) -> None:
    result = tmp_path / "run" / "results" / f"{runner.VARIANT}__seed501.json"
    result.parent.mkdir(parents=True)
    result.write_text(contents)

    with pytest.raises(RuntimeError, match="invalid completed-result artifact"):
        queue._is_complete(tmp_path, runner)


def test_queue_completion_honors_requested_epoch(tmp_path: Path) -> None:
    result = tmp_path / "run" / "results" / f"{runner.VARIANT}__seed501.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"history": [{"epoch": 4}]}')

    assert queue._is_complete(tmp_path, runner, epochs=4)
    assert not queue._is_complete(tmp_path, runner, epochs=100)


def test_strict_gated_postfusion_preserves_shape_and_residual_gradient() -> None:
    module = GatedStrictComplexPostFusion(4, 8)
    real = torch.randn(2, 3, 5, 4, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    output = module(real, imag)
    assert output[0].shape == real.shape
    assert output[1].shape == imag.shape
    assert type(module.value) is PackedComplexLinear
    assert type(module.out) is PackedComplexLinear
    assert not any("conjugate" in name for name, _parameter in module.named_parameters())
    (output[0].square().mean() + output[1].square().mean()).backward()
    assert real.grad is not None
    assert imag.grad is not None


def test_strict_projection_removes_exactly_four_c_h_weights() -> None:
    modes = 8
    hidden = 12
    widely = GatedComplexPostFusion(modes, hidden)
    strict = GatedStrictComplexPostFusion(modes, hidden)
    widely_parameters = sum(parameter.numel() for parameter in widely.parameters())
    strict_parameters = sum(parameter.numel() for parameter in strict.parameters())
    assert widely_parameters - strict_parameters == 4 * modes * hidden


def test_all_resolution_candidate_changes_only_seven_postfusions() -> None:
    model = runner._build(runner.VARIANT, runtime.model_config())
    owners = runner._postfusions(model)
    assert len(owners) == 7
    assert all(type(owner.post_fusion) is GatedStrictComplexPostFusion for owner in owners)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_084_388


def test_directpath_candidate_changes_only_the_seven_path_collapses() -> None:
    torch.manual_seed(501)
    baseline = runner._build(runner.VARIANT, runtime.model_config())
    torch.manual_seed(501)
    candidate = directpath_runner._build(
        directpath_runner.VARIANT,
        runtime.model_config(),
    )
    directpath_runner._assert_model(candidate)
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 1_941_028

    path_prefixes = (
        "stage1.quadrant_path_mode_combiner.",
        "stage2.quadrant_path_mode_combiner.",
        "stage3.quadrant_path_mode_combiner.",
        "same_resolution_blocks.56.path_collapse.",
        "same_resolution_blocks.28.path_collapse.",
        "same_resolution_blocks.14.path_collapse.",
        "same_resolution_blocks.7.path_collapse.",
    )
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    shared_keys = set(baseline_state) & set(candidate_state)
    for key in shared_keys:
        if not key.startswith(path_prefixes):
            torch.testing.assert_close(candidate_state[key], baseline_state[key], rtol=0, atol=0)
    assert set(baseline_state) - shared_keys
    assert set(candidate_state) - shared_keys


def test_pathh4_candidate_halves_only_the_seven_path_hidden_widths() -> None:
    torch.manual_seed(501)
    baseline = runner._build(runner.VARIANT, runtime.model_config())
    torch.manual_seed(501)
    candidate = pathh4_runner._build(pathh4_runner.VARIANT, runtime.model_config())
    pathh4_runner._assert_model(candidate)
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 2_005_540

    path_prefixes = (
        "stage1.quadrant_path_mode_combiner.",
        "stage2.quadrant_path_mode_combiner.",
        "stage3.quadrant_path_mode_combiner.",
        "same_resolution_blocks.56.path_collapse.",
        "same_resolution_blocks.28.path_collapse.",
        "same_resolution_blocks.14.path_collapse.",
        "same_resolution_blocks.7.path_collapse.",
    )
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    for key in set(baseline_state) & set(candidate_state):
        if not key.startswith(path_prefixes):
            torch.testing.assert_close(candidate_state[key], baseline_state[key], rtol=0, atol=0)


def test_pathswiglu_candidate_changes_only_the_seven_path_collapses() -> None:
    torch.manual_seed(501)
    baseline = runner._build(runner.VARIANT, runtime.model_config())
    torch.manual_seed(501)
    candidate = pathswiglu_runner._build(
        pathswiglu_runner.VARIANT,
        runtime.model_config(),
    )
    pathswiglu_runner._assert_model(candidate)
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 2_025_252

    path_prefixes = (
        "stage1.quadrant_path_mode_combiner.",
        "stage2.quadrant_path_mode_combiner.",
        "stage3.quadrant_path_mode_combiner.",
        "same_resolution_blocks.56.path_collapse.",
        "same_resolution_blocks.28.path_collapse.",
        "same_resolution_blocks.14.path_collapse.",
        "same_resolution_blocks.7.path_collapse.",
    )
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    for key in set(baseline_state) & set(candidate_state):
        if not key.startswith(path_prefixes):
            torch.testing.assert_close(candidate_state[key], baseline_state[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("variant", "parameters"),
    [
        (overnight_runner.PF15K_PATH_H4, 2_005_540),
        (overnight_runner.PF15K_PATH_PG_H4, 2_070_080),
        (overnight_runner.PF15K_PATH_MAG_H4, 2_005_568),
        (overnight_runner.PF2K_PATH_H4, 2_349_604),
        (overnight_runner.PF2K_PATH_PG_H4, 2_414_144),
        (overnight_runner.PF15K_DIRECT_MAG, 1_941_056),
    ],
)
def test_overnight_path_candidate_keeps_the_control_surface(
    variant: str,
    parameters: int,
) -> None:
    model = overnight_runner._build(variant, runtime.model_config())
    overnight_runner._assert_model(model, variant)
    assert len(overnight_runner._path_collapses(model)) == 7
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters


def test_all_resolution_cl320_exactly_matches_widely_linear_capacity() -> None:
    model = cl320_runner._build(cl320_runner.VARIANT, runtime.model_config())
    owners = cl320_runner._postfusions(model)
    assert len(owners) == 7
    for owner in owners:
        postfusion = owner.post_fusion
        assert type(postfusion) is GatedStrictComplexPostFusion
        assert type(postfusion.value) is PackedComplexLinear
        assert type(postfusion.out) is PackedComplexLinear
        assert postfusion.modes == 128
        assert postfusion.hidden_modes == 320
        assert postfusion.gate.in_features == 256
        assert postfusion.gate.out_features == 320
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_772_516


def test_all_resolution_cl256_is_the_exact_pf2k_intermediate() -> None:
    model = cl256_runner._build(cl256_runner.VARIANT, runtime.model_config())
    owners = cl256_runner._postfusions(model)
    assert len(owners) == 7
    assert all(owner.post_fusion.hidden_modes == 256 for owner in owners)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_428_452


def test_k128_p256_candidate_changes_only_pole_side_widths() -> None:
    model = p256_runner._build(p256_runner.VARIANT, runtime.model_config())
    p256_runner._assert_model(model)
    assert model.descriptor_dim == 4096
    assert model.classifier.input_dim == 1024
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_663_140


def test_rank4_candidate_changes_all_eight_readers_and_only_their_rank() -> None:
    model = rank4_runner._build(rank4_runner.VARIANT, runtime.model_config())
    readers = rank4_runner._readers(model)
    assert len(readers) == 8
    assert all(reader.rank == 4 for reader in readers)
    assert all(reader.input_modes == reader.output_modes == 128 for reader in readers)
    assert all(reader.kernel_size == 3 for reader in readers)
    assert all(reader.normalize_input for reader in readers[:3])
    assert not readers[3].normalize_input
    assert readers[3].match_input_rms
    assert all(reader.normalize_input for reader in readers[4:])
    assert all(not reader.match_input_rms for reader in (*readers[:3], *readers[4:]))
    assert all(
        type(owner.post_fusion) is GatedStrictComplexPostFusion
        for owner in rank4_runner._postfusions(model)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_645_540


def test_magnitude_gate_is_mean_one_and_phase_invariant() -> None:
    module = GatedStrictComplexMagnitudePostFusion(4, 8)
    real = torch.randn(2, 3, 5, 4)
    imag = torch.randn_like(real)
    unit = module.norm(real, imag)
    initial_gate = module._gate_values(*unit)
    torch.testing.assert_close(initial_gate, torch.ones_like(initial_gate))

    with torch.no_grad():
        nn.init.normal_(module.gate.weight)
    angle = torch.tensor(0.73)
    cosine, sine = angle.cos(), angle.sin()
    rotated = cosine * real - sine * imag, sine * real + cosine * imag
    rotated_unit = module.norm(*rotated)
    gate = module._gate_values(*unit)
    rotated_gate = module._gate_values(*rotated_unit)
    torch.testing.assert_close(gate.mean(-1), torch.ones_like(gate.mean(-1)))
    torch.testing.assert_close(rotated_gate, gate, rtol=1.0e-5, atol=1.0e-5)

    output = module(real, imag)
    rotated_output = module(*rotated)
    expected = (
        cosine * output[0] - sine * output[1],
        sine * output[0] + cosine * output[1],
    )
    torch.testing.assert_close(rotated_output[0], expected[0], rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(rotated_output[1], expected[1], rtol=2.0e-5, atol=2.0e-5)


def test_all_resolution_magnitude_gate_candidate_is_minimal() -> None:
    model = mag_runner._build(mag_runner.VARIANT, runtime.model_config())
    owners = mag_runner._postfusions(model)
    assert len(owners) == 7
    assert all(type(owner.post_fusion) is GatedStrictComplexMagnitudePostFusion for owner in owners)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_912_356
