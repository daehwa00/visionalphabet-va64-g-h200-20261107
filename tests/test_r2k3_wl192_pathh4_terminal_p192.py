from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false, reportOptionalMemberAccess=false
# pyright: reportPrivateUsage=false
import pytest
import torch

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from scripts import a2d_r2k3_capacity_factory as capacity
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as control
from scripts import run_a2d_r2k3_wl192_pathh4_terminal_p192_imagenet100 as runner


def test_terminal_p192_changes_only_terminal_and_classifier_shapes() -> None:
    torch.manual_seed(501)
    reference = control._build(control.VARIANT, runtime.model_config())
    torch.manual_seed(501)
    candidate = runner._build(runner.VARIANT, runtime.model_config())
    runner._assert_model(candidate)
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 2_754_596
    assert candidate.terminal.input_modes == 128
    assert candidate.terminal.modes == 192
    assert type(candidate.terminal.pole_input_projection) is FactorizedComplexConv2dReader
    assert candidate.descriptor_dim == 2304
    assert candidate.classifier.input_dim == 768
    assert candidate.same_resolution_blocks["7"].pole_modes == 128
    assert candidate.terminal.pole_input_projection.rms_reference_modes == 192

    candidate_reader = candidate.terminal.pole_input_projection
    primary = candidate_reader.point_weight_real[:, 0]
    torch.testing.assert_close(
        primary.T @ primary,
        torch.eye(128),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert torch.count_nonzero(candidate.classifier.affine.linear.weight[:, 512:]) > 0

    allowed = ("terminal.", "classifier.")
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    for key in set(reference_state).intersection(candidate_state):
        if not key.startswith(allowed):
            torch.testing.assert_close(candidate_state[key], reference_state[key], rtol=0, atol=0)


def test_terminal_p192_declares_one_gpu1_job() -> None:
    assert runner.JOBS_BY_GPU == {1: (runner.VARIANT,)}


def test_terminal_resize_is_an_exact_rng_transparent_noop_at_p128() -> None:
    torch.manual_seed(501)
    model = control._build(control.VARIANT, runtime.model_config())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng_state = torch.random.get_rng_state().clone()

    capacity.resize_terminal_poles_(model, target_poles=128)

    torch.testing.assert_close(torch.random.get_rng_state(), rng_state, rtol=0, atol=0)
    assert set(model.state_dict()) == set(state)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_terminal_p192_does_not_preserve_the_p128_initial_function() -> None:
    torch.manual_seed(501)
    reference = control._build(control.VARIANT, runtime.model_config()).eval()
    torch.manual_seed(501)
    candidate = runner._build(runner.VARIANT, runtime.model_config()).eval()
    torch.manual_seed(503)
    inputs = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        reference_output = reference(inputs)[0]
        candidate_output = candidate(inputs)[0]
    assert not torch.allclose(candidate_output, reference_output, rtol=0, atol=1.0e-6)


@pytest.mark.parametrize("target_poles", [160, 256])
def test_terminal_resize_is_shape_generic_and_jointly_initialized(
    target_poles: int,
) -> None:
    torch.manual_seed(601)
    candidate = control._build(control.VARIANT, runtime.model_config()).eval()
    capacity.resize_terminal_poles_(candidate, target_poles=target_poles)
    assert candidate.terminal.modes == target_poles
    assert candidate.terminal.pole_input_projection.rms_reference_modes == target_poles
    assert candidate.classifier.input_dim == 4 * target_poles
    primary = candidate.terminal.pole_input_projection.point_weight_real[:, 0]
    gram = primary @ primary.T if target_poles < 128 else primary.T @ primary
    torch.testing.assert_close(
        gram,
        torch.eye(gram.shape[0]),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
