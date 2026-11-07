from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
# pyright: reportOptionalMemberAccess=false, reportPrivateUsage=false
import pytest
import torch

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as control
from scripts import run_a2d_r2k3_terminal_path_followup_imagenet100 as runner

EXPECTED_PARAMETERS = {
    runner.P128_RANDOM_SQUARE: 2_693_668,
    runner.P96_PATH_H6: 2_702_628,
    runner.P96_PATH_H8: 2_742_052,
}


def test_followup_queue_partitions_all_variants_once() -> None:
    queued = tuple(variant for lane in runner.JOBS_BY_GPU.values() for variant in lane)
    assert set(runner.JOBS_BY_GPU) == {0, 1}
    assert len(queued) == len(set(queued))
    assert set(queued) == set(runner.VARIANTS)


def test_square_reader_can_select_identity_or_random_orthogonal_basis() -> None:
    identity_reader = FactorizedComplexConv2dReader(16, 16, rank=2)
    random_reader = FactorizedComplexConv2dReader(16, 16, rank=2)
    torch.manual_seed(701)
    identity_reader.initialize_semi_orthogonal_()
    torch.manual_seed(701)
    random_reader.initialize_semi_orthogonal_(identity_when_square=False)

    identity = torch.eye(16)
    torch.testing.assert_close(identity_reader.point_weight_real[:, 0], identity)
    primary = random_reader.point_weight_real[:, 0]
    assert not torch.equal(primary, identity)
    torch.testing.assert_close(primary.T @ primary, identity, rtol=1.0e-5, atol=1.0e-5)


def test_random_square_control_changes_only_terminal_reader_weights() -> None:
    torch.manual_seed(501)
    reference = control._build(control.VARIANT, runtime.model_config())
    torch.manual_seed(501)
    candidate = runner._build(runner.P128_RANDOM_SQUARE, runtime.model_config())
    changed = {
        name
        for name, value in reference.state_dict().items()
        if not torch.equal(value, candidate.state_dict()[name])
    }
    assert changed == {
        "terminal.pole_input_projection.point_weight_real",
        "terminal.pole_input_projection.point_weight_imag",
        "terminal.pole_input_projection.spatial_weight_real",
        "terminal.pole_input_projection.spatial_weight_imag",
    }


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_followup_variants_match_declared_contract(variant: str) -> None:
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())
    runner._assert_model(model, variant)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == EXPECTED_PARAMETERS[variant]


@pytest.mark.parametrize(
    ("variant", "hidden"),
    [(runner.P96_PATH_H6, 6), (runner.P96_PATH_H8, 8)],
)
def test_p96_path_followups_change_all_seven_hidden_widths(
    variant: str,
    hidden: int,
) -> None:
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())
    collapses = control._path_collapses(model)
    assert len(collapses) == 7
    assert all(collapse.path_input.output_paths == hidden for collapse in collapses)
    assert model.terminal.modes == 96
    assert model.classifier.input_dim == 384
