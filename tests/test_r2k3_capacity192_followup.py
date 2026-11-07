from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import json

import pytest
import torch

from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_capacity192_followup_imagenet100 as runner

EXPECTED_PARAMETERS = {
    runner.K192_P128: 4_445_092,
    runner.K128_P192: 3_386_276,
    runner.K192_P192: 4_580_644,
}


def test_capacity192_queue_partitions_every_variant_once() -> None:
    queued = tuple(variant for lane in runner.JOBS_BY_GPU.values() for variant in lane)
    assert set(runner.JOBS_BY_GPU) == {0, 1}
    assert len(queued) == len(set(queued))
    assert set(queued) == set(runner.VARIANTS)


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_capacity192_variants_match_declared_contract(variant: str) -> None:
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())
    runner._assert_model(model, variant)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == EXPECTED_PARAMETERS[variant]
    config = runner._variant_config(variant)
    assert json.loads(json.dumps(config)) == config


def test_capacity192_factorial_changes_only_declared_k_and_p_axes() -> None:
    assert runner.SPECS[runner.K192_P128].excitation_modes == (192,) * 4
    assert runner.SPECS[runner.K192_P128].pole_modes == (128,) * 4
    assert runner.SPECS[runner.K128_P192].excitation_modes == (128,) * 4
    assert runner.SPECS[runner.K128_P192].pole_modes == (192,) * 4
    assert runner.SPECS[runner.K192_P192].excitation_modes == (192,) * 4
    assert runner.SPECS[runner.K192_P192].pole_modes == (192,) * 4
