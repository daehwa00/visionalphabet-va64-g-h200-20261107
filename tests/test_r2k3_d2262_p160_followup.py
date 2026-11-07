from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
from typing import cast

import torch

from scripts import a2d_r2k3_runtime as runtime
from scripts import r2k3_runner_registry as registry
from scripts import run_a2d_r2k3_d2262_p160_followup_imagenet100 as runner

EXPECTED_PARAMETERS = 4_713_444


def test_followup_owns_one_gpu1_job() -> None:
    assert runner.VARIANTS == (runner.VARIANT,)
    assert runner.JOBS_BY_GPU == {1: (runner.VARIANT,)}


def test_followup_resolves_through_shared_registry() -> None:
    resolved = registry.resolve_runner("a2d_r2k3_d2262_p160_followup")
    assert resolved.__file__ == runner.__file__
    assert resolved.JOBS_BY_GPU == runner.JOBS_BY_GPU


def test_followup_matches_exact_k_p_depth_and_parameters() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, runtime.model_config())

    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    assert runner.SPEC.excitation_modes == (128,) * 4
    assert runner.SPEC.pole_modes == (160,) * 4
    assert runner.SPEC.depth == (2, 2, 6, 2)
    for name in runner.STAGE_NAMES:
        stage = getattr(model, name)
        assert stage.input_modes == 128
        assert stage.modes == 160
    extras = cast("torch.nn.ModuleDict", model.extra_same_resolution_blocks)
    assert set(extras) == {"14"}
    assert len(cast("torch.nn.ModuleList", extras["14"])) == 4


def test_followup_variant_config_is_json_exact() -> None:
    payload = runner._variant_config()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["experiment"]["depth_by_resolution"] == {
        "56": 2,
        "28": 2,
        "14": 6,
        "7": 2,
    }
    assert payload["backbone"]["pole_schedule"] == [160, 160, 160, 160]
