from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false
# pyright: reportPrivateUsage=false
import json
from typing import cast

import torch

from lnet.pac_complex_layers import ComplexLinear
from lnet.pac_complex_projection_adapter import PreNormScaledComplexProjection
from scripts import a2d_r2k3_runtime as runtime
from scripts import r2k3_runner_registry as registry
from scripts import run_a2d_r2k3_monotone_k_d2262_imagenet100 as runner

EXPECTED_PARAMETERS = 6_106_692


def test_monotone_k_runner_owns_one_gpu0_job() -> None:
    assert runner.VARIANTS == (runner.VARIANT,)
    assert runner.JOBS_BY_GPU == {0: (runner.VARIANT,)}
    resolved = registry.resolve_runner("a2d_r2k3_monotone_k_d2262")
    assert resolved.__file__ == runner.__file__


def test_monotone_k_model_keeps_k160_through_terminal_without_gain_adapters() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, runtime.model_config())

    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    assert runner.SPEC.excitation_modes == (128, 128, 160, 160)
    assert runner.SPEC.pole_modes == (160,) * 4
    assert runner.SPEC.depth == (2, 2, 6, 2)
    assert tuple(getattr(model, name).input_modes for name in runner.STAGE_NAMES) == (
        128,
        128,
        160,
        160,
    )
    assert not any(isinstance(module, PreNormScaledComplexProjection) for module in model.modules())

    stage2 = model.stage2.augmented
    stage3 = model.stage3.augmented
    assert isinstance(stage2.carry_projection, ComplexLinear)
    assert stage2.carry_projection.input_modes == 128
    assert stage2.carry_projection.output_modes == 160
    assert stage3.carry_projection is None
    assert isinstance(stage3.memory_projection, ComplexLinear)
    assert stage3.memory_projection.input_modes == stage3.memory_projection.output_modes == 160

    extras = cast("torch.nn.ModuleDict", model.extra_same_resolution_blocks)
    assert set(extras) == {"14"}
    assert len(cast("torch.nn.ModuleList", extras["14"])) == 4


def test_monotone_k_variant_config_records_the_single_width_change() -> None:
    payload = runner._variant_config()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["backbone"]["excitation_schedule"] == [128, 128, 160, 160]
    assert payload["backbone"]["pole_schedule"] == [160, 160, 160, 160]
    assert payload["experiment"]["projection_contract"].endswith(
        "no pre-normalization wrapper and no learned branch gamma"
    )
