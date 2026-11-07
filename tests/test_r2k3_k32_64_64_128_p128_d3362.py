from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
from typing import cast

import torch
from torch import nn

from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k32_64_64_128_p128_d3362_imagenet100 as runner

EXPECTED_PARAMETERS = 2_042_372


def test_k32_p128_d3362_schedule_and_depth_contract() -> None:
    assert runner.SPEC.excitation_modes == (32, 64, 64, 128)
    assert runner.SPEC.pole_modes == (128, 128, 128, 128)
    assert runner.SPEC.extra_blocks == (1, 1, 4, 0)
    assert runner.SPEC.depth == (3, 3, 6, 2)
    assert runner.JOBS_BY_GPU == {0: (runner.VARIANT,)}


def test_k32_p128_d3362_builds_generic_repeated_blocks() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, runtime.model_config())
    runner._assert_model(model, runner.VARIANT)
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    groups = {
        name: cast("nn.ModuleList", blocks)
        for name, blocks in extras.items()
    }
    assert {name: len(blocks) for name, blocks in groups.items()} == {
        "56": 1,
        "28": 1,
        "14": 4,
    }
    expected_hidden = {"56": 48, "28": 96, "14": 96}
    for resolution, blocks in groups.items():
        for block in blocks:
            assert block.post_fusion.hidden_modes == expected_hidden[resolution]
