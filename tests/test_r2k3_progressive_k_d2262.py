from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
# pyright: reportUnknownLambdaType=false
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import run_alphabet2d_imagenet100_nano as harness
import run_complex_scan_zero_init_imagenet100 as zero_init
import torch

from lnet.pac_complex_layers import ComplexLinear
from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import r2k3_runner_registry as registry
from scripts import run_a2d_r2k3_progressive_k_d2262_imagenet100 as runner

if TYPE_CHECKING:
    from torch import nn

    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


EXPECTED_PARAMETERS = {
    runner.K14_160: 5_809_636,
    runner.K14_192: 7_024_612,
    runner.K28_14_160: 6_147_812,
    runner.K56_96: 5_570_436,
}
EXPECTED_K = {
    runner.K14_160: (128, 128, 160, 128),
    runner.K14_192: (128, 128, 192, 128),
    runner.K28_14_160: (128, 160, 160, 128),
    runner.K56_96: (96, 128, 160, 128),
}
EXPECTED_H = {
    variant: tuple(round(1.5 * modes) for modes in modes_by_stage)
    for variant, modes_by_stage in EXPECTED_K.items()
}


def _assert_identity_projection(projection: ComplexLinear, modes: int) -> None:
    torch.testing.assert_close(
        projection.weight_real,
        torch.eye(
            modes,
            device=projection.weight_real.device,
            dtype=projection.weight_real.dtype,
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        projection.weight_imag,
        torch.zeros_like(projection.weight_imag),
        rtol=0.0,
        atol=0.0,
    )


@pytest.fixture
def contract_args(monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    monkeypatch.setattr(
        harness,
        "_dataset_digest",
        lambda _root: (zero_init.IMAGENET100_MANIFEST_SHA256, 130_000, 5_000),
    )
    return argparse.Namespace(
        root=Path("/nonexistent/run"),
        data_root=Path("/nonexistent/data"),
        epochs=100,
        batch_size=128,
        gradient_accumulation_steps=1,
        precision="bfloat16",
        workers=8,
        run_seeds=[501],
        variants=None,
        initialize_only=False,
    )


def test_progressive_k_queue_is_disjoint_complete_and_ordered() -> None:
    assert runner.JOBS_BY_GPU == {
        0: (runner.K14_160, runner.K28_14_160),
        1: (runner.K14_192, runner.K56_96),
    }
    queued = tuple(variant for lane in runner.JOBS_BY_GPU.values() for variant in lane)
    assert len(queued) == len(set(queued)) == 4
    assert set(queued) == set(runner.VARIANTS)


def test_progressive_k_runner_resolves_through_shared_registry() -> None:
    resolved = registry.resolve_runner("a2d_r2k3_progressive_k_d2262")
    assert resolved.__file__ == runner.__file__
    assert resolved.JOBS_BY_GPU == runner.JOBS_BY_GPU


def test_materialized_square_memory_projection_preserves_initial_function() -> None:
    torch.manual_seed(31)
    transition = GatedPoleExcitationS2DTransition(8, 6, 8, post_hidden=12).double()
    memory = (torch.randn(2, 3, 8, dtype=torch.double),) * 2
    carry = (torch.randn(2, 3, 24, dtype=torch.double),) * 2
    with torch.no_grad():
        expected = transition(*memory, *carry)

    rng_state = torch.random.get_rng_state()
    runner._materialize_square_memory_projection(transition)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    projection = cast("ComplexLinear", transition.memory_projection)
    _assert_identity_projection(projection, 8)
    actual = transition(*memory, *carry)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    (actual[0].square().mean() + actual[1].square().mean()).backward()
    assert projection.weight_real.grad is not None
    assert projection.weight_imag.grad is not None


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_progressive_k_variants_match_exact_stage_contract(variant: str) -> None:
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())
    spec = runner.SPECS[variant]
    hidden = EXPECTED_H[variant]

    assert spec.excitation_modes == EXPECTED_K[variant]
    assert spec.pole_modes == (160,) * 4
    assert spec.depth == (2, 2, 6, 2)
    assert runner._capacity_spec(variant).post_hidden_modes == hidden
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS[
        variant
    ]

    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    for index, name in enumerate(runner.STAGE_NAMES):
        stage = getattr(model, name)
        block = cast(
            "SameResolutionPoleScanBlock",
            blocks[str(runner.RESOLUTIONS[index])],
        )
        assert stage.input_modes == spec.excitation_modes[index]
        assert stage.modes == 160
        assert block.modes == spec.excitation_modes[index]
        assert block.pole_modes == 160
        assert block.post_fusion.hidden_modes == hidden[index]
        assert type(block.memory_projection) is ComplexLinear
        if block.pole_modes == block.modes:
            _assert_identity_projection(block.memory_projection, block.modes)
        if name != "terminal":
            assert type(stage.augmented.memory_projection) is ComplexLinear
            assert stage.augmented.output_modes == spec.excitation_modes[index + 1]
            assert stage.augmented.post_hidden == hidden[index + 1]
            if stage.augmented.input_modes == stage.augmented.output_modes:
                _assert_identity_projection(
                    stage.augmented.memory_projection,
                    stage.augmented.input_modes,
                )

    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    assert set(extras) == {"14"}
    extra_blocks = cast("nn.ModuleList", extras["14"])
    assert len(extra_blocks) == 4
    assert all(block.modes == spec.excitation_modes[2] for block in extra_blocks)
    assert all(block.post_fusion.hidden_modes == hidden[2] for block in extra_blocks)
    assert all(type(block.memory_projection) is ComplexLinear for block in extra_blocks)
    for item in extra_blocks:
        block = cast("SameResolutionPoleScanBlock", item)
        if block.pole_modes == block.modes:
            _assert_identity_projection(
                cast("ComplexLinear", block.memory_projection),
                block.modes,
            )


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_progressive_k_variant_config_is_json_exact(variant: str) -> None:
    payload = runner._variant_config(variant)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["backbone"]["excitation_schedule"] == list(EXPECTED_K[variant])
    assert payload["backbone"]["pole_schedule"] == [160] * 4
    assert payload["backbone"]["transition"]["post_hidden_modes_by_resolution"] == dict(
        zip(map(str, runner.RESOLUTIONS), EXPECTED_H[variant], strict=True)
    )


def test_progressive_k_contract_pins_queue_parameters_and_sources(
    contract_args: argparse.Namespace,
) -> None:
    payload = runner._contract(contract_args)

    assert payload["schema"] == "lnet.a2d.r2k3.progressive_k_d2262.v1"
    assert payload["variants"] == list(runner.VARIANTS)
    assert payload["seeds"] == [501]
    assert payload["jobs_by_gpu"] == {
        str(gpu): list(variants) for gpu, variants in runner.JOBS_BY_GPU.items()
    }
    assert payload["parameter_counts"] == EXPECTED_PARAMETERS
    assert len(payload["source_sha256"]["progressive_k_d2262_runner"]) == 64
