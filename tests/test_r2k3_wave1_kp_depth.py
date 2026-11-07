from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import run_alphabet2d_imagenet100_nano as harness
import run_complex_scan_zero_init_imagenet100 as zero_init
import torch

from scripts import a2d_r2k3_runtime as runtime
from scripts import r2k3_runner_registry as registry
from scripts import run_a2d_r2k3_wave1_kp_depth_imagenet100 as runner

if TYPE_CHECKING:
    from torch import nn

    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


EXPECTED_PARAMETERS = {
    runner.D2242_P128: 3_350_052,
    runner.D2242_P14_R15: 3_711_012,
    runner.D2242_P28_14_R15: 3_891_492,
    runner.D2242_PFRONT3_R15: 4_071_972,
    runner.D2262_P128: 4_006_436,
    runner.D2242_UNIFORM_R15: 4_223_140,
}


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


def test_wave1_lane_membership_and_order_are_exact() -> None:
    assert runner.JOBS_BY_GPU == {
        0: (
            runner.D2242_P128,
            runner.D2242_P14_R15,
            runner.D2242_P28_14_R15,
            runner.D2242_PFRONT3_R15,
        ),
        1: (runner.D2262_P128, runner.D2242_UNIFORM_R15),
    }
    queued = tuple(variant for lane in runner.JOBS_BY_GPU.values() for variant in lane)
    assert queued == runner.VARIANTS
    assert len(queued) == len(set(queued)) == 6


def test_wave1_runner_resolves_through_the_shared_registry() -> None:
    resolved = registry.resolve_runner("a2d_r2k3_wave1_kp_depth")
    assert resolved.__file__ == runner.__file__
    assert resolved.VARIANTS == runner.VARIANTS
    assert resolved.JOBS_BY_GPU == runner.JOBS_BY_GPU


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_wave1_variants_match_k_p_depth_and_parameter_contract(variant: str) -> None:
    spec = runner.SPECS[variant]
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())

    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS[
        variant
    ]
    assert spec.excitation_modes == (128,) * 4
    assert model.descriptor_dim == spec.descriptor_dim
    assert model.classifier.input_dim == spec.q4_dim
    same_resolution_blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    for index, name in enumerate(runner.STAGE_NAMES):
        stage = getattr(model, name)
        assert stage.input_modes == 128
        assert stage.modes == spec.pole_modes[index]
        block = cast(
            "SameResolutionPoleScanBlock",
            same_resolution_blocks[str(runner.RESOLUTIONS[index])],
        )
        assert block.modes == 128
        assert block.pole_modes == spec.pole_modes[index]

    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    assert set(extras) == {"14"}
    assert len(cast("nn.ModuleList", extras["14"])) == spec.extra_blocks[2]
    assert spec.depth in {(2, 2, 4, 2), (2, 2, 6, 2)}


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_wave1_variant_configs_are_json_exact(variant: str) -> None:
    payload = runner._variant_config(variant)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["experiment"]["wave"] == 1
    assert payload["experiment"]["main_ratio_family"] == [1.0, 1.5, 2.0]


def test_wave1_contract_pins_membership_parameters_and_source(
    contract_args: argparse.Namespace,
) -> None:
    payload = runner._contract(contract_args)

    assert payload["schema"] == "lnet.a2d.r2k3.wave1_kp_depth.v1"
    assert payload["variants"] == list(runner.VARIANTS)
    assert payload["seeds"] == [501]
    assert payload["jobs_by_gpu"] == {
        "0": list(runner.GPU0_VARIANTS),
        "1": list(runner.GPU1_VARIANTS),
    }
    assert payload["parameter_counts"] == EXPECTED_PARAMETERS
    assert list(payload["variant_configs"]) == list(runner.VARIANTS)
    assert list(payload["architecture"]) == list(runner.VARIANTS)
    digest = payload["source_sha256"]["wave1_kp_depth_runner"]
    assert len(digest) == 64
    assert set(payload["variants"]).isdisjoint(
        {
            "K128-P128x4-D2232-FullSR14x1",
            "K128-P128x4-D2223-FullSR7x1",
        }
    )


def test_wave1_repeated_sr14_blocks_are_distinct_and_ordered() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.D2242_P14_R15, runtime.model_config()).eval()
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    blocks = cast("nn.ModuleList", extras["14"])
    assert len(blocks) == 2
    assert blocks[0] is not blocks[1]
    assert all(
        left is not right
        for left, right in zip(blocks[0].parameters(), blocks[1].parameters(), strict=True)
    )

    calls: list[int] = []
    handles = [
        block.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.append(index)
        )
        for index, block in enumerate(blocks)
    ]
    try:
        with torch.no_grad():
            descriptor = model.raw_descriptor(torch.randn(1, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert descriptor.shape == (1, 2_304)
    assert calls == [0, 1]
