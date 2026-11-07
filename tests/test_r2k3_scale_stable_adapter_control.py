from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
# pyright: reportUnknownLambdaType=false
import argparse
import json
from pathlib import Path

import pytest
import run_alphabet2d_imagenet100_nano as harness
import run_complex_scan_zero_init_imagenet100 as zero_init
import torch

from lnet.pac_complex_layers import ScaleStableComplexLinear
from scripts import a2d_r2k3_runtime as runtime
from scripts import r2k3_runner_registry as registry
from scripts import run_a2d_r2k3_d2262_p160_followup_imagenet100 as base
from scripts import run_a2d_r2k3_scale_stable_adapter_control_imagenet100 as runner

EXPECTED_PARAMETERS = {
    runner.K128_CONTROL: 4_713_444,
    runner.K14_160: 5_809_636,
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


def test_scale_stable_queue_is_one_control_per_gpu() -> None:
    assert runner.JOBS_BY_GPU == {0: (runner.K128_CONTROL,), 1: (runner.K14_160,)}
    resolved = registry.resolve_runner("a2d_r2k3_scale_stable_adapter_control")
    assert resolved.__file__ == runner.__file__


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_scale_stable_variants_match_parameter_and_adapter_contract(variant: str) -> None:
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS[
        variant
    ]
    owners = runner._adapter_owners(model)
    assert len(owners) == 11
    assert all(type(owner.memory_projection) is ScaleStableComplexLinear for owner in owners)
    stable_carries = [
        owner.carry_projection
        for owner in owners
        if hasattr(owner, "carry_projection") and owner.carry_projection is not None
    ]
    expected_carries = 0 if variant == runner.K128_CONTROL else 2
    assert len(stable_carries) == expected_carries
    assert all(type(projection) is ScaleStableComplexLinear for projection in stable_carries)


def test_k128_control_preserves_unconstrained_initial_function_and_rng() -> None:
    torch.manual_seed(501)
    reference = base._build(base.VARIANT, runtime.model_config()).eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = reference(inputs)[0]
    rng_state = torch.random.get_rng_state()

    runner._install_scale_stable_adapters(reference)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    with torch.no_grad():
        actual = reference(inputs)[0]
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=2.0e-6)


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_scale_stable_variant_config_is_json_exact(variant: str) -> None:
    payload = runner._variant_config(variant)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["backbone"]["excitation_schedule"] == list(
        runner.SPECS[variant].excitation_modes
    )


def test_scale_stable_contract_pins_parameters_and_sources(
    contract_args: argparse.Namespace,
) -> None:
    payload = runner._contract(contract_args)
    assert payload["schema"] == "lnet.a2d.r2k3.scale_stable_adapter_control.v1"
    assert payload["parameter_counts"] == EXPECTED_PARAMETERS
    assert payload["jobs_by_gpu"] == {
        str(gpu): list(variants) for gpu, variants in runner.JOBS_BY_GPU.items()
    }
    assert len(payload["source_sha256"]["scale_stable_adapter_control_runner"]) == 64
