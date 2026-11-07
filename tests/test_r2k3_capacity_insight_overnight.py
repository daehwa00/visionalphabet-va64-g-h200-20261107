from __future__ import annotations

import json

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportIndexIssue=false
# pyright: reportOptionalMemberAccess=false, reportPrivateUsage=false
# pyright: reportPrivateLocalImportUsage=false
# pyright: reportUnknownLambdaType=false
from argparse import Namespace
from typing import TYPE_CHECKING

import pytest
import torch

from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as runner
from scripts import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as control
from scripts import smoke_r2k3_campaign as smoke

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_PARAMETERS = {
    runner.TERM_P96: 2_663_204,
    runner.TERM_P160: 2_724_132,
    runner.EARLY_THIN: 2_602_916,
    runner.K160_P128: 3_671_780,
    runner.K128_P160: 3_154_660,
    runner.K160_P160: 3_592_100,
    runner.EXTRA_SR14_7: 3_350_052,
}


def test_capacity_insight_queue_partitions_every_candidate_once() -> None:
    queued = tuple(variant for lane in runner.JOBS_BY_GPU.values() for variant in lane)
    assert set(runner.JOBS_BY_GPU) == {0, 1}
    assert len(queued) == len(set(queued))
    assert set(queued) == set(runner.VARIANTS)


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_variant_config_survives_json_roundtrip_exactly(variant: str) -> None:
    config = runner._variant_config(variant)
    assert json.loads(json.dumps(config)) == config


def test_generic_p128_builder_is_bit_exact_with_the_control() -> None:
    spec = runner.InsightSpec((128, 128, 128, 128), (128, 128, 128, 128))
    torch.manual_seed(501)
    reference = control._build(control.VARIANT, runtime.model_config())
    torch.manual_seed(501)
    candidate = runner._build_all_resolution(spec, runtime.model_config())

    assert set(candidate.state_dict()) == set(reference.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("variant", runner.VARIANTS)
def test_capacity_insight_variants_match_declared_contract(variant: str) -> None:
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())
    runner._assert_model(model, variant)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == EXPECTED_PARAMETERS[variant]


@pytest.mark.parametrize(
    ("variant", "target_poles"),
    [(runner.TERM_P96, 96), (runner.TERM_P160, 160)],
)
def test_terminal_sweep_uses_one_joint_reader_and_fully_active_head(
    variant: str,
    target_poles: int,
) -> None:
    torch.manual_seed(501)
    model = runner._build(variant, runtime.model_config())
    reader = model.terminal.pole_input_projection
    primary = reader.point_weight_real[:, 0]
    gram = primary @ primary.T if target_poles < 128 else primary.T @ primary
    torch.testing.assert_close(
        gram,
        torch.eye(gram.shape[0]),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    assert reader.rms_reference_modes == target_poles
    head_weight = model.classifier.affine.linear.weight
    assert torch.count_nonzero(head_weight) == head_weight.numel()


def test_extra_depth_executes_one_additional_block_at_14_and_7() -> None:
    torch.manual_seed(501)
    model = runner._build(runner.EXTRA_SR14_7, runtime.model_config()).eval()
    calls = {"14": 0, "7": 0}
    handles = []
    for resolution, block in model.extra_same_resolution_blocks.items():
        handles.append(
            block.register_forward_hook(
                lambda _module, _inputs, _output, key=resolution: calls.__setitem__(
                    key, calls[key] + 1
                )
            )
        )
    try:
        with torch.no_grad():
            descriptor = model.raw_descriptor(torch.randn(1, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert descriptor.shape == (1, 2048)
    assert calls == {"14": 1, "7": 1}


def test_full_batch_smoke_compiles_each_candidate_only_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compile_flags = []
    full_batch_calls = []
    monkeypatch.setattr(
        smoke.shared,
        "_run_candidate",
        lambda _variant, _device, _root, *, compile_model: (
            compile_flags.append(compile_model) or {"status": "passed"}
        ),
    )
    monkeypatch.setattr(
        smoke.shared,
        "_full_batch_cuda_step",
        lambda variant, _batch, _steps: full_batch_calls.append(variant) or {"steps": 2.0},
    )
    args = Namespace(
        root=tmp_path,
        compile=True,
        full_batch=True,
        full_batch_size=128,
        repeat_steps=2,
    )

    payload = smoke._multi_variant_payload(
        args,
        torch.device("cpu"),
        runner.VARIANTS,
    )

    assert compile_flags == [False] * len(runner.VARIANTS)
    assert full_batch_calls == list(runner.VARIANTS)
    assert set(payload["variants"]) == set(runner.VARIANTS)
