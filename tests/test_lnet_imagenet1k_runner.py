from __future__ import annotations

# pyright: reportPrivateUsage=false
import argparse
from typing import TYPE_CHECKING

import torch

from scripts import run_lnet_k128_p160_160_160_128_d2262_imagenet1k as runner

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_lnet_imagenet1k_contract_is_frozen() -> None:
    assert runner.VARIANT == "K128-P160-160-160-128-D2262"
    assert runner.SEED == 501
    assert runner.EPOCHS == 100
    assert runner.LEARNING_RATE == 3.0e-3
    assert runner.EXPECTED_PARAMETERS == 5_083_176


def test_lnet_imagenet1k_uses_matched_baseline_task(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_root=tmp_path / "output",
        data_root=tmp_path / "imagenet",
        batch_size=128,
        workers=8,
        wandb_mode="disabled",
    )
    task = runner._task(args)
    assert task.epochs == 100
    assert task.batch_size == 128
    assert task.gradient_accumulation_steps == 2
    assert task.workers == 8
    assert task.learning_rate == 3.0e-3
    assert task.seed == 501
    assert task.resume is False


def test_lnet_imagenet1k_task_uses_the_local_runtime_contract(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_root=tmp_path / "output",
        data_root=tmp_path / "imagenet",
        batch_size=128,
        workers=8,
        wandb_mode="disabled",
    )
    task = runner._task(args)

    assert isinstance(task, runner.imagenet_runtime.BaselineTask)
    assert task.num_classes == 1000
    assert set(task.source_sha256) == {"imagenet1k_runner", "lnet_internal_closure"}


def test_lnet_imagenet1k_main_submits_the_local_task_to_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        output_root=tmp_path / "output",
        data_root=tmp_path / "imagenet",
        batch_size=128,
        workers=8,
        wandb_mode="disabled",
    )
    submitted: list[tuple[runner.ImageNet1KTask, object]] = []
    monkeypatch.setattr(runner, "_arguments", lambda: args)
    monkeypatch.setattr(
        runner.imagenet_runtime,
        "run_task",
        lambda task, *, model_builder: submitted.append((task, model_builder)) or {"status": "ok"},
    )

    runner.main()

    assert submitted[0][0].model_key == runner.MODEL_KEY
    assert submitted[0][1] is runner._build_model


def test_lnet_imagenet1k_builder_has_exact_parameter_count() -> None:
    model = runner._build_model(runner.MODEL_KEY, None, 1000)
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_083_176


def test_lnet_imagenet1k_exposes_only_primary_logits() -> None:
    model = runner._build_model(runner.MODEL_KEY, None, 1000).eval()
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 1000)
    assert bool(torch.isfinite(logits).all())
