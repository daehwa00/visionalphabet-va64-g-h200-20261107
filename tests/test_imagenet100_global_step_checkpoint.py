from __future__ import annotations

# pyright: reportExplicitAny=false, reportMissingImports=false
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import json
import random
from typing import TYPE_CHECKING

import numpy as np
import pytest
import run_alphabet2d_imagenet100_nano as runner
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _checkpoint_components() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.Generator,
    np.random.Generator,
]:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return (
        model,
        optimizer,
        scheduler,
        torch.Generator().manual_seed(17),
        np.random.default_rng(19),
    )


def _write_checkpoint(path: Path, *, epoch: int, global_step: int | None) -> None:
    model, optimizer, scheduler, training_generator, mixup_generator = _checkpoint_components()
    payload: dict[str, Any] = {
        "variant": "fixture",
        "seed": 501,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": [{"epoch": float(epoch)}],
        "training_seconds": 12.5,
        "training_generator_state": training_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": [],
        "python_rng_state": random.getstate(),
        "mixup_rng_state": mixup_generator.bit_generator.state,
    }
    if global_step is not None:
        payload["global_step"] = global_step
    torch.save(payload, path)


def _restore(
    path: Path, progress: dict[str, int], *, steps: int | None
) -> tuple[
    int,
    list[dict[str, float]],
    float,
]:
    model, optimizer, scheduler, training_generator, mixup_generator = _checkpoint_components()
    return runner._restore_checkpoint(
        path,
        variant="fixture",
        seed=501,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_generator=training_generator,
        mixup_generator=mixup_generator,
        progress=progress,
        optimizer_steps_per_epoch=steps,
    )


def test_restore_keeps_tuple_contract_and_exposes_stored_global_step(tmp_path: Path) -> None:
    checkpoint = tmp_path / "current.pt"
    _write_checkpoint(checkpoint, epoch=3, global_step=47)
    progress: dict[str, int] = {}

    restored = _restore(checkpoint, progress, steps=None)

    assert restored == (3, [{"epoch": 3.0}], 12.5)
    assert progress == {"global_step": 47}


def test_restore_derives_legacy_global_step_from_completed_epochs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    _write_checkpoint(checkpoint, epoch=3, global_step=None)
    progress: dict[str, int] = {}

    restored = _restore(checkpoint, progress, steps=7)

    assert restored[0] == 3
    assert progress == {"global_step": 21}


def test_restore_refuses_to_guess_legacy_global_step_without_epoch_size(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    _write_checkpoint(checkpoint, epoch=2, global_step=None)

    with pytest.raises(RuntimeError, match="optimizer_steps_per_epoch"):
        _restore(checkpoint, {}, steps=None)


def test_missing_checkpoint_initializes_optional_progress_without_changing_tuple(
    tmp_path: Path,
) -> None:
    progress = {"global_step": 99}

    restored = _restore(tmp_path / "missing.pt", progress, steps=7)

    assert restored == (0, [], 0.0)
    assert progress == {"global_step": 0}


def test_run_job_checkpoints_exact_successful_optimizer_step_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = TensorDataset(torch.ones(8, 1), torch.zeros(8, dtype=torch.long))
    train_loader = DataLoader(dataset, batch_size=2)
    validation_loader = DataLoader(dataset, batch_size=2)
    observed_global_steps: list[int] = []
    atomic_torch = runner._atomic_torch

    def train_epoch(
        model: nn.Module,
        _runtime: nn.Module,
        _loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        **_kwargs: object,
    ) -> dict[str, float]:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([parameter.sum() for parameter in model.parameters()]).sum()
            loss.backward()
            optimizer.step()
        return {"loss": 1.0, "mixed_accuracy": 0.5}

    def record_checkpoint(path: Path, payload: object) -> None:
        assert isinstance(payload, dict)
        global_step = payload["global_step"]
        assert isinstance(global_step, int)
        observed_global_steps.append(global_step)
        atomic_torch(path, payload)

    monkeypatch.setattr(
        runner,
        "_loaders",
        lambda *_args, **_kwargs: (train_loader, validation_loader),
    )
    monkeypatch.setattr(runner, "_build_runtime", lambda model, _recipe: model)
    monkeypatch.setattr(runner, "_initialize_wandb_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_atomic_torch", record_checkpoint)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    bindings = runner.RunnerBindings(
        variants=("fixture",),
        seeds=(501,),
        model_config=lambda **_kwargs: object(),
        build_model=lambda _variant, _config: nn.Linear(1, 1),
        contract=lambda _args: {},
        build_optimizer=lambda model, recipe: torch.optim.AdamW(
            model.parameters(),
            lr=recipe["learning_rate"],
        ),
        prepare_model=lambda model, _recipe: model,
        train_epoch=train_epoch,
        evaluate=lambda *_args, **_kwargs: {"accuracy": 0.5, "cross_entropy": 1.0},
        wandb_model_metrics=lambda _model: {},
        summarize=lambda _root, _contract: None,
    )
    contract = {
        "model": {},
        "recipe": {
            "epochs": 2,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "mixup_alpha": 0.8,
            "precision": "float32",
            "channels_last": False,
        },
    }

    runner._run_job(
        tmp_path,
        contract,
        variant="fixture",
        seed=501,
        data_root=tmp_path,
        workers=0,
        device=torch.device("cpu"),
        bindings=bindings,
    )

    assert observed_global_steps == [3, 6]
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "fixture__seed501.pt",
        map_location="cpu",
        weights_only=True,
    )
    result = json.loads((tmp_path / "results" / "fixture__seed501.json").read_text())
    assert checkpoint["global_step"] == 6
    assert result["global_step"] == 6
