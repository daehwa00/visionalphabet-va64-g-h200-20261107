from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import imagenet1k_runtime as runtime
from scripts import run_lnet_k128_p160_160_160_128_d2262_imagenet1k as runner

if TYPE_CHECKING:
    from pathlib import Path


def _synthetic_loaders(
    task: runtime.BaselineTask, _device: torch.device,
) -> runtime.LoaderBundle:
    generator = torch.Generator().manual_seed(902)
    dataset = TensorDataset(
        torch.randn(2, 3, 224, 224, generator=generator), torch.tensor([0, 1]),
    )
    train_generator = torch.Generator().manual_seed(task.seed)
    validation_generator = torch.Generator().manual_seed(task.seed + 1)
    return runtime.LoaderBundle(
        DataLoader(dataset, batch_size=2, shuffle=True, generator=train_generator),
        DataLoader(dataset, batch_size=2, generator=validation_generator),
        train_generator,
        validation_generator,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_frozen_lnet_trains_and_checkpoints_with_the_restored_cuda_runtime(tmp_path: Path) -> None:
    task = runtime.BaselineTask(
        phase="full",
        model_key=runner.MODEL_KEY,
        seed=runner.SEED,
        learning_rate=runner.LEARNING_RATE,
        epochs=1,
        data_root=tmp_path / "synthetic-data",
        output_dir=tmp_path / "run",
        result_path=tmp_path / "run/result.json",
        checkpoint_path=tmp_path / "run/checkpoint.pt",
        batch_size=2,
        workers=0,
        wandb_mode="disabled",
    )

    result = runtime.run_task(
        task,
        device=torch.device("cuda"),
        model_builder=runner._build_model,
        loader_builder=_synthetic_loaders,
    )

    assert result["parameters"] == runner.EXPECTED_PARAMETERS
    assert result["completed_epochs"] == result["global_step"] == 1
    assert math.isfinite(result["final_validation"]["cross_entropy"])
    checkpoint = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
    optimizer_states = checkpoint["optimizer"]["state"].values()
    assert any(bool(state["exp_avg"].count_nonzero()) for state in optimizer_states)
    assert all(bool(value.isfinite().all()) for value in checkpoint["model"].values())
