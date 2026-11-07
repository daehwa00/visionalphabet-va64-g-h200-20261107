"""Scientific pairing and durable runtime hooks for the H200 K128 experiment."""

import gzip
import json
from dataclasses import replace

import imagenet1k_runtime as runtime
import pytest
import run_k128_optimization_campaign as campaign
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def test_paired_build_preserves_non_geometry_weights_and_rng() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(501)
    baseline = campaign.build_model("baseline")
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(501)
    improved = campaign.build_model("improved")
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    assert sum(p.numel() for p in baseline.parameters()) == campaign.PARAMETERS
    assert sum(p.numel() for p in improved.parameters()) == campaign.PARAMETERS
    other = improved.state_dict()
    for name, value in baseline.state_dict().items():
        if "damping_logits_" not in name:
            assert name in other, name
            assert torch.equal(value, other[name]), name
    assert len(campaign.pole_snapshot(baseline)["banks"]) == 12
    assert len(campaign.pole_snapshot(improved)["banks"]) == 12


def _task(root, *, resume=False):
    return runtime.BaselineTask(
        phase="full",
        model_key="tiny-hooks",
        seed=501,
        learning_rate=0.001,
        epochs=2,
        data_root=root / "data",
        output_dir=root,
        result_path=root / "result.json",
        checkpoint_path=root / "checkpoint.pt",
        batch_size=128,
        workers=0,
        num_classes=3,
        precision="float32",
        resume=resume,
    )


def _model(_key, _source, classes):
    return nn.Sequential(nn.Flatten(), nn.Linear(12, classes))


def _loaders(task, _device):
    fixed = torch.Generator().manual_seed(11)
    dataset = TensorDataset(torch.randn(8, 3, 2, 2, generator=fixed), torch.arange(8) % 3)
    train_rng = torch.Generator().manual_seed(task.seed)
    val_rng = torch.Generator().manual_seed(task.seed + 1)
    return runtime.LoaderBundle(
        DataLoader(dataset, batch_size=4, shuffle=True, generator=train_rng),
        DataLoader(dataset, batch_size=4, generator=val_rng),
        train_rng,
        val_rng,
    )


def test_epoch_boundary_pause_resume_and_callback_checkpoint(tmp_path) -> None:
    events = []

    def callback(model, row):
        saved = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
        assert saved["completed_epochs"] == row["epoch"]
        assert all(torch.equal(v, saved["model"][k]) for k, v in model.state_dict().items())
        events.append(row["epoch"])

    kwargs = {
        "device": torch.device("cpu"),
        "model_builder": _model,
        "loader_builder": _loaders,
        "mixup_factory": lambda task: (
            lambda inputs, labels: (
                inputs,
                torch.nn.functional.one_hot(labels, task.num_classes).float(),
            )
        ),
    }
    uninterrupted_task = _task(tmp_path / "uninterrupted")
    runtime.run_task(uninterrupted_task, **kwargs)
    task = _task(tmp_path / "paused")
    paused = runtime.run_task(task, epoch_callback=callback, stop_requested=lambda: True, **kwargs)
    assert paused["status"] == "paused"
    assert events == [1]
    assert not task.result_path.exists()
    resumed = runtime.run_task(replace(task, resume=True), epoch_callback=callback, **kwargs)
    assert resumed["status"] == "completed"
    assert events == [1, 1, 2]
    left = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
    right = torch.load(uninterrupted_task.checkpoint_path, map_location="cpu", weights_only=True)
    assert all(torch.equal(value, right["model"][name]) for name, value in left["model"].items())


def test_trajectory_snapshot_is_atomic_and_loadable(tmp_path) -> None:
    torch.manual_seed(501)
    model = campaign.build_model("improved")
    campaign.save_poles(tmp_path, model, 0, 0)
    with gzip.open(tmp_path / "poles/epoch-0000.json.gz", "rt") as stream:
        data = json.load(stream)
    assert data["epoch"] == 0
    assert data["global_step"] == 0
    assert len(data["banks"]) == 12


def test_resume_repairs_epoch_artifact_after_callback_failure(tmp_path) -> None:
    task = _task(tmp_path)
    kwargs = {
        "device": torch.device("cpu"),
        "model_builder": _model,
        "loader_builder": _loaders,
        "mixup_factory": lambda active: (
            lambda images, labels: (
                images,
                torch.nn.functional.one_hot(labels, active.num_classes).float(),
            )
        ),
    }

    def failed_callback(_model, _row):
        raise RuntimeError("interrupted before derived artifact")

    with pytest.raises(RuntimeError, match="before derived artifact"):
        runtime.run_task(task, epoch_callback=failed_callback, **kwargs)
    saved = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
    assert saved["completed_epochs"] == 1
    seen = []

    def repair(model, row):
        if row["epoch"] == 1:
            assert all(
                torch.equal(value, saved["model"][name])
                for name, value in model.state_dict().items()
            )
        seen.append(row["epoch"])

    runtime.run_task(replace(task, resume=True), epoch_callback=repair, **kwargs)
    assert seen == [1, 2]
