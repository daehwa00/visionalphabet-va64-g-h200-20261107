from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import imagenet1k_runtime as runtime
import k128_runtime_migration as migration
import pytest
import torch


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _make_task(
    variant: str, *, batch_size: int = 256, resume: bool = True
) -> runtime.BaselineTask:
    campaign_root = migration.CAMPAIGN_ROOT
    output = campaign_root / variant
    return runtime.BaselineTask(
        phase="full",
        model_key=f"k128_{variant}",
        seed=501,
        learning_rate=0.003,
        epochs=200,
        data_root=migration.DATA_ROOT,
        output_dir=output,
        result_path=output / "result.json",
        checkpoint_path=output / "checkpoint.pt",
        source_root=campaign_root / "code" / ("b" * 40) / "vision-runtime",
        batch_size=batch_size,
        effective_batch_size=batch_size,
        workers=8,
        execution=runtime.ExecutionOptions(
            compile_model=True,
            gpu_mixup=True,
            non_blocking_transfer=True,
            prefetch_factor=2,
            metric_sync_interval=100,
        ),
        wandb_mode="disabled",
        resume=resume,
        num_classes=1000,
        precision="bfloat16",
        source_sha256={"campaign_runtime_closure": "c" * 64},
    )


def _target_contract(task: runtime.BaselineTask, variant: str) -> dict[str, Any]:
    contract = copy.deepcopy(migration.legacy_contract(variant))
    new_task = contract["task"]
    new_task.update(
        {
            "source_root": str(task.source_root.resolve()),
            "source_sha256": task.source_sha256,
            "batch_size": task.batch_size,
            "effective_batch_size": task.effective_batch_size,
            "workers": task.workers,
        }
    )
    contract["task_sha256"] = _digest_json(new_task)
    contract["recipe"].update(
        {
            "schedule": migration.NEW_SCHEDULE,
            "batch_size": task.batch_size,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": task.effective_batch_size,
        }
    )
    contract["source_sha256"] = {
        **task.source_sha256,
        "imagenet1k_runtime": hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
    }
    contract["source_digest_sha256"] = _digest_json(contract["source_sha256"])
    contract["execution"] = {
        "compile_model": True,
        "gpu_mixup": True,
        "non_blocking_transfer": True,
        "prefetch_factor": 2,
        "metric_sync_interval": 100,
    }
    return contract


def _make_origin(
    task: runtime.BaselineTask,
    variant: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    completed_epochs: int = 2,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    old_contract = migration.legacy_contract(variant)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = task.output_dir / "contracts" / f"{task.task_name}.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(old_contract), encoding="utf-8")

    dataset_manifest = {
        "train_images": migration.TRAIN_IMAGES,
        "validation_images": migration.VALIDATION_IMAGES,
        "classes": migration.NUM_CLASSES,
        "relative_path_label_size_sha256": "d" * 64,
    }
    (task.output_dir / "dataset.json").write_text(json.dumps(dataset_manifest), encoding="utf-8")
    origin = {
        "checkpoint_sha256": "0" * 64,
        "completed_epochs": completed_epochs,
        "global_step": completed_epochs * migration.EPOCH_UPDATES,
        "archive_sha256": "a" * 64,
        "code_sha": migration.OLD_CODE_SHA,
        "dataset_manifest_sha256": dataset_manifest["relative_path_label_size_sha256"],
    }
    history = [
        {
            "epoch": epoch,
            "global_step": epoch * migration.EPOCH_UPDATES,
            "train": {"loss": 1.25, "optimizer_steps": float(migration.EPOCH_UPDATES)},
            "validation": {"accuracy": 0.5},
            "training_seconds": 60.0 * epoch,
        }
        for epoch in range(1, completed_epochs + 1)
    ]
    checkpoint = {
        "schema": runtime.SCHEMA,
        "contract_sha256": _digest_json(old_contract),
        "task_sha256": old_contract["task_sha256"],
        "source_digest_sha256": old_contract["source_digest_sha256"],
        "source_sha256": old_contract["source_sha256"],
        "completed_epochs": completed_epochs,
        "global_step": origin["global_step"],
        "parameters": 5_083_176,
        "history": history,
        "training_seconds": 123.5,
        "model": {"weight": torch.arange(8, dtype=torch.float32).reshape(2, 4)},
        "optimizer": {
            "state": {3: {"exp_avg": torch.tensor([0.1, 0.2]), "step": torch.tensor(17.0)}},
            "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999)}],
        },
        "rng": {
            "python": (3, (1, 2, 3), None),
            "numpy": {"bit_generator": "MT19937", "state": [1, 2, 3], "position": 3},
            "torch": torch.tensor([4, 5], dtype=torch.uint8),
            "cuda": [torch.tensor([6], dtype=torch.uint8)],
            "train_generator": torch.tensor([7], dtype=torch.uint8),
            "validation_generator": torch.tensor([8], dtype=torch.uint8),
        },
    }
    torch.save(checkpoint, task.checkpoint_path)
    origin["checkpoint_sha256"] = hashlib.sha256(task.checkpoint_path.read_bytes()).hexdigest()
    new_contract = _target_contract(task, variant)
    monkeypatch.setattr(runtime, "_contract", lambda _task, _device: new_contract)
    return origin, checkpoint, new_contract


@pytest.fixture(autouse=True)
def migration_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(migration, "CAMPAIGN_ROOT", tmp_path / "campaign")
    monkeypatch.setattr(migration, "DATA_ROOT", tmp_path / "imagenet")


def test_legacy_contract_migrates_without_changing_training_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task("baseline")
    origin, original, new_contract = _make_origin(task, "baseline", monkeypatch)

    assert migration.migrate_checkpoint(task, "baseline", torch.device("cuda"), origin) is True

    migrated = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
    runtime._validate_checkpoint(migrated, new_contract, task)
    assert migrated["contract_sha256"] == _digest_json(new_contract)
    assert migrated["task_sha256"] == new_contract["task_sha256"]
    assert migrated["source_digest_sha256"] == new_contract["source_digest_sha256"]
    assert migrated["source_sha256"] == new_contract["source_sha256"]
    preserved_fields = (
        "schema",
        "completed_epochs",
        "global_step",
        "parameters",
        "history",
        "training_seconds",
        "model",
        "optimizer",
        "rng",
    )
    for name in preserved_fields:
        assert migration._nested_values_equal(migrated[name], original[name])
    receipt_paths = list((task.output_dir / "contracts" / "migrations").glob("*.json"))
    assert len(receipt_paths) == 1
    receipt = json.loads(receipt_paths[0].read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["origin"] == origin
    assert receipt["old_checkpoint_sha256"] == origin["checkpoint_sha256"]
    actual_checkpoint_sha = hashlib.sha256(task.checkpoint_path.read_bytes()).hexdigest()
    assert receipt["new_checkpoint_sha256"] == actual_checkpoint_sha
    assert receipt["completed_epochs"] == origin["completed_epochs"]
    assert receipt["global_step"] == origin["global_step"]


def test_interrupted_after_checkpoint_replace_recovers_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task("baseline", batch_size=512)
    origin, _original, new_contract = _make_origin(
        task, "baseline", monkeypatch, completed_epochs=1
    )
    replace = migration._replace_checkpoint

    def replace_then_interrupt(stage: Path, checkpoint: Path) -> None:
        replace(stage, checkpoint)
        raise RuntimeError("simulated process interruption after checkpoint replace")

    monkeypatch.setattr(migration, "_replace_checkpoint", replace_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        migration.migrate_checkpoint(task, "baseline", torch.device("cuda"), origin)
    monkeypatch.setattr(migration, "_replace_checkpoint", replace)

    assert migration.migrate_checkpoint(task, "baseline", torch.device("cuda"), origin) is True
    payload = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
    runtime._validate_checkpoint(payload, new_contract, task)
    payload["model"]["weight"][0, 0] = 99.0  # Represents a later new-contract training update.
    torch.save(payload, task.checkpoint_path)
    assert migration.migrate_checkpoint(task, "baseline", torch.device("cuda"), {}) is False
    resumed = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
    assert resumed["model"]["weight"][0, 0] == 99.0


@pytest.mark.parametrize(
    "failure", ["old_recipe", "target_recipe", "target_source", "checkpoint_sha"]
)
def test_rejects_changed_recipe_source_or_origin_hash(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    task = _make_task("baseline")
    origin, _checkpoint, new_contract = _make_origin(task, "baseline", monkeypatch)
    contract_path = task.output_dir / "contracts" / f"{task.task_name}.json"
    if failure == "old_recipe":
        stored = json.loads(contract_path.read_text(encoding="utf-8"))
        stored["recipe"]["mixup_alpha"] = 0.7
        contract_path.write_text(json.dumps(stored), encoding="utf-8")
    elif failure == "target_recipe":
        new_contract["recipe"]["label_smoothing"] = 0.2
    elif failure == "target_source":
        new_contract["task"]["source_root"] = str(migration.CAMPAIGN_ROOT / "outside")
    else:
        origin["checkpoint_sha256"] = "f" * 64

    with pytest.raises(RuntimeError):
        migration.migrate_checkpoint(task, "baseline", torch.device("cuda"), origin)


def test_existing_result_is_never_treated_as_a_paused_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task("baseline")
    origin, _checkpoint, _contract = _make_origin(task, "baseline", monkeypatch)
    task.result_path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"result\.json exists"):
        migration.migrate_checkpoint(task, "baseline", torch.device("cuda"), origin)


def test_completed_target_result_is_validated_and_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task("baseline")
    task.output_dir.mkdir(parents=True, exist_ok=True)
    new_contract = _target_contract(task, "baseline")
    monkeypatch.setattr(runtime, "_contract", lambda _task, _device: new_contract)
    contract_path = task.output_dir / "contracts" / f"{task.task_name}.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(new_contract), encoding="utf-8")

    history = [
        {"epoch": epoch, "global_step": epoch * migration.EPOCH_UPDATES}
        for epoch in range(1, task.epochs + 1)
    ]
    completed_epochs = task.epochs
    global_step = task.epochs * migration.EPOCH_UPDATES
    binding = {
        "schema": runtime.SCHEMA,
        "contract_sha256": _digest_json(new_contract),
        "task_sha256": new_contract["task_sha256"],
        "source_digest_sha256": new_contract["source_digest_sha256"],
        "source_sha256": new_contract["source_sha256"],
    }
    torch.save(
        {
            **binding,
            "completed_epochs": completed_epochs,
            "global_step": global_step,
            "history": history,
            "training_seconds": 12.5,
            "model": {"weight": torch.ones(2)},
            "optimizer": {"state": {}, "param_groups": []},
            "rng": {"torch": torch.zeros(1, dtype=torch.uint8)},
        },
        task.checkpoint_path,
    )
    result = {
        **binding,
        "status": "completed",
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "history": history,
        "final_validation": {"accuracy": 0.5},
    }
    task.result_path.write_text(json.dumps(result), encoding="utf-8")

    assert migration.migrate_checkpoint(task, "baseline", torch.device("cuda"), {}) is False

    result["history"][-1]["global_step"] -= 1
    task.result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError):
        migration.migrate_checkpoint(task, "baseline", torch.device("cuda"), {})


def test_fresh_improved_arm_does_not_need_frozen_origin_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task("improved", resume=False)
    new_contract = _target_contract(task, "improved")
    monkeypatch.setattr(runtime, "_contract", lambda _task, _device: new_contract)

    assert migration.migrate_checkpoint(task, "improved", torch.device("cuda"), {}) is False
