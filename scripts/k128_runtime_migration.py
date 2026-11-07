"""Strict, one-time migration of the frozen K128 ImageNet checkpoints.

The old contract is reconstructed from the frozen runtime's formulas and pinned
values. Its stored JSON is checked against that reconstruction before it is
used. Only checkpoint binding fields are changed; all training state is copied
verbatim into the staged checkpoint.
"""

# pyright: reportExplicitAny=false, reportPrivateUsage=false, reportImplicitRelativeImport=false
# Checkpoint and contract payloads are intentionally dynamic serialization dictionaries.
# Runtime imports are top-level because the deployed CLI puts `scripts/` on PYTHONPATH;
# private helpers preserve the worker's exact contract and checkpoint validation rules.

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CAMPAIGN_ROOT = Path("/app/output/k128-optimization-20260929")
DATA_ROOT = Path("/app/data/ImageNet-2012")
OLD_CODE_SHA = "44ea5c788e84d148855343544b9c8a8ea3fb584e"
OLD_CLOSURE_SHA = "e4f24f17e85504afb96e0ca85e1a1361df6f787b7dd48aaf88e453651bfd037c"
OLD_RUNTIME_SHA = "a2a50f5d78862eaebe50c8791d62ebb882fcf1dc96d07fe92ef88a7577f124eb"
OLD_SCHEMA = "lnet.h200.imagenet1k.matched_baseline.worker.v1"
NEW_SCHEDULE = "linear warmup plus cosine decay per optimizer step using epoch-normalized schedule position"
EPOCH_UPDATES = 5_005
TRAIN_IMAGES = 1_281_167
VALIDATION_IMAGES = 50_000
NUM_CLASSES = 1_000
RECEIPT_SCHEMA = "visionalphabet.k128.checkpoint_migration.v1"
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_ORIGIN = {
    "checkpoint_sha256",
    "completed_epochs",
    "global_step",
    "archive_sha256",
    "code_sha",
    "dataset_manifest_sha256",
}
_ALLOWED_CHANGES = [
    "task.source_root",
    "task.source_sha256",
    "task.batch_size",
    "task.effective_batch_size",
    "task.workers",
    "recipe.schedule",
    "recipe.batch_size",
    "recipe.gradient_accumulation_steps",
    "recipe.effective_batch_size",
    "source_sha256",
    "source_digest_sha256",
    "execution",
    "task_sha256",
    "checkpoint.contract_sha256",
    "checkpoint.task_sha256",
    "checkpoint.source_digest_sha256",
    "checkpoint.source_sha256",
]


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    ).encode("utf-8")


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_task(variant: str) -> dict[str, Any]:
    if variant not in {"baseline", "improved"}:
        raise ValueError("K128 migration variant must be baseline or improved")
    output = CAMPAIGN_ROOT / variant
    return {
        "phase": "full",
        "model_key": f"k128_{variant}",
        "seed": 501,
        "learning_rate": 0.003,
        "epochs": 200,
        "data_root": str(DATA_ROOT),
        "output_dir": str(output),
        "result_path": str(output / "result.json"),
        "checkpoint_path": str(output / "checkpoint.pt"),
        "source_root": str(CAMPAIGN_ROOT / "code" / OLD_CODE_SHA / "vision-runtime"),
        "batch_size": 128,
        "workers": 4,
        "wandb_mode": "disabled",
        "num_classes": NUM_CLASSES,
        "precision": "bfloat16",
        "source_sha256": {"campaign_runtime_closure": OLD_CLOSURE_SHA},
    }


def legacy_contract(variant: str) -> dict[str, Any]:
    """Reconstruct one original worker contract from pinned frozen formulas."""
    task = _legacy_task(variant)
    source_sha256 = {
        "campaign_runtime_closure": OLD_CLOSURE_SHA,
        "imagenet1k_runtime": OLD_RUNTIME_SHA,
    }
    recipe = {
        "image_size": 224,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.05,
        "configured_full_run_warmup_epochs": 5,
        "schedule": "linear warmup plus cosine decay per optimizer step",
        "label_smoothing": 0.1,
        "mixup_alpha": 0.8,
        "cutmix_alpha": 0.0,
        "randaugment": "rand-m9-mstd0.5-inc1 (N=2, M=9)",
        "random_erasing_probability": 0.25,
        "batch_size": 128,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 256,
        "precision": "bfloat16",
    }
    return {
        "schema": OLD_SCHEMA,
        "task": task,
        "task_sha256": _sha256_payload(task),
        "model": {"key": f"k128_{variant}", "num_classes": NUM_CLASSES, "pretrained": False},
        "recipe": recipe,
        "dataset": {"name": "ImageNet-1K", "root": str(DATA_ROOT)},
        "source_sha256": source_sha256,
        "source_digest_sha256": _sha256_payload(source_sha256),
        "runtime": {
            "device_type": "cuda",
            "torch": "2.9.1+cu128",
            "cuda_runtime": "12.8",
        },
    }


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256.fullmatch(value):
        raise RuntimeError(f"migration origin {name} must be a lowercase SHA-256")
    return value


def _validate_origin(origin: object) -> dict[str, Any]:
    if not isinstance(origin, dict) or not _REQUIRED_ORIGIN.issubset(origin):
        missing = sorted(_REQUIRED_ORIGIN - set(origin) if isinstance(origin, dict) else _REQUIRED_ORIGIN)
        raise RuntimeError(f"migration origin is incomplete; missing {missing}")
    # Keep extra freeze metadata in the receipt, but require it to be JSON-safe.
    try:
        json.loads(_canonical_json(origin))
    except (TypeError, ValueError) as error:
        raise RuntimeError("migration origin is not valid finite JSON") from error
    result = dict(origin)
    _require_sha256(result["checkpoint_sha256"], "checkpoint_sha256")
    _require_sha256(result["archive_sha256"], "archive_sha256")
    _require_sha256(result["dataset_manifest_sha256"], "dataset_manifest_sha256")
    if result["code_sha"] != OLD_CODE_SHA:
        raise RuntimeError("migration origin code SHA does not identify the frozen K128 source")
    completed_epochs = result["completed_epochs"]
    global_step = result["global_step"]
    if (
        isinstance(completed_epochs, bool)
        or not isinstance(completed_epochs, int)
        or not 1 <= completed_epochs <= 200
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step != completed_epochs * EPOCH_UPDATES
    ):
        raise RuntimeError("migration origin epoch or optimizer-step count is inconsistent")
    return result


def _check_task_target(task: Any, variant: str) -> None:
    import imagenet1k_runtime as runtime

    runtime._validate_task(task)
    if variant not in {"baseline", "improved"}:
        raise ValueError("K128 migration variant must be baseline or improved")
    if (
        task.phase != "full"
        or task.model_key != f"k128_{variant}"
        or task.seed != 501
        or task.learning_rate != 0.003
        or task.epochs != 200
        or task.data_root.resolve() != DATA_ROOT.resolve()
        or task.output_dir.resolve() != (CAMPAIGN_ROOT / variant).resolve()
        or task.result_path.resolve() != (CAMPAIGN_ROOT / variant / "result.json").resolve()
        or task.checkpoint_path.resolve() != (CAMPAIGN_ROOT / variant / "checkpoint.pt").resolve()
        or task.batch_size not in {256, 512}
        or task.effective_batch_size != task.batch_size
        or task.workers != 8
        or task.wandb_mode != "disabled"
        or task.num_classes != NUM_CLASSES
        or task.precision != "bfloat16"
        or task.source_root is None
    ):
        raise RuntimeError("target K128 task changes a pinned semantic or migration invariant")
    if (
        task.output_dir.is_symlink()
        or task.result_path.is_symlink()
        or task.checkpoint_path.is_symlink()
    ):
        raise RuntimeError("K128 output, result, and checkpoint paths cannot be symlinks")
    expected_source = {
        "campaign_runtime_closure": task.source_sha256.get("campaign_runtime_closure")
    }
    if (
        set(task.source_sha256) != {"campaign_runtime_closure"}
        or not isinstance(expected_source["campaign_runtime_closure"], str)
        or not _HEX_SHA256.fullmatch(expected_source["campaign_runtime_closure"])
    ):
        raise RuntimeError("target task source identity is malformed")
    source_root = Path(task.source_root).resolve().as_posix()
    campaign_root = CAMPAIGN_ROOT.resolve().as_posix()
    if not re.fullmatch(
        re.escape(campaign_root) + r"/code/[0-9a-f]{40}/vision-runtime", source_root
    ):
        raise RuntimeError("target source root is outside the K128 source relocation boundary")
    execution = task.execution
    if (
        execution.compile_model is not True
        or execution.gpu_mixup is not True
        or execution.non_blocking_transfer is not True
        or execution.prefetch_factor != 2
        or execution.metric_sync_interval != 100
    ):
        raise RuntimeError("target execution options differ from the approved K128 runtime profile")


def _target_contract(task: Any, variant: str, device: Any) -> dict[str, Any]:
    import imagenet1k_runtime as runtime

    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("K128 checkpoint migration requires the pinned CUDA runtime")
    contract = runtime._contract(task, device)
    old = legacy_contract(variant)
    _assert_only_approved_contract_changes(old, contract, task, variant)
    return contract


def _assert_only_approved_contract_changes(
    old: dict[str, Any], target: dict[str, Any], task: Any, variant: str
) -> None:
    import imagenet1k_runtime as runtime

    if target.get("recipe", {}).get("schedule") != NEW_SCHEDULE:
        raise RuntimeError("target runtime schedule is not the approved epoch-normalized schedule")
    execution = target.get("execution")
    expected_execution = {
        "compile_model": True,
        "gpu_mixup": True,
        "non_blocking_transfer": True,
        "prefetch_factor": 2,
        "metric_sync_interval": 100,
    }
    if execution != expected_execution:
        raise RuntimeError("target contract execution options differ from the approved profile")

    normalized = copy.deepcopy(target)
    target_task = normalized.get("task")
    if not isinstance(target_task, dict):
        raise TypeError("target contract task is missing")
    old_task_keys = set(old["task"])
    if set(target_task) != old_task_keys | {"effective_batch_size"}:
        raise RuntimeError("target contract task contains fields outside the approved migration boundary")
    for name in ("source_root", "source_sha256", "batch_size", "effective_batch_size", "workers"):
        if name == "effective_batch_size":
            target_task.pop(name, None)
        else:
            target_task[name] = old["task"][name]
    # `effective_batch_size` was added to the task representation in the new runtime;
    # it was already pinned by the old recipe's constant.
    normalized["task"] = target_task
    normalized["task_sha256"] = _sha256_payload(target_task)

    recipe = normalized.get("recipe")
    if not isinstance(recipe, dict):
        raise TypeError("target contract recipe is missing")
    recipe["schedule"] = old["recipe"]["schedule"]
    recipe["batch_size"] = old["recipe"]["batch_size"]
    recipe["gradient_accumulation_steps"] = old["recipe"]["gradient_accumulation_steps"]
    recipe["effective_batch_size"] = old["recipe"]["effective_batch_size"]

    normalized["source_sha256"] = old["source_sha256"]
    normalized["source_digest_sha256"] = old["source_digest_sha256"]
    normalized.pop("execution", None)
    if normalized != old:
        raise RuntimeError("target contract changes a semantic K128 recipe/model/data invariant")

    actual_task = target["task"]
    if (
        actual_task.get("batch_size") != task.batch_size
        or actual_task.get("effective_batch_size") != task.effective_batch_size
        or actual_task.get("workers") != 8
        or actual_task.get("model_key") != f"k128_{variant}"
        or actual_task.get("source_root") != str(Path(task.source_root).resolve())
        or actual_task.get("source_sha256") != task.source_sha256
    ):
        raise RuntimeError("target contract task fields do not match the requested target")
    expected_source_sha256 = {
        **task.source_sha256,
        "imagenet1k_runtime": hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
    }
    if target.get("source_sha256") != expected_source_sha256:
        raise RuntimeError("target source fingerprint does not match the active runtime and task")


def _legacy_validation_task(variant: str) -> Any:
    import imagenet1k_runtime as runtime

    task_data = _legacy_task(variant)
    return runtime.BaselineTask(
        phase=task_data["phase"],
        model_key=task_data["model_key"],
        seed=task_data["seed"],
        learning_rate=task_data["learning_rate"],
        epochs=task_data["epochs"],
        data_root=Path(task_data["data_root"]),
        output_dir=Path(task_data["output_dir"]),
        result_path=Path(task_data["result_path"]),
        checkpoint_path=Path(task_data["checkpoint_path"]),
        source_root=Path(task_data["source_root"]),
        batch_size=128,
        effective_batch_size=256,
        workers=4,
        wandb_mode="disabled",
        resume=True,
        num_classes=NUM_CLASSES,
        precision="bfloat16",
        source_sha256={"campaign_runtime_closure": OLD_CLOSURE_SHA},
    )


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} is not a JSON object: {path}")
    return value


def _validate_dataset_manifest(output_dir: Path, origin: dict[str, Any]) -> None:
    manifest_path = output_dir / "dataset.json"
    manifest = _load_json_object(manifest_path, "variant dataset manifest")
    expected_keys = {
        "train_images",
        "validation_images",
        "classes",
        "relative_path_label_size_sha256",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("train_images") != TRAIN_IMAGES
        or manifest.get("validation_images") != VALIDATION_IMAGES
        or manifest.get("classes") != NUM_CLASSES
        or not isinstance(manifest.get("relative_path_label_size_sha256"), str)
        or not _HEX_SHA256.fullmatch(manifest["relative_path_label_size_sha256"])
    ):
        raise RuntimeError("K128 dataset manifest does not match the frozen ImageNet-1K inventory")
    if manifest["relative_path_label_size_sha256"] != origin["dataset_manifest_sha256"]:
        raise RuntimeError("K128 dataset manifest digest differs from the frozen archive origin")
    shared_path = output_dir.parent / "dataset.json"
    if shared_path.exists() and _load_json_object(shared_path, "shared dataset manifest") != manifest:
        raise RuntimeError("shared and variant K128 dataset manifests disagree")


def _validate_checkpoint_structure(
    payload: dict[str, Any], contract: dict[str, Any], task: Any
) -> None:
    import imagenet1k_runtime as runtime

    runtime._validate_checkpoint(payload, contract, task)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise RuntimeError(f"could not safely load checkpoint: {path}") from error
    if not isinstance(value, dict):
        raise TypeError("K128 checkpoint payload is not an object")
    return value


def _validate_epoch_step(payload: dict[str, Any], origin: dict[str, Any]) -> None:
    completed_epochs = origin["completed_epochs"]
    global_step = origin["global_step"]
    if payload.get("completed_epochs") != completed_epochs or payload.get("global_step") != global_step:
        raise RuntimeError("checkpoint epoch or optimizer-step count differs from the frozen origin")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != completed_epochs:
        raise RuntimeError("checkpoint history length differs from the frozen completed epoch")
    for epoch, row in enumerate(history, 1):
        if (
            not isinstance(row, dict)
            or row.get("epoch") != epoch
            or row.get("global_step") != epoch * EPOCH_UPDATES
        ):
            raise RuntimeError("checkpoint history has an unexpected epoch or update count")
    training_seconds = payload.get("training_seconds")
    if (
        isinstance(training_seconds, bool)
        or not isinstance(training_seconds, (int, float))
        or not math.isfinite(training_seconds)
        or training_seconds < 0
    ):
        raise TypeError("checkpoint training_seconds is missing or invalid")


def _validate_legacy_checkpoint(
    path: Path,
    variant: str,
    contract: dict[str, Any],
    origin: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("frozen K128 checkpoint is missing or is not a regular file")
    actual_sha = _file_sha256(path)
    if actual_sha != origin["checkpoint_sha256"]:
        raise RuntimeError("checkpoint raw SHA-256 differs from the frozen archive origin")
    payload = _load_checkpoint(path)
    _validate_checkpoint_structure(payload, contract, _legacy_validation_task(variant))
    _validate_epoch_step(payload, origin)
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise RuntimeError("migration metadata directory cannot be a symlink")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(_canonical_json(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)
    if path.parent.parent.exists():
        _fsync_directory(path.parent.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_staged_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    import torch

    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _replace_checkpoint(stage: Path, checkpoint: Path) -> None:
    stage.replace(checkpoint)
    _fsync_directory(checkpoint.parent)


@contextmanager
def _migration_lock(checkpoint: Path) -> Generator[None]:
    lock_id = hashlib.sha256(str(checkpoint.resolve()).encode("utf-8")).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"visionalphabet-k128-migration-{lock_id}.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _read_contract(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        raise RuntimeError("K128 contract path cannot be a symlink")
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError("K128 contract path must be a regular file")
    return _load_json_object(path, "K128 runtime contract")


def _read_receipts(migration_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    if migration_dir.is_symlink():
        raise RuntimeError("K128 migration receipt directory cannot be a symlink")
    if not migration_dir.exists():
        return []
    if not migration_dir.is_dir():
        raise RuntimeError("K128 migration receipt directory is invalid")
    receipts = []
    for path in sorted(migration_dir.glob("*.json")):
        record = _load_json_object(path, "K128 migration receipt")
        if (
            record.get("schema") != RECEIPT_SCHEMA
            or record.get("migration_id") != path.stem
            or record.get("status") not in {"prepared", "completed"}
        ):
            raise RuntimeError(f"unrecognized K128 migration receipt: {path}")
        receipts.append((path, record))
    return receipts


def _validate_receipt(
    path: Path,
    receipt: dict[str, Any],
    *,
    variant: str,
    checkpoint_path: Path,
    contract_path: Path,
    old_contract: dict[str, Any],
    new_contract: dict[str, Any],
    origin: dict[str, Any] | None,
) -> None:
    if receipt.get("status") not in {"prepared", "completed"}:
        raise RuntimeError("K128 migration receipt status is invalid")
    if (
        receipt.get("variant") != variant
        or Path(str(receipt.get("checkpoint_path", ""))) != checkpoint_path
        or Path(str(receipt.get("contract_path", ""))) != contract_path
        or receipt.get("old_contract") != old_contract
        or receipt.get("new_contract") != new_contract
        or receipt.get("old_contract_sha256") != _sha256_payload(old_contract)
        or receipt.get("new_contract_sha256") != _sha256_payload(new_contract)
        or receipt.get("allowed_changes") != _ALLOWED_CHANGES
    ):
        raise RuntimeError(f"K128 migration receipt identity or contract changed: {path}")
    for name in ("old_checkpoint_sha256", "new_checkpoint_sha256"):
        _require_sha256(receipt.get(name), name)
    receipt_origin = receipt.get("origin")
    if not isinstance(receipt_origin, dict):
        raise TypeError("K128 migration receipt origin is invalid")
    validated_receipt_origin = _validate_origin(receipt_origin)
    if (
        receipt.get("completed_epochs") != validated_receipt_origin["completed_epochs"]
        or receipt.get("global_step") != validated_receipt_origin["global_step"]
        or receipt.get("old_checkpoint_sha256")
        != validated_receipt_origin["checkpoint_sha256"]
    ):
        raise RuntimeError("K128 migration receipt epoch or step metadata is inconsistent")
    if origin is not None and validated_receipt_origin != origin:
        raise RuntimeError("K128 migration receipt does not match the verified archive origin")


def _finish_prepared_transaction(
    receipt_path: Path,
    receipt: dict[str, Any],
    *,
    checkpoint_path: Path,
    contract_path: Path,
    old_contract: dict[str, Any],
    new_contract: dict[str, Any],
    task: Any,
    origin: dict[str, Any],
) -> None:
    old_sha = receipt["old_checkpoint_sha256"]
    new_sha = receipt["new_checkpoint_sha256"]
    current_sha = _file_sha256(checkpoint_path)
    stage = checkpoint_path.with_name(f".{checkpoint_path.name}.{receipt['migration_id']}.migrating")

    contract = _read_contract(contract_path)
    if contract not in (old_contract, new_contract):
        raise RuntimeError("K128 contract changed during an interrupted migration")

    if current_sha == old_sha:
        if contract != old_contract:
            raise RuntimeError("old checkpoint is paired with an already replaced K128 contract")
        if _file_sha256(stage) != new_sha:
            raise RuntimeError("prepared K128 migrated checkpoint stage is missing or corrupt")
        old_payload = _validate_legacy_checkpoint(
            checkpoint_path, str(receipt["variant"]), old_contract, origin
        )
        staged_payload = _load_checkpoint(stage)
        _validate_checkpoint_structure(staged_payload, new_contract, task)
        _validate_epoch_step(staged_payload, origin)
        _assert_training_state_equal(old_payload, staged_payload)
        _replace_checkpoint(stage, checkpoint_path)
        current_sha = _file_sha256(checkpoint_path)
    elif current_sha == new_sha:
        if stage.exists():
            if _file_sha256(stage) != new_sha:
                raise RuntimeError("leftover K128 migration stage has unexpected content")
            stage.unlink()
            _fsync_directory(stage.parent)
        payload = _load_checkpoint(checkpoint_path)
        _validate_checkpoint_structure(payload, new_contract, task)
        _validate_epoch_step(payload, origin)
    else:
        raise RuntimeError("K128 checkpoint content matches neither side of the prepared transaction")

    if current_sha != new_sha:
        raise RuntimeError("K128 checkpoint replacement did not produce the prepared content")
    _write_json_atomic(contract_path, new_contract)
    receipt["status"] = "completed"
    receipt["completed_at_unix_ns"] = time.time_ns()
    _write_json_atomic(receipt_path, receipt)


def _assert_training_state_equal(old: dict[str, Any], new: dict[str, Any]) -> None:
    binding_fields = {
        "contract_sha256",
        "task_sha256",
        "source_digest_sha256",
        "source_sha256",
    }
    if old.keys() != new.keys():
        raise RuntimeError("K128 checkpoint migration changed the payload key set")
    for name, value in old.items():
        if name not in binding_fields and not _nested_values_equal(value, new[name]):
            raise RuntimeError(f"K128 checkpoint migration changed training state field {name!r}")


def _nested_values_equal(left: Any, right: Any) -> bool:
    import math

    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.layout == right.layout
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_nested_values_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left):
        return math.isnan(right)
    return type(left) is type(right) and left == right


def _migrate_locked(
    task: Any,
    variant: str,
    device: Any,
    expected_origin: object,
) -> bool:
    import imagenet1k_runtime as runtime

    checkpoint_path = task.checkpoint_path.resolve()
    contract_path = task.output_dir.resolve() / "contracts" / f"{task.task_name}.json"
    result_path = task.result_path.resolve()
    migration_dir = task.output_dir.resolve() / "contracts" / "migrations"
    result_exists = result_path.exists() or result_path.is_symlink()
    if result_path.is_symlink():
        raise RuntimeError("K128 result path cannot be a symlink")

    old_contract = legacy_contract(variant)
    new_contract = _target_contract(task, variant, device)
    if checkpoint_path.is_symlink():
        raise RuntimeError("K128 checkpoint path cannot be a symlink")
    if task.resume != checkpoint_path.is_file():
        raise RuntimeError("K128 resume flag and checkpoint presence disagree")
    contract = _read_contract(contract_path)
    receipts = _read_receipts(migration_dir)
    prepared = [(path, item) for path, item in receipts if item.get("status") == "prepared"]
    completed = [(path, item) for path, item in receipts if item.get("status") == "completed"]
    if len(receipts) > 1:
        raise RuntimeError("multiple K128 migration transactions exist for one checkpoint")

    # A target-contract checkpoint has already crossed the migration boundary.
    # This path intentionally needs no old-origin metadata (for example, the
    # improved arm is fresh or the baseline has resumed under the new contract).
    if contract == new_contract and checkpoint_path.is_file() and not prepared:
        payload = _load_checkpoint(checkpoint_path)
        _validate_checkpoint_structure(payload, new_contract, task)
        if result_exists:
            result = _load_json_object(result_path, "completed K128 result")
            runtime._validate_result(result, new_contract, task)
            if (
                result.get("completed_epochs") != payload.get("completed_epochs")
                or result.get("global_step") != payload.get("global_step")
                or result.get("history") != payload.get("history")
            ):
                raise RuntimeError("completed K128 result does not match its final checkpoint")
        if completed:
            receipt_path, receipt = completed[0]
            _validate_receipt(
                receipt_path,
                receipt,
                variant=variant,
                checkpoint_path=checkpoint_path,
                contract_path=contract_path,
                old_contract=old_contract,
                new_contract=new_contract,
                origin=None,
            )
        return False

    if result_exists:
        raise RuntimeError("result.json exists; K128 paused-checkpoint migration is no longer valid")

    if not checkpoint_path.exists() and not contract_path.exists() and not receipts:
        if expected_origin in ({}, None):
            return False
        raise RuntimeError("verified frozen K128 origin has no checkpoint to migrate")
    if not checkpoint_path.is_file():
        raise RuntimeError("K128 checkpoint is missing or is not a regular file")

    origin = _validate_origin(expected_origin)
    _validate_dataset_manifest(task.output_dir.resolve(), origin)

    if prepared:
        receipt_path, receipt = prepared[0]
        _validate_receipt(
            receipt_path,
            receipt,
            variant=variant,
            checkpoint_path=checkpoint_path,
            contract_path=contract_path,
            old_contract=old_contract,
            new_contract=new_contract,
            origin=origin,
        )
        _finish_prepared_transaction(
            receipt_path,
            receipt,
            checkpoint_path=checkpoint_path,
            contract_path=contract_path,
            old_contract=old_contract,
            new_contract=new_contract,
            task=task,
            origin=origin,
        )
        return True

    if completed:
        # A completed receipt with a still-old contract is impossible under the
        # write order and therefore signals external modification.
        raise RuntimeError("completed K128 migration receipt is paired with a non-target contract")
    if contract != old_contract:
        if contract is None:
            raise RuntimeError("frozen K128 checkpoint has no original contract")
        raise RuntimeError("stored K128 contract does not match the reconstructed frozen contract")
    if _file_sha256(checkpoint_path) != origin["checkpoint_sha256"]:
        raise RuntimeError("checkpoint raw SHA-256 differs from the frozen archive origin")

    old_payload = _validate_legacy_checkpoint(checkpoint_path, variant, old_contract, origin)
    migrated = copy.copy(old_payload)
    migrated["contract_sha256"] = _sha256_payload(new_contract)
    migrated["task_sha256"] = new_contract["task_sha256"]
    migrated["source_digest_sha256"] = new_contract["source_digest_sha256"]
    migrated["source_sha256"] = new_contract["source_sha256"]
    _validate_checkpoint_structure(migrated, new_contract, task)
    _validate_epoch_step(migrated, origin)
    _assert_training_state_equal(old_payload, migrated)

    migration_id = uuid.uuid4().hex
    stage = checkpoint_path.with_name(f".{checkpoint_path.name}.{migration_id}.migrating")
    _save_staged_checkpoint(stage, migrated)
    staged_sha = _file_sha256(stage)
    receipt_path = migration_dir / f"{migration_id}.json"
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "migration_id": migration_id,
        "status": "prepared",
        "variant": variant,
        "checkpoint_path": str(checkpoint_path),
        "contract_path": str(contract_path),
        "origin": origin,
        "old_contract": old_contract,
        "new_contract": new_contract,
        "old_contract_sha256": _sha256_payload(old_contract),
        "new_contract_sha256": _sha256_payload(new_contract),
        "old_checkpoint_sha256": origin["checkpoint_sha256"],
        "new_checkpoint_sha256": staged_sha,
        "allowed_changes": _ALLOWED_CHANGES,
        "completed_epochs": origin["completed_epochs"],
        "global_step": origin["global_step"],
        "prepared_at_unix_ns": time.time_ns(),
    }
    _write_json_atomic(receipt_path, receipt)
    _finish_prepared_transaction(
        receipt_path,
        receipt,
        checkpoint_path=checkpoint_path,
        contract_path=contract_path,
        old_contract=old_contract,
        new_contract=new_contract,
        task=task,
        origin=origin,
    )
    return True


def migrate_checkpoint(task: Any, variant: str, device: Any, expected_origin: dict[str, Any]) -> bool:
    """Migrate a verified frozen K128 checkpoint, returning whether work occurred.

    Returns ``False`` for a fresh arm or an already target-bound checkpoint.
    Returns ``True`` after applying a migration or completing its interrupted
    prepared transaction. Raises on any contract, origin, or file mismatch.
    """
    _check_task_target(task, variant)
    with _migration_lock(task.checkpoint_path.resolve()):
        return _migrate_locked(task, variant, device, expected_origin)


__all__ = ["legacy_contract", "migrate_checkpoint"]
