"""Validate and quarantine R2K3 training artifacts."""

from __future__ import annotations

# Checkpoint payloads and optimizer state dictionaries are intentionally dynamic.
# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

import a2d_r2k3_runtime as runtime
import numpy as np
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as runner
import run_alphabet2d_imagenet100_nano as harness
import torch


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_checkpoint_identity(checkpoint: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError):
        return None
    return payload if isinstance(payload, dict) else None


def rng_states_are_restorable(payload: dict[str, Any]) -> bool:
    training_rng = payload.get("training_generator_state")
    torch_rng = payload.get("torch_rng_state")
    cuda_rng = payload.get("cuda_rng_state")
    python_rng = payload.get("python_rng_state")
    mixup_rng = payload.get("mixup_rng_state")
    if not (
        isinstance(training_rng, torch.Tensor)
        and isinstance(torch_rng, torch.Tensor)
        and isinstance(python_rng, tuple)
        and isinstance(mixup_rng, dict)
    ):
        return False
    try:
        torch.Generator().set_state(training_rng)
        torch.Generator().set_state(torch_rng)
        random.Random().setstate(python_rng)  # noqa: S311 - validate training RNG state
        generator = np.random.default_rng()
        generator.bit_generator.state = mixup_rng
        cuda_state_size = (
            torch.Generator(device="cuda").get_state().numel() if torch.cuda.is_available() else 16
        )
        if not (
            isinstance(cuda_rng, list)
            and bool(cuda_rng)
            and all(
                isinstance(state, torch.Tensor)
                and state.dtype == torch.uint8
                and state.numel() == cuda_state_size
                for state in cuda_rng
            )
        ):
            return False
        if torch.cuda.is_available():
            if len(cuda_rng) > torch.cuda.device_count():
                return False
            for index, state in enumerate(cuda_rng):
                torch.Generator(device=f"cuda:{index}").set_state(state)
    except (RuntimeError, TypeError, ValueError):
        return False
    return True


def contract_recipe(path: Path) -> dict[str, Any] | None:
    try:
        contract = json.loads(path.read_text())
        recipe = contract["recipe"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return recipe if isinstance(recipe, dict) else None


def optimizer_groups_restore_exactly(
    archived_groups: list[dict[str, Any]],
    expected_groups: list[dict[str, Any]],
    archived_scheduler: dict[str, Any],
    expected_scheduler: dict[str, Any],
) -> bool:
    archived_lrs = archived_scheduler["_last_lr"]
    immutable_options = (
        "initial_lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
    )
    if len(archived_groups) != len(expected_groups) or len(archived_lrs) != len(expected_groups):
        return False
    if (
        archived_scheduler["base_lrs"] != expected_scheduler["base_lrs"]
        or len(archived_scheduler["base_lrs"]) != len(expected_groups)
        or len(archived_scheduler["lr_lambdas"]) != len(expected_groups)
    ):
        return False
    return all(
        archived["params"] == expected["params"]
        and archived["lr"] == archived_lr
        and all(archived.get(name) == expected.get(name) for name in immutable_options)
        for archived, expected, archived_lr in zip(
            archived_groups,
            expected_groups,
            archived_lrs,
            strict=True,
        )
    )


def training_state_loads(
    variant: str,
    payload: dict[str, Any],
    recipe: dict[str, Any],
) -> bool:
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(501)
            model = runner._build(variant, runtime.model_config())
        model.load_state_dict(payload["model"], strict=True)
        optimizer = runtime.build_optimizer(model, recipe)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: harness._learning_rate_factor(epoch, recipe["epochs"]),
        )
        expected_groups = optimizer.state_dict()["param_groups"]
        expected_scheduler = scheduler.state_dict()
        archived_groups = payload["optimizer"]["param_groups"]
        expected_state_ids = {
            parameter_id for group in expected_groups for parameter_id in group["params"]
        }
        if set(payload["optimizer"]["state"]) != expected_state_ids:
            return False
        archived_scheduler = payload["scheduler"]
        if not optimizer_groups_restore_exactly(
            archived_groups,
            expected_groups,
            archived_scheduler,
            expected_scheduler,
        ):
            return False
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(archived_scheduler)
        for group in optimizer.param_groups:
            group["fused"] = False
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False
    return True


def checkpoint_is_resumable(
    payload: dict[str, Any] | None,
    variant: str,
    epochs: int,
    *,
    require_final: bool,
    validate_model: bool = False,
    require_global_step: bool = False,
    recipe: dict[str, Any] | None = None,
) -> bool:
    required = {
        "optimizer",
        "scheduler",
        "history",
        "training_seconds",
        "training_generator_state",
        "torch_rng_state",
        "cuda_rng_state",
        "python_rng_state",
        "mixup_rng_state",
    }
    if payload is None or not required.issubset(payload):
        return False
    epoch = payload.get("epoch")
    model = payload.get("model")
    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    python_rng = payload.get("python_rng_state")
    mixup_rng = payload.get("mixup_rng_state")
    global_step = payload.get("global_step")
    return bool(
        payload.get("variant") == variant
        and type(payload.get("seed")) is int
        and payload["seed"] == 501
        and type(epoch) is int
        and (epoch == epochs if require_final else 0 <= epoch <= epochs)
        and (
            type(global_step) is int and global_step >= 0
            if require_global_step
            else global_step is None or (type(global_step) is int and global_step >= 0)
        )
        and isinstance(model, dict)
        and bool(model)
        and all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in model.items()
        )
        and isinstance(optimizer, dict)
        and {"state", "param_groups"}.issubset(optimizer)
        and isinstance(optimizer["state"], dict)
        and bool(optimizer["state"])
        and isinstance(optimizer["param_groups"], list)
        and bool(optimizer["param_groups"])
        and all(
            isinstance(group, dict)
            and isinstance(group.get("params"), list)
            and bool(group["params"])
            for group in optimizer["param_groups"]
        )
        and set(optimizer["state"]).issubset(
            parameter for group in optimizer["param_groups"] for parameter in group["params"]
        )
        and all(
            type(parameter) is int and isinstance(state, dict) and bool(state)
            for parameter, state in optimizer["state"].items()
        )
        and isinstance(scheduler, dict)
        and {"base_lrs", "last_epoch", "_step_count"}.issubset(scheduler)
        and isinstance(scheduler["base_lrs"], list)
        and bool(scheduler["base_lrs"])
        and type(scheduler["last_epoch"]) is int
        and scheduler["last_epoch"] == epoch
        and type(scheduler["_step_count"]) is int
        and scheduler["_step_count"] == epoch + 1
        and isinstance(payload.get("history"), list)
        and len(payload["history"]) == epoch
        and all(
            isinstance(row, dict) and row.get("epoch") == index
            for index, row in enumerate(payload["history"], start=1)
        )
        and finite(payload.get("training_seconds"))
        and float(payload["training_seconds"]) >= 0.0
        and rng_states_are_restorable(payload)
        and isinstance(python_rng, tuple)
        and len(python_rng) == 3
        and type(python_rng[0]) is int
        and isinstance(python_rng[1], tuple)
        and isinstance(mixup_rng, dict)
        and {"bit_generator", "state", "has_uint32", "uinteger"}.issubset(mixup_rng)
        and isinstance(mixup_rng["state"], dict)
        and bool(mixup_rng["state"])
        and (
            not validate_model
            or (recipe is not None and training_state_loads(variant, payload, recipe))
        )
    )


def valid_contract(
    path: Path,
    variant: str,
    epochs: int,
    *,
    campaign: dict[str, Any] | None,
    source_sha256: dict[str, str] | None,
) -> bool:
    try:
        contract = json.loads(path.read_text())
        recipe = contract["recipe"]
        if not (
            contract.get("variants") == [variant]
            and contract.get("seeds") == [501]
            and isinstance(recipe, dict)
            and type(recipe.get("epochs")) is int
            and recipe.get("epochs") == epochs
        ):
            return False
        if campaign is not None and not all(
            recipe.get(name) == campaign[name]
            for name in (
                "epochs",
                "batch_size",
                "gradient_accumulation_steps",
                "precision",
                "loader_workers",
                "loader_persistent_workers",
            )
        ):
            return False
        if campaign is not None:
            data = contract.get("data")
            if not isinstance(data, dict) or not (
                data.get("manifest_sha256") == campaign["data_manifest_sha256"]
                and data.get("train_images") == campaign["train_images"]
                and data.get("validation_images") == campaign["validation_images"]
            ):
                return False
        if source_sha256 is not None:
            recorded = contract.get("source_sha256")
            if not isinstance(recorded, dict) or not all(
                recorded.get(name) == digest for name, digest in source_sha256.items()
            ):
                return False
    except (AttributeError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def valid_run(
    run_root: Path,
    variant: str,
    epochs: int,
    *,
    campaign: dict[str, Any] | None = None,
    source_sha256: dict[str, str] | None = None,
) -> bool:
    result = run_root / "results" / f"{variant}__seed501.json"
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    contract_path = run_root / "contract.json"
    if not result.is_file() or not checkpoint.is_file() or not contract_path.is_file():
        return False
    checkpoint_payload = load_checkpoint_identity(checkpoint)
    if checkpoint_payload is None:
        return False
    recipe = contract_recipe(contract_path)
    if not checkpoint_is_resumable(
        checkpoint_payload,
        variant,
        epochs,
        require_final=True,
        validate_model=campaign is not None,
        require_global_step=campaign is not None,
        recipe=recipe,
    ) or not valid_contract(
        contract_path,
        variant,
        epochs,
        campaign=campaign,
        source_sha256=source_sha256,
    ):
        return False
    try:
        payload = json.loads(result.read_text())
        history = payload["history"]
        final = payload["final_validation"]
        valid = bool(
            payload.get("variant") == variant
            and payload.get("seed") == 501
            and isinstance(history, list)
            and len(history) == epochs
            and isinstance(history[-1], dict)
            and history[-1].get("epoch") == epochs
            and isinstance(final, dict)
            and finite(final.get("accuracy"))
            and finite(final.get("cross_entropy"))
            and type(payload.get("parameters")) is int
            and payload["parameters"] > 0
            and type(payload.get("global_step")) is int
            and (
                checkpoint_payload.get("global_step") == payload.get("global_step")
                if campaign is not None
                else checkpoint_payload.get("global_step") is None
                or checkpoint_payload.get("global_step") == payload.get("global_step")
            )
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    else:
        return valid


def quarantine_invalid_completion(
    run_root: Path,
    variant: str,
    epochs: int,
    *,
    campaign: dict[str, Any],
    source_sha256: dict[str, str],
) -> None:
    result = run_root / "results" / f"{variant}__seed501.json"
    checkpoint = run_root / "checkpoints" / f"{variant}__seed501.pt"
    contract = run_root / "contract.json"
    if valid_run(
        run_root,
        variant,
        epochs,
        campaign=campaign,
        source_sha256=source_sha256,
    ):
        return
    suffix = f"invalid-{int(time.time())}-{os.getpid()}"
    if result.is_file():
        result.replace(result.with_name(f"{result.stem}.{suffix}{result.suffix}"))
    contract_valid = valid_contract(
        contract,
        variant,
        epochs,
        campaign=campaign,
        source_sha256=source_sha256,
    )
    if contract.is_file() and not contract_valid:
        contract.replace(contract.with_name(f"{contract.stem}.{suffix}{contract.suffix}"))
    checkpoint_payload = load_checkpoint_identity(checkpoint)
    recipe = contract_recipe(contract)
    resumable = checkpoint_is_resumable(
        checkpoint_payload,
        variant,
        epochs,
        require_final=False,
        validate_model=True,
        require_global_step=False,
        recipe=recipe,
    )
    if checkpoint.is_file() and (not resumable or not contract_valid):
        checkpoint.replace(checkpoint.with_name(f"{checkpoint.stem}.{suffix}{checkpoint.suffix}"))


__all__ = [
    "checkpoint_is_resumable",
    "contract_recipe",
    "finite",
    "load_checkpoint_identity",
    "optimizer_groups_restore_exactly",
    "quarantine_invalid_completion",
    "rng_states_are_restorable",
    "training_state_loads",
    "valid_contract",
    "valid_run",
]
