# Match the archived timm recipe's global NumPy/Mixup RNG protocol.
# ruff: noqa: NPY002
# Checkpoint and DataLoader payloads are intentionally serialized/dynamic.
# pyright: reportExplicitAny=false
"""Durable matched-recipe runtime for the retained internal ImageNet-1K LNet run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Callable


SCHEMA = "lnet.h200.imagenet1k.matched_baseline.worker.v1"
EFFECTIVE_BATCH_SIZE = 256
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.8
RESULT_PREFIX = "LNET_IMAGENET1K_RESULT_JSON="
PROGRESS_PREFIX = "LNET_IMAGENET1K_PROGRESS_JSON="


class Mixup(Protocol):
    def __call__(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]: ...


class BatchMixup(Protocol):
    def __call__(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]: ...

    def _mix_batch(self, inputs: Tensor) -> float: ...


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Opt-in runtime changes that affect execution while preserving the recipe."""

    compile_model: bool = False
    gpu_mixup: bool = False
    non_blocking_transfer: bool = False
    prefetch_factor: int = 1
    # Zero retains per-microbatch metric reads; a positive value reads at optimizer-step
    # intervals, with an unconditional read at the end of each epoch.
    metric_sync_interval: int = 0


@dataclass(slots=True)
class BatchMixupAdapter:
    """Preserve timm batch Mixup when an odd tail self-pairs at its center."""

    mixup: BatchMixup
    target_mixer: Callable[[Tensor, int, float, float], Tensor]
    num_classes: int
    label_smoothing: float

    def __call__(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        if inputs.shape[0] % 2 == 0:
            return self.mixup(inputs, targets)
        coefficient = self.mixup._mix_batch(inputs)  # pyright: ignore[reportPrivateUsage]
        mixed_targets = self.target_mixer(
            targets,
            self.num_classes,
            coefficient,
            self.label_smoothing,
        )
        return inputs, mixed_targets


@dataclass(frozen=True, slots=True)
class BaselineTask:
    """All result-affecting inputs for the retained internal LNet recipe."""

    phase: Literal["full"]
    model_key: str
    seed: int
    learning_rate: float
    epochs: int
    data_root: Path
    output_dir: Path
    result_path: Path
    checkpoint_path: Path
    source_root: Path | None = None
    batch_size: int = 128
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE
    workers: int = 2
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)
    wandb_mode: Literal["disabled", "online"] = "disabled"
    resume: bool = False
    num_classes: int = 1000
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    source_sha256: dict[str, str] = field(default_factory=dict)

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.effective_batch_size // self.batch_size

    @property
    def task_name(self) -> str:
        lr_label = f"{self.learning_rate:.8g}".replace(".", "p")
        return f"{self.model_key}__{self.phase}__seed{self.seed}__lr{lr_label}"


@dataclass(slots=True)
class LoaderBundle:
    train: DataLoader[Any]
    validation: DataLoader[Any]
    train_generator: torch.Generator
    validation_generator: torch.Generator


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _progress_path(task: BaselineTask) -> Path:
    return task.output_dir / "progress" / f"{task.task_name}.json"


def _emit_progress(task: BaselineTask, payload: dict[str, object]) -> None:
    _atomic_json(_progress_path(task), payload)
    print(f"{PROGRESS_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)


@dataclass(slots=True)
class LocalLogger:
    """Durable telemetry when online reporting is disabled."""

    task: BaselineTask

    def log(self, record: dict[str, object]) -> None:
        path = self.task.output_dir / "telemetry" / f"{self.task.task_name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def finish(self, result: dict[str, object]) -> None:
        self.log({"kind": "final", "online_requested": self.task.wandb_mode == "online", **result})


class Logger(Protocol):
    def log(self, record: dict[str, object]) -> None: ...

    def finish(self, result: dict[str, object]) -> None: ...


class WandbRun(Protocol):
    def log(self, record: dict[str, object]) -> None: ...

    def finish(self, exit_code: int = 0) -> None: ...


@dataclass(slots=True)
class WandbLogger:
    run: WandbRun

    def log(self, record: dict[str, object]) -> None:
        self.run.log(record)

    def finish(self, result: dict[str, object]) -> None:
        try:
            self.run.log(result)
        finally:
            self.run.finish(exit_code=0 if result.get("status") == "completed" else 1)


def _make_logger(task: BaselineTask, contract: dict[str, object]) -> Logger:
    if task.wandb_mode == "disabled":
        return LocalLogger(task)
    try:
        import wandb
    except ModuleNotFoundError as error:
        raise RuntimeError("--wandb-mode online requires wandb==0.22.3") from error
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "alphabet2d-imagenet1k-h200-baselines"),
        entity=os.environ.get("WANDB_ENTITY", "daehwa"),
        group=os.environ.get("WANDB_GROUP", "lnet-imagenet1k-matched-100ep-s501-v1"),
        name=os.environ.get("WANDB_NAME", f"{task.model_key}-seed{task.seed}"),
        id=_sha256_payload(contract)[:16],
        resume="allow",
        mode="online",
        config=contract,
    )
    return WandbLogger(cast("WandbRun", cast("object", run)))


def _task_payload(task: BaselineTask) -> dict[str, Any]:
    payload = asdict(task)
    for name in ("data_root", "output_dir", "result_path", "checkpoint_path", "source_root"):
        if payload[name] is not None:
            payload[name] = str(Path(cast("Path", payload[name])).resolve())
    payload.pop("resume")
    # Keep execution tuning in the top-level contract, separate from task identity.
    payload.pop("execution")
    return payload


def _validate_task(task: BaselineTask) -> None:
    if task.phase != "full":
        raise ValueError("the retained ImageNet-1K runtime supports full tasks only")
    if task.seed < 0 or task.epochs < 1 or task.workers < 0 or task.num_classes < 2:
        raise ValueError("task seed, epochs, workers, or class count is invalid")
    if not math.isfinite(task.learning_rate) or task.learning_rate <= 0.0:
        raise ValueError("task learning rate must be positive and finite")
    if (
        task.batch_size < 1
        or task.effective_batch_size < 1
        or task.effective_batch_size % task.batch_size
    ):
        raise ValueError("physical batch size must be a positive divisor of effective batch size")
    if task.precision not in {"bfloat16", "float32"}:
        raise ValueError("task precision is invalid")
    if task.execution.prefetch_factor < 1 or task.execution.metric_sync_interval < 0:
        raise ValueError("execution prefetch factor or metric sync interval is invalid")


def _contract(task: BaselineTask, device: torch.device) -> dict[str, Any]:
    task_payload = _task_payload(task)
    source_sha256 = {
        **task.source_sha256,
        "imagenet1k_runtime": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return {
        "schema": SCHEMA,
        "task": task_payload,
        "task_sha256": _sha256_payload(task_payload),
        "model": {"key": task.model_key, "num_classes": task.num_classes, "pretrained": False},
        "recipe": {
            "image_size": 224,
            "optimizer": "AdamW",
            "learning_rate": task.learning_rate,
            "weight_decay": WEIGHT_DECAY,
            "configured_full_run_warmup_epochs": WARMUP_EPOCHS,
            "schedule": (
                "linear warmup plus cosine decay per optimizer step using "
                "epoch-normalized schedule position"
            ),
            "label_smoothing": LABEL_SMOOTHING,
            "mixup_alpha": MIXUP_ALPHA,
            "cutmix_alpha": 0.0,
            "randaugment": "rand-m9-mstd0.5-inc1 (N=2, M=9)",
            "random_erasing_probability": 0.25,
            "batch_size": task.batch_size,
            "gradient_accumulation_steps": task.gradient_accumulation_steps,
            "effective_batch_size": task.effective_batch_size,
            "precision": task.precision,
        },
        "dataset": {"name": "ImageNet-1K", "root": str(task.data_root.resolve())},
        "source_sha256": source_sha256,
        "source_digest_sha256": _sha256_payload(source_sha256),
        "runtime": {
            "device_type": device.type,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "execution": asdict(task.execution),
    }


def _build_loaders(task: BaselineTask, device: torch.device) -> LoaderBundle:
    try:
        from timm.data.transforms_factory import create_transform
        from torchvision.datasets import ImageFolder
        from torchvision.transforms import InterpolationMode, v2
    except ModuleNotFoundError as error:
        raise RuntimeError("ImageNet-1K default recipe requires timm==1.0.26") from error
    train_dataset = ImageFolder(
        task.data_root / "train",
        transform=cast(
            "Callable[..., object]",
            create_transform(
                input_size=(3, 224, 224),
                is_training=True,
                auto_augment="rand-m9-mstd0.5-inc1",
                interpolation="bicubic",
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
            ),
        ),
    )
    validation_dataset = ImageFolder(
        task.data_root / "val",
        transform=v2.Compose(
            [
                v2.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
                v2.CenterCrop(224),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    )
    if len(train_dataset.classes) != task.num_classes:
        raise RuntimeError(f"training split has {len(train_dataset.classes)} classes")
    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise RuntimeError("training and validation class mappings differ")
    train_generator = torch.Generator().manual_seed(task.seed)
    validation_generator = torch.Generator().manual_seed(task.seed + 1)
    common: dict[str, Any] = {
        "batch_size": task.batch_size,
        "num_workers": task.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    if task.workers:
        common["prefetch_factor"] = task.execution.prefetch_factor
        # Model construction initializes CUDA before the first iterator. Spawn
        # avoids inheriting CUDA/CPU thread-pool state in image decode workers.
        common["multiprocessing_context"] = "spawn"
    return LoaderBundle(
        DataLoader(train_dataset, shuffle=True, generator=train_generator, **common),
        DataLoader(validation_dataset, shuffle=False, generator=validation_generator, **common),
        train_generator,
        validation_generator,
    )


def _make_mixup(task: BaselineTask) -> Mixup:
    try:
        from timm.data.mixup import Mixup as TimmMixup
        from timm.data.mixup import mixup_target
    except ModuleNotFoundError as error:
        raise RuntimeError("ImageNet-1K default recipe requires timm==1.0.26") from error
    mixup = TimmMixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=0.0,
        prob=1.0,
        switch_prob=0.0,
        mode="batch",
        label_smoothing=LABEL_SMOOTHING,
        num_classes=task.num_classes,
    )
    return BatchMixupAdapter(
        cast("BatchMixup", cast("object", mixup)),
        cast("Callable[[Tensor, int, float, float], Tensor]", mixup_target),
        task.num_classes,
        LABEL_SMOOTHING,
    )


def _build_optimizer(
    model: nn.Module, learning_rate: float, device: torch.device
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    kwargs: dict[str, Any] = {"lr": learning_rate, "weight_decay": WEIGHT_DECAY}
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        **kwargs,
    )


def _compile_model_in_place(model: nn.Module, options: ExecutionOptions) -> None:
    """Compile only the module call path, preserving the module and parameter objects."""
    if not options.compile_model:
        return
    compile_module = getattr(model, "compile", None)
    if not callable(compile_module):
        raise RuntimeError(  # noqa: TRY004
            "this PyTorch version does not support nn.Module.compile"
        )
    torch._dynamo.config.recompile_limit = 64  # pyright: ignore[reportPrivateUsage]  # Match measured compile limit.
    compile_module(mode="default", dynamic=False)


def _learning_rate(task: BaselineTask, global_step: int, steps_per_epoch: int) -> float:
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = task.epochs * steps_per_epoch
    completed = global_step + 1
    if completed <= warmup_steps:
        return task.learning_rate * completed / warmup_steps
    progress = min((completed - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return task.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _logits(output: object, batch_size: int, num_classes: int) -> Tensor:
    if not isinstance(output, Tensor) or output.shape != (batch_size, num_classes):
        raise RuntimeError(f"classifier logits must have shape {(batch_size, num_classes)}")
    return output


class _CudaBatchTransfer:
    """Queue pinned host batches on a copy stream and retain them until DMA completes."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.pending_hosts: list[tuple[torch.cuda.Event, tuple[Tensor, ...]]] = []

    def copy(self, batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        with torch.cuda.stream(self.stream):
            copied = tuple(_to_device(value, self.device, non_blocking=True) for value in batch)
            event = torch.cuda.Event()
            event.record(self.stream)
        self.pending_hosts = [
            (pending, host_batch)
            for pending, host_batch in self.pending_hosts
            if not pending.query()
        ]
        self.pending_hosts.append((event, batch))
        return copied

    def wait(self, *tensors: Tensor) -> None:
        compute_stream = torch.cuda.current_stream(self.device)
        compute_stream.wait_stream(self.stream)
        for tensor in tensors:
            tensor.record_stream(compute_stream)


def _move_batch(batch: tuple[Tensor, ...], device: torch.device) -> tuple[Tensor, ...]:
    return tuple(_to_device(value, device) for value in batch)


def _to_device(value: Tensor, device: torch.device, *, non_blocking: bool = False) -> Tensor:
    if value.ndim == 4:
        return value.to(
            device=device,
            non_blocking=non_blocking,
            memory_format=torch.channels_last,
        )
    return value.to(device=device, non_blocking=non_blocking)


def _step_callback_for_epoch(
    callback: Callable[[dict[str, Any]], None] | None, epoch: int
) -> Callable[[dict[str, Any]], None] | None:
    if callback is None:
        return None

    def report_step(record: dict[str, Any]) -> None:
        callback({"epoch": epoch, **record})

    return report_step


class TrainingInterrupted(RuntimeError):
    """Stop at an optimizer boundary without validation or a new recovery point."""


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    mixup: Mixup,
    task: BaselineTask,
    device: torch.device,
    global_step: int,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
    epoch: int = 1,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[dict[str, float], int]:
    if stop_requested is not None and stop_requested():
        raise TrainingInterrupted("stop requested before optimizer update")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = task.gradient_accumulation_steps
    batch_count = len(loader)
    steps_per_epoch = math.ceil(batch_count / accumulation)
    loss_sum = 0.0
    correct = total = optimizer_steps = window_start = window_size = 0
    metric_loss = torch.zeros((), device=device, dtype=torch.float64)
    metric_correct = torch.zeros((), device=device, dtype=torch.int64)
    gpu_mixup = task.execution.gpu_mixup and device.type == "cuda"
    async_transfer = task.execution.non_blocking_transfer and device.type == "cuda"
    transfer = _CudaBatchTransfer(device) if async_transfer else None
    iterator = iter(loader)

    def next_batch() -> tuple[Tensor, ...]:
        raw_inputs, raw_targets = next(iterator)
        if gpu_mixup:
            batch = (raw_inputs, raw_targets)
        else:
            mixed_inputs, mixed_targets = mixup(raw_inputs, raw_targets)
            batch = (mixed_inputs, raw_targets, mixed_targets)
        if transfer is not None:
            return transfer.copy(batch)
        return _move_batch(batch, device)

    prepared = next_batch() if batch_count and transfer is not None else None
    for index in range(batch_count):
        if transfer is None:
            prepared = next_batch()
        if prepared is None:
            raise RuntimeError("training loader ended before its reported length")
        if transfer is not None:
            transfer.wait(*prepared)
        if gpu_mixup:
            inputs, hard_targets = prepared
            inputs, soft_targets = mixup(inputs, hard_targets)
        else:
            inputs, hard_targets, soft_targets = prepared
        if index == window_start:
            window_size = min(accumulation, batch_count - window_start)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and task.precision == "bfloat16",
        ):
            logits = _logits(model(inputs), hard_targets.numel(), task.num_classes)
            loss = (
                -(soft_targets.float() * functional.log_softmax(logits.float(), dim=-1))
                .sum(dim=-1)
                .mean()
            )
        (loss / window_size).backward()
        batch_examples = hard_targets.numel()
        metric_loss.add_(loss.detach().to(torch.float64), alpha=batch_examples)
        metric_correct.add_(logits.detach().argmax(dim=-1).eq(hard_targets).sum())
        total += batch_examples

        optimizer_boundary = index + 1 == window_start + window_size
        if optimizer_boundary:
            optimizer_steps += 1
            schedule_step = (epoch - 1) * steps_per_epoch + optimizer_steps
            for group in optimizer.param_groups:
                group["lr"] = _learning_rate(task, schedule_step - 1, steps_per_epoch)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            window_start = index + 1
            if stop_requested is not None and stop_requested():
                raise TrainingInterrupted("stop requested after optimizer update; no validation/checkpoint")

        # Queue the complete update before a loader wait. Copying the next batch
        # still overlaps GPU work and precedes periodic CPU metric reads.
        if transfer is not None:
            prepared = next_batch() if index + 1 < batch_count else None

        sync_interval = task.execution.metric_sync_interval
        sync_metrics = sync_interval == 0 or (
            optimizer_boundary and optimizer_steps % sync_interval == 0
        )
        callback_due = (
            optimizer_boundary and step_callback is not None and optimizer_steps % 100 == 0
        )
        if sync_metrics or callback_due:
            loss_sum = float(metric_loss.item())
            correct = int(metric_correct.item())
        if callback_due and step_callback is not None:
            step_callback(
                {
                    "global_step": global_step,
                    "optimizer_steps": optimizer_steps,
                    "loss": loss_sum / total,
                    "examples": total,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    # A positive sync interval intentionally defers all other metric reads to this point.
    loss_sum = float(metric_loss.item())
    correct = int(metric_correct.item())
    if not total or not math.isfinite(loss_sum):
        raise RuntimeError("training loader produced an invalid loss")
    return {
        "loss": loss_sum / total,
        "mixed_accuracy": correct / total,
        "examples": float(total),
        "optimizer_steps": float(optimizer_steps),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }, global_step


def _evaluate(
    model: nn.Module, loader: DataLoader[Any], task: BaselineTask, device: torch.device,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, float]:
    model.eval()
    metric_loss = torch.zeros((), device=device, dtype=torch.float64)
    metric_correct = torch.zeros((), device=device, dtype=torch.int64)
    metric_top5 = torch.zeros((), device=device, dtype=torch.int64)
    total = 0
    async_transfer = task.execution.non_blocking_transfer and device.type == "cuda"
    transfer = _CudaBatchTransfer(device) if async_transfer else None
    iterator = iter(loader)

    def next_batch() -> tuple[Tensor, ...]:
        inputs, targets = next(iterator)
        batch = (inputs, targets)
        return transfer.copy(batch) if transfer is not None else _move_batch(batch, device)

    batch_count = len(loader)
    prepared = next_batch() if batch_count and transfer is not None else None
    with torch.inference_mode():
        for index in range(batch_count):
            if stop_requested is not None and stop_requested():
                raise TrainingInterrupted("stop requested during validation")
            if transfer is None:
                prepared = next_batch()
            if prepared is None:
                raise RuntimeError("validation loader ended before its reported length")
            if transfer is not None:
                transfer.wait(*prepared)
            inputs, targets = prepared
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and task.precision == "bfloat16",
            ):
                logits = _logits(model(inputs), targets.numel(), task.num_classes).float()
            metric_loss.add_(
                functional.cross_entropy(logits, targets, reduction="sum").to(torch.float64)
            )
            metric_correct.add_(logits.argmax(dim=-1).eq(targets).sum())
            metric_top5.add_(
                logits.topk(min(5, task.num_classes), dim=-1)
                .indices.eq(targets[:, None])
                .any(dim=-1)
                .sum()
            )
            total += targets.numel()
            if transfer is not None:
                prepared = next_batch() if index + 1 < batch_count else None
    loss_sum = float(metric_loss.item())
    correct = int(metric_correct.item())
    top5_correct = int(metric_top5.item())
    if not total or not math.isfinite(loss_sum):
        raise RuntimeError("validation loader produced an invalid loss")
    return {
        "accuracy": correct / total,
        "top5_accuracy": top5_correct / total,
        "cross_entropy": loss_sum / total,
        "examples": float(total),
    }


def _capture_rng(bundle: LoaderBundle, device: torch.device) -> dict[str, Any]:
    state = cast(
        "tuple[str, np.ndarray[Any, Any], int, int, float]",
        cast("object", np.random.get_state()),
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": state[0],
            "state": state[1].tolist(),
            "position": state[2],
            "has_gauss": state[3],
            "cached_gaussian": state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "train_generator": bundle.train_generator.get_state(),
        "validation_generator": bundle.validation_generator.get_state(),
    }


def _restore_rng(payload: dict[str, Any], bundle: LoaderBundle, device: torch.device) -> None:
    random.setstate(payload["python"])
    state = payload["numpy"]
    np.random.set_state(
        (
            state["bit_generator"],
            np.asarray(state["state"], dtype=np.uint32),
            int(state["position"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(payload["torch"])
    if device.type == "cuda" and payload["cuda"]:
        torch.cuda.set_rng_state_all(payload["cuda"])
    bundle.train_generator.set_state(payload["train_generator"])
    bundle.validation_generator.set_state(payload["validation_generator"])


def _validate_binding(payload: dict[str, Any], contract: dict[str, Any]) -> None:
    expected = {
        "schema": SCHEMA,
        "contract_sha256": _sha256_payload(contract),
        "task_sha256": contract["task_sha256"],
        "source_digest_sha256": contract["source_digest_sha256"],
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RuntimeError(
            "checkpoint belongs to an incompatible historical ImageNet-1K worker contract"
        )


def _history_is_consistent(history: object, completed_epochs: int, global_step: int) -> bool:
    return bool(
        isinstance(history, list)
        and len(history) == completed_epochs
        and all(
            isinstance(row, dict) and row.get("epoch") == index
            for index, row in enumerate(history, 1)
        )
        and (
            completed_epochs == 0
            or (isinstance(history[-1], dict) and history[-1].get("global_step") == global_step)
        )
    )


def _validate_checkpoint(
    payload: dict[str, Any], contract: dict[str, Any], task: BaselineTask
) -> None:
    _validate_binding(payload, contract)
    completed_epochs = payload.get("completed_epochs")
    global_step = payload.get("global_step")
    if (
        not isinstance(completed_epochs, int)
        or not 0 <= completed_epochs <= task.epochs
        or not isinstance(global_step, int)
        or global_step < 0
        or not _history_is_consistent(payload.get("history"), completed_epochs, global_step)
        or not isinstance(payload.get("model"), dict)
        or not isinstance(payload.get("optimizer"), dict)
        or not isinstance(payload.get("rng"), dict)
    ):
        raise RuntimeError("checkpoint is structurally incomplete")


def _validate_result(payload: dict[str, Any], contract: dict[str, Any], task: BaselineTask) -> None:
    _validate_binding(payload, contract)
    if (
        payload.get("status") != "completed"
        or payload.get("completed_epochs") != task.epochs
        or not isinstance(payload.get("global_step"), int)
        or not _history_is_consistent(payload.get("history"), task.epochs, payload["global_step"])
        or not isinstance(payload.get("final_validation"), dict)
    ):
        raise RuntimeError("result artifact is structurally incomplete")


def _run_task(
    task: BaselineTask,
    *,
    device: torch.device | None = None,
    model_builder: Callable[[str, str | Path | None, int], nn.Module],
    loader_builder: Callable[[BaselineTask, torch.device], LoaderBundle] = _build_loaders,
    mixup_factory: Callable[[BaselineTask], Mixup] = _make_mixup,
    logger_factory: Callable[[BaselineTask, dict[str, object]], Logger] = _make_logger,
    optimizer_builder: Callable[[nn.Module, float, torch.device], torch.optim.Optimizer]
    | None = None,
    epoch_callback: Callable[[nn.Module, dict[str, Any]], None] | None = None,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    stop_without_checkpoint: bool = False,
) -> dict[str, Any]:
    """Run or exact-contract resume a local, authoritative LNet task."""
    _validate_task(task)
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = _contract(task, resolved_device)
    contract_sha256 = _sha256_payload(contract)
    contract_path = task.output_dir / "contracts" / f"{task.task_name}.json"
    if (
        contract_path.exists()
        and _sha256_payload(json.loads(contract_path.read_text())) != contract_sha256
    ):
        raise RuntimeError("existing contract changed for this ImageNet-1K task")
    if task.result_path.exists():
        result = json.loads(task.result_path.read_text())
        if not isinstance(result, dict):
            raise RuntimeError("result artifact is not a JSON object")
        if not contract_path.exists():
            raise RuntimeError("result artifact has no contract")
        _validate_result(result, contract, task)
        return result
    if task.resume and not task.checkpoint_path.exists():
        raise RuntimeError("resume was requested but the checkpoint is missing")
    if not task.resume and task.checkpoint_path.exists():
        raise RuntimeError("resume flag and checkpoint presence disagree")
    checkpoint: dict[str, Any] | None = None
    if task.resume:
        candidate = torch.load(task.checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(candidate, dict):
            raise RuntimeError("checkpoint payload is not an object")
        if not contract_path.exists():
            raise RuntimeError("checkpoint has no contract")
        _validate_checkpoint(candidate, contract, task)
        checkpoint = candidate
    if not contract_path.exists():
        _atomic_json(contract_path, contract)
    logger = logger_factory(task, cast("dict[str, object]", contract))
    random.seed(task.seed)
    np.random.seed(task.seed)
    torch.manual_seed(task.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(task.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    model = model_builder(task.model_key, task.source_root, task.num_classes).to(resolved_device)
    model = model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    _compile_model_in_place(model, task.execution)
    optimizer = (optimizer_builder or _build_optimizer)(model, task.learning_rate, resolved_device)
    loaders = loader_builder(task, resolved_device)
    mixup = mixup_factory(task)
    history: list[dict[str, Any]] = []
    start_epoch = global_step = 0
    elapsed_before_resume = 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch, global_step = (
            int(checkpoint["completed_epochs"]),
            int(checkpoint["global_step"]),
        )
        elapsed_before_resume = float(checkpoint["training_seconds"])
        _restore_rng(checkpoint["rng"], loaders, resolved_device)
        # A checkpoint is committed before its derived epoch artifacts. Repair
        # those artifacts if interruption occurred between the two writes.
        if epoch_callback is not None and history:
            epoch_callback(model, history[-1])
    started = time.monotonic()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    for epoch in range(start_epoch + 1, task.epochs + 1):
        active_step_callback = _step_callback_for_epoch(step_callback, epoch)
        train, global_step = _train_epoch(
            model,
            loaders.train,
            optimizer,
            mixup,
            task,
            resolved_device,
            global_step,
            active_step_callback,
            epoch=epoch,
            stop_requested=stop_requested if stop_without_checkpoint else None,
        )
        if stop_without_checkpoint and stop_requested is not None and stop_requested():
            raise TrainingInterrupted("stop requested before validation; no new checkpoint")
        if stop_without_checkpoint:
            validation = _evaluate(model, loaders.validation, task, resolved_device, stop_requested=stop_requested)
        else:
            validation = _evaluate(model, loaders.validation, task, resolved_device)
        if stop_without_checkpoint and stop_requested is not None and stop_requested():
            raise TrainingInterrupted("stop requested before checkpoint")
        training_seconds = elapsed_before_resume + time.monotonic() - started
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train,
            "validation": validation,
            "training_seconds": training_seconds,
        }
        history.append(row)
        checkpoint = {
            "schema": SCHEMA,
            "contract_sha256": contract_sha256,
            "task_sha256": contract["task_sha256"],
            "source_digest_sha256": contract["source_digest_sha256"],
            "source_sha256": contract["source_sha256"],
            "completed_epochs": epoch,
            "global_step": global_step,
            "parameters": parameters,
            "history": history,
            "training_seconds": training_seconds,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": _capture_rng(loaders, resolved_device),
        }
        _atomic_torch(task.checkpoint_path, checkpoint)
        if epoch_callback is not None:
            epoch_callback(model, row)
        logger.log(
            {
                "kind": "epoch",
                "epoch": epoch,
                "global_step": global_step,
                "train": train,
                "validation": validation,
            }
        )
        _emit_progress(
            task,
            {
                "schema": SCHEMA,
                "status": "running",
                "model_key": task.model_key,
                "seed": task.seed,
                "epoch": epoch,
                "epochs": task.epochs,
                "global_step": global_step,
                "validation_accuracy": validation["accuracy"],
                "checkpoint_path": str(task.checkpoint_path.resolve()),
                "contract_sha256": contract_sha256,
            },
        )
        if stop_requested is not None and stop_requested() and epoch < task.epochs:
            paused: dict[str, object] = {
                "status": "paused",
                "model_key": task.model_key,
                "completed_epochs": epoch,
                "global_step": global_step,
                "checkpoint_path": str(task.checkpoint_path),
            }
            logger.finish(paused)
            return paused
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "model_key": task.model_key,
        "seed": task.seed,
        "parameters": parameters,
        "completed_epochs": task.epochs,
        "global_step": global_step,
        "training_seconds": elapsed_before_resume + time.monotonic() - started,
        "final_validation": _evaluate(model, loaders.validation, task, resolved_device),
        "history": history,
        "contract_sha256": contract_sha256,
        "task_sha256": contract["task_sha256"],
        "source_digest_sha256": contract["source_digest_sha256"],
        "source_sha256": contract["source_sha256"],
    }
    _atomic_json(task.result_path, result)
    logger.finish(cast("dict[str, object]", result))
    _emit_progress(
        task,
        {
            "schema": SCHEMA,
            "status": "completed",
            "model_key": task.model_key,
            "seed": task.seed,
            "epoch": task.epochs,
            "epochs": task.epochs,
            "global_step": global_step,
            "result_path": str(task.result_path.resolve()),
            "contract_sha256": contract_sha256,
        },
    )
    print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)
    return result


def run_task(
    task: BaselineTask,
    *,
    device: torch.device | None = None,
    model_builder: Callable[[str, str | Path | None, int], nn.Module],
    loader_builder: Callable[[BaselineTask, torch.device], LoaderBundle] = _build_loaders,
    mixup_factory: Callable[[BaselineTask], Mixup] = _make_mixup,
    logger_factory: Callable[[BaselineTask, dict[str, object]], Logger] = _make_logger,
    optimizer_builder: Callable[[nn.Module, float, torch.device], torch.optim.Optimizer]
    | None = None,
    epoch_callback: Callable[[nn.Module, dict[str, Any]], None] | None = None,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    stop_without_checkpoint: bool = False,
) -> dict[str, Any]:
    """Run a task and close its logger on successful, failed, or interrupted execution."""
    active_logger: Logger | None = None

    def tracked_logger(task: BaselineTask, contract: dict[str, object]) -> Logger:
        nonlocal active_logger
        active_logger = logger_factory(task, contract)
        return active_logger

    try:
        return _run_task(
            task,
            device=device,
            model_builder=model_builder,
            loader_builder=loader_builder,
            mixup_factory=mixup_factory,
            logger_factory=tracked_logger,
            optimizer_builder=optimizer_builder,
            epoch_callback=epoch_callback,
            step_callback=step_callback,
            stop_requested=stop_requested,
            stop_without_checkpoint=stop_without_checkpoint,
        )
    except BaseException as error:
        if active_logger is not None:
            status = (
                "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit, TrainingInterrupted)) else "failed"
            )
            with suppress(BaseException):
                active_logger.finish(
                    {"kind": status, "status": status, "model_key": task.model_key}
                )
        raise


__all__ = [
    "EFFECTIVE_BATCH_SIZE",
    "SCHEMA",
    "BaselineTask",
    "ExecutionOptions",
    "LoaderBundle",
    "run_task",
]
