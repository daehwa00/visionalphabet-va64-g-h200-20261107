# ruff: noqa: ANN001, ANN201, ANN202, ANN401, C901, E501, PLC0415, PLR0912, PLR0915, PTH105, SIM102, SLF001, T201
"""Standalone ImageNet-100 training, evaluation, and compile runtime for VA-Ablation-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import imagenet100_checkpoint_runtime as checkpoint_runtime
import imagenet100_data_runtime as data_runtime
import numpy as np
import torch
from alphabet_model_builder import model_config as default_model_config
from h200_telemetry import Telemetry
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import parametrize

from alphabet.capture_safe_orthogonal import prepare_capture_safe_orthogonal_
from alphabet.classifier_heads import AffineCorrelationClassifier, expected_calibration_error

if TYPE_CHECKING:
    from collections.abc import Callable, Sized

    from torch.utils.data import DataLoader
    from wandb.sdk.wandb_run import Run as WandbRun

_CUDA_GRAPH_COMPILE_MODES = data_runtime.CUDA_GRAPH_COMPILE_MODES
atomic_json = checkpoint_runtime.atomic_json
atomic_torch = checkpoint_runtime.atomic_torch


@dataclass(frozen=True, slots=True)
class RunnerBindings:
    """Explicit experiment callbacks consumed by the ImageNet-100 runner."""

    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    model_config: Callable[..., Any]
    build_model: Callable[..., nn.Module]
    contract: Callable[[argparse.Namespace], dict[str, Any]]
    build_optimizer: Callable[..., torch.optim.Optimizer]
    prepare_model: Callable[[nn.Module, dict[str, Any]], nn.Module]
    train_epoch: Callable[..., dict[str, float]]
    evaluate: Callable[..., dict[str, float]]
    wandb_model_metrics: Callable[[nn.Module], dict[str, float]]
    summarize: Callable[[Path, dict[str, Any]], dict[str, Any] | None]
    calibrated_validation: bool = False
    close_evaluator: Callable[[], None] | None = None


def parse_args(*, variants: tuple[str, ...], seeds: tuple[int, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--variants", choices=variants, nargs="+", default=list(variants))
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(seeds))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Checkpoint and stop at this epoch without changing the schedule",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--effective-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument(
        "--fast-gpu-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefetch uint8 training images and normalize/erase on CUDA",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    if len(variants) == 1 and variants[0] in {
        "pico-600-opt",
        "VA-Ablation-pico-v11-in1k",
        "VA-Ablation-pico-v11-in1k-lightA",
        "VA-Ablation-pico-v11-in1k-lightB",
        "VA-Ablation-lower-nano-v6-in1k",
    }:
        parser.add_argument("--compact-manifest", action="store_true")
        parser.add_argument(
            "--clean-loader-workers",
            action="store_true",
            help="Use mmap manifests and start workers from a clean fork server",
        )
        parser.add_argument("--gpu-randaugment", action="store_true")
        parser.add_argument("--tensor-crop", action="store_true")
        parser.add_argument("--compile", action="store_true")
        parser.add_argument(
            "--compile-mode",
            default="default",
            choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        )
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-dense-from", type=int, default=85)
    parser.add_argument("--dual-eval", action="store_true")
    parser.add_argument(
        "--val-persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the existing epoch checkpoint (also automatic)",
    )
    args = parser.parse_args()
    if getattr(args, "clean_loader_workers", False) and not args.compact_manifest:
        parser.error("--clean-loader-workers requires --compact-manifest")
    if getattr(args, "gpu_randaugment", False) and not args.fast_gpu_augment:
        parser.error("--gpu-randaugment requires --fast-gpu-augment")
    if getattr(args, "tensor_crop", False) and not args.fast_gpu_augment:
        parser.error("--tensor-crop requires --fast-gpu-augment")
    for option in (
        "epochs",
        "eval_every",
        "eval_dense_from",
        "checkpoint_every",
        "progress_every",
        "prefetch_factor",
    ):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch < args.epochs:
        parser.error("--stop-after-epoch must be between 1 and --epochs minus 1")
    # Preserve explicit historical microbatch commands. A plain invocation now targets 1024.
    resume_recipe = {}
    if (
        args.batch_size is None
        and args.gradient_accumulation_steps is None
        and args.effective_batch_size is None
    ):
        contract_path = args.root / "contract.json"
        if contract_path.is_file():
            resume_recipe = json.loads(contract_path.read_text()).get("recipe", {})
    nominal = (
        args.batch_size if args.batch_size is not None else resume_recipe.get("batch_size", 128)
    )
    if nominal <= 0:
        parser.error("--batch-size must be positive")
    target = args.effective_batch_size
    if target is None and args.batch_size is None and args.gradient_accumulation_steps is None:
        target = (
            nominal * resume_recipe["gradient_accumulation_steps"]
            if "gradient_accumulation_steps" in resume_recipe
            else 1024
        )
    if target is not None:
        if target <= 0 or target % nominal:
            parser.error("--effective-batch-size must be a positive multiple of --batch-size")
        if (
            args.gradient_accumulation_steps is not None
            and nominal * args.gradient_accumulation_steps != target
        ):
            parser.error("effective batch conflicts with batch size and gradient accumulation")
    args.batch_size = nominal
    args.gradient_accumulation_steps = (
        target // nominal
        if target is not None
        else (
            args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else 1
        )
    )
    return args


def base_contract(
    args: argparse.Namespace,
    *,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    num_classes: int = 100,
) -> dict[str, Any]:
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    loader_workers = data_runtime.active_loader_workers(args.workers)
    loader_persistent_workers = data_runtime.persistent_loader_workers(loader_workers)
    data_digest, train_count, validation_count = data_runtime.dataset_digest(args.data_root)
    config = default_model_config(output_dim=num_classes)
    reduced_weight_decay = variants == ("VA-Ablation-17-p1-nr4-r1-1024d-v3",)
    relaxed = reduced_weight_decay or variants in {
        ("VA-Ablation-scale2-pico-1m",),
        ("VA-Ablation-lower-nano-v2",),
        ("VA-Ablation-lower-nano-v3",),
        ("VA-Ablation-lower-nano-v4",),
        ("VA-Ablation-lower-nano-v5",),
        ("VA-Ablation-lower-nano-v6",),
        ("VA-Ablation-lower-nano-v7",),
        ("VA-Ablation-lower-nano-v8",),
        ("VA-Ablation-lower-nano-v9",),
        ("VA-Ablation-lower-nano-v10",),
        ("VA-Ablation-lower-nano-v11",),
        ("VA-Ablation-lower-nano-v12",),
        ("VA-Ablation-lower-nano-v13",),
        ("VA-Ablation-lower-nano-v14",),
        ("VA-Ablation-lower-nano-v15",),
        ("VA-Ablation-scale1-depth-5m",),
        ("VA-Ablation-17-p1-nr4-r1b-1024d-v3",),
        ("VA-Ablation-17-p1-nr4-r1b-cm04-1024d-v3",),
        ("VA-Ablation-17-p1-nr4-r1b-cm04-ema9992-1024d-v3",),
        ("VA-Ablation-17-p1-nr4-r1b-cm10-ema9998-1024d-v3",),
        ("VA-Ablation-17-p1-nr4-r1b-cm04-dp01-ema9998-1024d-v3",),
        ("VA-Ablation-17-p1-nr4-r1b-cm04-dp01-ep200-ema99985-1024d-v3",),
    }
    return {
        "variants": list(variants),
        "priority": list(variants),
        "seeds": list(seeds),
        "num_classes": num_classes,
        "model": asdict(config),
        "recipe": {
            "epochs": args.epochs,
            "planned_epochs": args.epochs,
            "schedule_horizon_epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "optimizer": "AdamW (fused, pole-aware parameter groups)",
            "fused_optimizer": True,
            "learning_rate": 3.0e-3,
            "modal_learning_rate_multiplier": 1.0 / 3.0,
            "pole_geometry_learning_rate_multiplier": 1.0,
            "weight_decay": 0.03 if reduced_weight_decay else 0.05,
            "warmup_epochs": 5,
            "schedule": "warmup plus cosine",
            "label_smoothing": 0.05 if relaxed else 0.1,
            "mixup_alpha": 0.4 if relaxed else 0.8,
            "precision": args.precision,
            "compile_mode": "default",
            "compiled_training_preparation": True,
            "channels_last": True,
            "emulate_precision_casts": True,
            "matmul_precision": "high (TF32 enabled)",
            "loader_workers": loader_workers,
            "loader_persistent_workers": loader_persistent_workers,
            "validation_loader_persistent_workers": bool(
                args.val_persistent_workers and loader_workers
            ),
            "validation_loader_workers": min(4, loader_workers),
            "eval_every": args.eval_every,
            "eval_dense_from": args.eval_dense_from,
            "dual_eval": args.dual_eval,
            "final_dual_eval": True,
            "checkpoint_every": args.checkpoint_every,
            "progress_every": args.progress_every,
            "history_jsonl": "all epochs; reconcile to checkpoint on resume",
            "evaluation_rng": "forked; preserve historical validation iterator RNG advance at each epoch boundary",
            "loader_prefetch_factor": args.prefetch_factor if loader_workers else None,
            "fast_gpu_augment": args.fast_gpu_augment,
            "cpu_affinity": os.environ.get("LNET_CPU_AFFINITY_ACTIVE"),
            "device_prefetch_stream": True,
            "device_prefetch_scope": "copy_and_augment" if args.fast_gpu_augment else "copy_only",
            "fused_h2d_channels_last": False,
            "augmentation": "matched A2D ImageNet-100 public recipe"
            + (" with uint8 CUDA prefetch" if args.fast_gpu_augment else ""),
            "selection": f"fixed epoch {args.epochs}; no within-run validation selection",
            "resume": "epoch-boundary exact RNG restore",
        },
        "data": {
            "manifest_sha256": data_digest,
            "root": str(args.data_root),
            "classes": num_classes,
            "train_images": train_count,
            "validation_images": validation_count,
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "source_sha256": {},
    }


def _initialize(root: Path, contract: dict[str, Any]) -> None:
    path = root / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(contract)):
            raise RuntimeError("existing experiment root has a different immutable contract")
    else:
        atomic_json(path, contract)


def make_scheduler(optimizer, recipe, steps_per_epoch):
    warmup_epochs = int(recipe.get("warmup_epochs", 5))
    if "min_lr" not in recipe:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: _learning_rate_factor(epoch, recipe["epochs"], warmup_epochs),
        )
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = recipe["epochs"] * steps_per_epoch

    # Scale all groups together to preserve the pole-aware LR multipliers.
    floor = recipe["min_lr"] / recipe["learning_rate"]

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps + 1) / (total_steps - warmup_steps))
        return floor + (1.0 - floor) * (1.0 + math.cos(math.pi * progress)) / 2.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _learning_rate_factor(epoch: int, epochs: int, warmup_epochs: int = 5) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _mixed_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    permuted_targets: Tensor,
    mixing: float | Tensor,
    *,
    label_smoothing: float,
) -> Tensor:
    if torch.is_tensor(mixing):
        first = functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=label_smoothing
        )
        second = functional.cross_entropy(
            logits, permuted_targets, reduction="none", label_smoothing=label_smoothing
        )
        return (mixing * first + (1.0 - mixing) * second).mean()
    if mixing == 1.0:
        return functional.cross_entropy(logits, targets, label_smoothing=label_smoothing)
    return mixing * functional.cross_entropy(
        logits,
        targets,
        label_smoothing=label_smoothing,
    ) + (1.0 - mixing) * functional.cross_entropy(
        logits,
        permuted_targets,
        label_smoothing=label_smoothing,
    )


def training_objective(
    model: nn.Module,
    output: Tensor | tuple[Tensor, ...],
    targets: Tensor,
    permuted_targets: Tensor,
    mixing: float | Tensor,
    *,
    label_smoothing: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    if not isinstance(output, tuple) or len(output) != 5:
        raise RuntimeError("affine Q-head model returned an invalid training tuple")
    joint, affine, _fusion, _lrq, _descriptor = output
    classifier = cast("AffineCorrelationClassifier", cast("Any", model).classifier)
    joint_loss = _mixed_cross_entropy(
        joint, targets, permuted_targets, mixing, label_smoothing=label_smoothing
    )
    auxiliary = (
        _mixed_cross_entropy(
            affine, targets, permuted_targets, mixing, label_smoothing=label_smoothing
        )
        if classifier.affine_auxiliary_weight > 0.0
        else joint_loss.detach()
    )
    loss = joint_loss + classifier.affine_auxiliary_weight * auxiliary
    return joint, loss, {"joint_ce": joint_loss, "affine_aux_ce": auxiliary}


def train_epoch(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mixup_generator: Any,
    mixup_alpha: float,
    label_smoothing: float,
    precision: str,
    gradient_accumulation_steps: int = 1,
    channels_last: bool = False,
) -> dict[str, float]:
    """Run one ImageNet-100 MixUp training epoch."""
    from automatic_training_layout import training_microbatch

    if hasattr(runtime, "logical_update"):
        from effective_batch_epoch import train_epoch as effective_epoch

        return effective_epoch(
            model,
            runtime,
            loader,
            device=device,
            mixup_generator=mixup_generator,
            mixup_alpha=mixup_alpha,
            label_smoothing=label_smoothing,
            group_size=gradient_accumulation_steps,
            channels_last=channels_last,
        )
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    model.train()
    runtime.train()
    loss_terms: list[Tensor] = []
    correct_terms: list[Tensor] = []
    diagnostics: dict[str, list[Tensor]] = {}
    count = 0
    batch_count = len(loader)
    batches = data_runtime.device_batches(loader, device, channels_last=channels_last)
    for batch_index, (inputs, targets) in enumerate(batches):
        group_offset = batch_index % gradient_accumulation_steps
        if group_offset == 0:
            optimizer.zero_grad(set_to_none=True)
        group_size = min(
            gradient_accumulation_steps,
            batch_count - (batch_index - group_offset),
        )
        permutation = torch.randperm(targets.numel(), device=device)
        permuted_targets = targets[permutation]
        mixing = float(mixup_generator.beta(mixup_alpha, mixup_alpha))
        mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
        with training_microbatch(
            runtime,
            mixed_inputs,
            targets,
            permuted_targets,
            mixing,
            group_size,
            group_offset,
        ) as active_runtime:
            data_runtime.begin_cudagraph_step(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bfloat16",
            ):
                output = active_runtime(mixed_inputs)
                logits, loss, active_diagnostics = training_objective(
                    model,
                    output,
                    targets,
                    permuted_targets,
                    mixing,
                    label_smoothing=label_smoothing,
                )
            (loss / group_size).backward()
            group_complete = group_offset + 1 == group_size or batch_index + 1 == batch_count
            if not group_complete and group_offset == 0 and device.type == "cuda":
                # AccumulateGrad can adopt buffers owned by the compiled backward.
                # Keep the accumulated gradients alive across the next graph step.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad = parameter.grad.clone()
            if group_complete:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        loss_terms.append(loss.detach() * targets.numel())
        correct_terms.append(logits.argmax(dim=-1).eq(targets).sum())
        for name, value in active_diagnostics.items():
            diagnostics.setdefault(name, []).append(value.detach() * targets.numel())
        del output, logits, loss, active_diagnostics
        count += targets.numel()
    result = {
        "loss": float(torch.stack(loss_terms).double().sum()) / count,
        "mixed_accuracy": int(torch.stack(correct_terms).sum()) / count,
    }
    cast("Any", model)._latest_training_diagnostics = {
        name: float(torch.stack(values).double().sum()) / count
        for name, values in diagnostics.items()
    }
    return result


def evaluate(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    """Evaluate ImageNet-100 validation accuracy, cross-entropy, and ECE."""
    classifier = cast("Any", model).classifier
    if not isinstance(classifier, AffineCorrelationClassifier):
        raise TypeError("expected AffineCorrelationClassifier for evaluation")
    model.eval()
    runtime.eval()
    outputs: list[list[Tensor]] = [[], [], [], [], []]
    labels: list[Tensor] = []
    with torch.inference_mode():
        for inputs, targets in data_runtime.device_batches(
            loader,
            device,
            channels_last=channels_last,
        ):
            data_runtime.begin_cudagraph_step(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bfloat16",
            ):
                output = runtime(inputs)
            if not isinstance(output, tuple) or len(output) != 5:
                raise RuntimeError("affine Q-head evaluation lost branch outputs")
            cpu_outputs: dict[int, Tensor] = {}
            for index, value in enumerate(output):
                identity = id(value)
                if identity not in cpu_outputs:
                    cpu_outputs[identity] = value.detach().float().cpu()
                outputs[index].append(cpu_outputs[identity])
            labels.append(targets.cpu())
    joint, affine, _fusion, _lrq, descriptor = [torch.cat(values, dim=0) for values in outputs]
    target = torch.cat(labels)
    cross_entropy = float(functional.cross_entropy(joint, target))
    result = {
        "accuracy": float(joint.argmax(dim=-1).eq(target).float().mean()),
        "cross_entropy": cross_entropy,
        "nll": cross_entropy,
        "ece": expected_calibration_error(joint, target),
        "samples": int(target.numel()),
    }
    if classifier.affine is not None:
        result["affine_only_accuracy"] = float(affine.argmax(dim=-1).eq(target).float().mean())
        result["affine_only_nll"] = float(functional.cross_entropy(affine.float(), target))
        with torch.inference_mode():
            standardized = classifier.affine.standardizer(descriptor.to(device))
            reconstructed = classifier.affine.linear(standardized).cpu()
        result["affine_reconstruction_max_error"] = float((affine - reconstructed).abs().max())
    cast("Any", model)._latest_evaluation_metrics = result
    return result


def prepare_model(model: nn.Module, recipe: dict[str, Any]) -> nn.Module:
    if bool(recipe.get("channels_last", False)):
        model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    prepare_capture_safe_orthogonal_(model)
    method = getattr(model, "prepare_for_compiled_training_", None)
    if callable(method):
        return cast("nn.Module", method())
    return model


def _exclude_capture_hostile_submodules(
    model: nn.Module,
    *,
    exclude_classifier: bool = False,
) -> list[str]:
    excluded: list[str] = []
    for name, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        if not getattr(module.forward, "_torchdynamo_disable", False):
            module.forward = torch.compiler.disable(module.forward)
        excluded.append(name)
    classifier = getattr(model, "classifier", None)
    if exclude_classifier and isinstance(classifier, nn.Module):
        if not getattr(classifier.forward, "_torchdynamo_disable", False):
            classifier.forward = torch.compiler.disable(classifier.forward)
        excluded.append("classifier")
    return excluded


def _has_replayed_cudagraph_stages(model: nn.Module) -> bool:
    if bool(getattr(model, "_lnet_requires_classic_cudagraph", False)):
        return True
    stage_count = sum(
        callable(getattr(module, "_damping_fields", None)) for module in model.modules()
    )
    return stage_count > 1


def _replayed_stage_compile_options(compile_mode: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "triton.cudagraphs": True,
        "triton.cudagraph_trees": False,
    }
    if compile_mode == "max-autotune":
        options.update({"coordinate_descent_tuning": True, "max_autotune": True})
    return options


def _build_runtime(model: nn.Module, recipe: dict[str, Any]) -> nn.Module:
    from alphabet.training_compiler import compile_training_model

    return compile_training_model(model) if recipe.get("compile_mode") else model


def _configure_compile_runtime(recipe: dict[str, Any]) -> str | None:
    compile_mode = os.environ.get("LNET_COMPILE_MODE") or recipe.get("compile_mode")
    if compile_mode is None:
        return None
    profile = json.dumps(
        {"dynamic": False, "fullgraph": False, "mode": str(compile_mode)},
        separators=(",", ":"),
        sort_keys=True,
    )
    os.environ["LNET_COMPILE_PROFILE"] = profile
    os.environ.setdefault("LNET_LAUNCH_CACHE", str(Path.home() / ".cache" / "lnet"))
    return profile


def _initialize_wandb_run(
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    parameters: int,
) -> WandbRun | None:
    project = os.environ.get("WANDB_PROJECT")
    if not project or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError as error:
        raise RuntimeError("WANDB_PROJECT is set but the wandb package is not installed") from error
    run_key = f"{root.resolve()}::{variant}::seed{seed}"
    run_id = os.environ.get("WANDB_RUN_ID") or hashlib.sha256(run_key.encode()).hexdigest()[:16]
    tracking_root = root / "wandb"
    tracking_root.mkdir(parents=True, exist_ok=True)
    variant_config = contract.get("variant_configs", {}).get(variant)
    return wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        group=os.environ.get("WANDB_GROUP", root.name),
        name=os.environ.get("WANDB_NAME", data_runtime.compact_wandb_name(variant)),
        id=run_id,
        resume="allow",
        dir=str(tracking_root),
        settings=wandb.Settings(
            init_timeout=float(os.environ.get("WANDB_INIT_TIMEOUT", "300")),
        ),
        config={
            "variant": variant,
            "seed": seed,
            "parameters": parameters,
            "model": variant_config or contract["model"],
            "model_template": contract["model"],
            "variant_config": variant_config,
            "recipe": contract["recipe"],
            "schema": contract["schema"],
        },
    )


def wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    metrics: dict[str, float] = {
        "model/parameters": float(sum(parameter.numel() for parameter in model.parameters())),
        "model/trainable_parameters": float(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    metrics.update(
        {
            f"train/{name}": float(value)
            for name, value in getattr(model, "_latest_training_diagnostics", {}).items()
        }
    )
    metrics.update(
        {
            f"validation/{name}": float(value)
            for name, value in getattr(model, "_latest_evaluation_metrics", {}).items()
            if name not in {"accuracy", "cross_entropy"}
        }
    )
    return metrics


def summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    rows = []
    for variant in contract["variants"]:
        for seed in contract["seeds"]:
            path = root / "results" / f"{variant}__seed{seed}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
    if not rows:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    summary = {
        "schema": contract["schema"],
        "completed_runs": len(rows),
        "variants": {
            variant: {
                "runs": len(active),
                "mean_accuracy": sum(row["final_validation"]["accuracy"] for row in active)
                / len(active),
                "mean_affine_only_accuracy": sum(
                    row["final_validation"].get("affine_only_accuracy", math.nan) for row in active
                )
                / len(active),
            }
            for variant, active in grouped.items()
        },
    }
    atomic_json(root / "summary.json", summary)
    return summary


def should_evaluate(epoch: int, *, epochs: int, eval_every: int, eval_dense_from: int) -> bool:
    return epoch in (1, epochs) or epoch % eval_every == 0 or epoch >= eval_dense_from


def _reconcile_history(path: Path, variant: str, seed: int, history: list) -> None:
    """Replace this job's rows with the authoritative checkpoint history."""
    lines = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # Interrupted append.
            if (row["variant"], row["seed"]) != (variant, seed):
                lines.append(line)
    lines.extend(json.dumps({"variant": variant, "seed": seed, **row}) for row in history)
    temporary = path.with_suffix(f".jsonl.tmp-{os.getpid()}")
    temporary.write_text("".join(line + "\n" for line in lines))
    os.replace(temporary, path)


def _run_job(
    root: Path,
    contract: dict[str, Any],
    **kwargs: Any,
) -> bool | None:
    with ExitStack() as cleanup:
        return _run_job_with_loaders(root, contract, cleanup, **kwargs)


def _run_job_with_loaders(
    root: Path,
    contract: dict[str, Any],
    cleanup: ExitStack,
    *,
    variant: str,
    seed: int,
    data_root: Path,
    workers: int,
    device: torch.device,
    bindings: RunnerBindings,
    stop_after_epoch: int | None = None,
) -> bool | None:
    result_path = root / "results" / f"{variant}__seed{seed}.json"
    if result_path.exists():
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = bindings.build_model(variant, bindings.model_config(**contract["model"])).to(device)
    recipe = contract["recipe"]
    model = bindings.prepare_model(model, recipe)
    training_generator = torch.Generator().manual_seed(seed)
    mixup_generator = np.random.default_rng(seed)
    train_loader, validation_loader = data_runtime.loaders(
        data_root,
        batch_size=recipe["batch_size"],
        validation_batch_size=min(128, recipe["batch_size"])
        if recipe.get("effective_batch_size") == 1024
        else None,
        workers=workers,
        training_generator=training_generator,
        val_persistent_workers=recipe["validation_loader_persistent_workers"],
        isolated_validation=True,
        prefetch_factor=recipe["loader_prefetch_factor"],
        fast_gpu_augment=recipe["fast_gpu_augment"],
        train_persistent_workers=recipe["loader_persistent_workers"],
        randaugment_magnitude=recipe.get("randaugment_magnitude", 9),
        randaugment_num_ops=recipe.get("randaugment_num_ops", 2),
        color_jitter=recipe.get("color_jitter", 0.0),
        random_erasing_prob=recipe.get("random_erasing_prob", 0.25),
        compact_manifest=recipe.get("compact_manifest", False),
        gpu_randaugment=recipe.get("gpu_randaugment", False),
        tensor_crop=recipe.get("tensor_crop", False),
        clean_loader_workers=recipe.get("clean_loader_workers", False),
    )
    cleanup.callback(data_runtime.close_persistent_loader, train_loader)
    cleanup.callback(data_runtime.close_persistent_loader, validation_loader)
    if bindings.close_evaluator is not None:
        cleanup.callback(bindings.close_evaluator)
    optimizer = bindings.build_optimizer(
        model, {**recipe, "progress_path": str(root / "optimizer-progress.json")}
    )
    gradient_accumulation_steps = int(recipe.get("gradient_accumulation_steps", 1))
    steps_per_epoch = checkpoint_runtime.optimizer_steps_per_epoch(
        train_loader,
        gradient_accumulation_steps,
    )
    step_schedule = "min_lr" in recipe
    if step_schedule:
        optimizer._va_track_gradient_norms = True
    scheduler = make_scheduler(optimizer, recipe, steps_per_epoch)
    if step_schedule:
        cleanup.callback(optimizer.register_step_post_hook(lambda *_: scheduler.step()).remove)
    checkpoint_path = root / "checkpoints" / f"{variant}__seed{seed}.pt"
    from va_model_ema import ModelEMA

    ema_decay = recipe.get("ema_decay")
    ema = (
        ModelEMA(model, ema_decay, warmup=recipe.get("ema_warmup", True))
        if ema_decay is not None
        else None
    )
    if ema is not None:
        cleanup.callback(ema.attach(optimizer).remove)
    progress = {"global_step": 0}
    start_epoch, history, training_seconds = checkpoint_runtime.restore_checkpoint(
        checkpoint_path,
        variant=variant,
        seed=seed,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_generator=training_generator,
        mixup_generator=mixup_generator,
        progress=progress,
        optimizer_steps_per_epoch=steps_per_epoch,
        ema=ema,
    )
    global_step = progress["global_step"]
    history_path = root / "history.jsonl"
    _reconcile_history(history_path, variant, seed, history)
    if stop_after_epoch is not None and start_epoch >= stop_after_epoch:
        return
    cuda_devices = [device] if device.type == "cuda" else []
    checkpoint_runtime.restore_optimizer_runtime_options(optimizer, recipe)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    wandb_run = _initialize_wandb_run(
        root, contract, variant=variant, seed=seed, parameters=parameters
    )
    runtime = _build_runtime(model, recipe)
    channels_last = bool(recipe.get("channels_last", False))
    telemetry = Telemetry(
        f"{variant}-seed{seed}", recipe["epochs"],
        torch.cuda.get_device_name(device) if cuda_devices else device.type,
    )
    telemetry.report("started", start_epoch, global_step)

    def evaluate_epoch(eval_model, engine, *, dual: bool) -> dict[str, float]:
        options = {"dual_eval": dual} if bindings.calibrated_validation else {}
        with torch.random.fork_rng(devices=cuda_devices):
            return bindings.evaluate(
                eval_model,
                engine,
                validation_loader,
                device,
                precision=recipe["precision"],
                channels_last=channels_last,
                **options,
            )

    start_epoch_step = global_step
    if step_schedule:
        torch.cuda.reset_peak_memory_stats(device)
    final_validation = raw_final_validation = None
    for epoch in range(start_epoch, recipe["epochs"]):
        epoch_number = epoch + 1
        final_epoch = epoch_number == recipe["epochs"]
        early_stop = epoch_number == stop_after_epoch
        optimizer._va_progress_epoch = (
            start_epoch_step,
            epoch_number,
            global_step + steps_per_epoch,
        )
        started = time.perf_counter()

        def observe_batch(batch_number, loss_terms, samples, current_epoch=epoch_number):
            if global_step == 0 and batch_number == min(gradient_accumulation_steps, len(train_loader)):
                print(f"epoch={current_epoch} first optimizer step completed", flush=True)
                telemetry.report("first_step", current_epoch, 1)
            if batch_number % recipe["progress_every"] == 0:
                print(
                    f"epoch={current_epoch} batch={batch_number} samples={samples} loss={float(torch.stack(loss_terms).sum()) / samples:.5f}",
                    flush=True,
                )

        optimizer._va_observe_batch = observe_batch
        train, optimizer_steps = checkpoint_runtime.train_epoch_with_step_count(
            bindings.train_epoch,
            model,
            runtime,
            train_loader,
            optimizer,
            device=device,
            mixup_generator=mixup_generator,
            mixup_alpha=recipe["mixup_alpha"],
            label_smoothing=recipe["label_smoothing"],
            precision=recipe["precision"],
            gradient_accumulation_steps=gradient_accumulation_steps,
            channels_last=channels_last,
        )
        if hasattr(optimizer, "_va_observe_batch"):
            del optimizer._va_observe_batch
        global_step += optimizer_steps
        # Epoch loss and accuracy materialization already waits for the training stream.
        epoch_training_seconds = time.perf_counter() - started
        training_seconds += epoch_training_seconds
        # v2's unseeded validation iterator consumed one CPU RNG draw. Keep that
        # epoch-boundary advance even when evaluation is skipped or persistent.
        torch.empty((), dtype=torch.int64, device="cpu").random_()
        validation = raw_validation = None
        if (
            early_stop
            or (epoch_number == 1 and recipe.get("evaluate_first_epoch"))
            or should_evaluate(
                epoch_number,
                epochs=recipe["epochs"],
                eval_every=recipe["eval_every"],
                eval_dense_from=recipe["eval_dense_from"],
            )
        ):
            dual = final_epoch or recipe["dual_eval"]
            validation = evaluate_epoch(model, model if step_schedule else runtime, dual=dual)
            if ema is not None:
                ema.sync_buffers(model)
                raw_validation = validation
                validation = evaluate_epoch(ema.model, ema.model, dual=dual)
            if step_schedule:
                if not (
                    validation["samples"]
                    == raw_validation["samples"]
                    == contract["data"]["validation_images"]
                ):
                    raise RuntimeError("Incomplete Raw/EMA validation")
            if final_epoch:
                final_validation = validation
                raw_final_validation = raw_validation
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train["loss"],
            "train_mixed_accuracy": train["mixed_accuracy"],
        }
        if variant in {
            "VA-Ablation-lower-nano-v2",
            "VA-Ablation-lower-nano-v3",
            "VA-Ablation-lower-nano-v4",
            "VA-Ablation-lower-nano-v5",
            "VA-Ablation-lower-nano-v6",
            "VA-Ablation-lower-nano-v7",
            "VA-Ablation-lower-nano-v8",
            "VA-Ablation-lower-nano-v9",
            "VA-Ablation-lower-nano-v10",
            "VA-Ablation-lower-nano-v11",
            "VA-Ablation-lower-nano-v12",
            "VA-Ablation-lower-nano-v13",
            "VA-Ablation-lower-nano-v14",
            "VA-Ablation-lower-nano-v15",
        }:
            row.update(
                training_seconds=epoch_training_seconds,
                ms_per_step=1000 * epoch_training_seconds / optimizer_steps,
                images_per_second=optimizer_steps
                * recipe["effective_batch_size"]
                / epoch_training_seconds,
            )
        if step_schedule:
            if ema.num_updates != global_step:
                raise RuntimeError("EMA and optimizer step counts diverged")
            row.update(
                global_step=global_step,
                training_seconds=epoch_training_seconds,
                images_per_second=train["samples"] / epoch_training_seconds,
                peak_vram_bytes=torch.cuda.max_memory_reserved(device),
                ema_num_updates=ema.num_updates,
                finite_gradients=train["finite_gradients"],
                validation_samples=validation["samples"] if validation else None,
            )
        if recipe.get("drop_path_rate", 0.0) or variant in {
            "VA-Ablation-scale1-depth-5m",
            "VA-Ablation-scale2-pico-1m",
        }:
            row["training_ms_per_step"] = 1000 * epoch_training_seconds / optimizer_steps
            row["training_images_per_second"] = len(train_loader.dataset) / epoch_training_seconds
        if validation is not None:
            row.update(
                validation_accuracy=validation["accuracy"],
                validation_cross_entropy=validation["cross_entropy"],
            )
        if raw_validation is not None:
            row.update(
                raw_validation_accuracy=raw_validation["accuracy"],
                raw_validation_cross_entropy=raw_validation["cross_entropy"],
            )
        history.append(row)
        _reconcile_history(history_path, variant, seed, history)
        print(json.dumps(row), flush=True)
        remote_stop = telemetry.should_stop(epoch_number, global_step, row)
        if wandb_run is not None:
            metrics = {
                "epoch": row["epoch"],
                "learning_rate": row["learning_rate"],
                "train/loss": row["train_loss"],
                "train/mixed_accuracy": row["train_mixed_accuracy"],
                "time/training_seconds": training_seconds,
                "global_step": global_step,
                "optimizer_steps/epoch": optimizer_steps,
            }
            if validation is not None:
                metrics["validation/accuracy"] = validation["accuracy"]
                metrics["validation/cross_entropy"] = validation["cross_entropy"]
                metrics.update(bindings.wandb_model_metrics(model))
            wandb_run.log(metrics, step=epoch + 1)
        if not step_schedule:
            scheduler.step()
        if not (
            remote_stop or early_stop or final_epoch or epoch_number % recipe["checkpoint_every"] == 0
        ):
            continue
        if ema is not None:
            ema.sync_buffers(model)
            if ema.num_updates != global_step:
                raise RuntimeError("EMA and optimizer step counts diverged")
        atomic_torch(
            checkpoint_path,
            {
                "variant": variant,
                "seed": seed,
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "training_seconds": training_seconds,
                "training_generator_state": training_generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if cuda_devices else [],
                "python_rng_state": random.getstate(),
                "mixup_rng_state": mixup_generator.bit_generator.state,
                **(ema.checkpoint() if ema is not None else {}),
            },
        )
        if remote_stop or early_stop:
            event = "stopped_by_kill_switch" if remote_stop else "stopped"
            if remote_stop:
                print(f"H200 kill switch requested at epoch {epoch_number}. Saved {checkpoint_path}", flush=True)
            telemetry.report(event, epoch_number, global_step, row)
            if wandb_run is not None:
                wandb_run.finish()
            return remote_stop
    if final_validation is None:
        # A final training checkpoint can exist without completed exports.
        final_validation = evaluate_epoch(model, runtime, dual=True)
        if ema is not None:
            ema.sync_buffers(model)
            raw_final_validation = final_validation
            final_validation = evaluate_epoch(ema.model, ema.model, dual=True)
    best_accuracy = max(
        row["validation_accuracy"] for row in history if "validation_accuracy" in row
    )
    from calibrated_evaluation import freeze_full_dataset_rms

    from alphabet.image_layers import LivePolewiseComplexRMS, fold_terminal_rms_for_inference

    models = [model, ema.model] if ema is not None else [model]
    moments = iter(
        freeze_full_dataset_rms(
            models,
            data_root,
            device,
            batch_size=recipe["batch_size"],
            workers=workers,
            prefetch_factor=recipe["loader_prefetch_factor"],
            precision="bfloat16",
            channels_last=channels_last,
            compact_manifest=recipe["compact_manifest"],
            clean_loader_workers=recipe["clean_loader_workers"],
        )
    )

    def export_rms(export_model, engine, suffix, validation):
        affine = getattr(getattr(export_model, "classifier", None), "affine", None)
        standardizer = getattr(affine, "standardizer", None)
        if isinstance(standardizer, LivePolewiseComplexRMS):
            precision = "float32"
            _, sample_count, stats = next(moments)
            if step_schedule and sample_count != contract["data"]["train_images"]:
                raise RuntimeError(f"incomplete clean RMS calibration: {sample_count}")
            export_metadata = {
                "variant": variant,
                "seed": seed,
                "epoch": recipe["epochs"],
                "frozen_pole_rms": {
                    "samples": sample_count,
                    "split": "train_clean_eval_transform",
                    "eps": standardizer.eps,
                    "output_scale": standardizer.output_scale,
                    "stats": stats,
                },
            }
            atomic_torch(
                checkpoint_path.with_name(f"{checkpoint_path.stem}{suffix}_frozen_clean_rms.pt"),
                {**export_metadata, "model": export_model.state_dict()},
            )
            if ema is not None:
                from calibrated_evaluation import verify_folded_affine

                with torch.random.fork_rng(devices=cuda_devices), ExitStack() as precision_scope:
                    if "clean_rms_precision" in recipe:
                        precision_scope.callback(
                            torch.set_float32_matmul_precision, torch.get_float32_matmul_precision()
                        )
                        torch.set_float32_matmul_precision("highest")
                    validation.update(
                        verify_folded_affine(
                            export_model,
                            engine,
                            validation_loader,
                            device,
                            precision=precision,
                            channels_last=channels_last,
                        )
                    )
            else:
                fold_terminal_rms_for_inference(export_model)
            atomic_torch(
                checkpoint_path.with_name(f"{checkpoint_path.stem}{suffix}_folded_inference.pt"),
                {
                    **export_metadata,
                    "model": export_model.state_dict(),
                    "folded_standardizer": True,
                },
            )

    export_rms(
        model,
        model if step_schedule else runtime,
        "",
        raw_final_validation if ema is not None else final_validation,
    )
    if ema is not None:
        export_rms(ema.model, ema.model, "_ema", final_validation)

    atomic_json(
        result_path,
        {
            "variant": variant,
            "seed": seed,
            "parameters": parameters,
            "global_step": global_step,
            "final_validation": final_validation,
            **({"raw_final_validation": raw_final_validation} if ema is not None else {}),
            "best_validation_accuracy_diagnostic": best_accuracy,
            "training_seconds": training_seconds,
            "complete_training_examples_per_second": (
                recipe["epochs"]
                * len(cast("Sized", cast("object", train_loader.dataset)))
                / training_seconds
            ),
            "history": history,
        },
    )
    telemetry.report("finished", recipe["epochs"], global_step, history[-1] if history else None)
    if wandb_run is not None:
        wandb_run.summary["final_validation_accuracy"] = final_validation["accuracy"]
        wandb_run.summary["final_validation_cross_entropy"] = final_validation["cross_entropy"]
        wandb_run.summary["best_validation_accuracy"] = best_accuracy
        wandb_run.summary["training_seconds"] = training_seconds
        wandb_run.finish()


def main(bindings: RunnerBindings) -> None:
    args = parse_args(variants=bindings.variants, seeds=bindings.seeds)
    if not args.initialize_only and not torch.cuda.is_available():
        raise RuntimeError("ImageNet-100 runner requires CUDA")
    data_runtime.configure_cpu_affinity()
    if not set(args.run_seeds) <= set(bindings.seeds):
        raise ValueError("run seeds fall outside the configured contract")
    contract = bindings.contract(args)
    _configure_compile_runtime(contract["recipe"])
    args.root.mkdir(parents=True, exist_ok=True)
    _initialize(args.root, contract)
    if args.initialize_only:
        if "drop_path_rate" in contract["recipe"]:
            (args.root / "history.jsonl").touch(exist_ok=True)
        return
    device = torch.device("cuda")
    for variant in args.variants:
        for seed in args.run_seeds:
            stopped = _run_job(
                args.root,
                contract,
                variant=variant,
                seed=seed,
                data_root=args.data_root,
                workers=args.workers,
                device=device,
                bindings=bindings,
                stop_after_epoch=args.stop_after_epoch,
            )
            if stopped:
                return
    summary = bindings.summarize(args.root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))
