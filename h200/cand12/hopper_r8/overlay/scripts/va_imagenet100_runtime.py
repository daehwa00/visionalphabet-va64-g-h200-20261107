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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import imagenet100_checkpoint_runtime as checkpoint_runtime
import imagenet100_data_runtime as data_runtime
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import parametrize
from va_model_builder import model_config as default_model_config

from lnet.capture_safe_orthogonal import prepare_capture_safe_orthogonal_
from lnet.classifier_heads import AffineCorrelationClassifier, expected_calibration_error

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


def parse_args(*, variants: tuple[str, ...], seeds: tuple[int, ...]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--variants", choices=variants, nargs="+", default=list(variants))
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(seeds))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--effective-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    # Preserve explicit historical microbatch commands. A plain invocation now targets 1024.
    resume_recipe = {}
    if args.batch_size is None and args.gradient_accumulation_steps is None and args.effective_batch_size is None:
        contract_path = args.root / "contract.json"
        if contract_path.is_file():
            resume_recipe = json.loads(contract_path.read_text()).get("recipe", {})
    nominal = args.batch_size if args.batch_size is not None else resume_recipe.get("batch_size", 128)
    if nominal <= 0:
        parser.error("--batch-size must be positive")
    target = args.effective_batch_size
    if target is None and args.batch_size is None and args.gradient_accumulation_steps is None:
        target = nominal * resume_recipe["gradient_accumulation_steps"] if "gradient_accumulation_steps" in resume_recipe else 1024
    if target is not None:
        if target <= 0 or target % nominal:
            parser.error("--effective-batch-size must be a positive multiple of --batch-size")
        if args.gradient_accumulation_steps is not None and nominal * args.gradient_accumulation_steps != target:
            parser.error("effective batch conflicts with batch size and gradient accumulation")
    args.batch_size = nominal
    args.gradient_accumulation_steps = target // nominal if target is not None else (args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else 1)
    return args


def base_contract(
    args: argparse.Namespace,
    *,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    loader_workers = data_runtime.active_loader_workers(args.workers)
    loader_persistent_workers = data_runtime.persistent_loader_workers(loader_workers)
    data_digest, train_count, validation_count = data_runtime.dataset_digest(args.data_root)
    config = default_model_config(output_dim=100)
    return {
        "variants": list(variants),
        "priority": list(variants),
        "seeds": list(seeds),
        "num_classes": 100,
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
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "warmup plus cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "precision": args.precision,
            "compile_mode": "default",
            "compiled_training_preparation": True,
            "channels_last": True,
            "emulate_precision_casts": True,
            "matmul_precision": "high (TF32 enabled)",
            "loader_workers": loader_workers,
            "loader_persistent_workers": loader_persistent_workers,
            "validation_loader_persistent_workers": False,
            "loader_prefetch_factor": data_runtime.PREFETCH_FACTOR,
            "cpu_affinity": os.environ.get("LNET_CPU_AFFINITY_ACTIVE"),
            "device_prefetch_stream": True,
            "device_prefetch_scope": "copy_only",
            "fused_h2d_channels_last": False,
            "augmentation": "matched A2D ImageNet-100 public recipe",
            "selection": f"fixed epoch {args.epochs}; no within-run validation selection",
            "resume": "epoch-boundary exact RNG restore",
        },
        "data": {
            "manifest_sha256": data_digest,
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


def _learning_rate_factor(epoch: int, epochs: int) -> float:
    if epoch < 5:
        return (epoch + 1) / 5.0
    progress = (epoch - 5) / max(1, epochs - 5)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _mixed_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    permuted_targets: Tensor,
    mixing: float,
) -> Tensor:
    if torch.is_tensor(mixing):
        first = functional.cross_entropy(logits, targets, reduction="none", label_smoothing=0.1)
        second = functional.cross_entropy(logits, permuted_targets, reduction="none", label_smoothing=0.1)
        return (mixing * first + (1.0 - mixing) * second).mean()
    return mixing * functional.cross_entropy(
        logits,
        targets,
        label_smoothing=0.1,
    ) + (1.0 - mixing) * functional.cross_entropy(
        logits,
        permuted_targets,
        label_smoothing=0.1,
    )


def training_objective(
    model: nn.Module,
    output: Tensor | tuple[Tensor, ...],
    targets: Tensor,
    permuted_targets: Tensor,
    mixing: float,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    if not isinstance(output, tuple) or len(output) != 5:
        raise RuntimeError("affine Q-head model returned an invalid training tuple")
    joint, affine, _fusion, _lrq, _descriptor = output
    classifier = cast("AffineCorrelationClassifier", cast("Any", model).classifier)
    joint_loss = _mixed_cross_entropy(joint, targets, permuted_targets, mixing)
    auxiliary = (
        _mixed_cross_entropy(affine, targets, permuted_targets, mixing)
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
    precision: str,
    gradient_accumulation_steps: int = 1,
    channels_last: bool = False,
) -> dict[str, float]:
    """Run one ImageNet-100 MixUp training epoch."""
    from automatic_training_layout import training_microbatch

    if hasattr(runtime, "logical_update"):
        from effective_batch_epoch import train_epoch as effective_epoch
        return effective_epoch(model, runtime, loader, device=device, mixup_generator=mixup_generator,
                               mixup_alpha=mixup_alpha, group_size=gradient_accumulation_steps,
                               channels_last=channels_last)
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
            runtime, mixed_inputs, targets, permuted_targets, mixing, group_size, group_offset,
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
                )
            (loss / group_size).backward()
            group_complete = group_offset + 1 == group_size or batch_index + 1 == batch_count
            if group_complete:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        loss_terms.append(loss.detach() * targets.numel())
        correct_terms.append(logits.argmax(dim=-1).eq(targets).sum())
        for name, value in active_diagnostics.items():
            diagnostics.setdefault(name, []).append(value.detach() * targets.numel())
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
    compile_mode = os.environ.get("LNET_COMPILE_MODE") or recipe.get("compile_mode")
    if compile_mode is None:
        return model
    from lnet.path_backward_layout import layout_graph_pass
    from lnet.retained_path_dispatch import step_graph_pass
    from lnet.segmented_path_scan import segment_graph_pass
    from lnet.output_dot_dispatch import output_dot_graph_pass

    _configure_compile_runtime(recipe)
    requested_cudagraphs = str(compile_mode) in _CUDA_GRAPH_COMPILE_MODES
    repeated_stages = requested_cudagraphs and _has_replayed_cudagraph_stages(model)
    os.environ["LNET_CUDAGRAPHS_ACTIVE"] = "1" if requested_cudagraphs else "0"
    _exclude_capture_hostile_submodules(model, exclude_classifier=requested_cudagraphs)
    options = (
        _replayed_stage_compile_options(str(compile_mode))
        if repeated_stages else dict(torch._inductor.list_mode_options(str(compile_mode)))
    )
    options["post_grad_custom_pre_pass"] = output_dot_graph_pass(
        segment_graph_pass(step_graph_pass(layout_graph_pass()))
    )
    if recipe.get("effective_batch_size") == 1024:
        from lnet.effective_output_vjp import vjp_graph_pass
        from lnet.effective_structural_scan import structural_graph_pass

        options["post_grad_custom_pre_pass"] = structural_graph_pass(
            vjp_graph_pass(options["post_grad_custom_pre_pass"])
        )
    # BEGIN R6 DISPATCH
    from lnet.r6_dispatch import r6_graph_pass

    options["post_grad_custom_pre_pass"] = r6_graph_pass(options["post_grad_custom_pre_pass"])
    # END R6 DISPATCH
    return cast(
        "nn.Module",
        torch.compile(model, options=options, fullgraph=False, dynamic=False),
    )


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
                    row["final_validation"].get("affine_only_accuracy", math.nan)
                    for row in active
                )
                / len(active),
            }
            for variant, active in grouped.items()
        },
    }
    atomic_json(root / "summary.json", summary)
    return summary


def _run_job(
    root: Path,
    contract: dict[str, Any],
    *,
    variant: str,
    seed: int,
    data_root: Path,
    workers: int,
    device: torch.device,
    bindings: RunnerBindings,
) -> None:
    result_path = root / "results" / f"{variant}__seed{seed}.json"
    if result_path.exists():
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = bindings.build_model(variant, bindings.model_config(**contract["model"])).to(device)
    recipe = contract["recipe"]
    model = bindings.prepare_model(model, recipe)
    training_generator = torch.Generator().manual_seed(seed)
    mixup_generator = np.random.default_rng(seed)
    train_loader, validation_loader = data_runtime.loaders(
        data_root,
        batch_size=recipe["batch_size"],
        validation_batch_size=min(128, recipe["batch_size"]) if recipe.get("effective_batch_size") == 1024 else None,
        workers=workers,
        training_generator=training_generator,
    )
    optimizer = bindings.build_optimizer(model, recipe)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _learning_rate_factor(epoch, recipe["epochs"]),
    )
    gradient_accumulation_steps = int(recipe.get("gradient_accumulation_steps", 1))
    steps_per_epoch = checkpoint_runtime.optimizer_steps_per_epoch(
        train_loader,
        gradient_accumulation_steps,
    )
    checkpoint_path = root / "checkpoints" / f"{variant}__seed{seed}.pt"
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
    )
    global_step = progress["global_step"]
    checkpoint_runtime.restore_optimizer_runtime_options(optimizer, recipe)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    wandb_run = _initialize_wandb_run(
        root, contract, variant=variant, seed=seed, parameters=parameters
    )
    from automatic_training_layout import AutomaticTrainingRuntime

    runtime_type = AutomaticTrainingRuntime
    if recipe.get("effective_batch_size") == 1024:
        from effective_batch_runtime import EffectiveTrainingRuntime
        runtime_type = EffectiveTrainingRuntime
    runtime = runtime_type(
        model, optimizer, recipe, contract["model"], variant, bindings, root, training_generator, mixup_generator,
    )
    channels_last = bool(recipe.get("channels_last", False))
    for epoch in range(start_epoch, recipe["epochs"]):
        runtime.epoch_index=epoch
        started = time.perf_counter()
        train, optimizer_steps = checkpoint_runtime.train_epoch_with_step_count(
            bindings.train_epoch,
            model,
            runtime,
            train_loader,
            optimizer,
            device=device,
            mixup_generator=mixup_generator,
            mixup_alpha=recipe["mixup_alpha"],
            precision=recipe["precision"],
            gradient_accumulation_steps=gradient_accumulation_steps,
            channels_last=channels_last,
        )
        global_step += optimizer_steps
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - started
        validation = bindings.evaluate(
            model,
            runtime,
            validation_loader,
            device,
            precision=recipe["precision"],
            channels_last=channels_last,
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train["loss"],
            "train_mixed_accuracy": train["mixed_accuracy"],
            "validation_accuracy": validation["accuracy"],
            "validation_cross_entropy": validation["cross_entropy"],
        }
        history.append(row)
        if wandb_run is not None:
            metrics = {
                "epoch": row["epoch"],
                "learning_rate": row["learning_rate"],
                "train/loss": row["train_loss"],
                "train/mixed_accuracy": row["train_mixed_accuracy"],
                "validation/accuracy": row["validation_accuracy"],
                "validation/cross_entropy": row["validation_cross_entropy"],
                "time/training_seconds": training_seconds,
                "global_step": global_step,
                "optimizer_steps/epoch": optimizer_steps,
            }
            metrics.update(bindings.wandb_model_metrics(model))
            wandb_run.log(metrics, step=epoch + 1)
        scheduler.step()
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
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "python_rng_state": random.getstate(),
                "mixup_rng_state": mixup_generator.bit_generator.state,
            },
        )
    final_validation = bindings.evaluate(
        model,
        runtime,
        validation_loader,
        device,
        precision=recipe["precision"],
        channels_last=channels_last,
    )
    atomic_json(
        result_path,
        {
            "variant": variant,
            "seed": seed,
            "parameters": parameters,
            "global_step": global_step,
            "final_validation": final_validation,
            "best_validation_accuracy_diagnostic": max(
                row["validation_accuracy"] for row in history
            ),
            "training_seconds": training_seconds,
            "complete_training_examples_per_second": (
                recipe["epochs"]
                * len(cast("Sized", cast("object", train_loader.dataset)))
                / training_seconds
            ),
            "history": history,
        },
    )
    if wandb_run is not None:
        wandb_run.summary["final_validation_accuracy"] = final_validation["accuracy"]
        wandb_run.summary["final_validation_cross_entropy"] = final_validation["cross_entropy"]
        wandb_run.summary["best_validation_accuracy"] = max(
            row["validation_accuracy"] for row in history
        )
        wandb_run.summary["training_seconds"] = training_seconds
        wandb_run.finish()


def main(bindings: RunnerBindings) -> None:
    args = parse_args(variants=bindings.variants, seeds=bindings.seeds)
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet-100 runner requires CUDA")
    data_runtime.configure_cpu_affinity()
    if not set(args.run_seeds) <= set(bindings.seeds):
        raise ValueError("run seeds fall outside the configured contract")
    contract = bindings.contract(args)
    _configure_compile_runtime(contract["recipe"])
    args.root.mkdir(parents=True, exist_ok=True)
    _initialize(args.root, contract)
    if args.initialize_only:
        return
    device = torch.device("cuda")
    for variant in args.variants:
        for seed in args.run_seeds:
            _run_job(
                args.root,
                contract,
                variant=variant,
                seed=seed,
                data_root=args.data_root,
                workers=args.workers,
                device=device,
                bindings=bindings,
            )
    summary = bindings.summarize(args.root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))
