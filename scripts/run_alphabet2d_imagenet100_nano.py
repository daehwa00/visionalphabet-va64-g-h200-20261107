# pyright: reportExplicitAny=false, reportMissingImports=false
"""Train matched four-scan and pole-free Nano models on ImageNet-100."""

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

from lnet.alphabet2d_cifar import ImageNetNanoConfig, build_imagenet_nano

if TYPE_CHECKING:
    from collections.abc import Callable, Sized

    from torch.utils.data import DataLoader
    from wandb.sdk.wandb_run import Run as WandbRun

VARIANTS = ("product_four", "pole_free")
SEEDS = (501, 509, 521)

_atomic_json = checkpoint_runtime.atomic_json
_atomic_torch = checkpoint_runtime.atomic_torch
_build_optimizer = checkpoint_runtime.build_optimizer
_optimizer_steps_per_epoch = checkpoint_runtime.optimizer_steps_per_epoch
_restore_checkpoint = checkpoint_runtime.restore_checkpoint
_restore_optimizer_runtime_options = checkpoint_runtime.restore_optimizer_runtime_options
_train_epoch_with_step_count = checkpoint_runtime.train_epoch_with_step_count

PREFETCH_FACTOR = data_runtime.PREFETCH_FACTOR
_CUDA_GRAPH_COMPILE_MODES = data_runtime.CUDA_GRAPH_COMPILE_MODES
_active_loader_workers = data_runtime.active_loader_workers
_begin_cudagraph_step = data_runtime.begin_cudagraph_step
_compact_wandb_name = data_runtime.compact_wandb_name
_configure_cpu_affinity = data_runtime.configure_cpu_affinity
_dataset_digest = data_runtime.dataset_digest
_device_batches = data_runtime.device_batches
_digest = data_runtime.digest
_loaders = data_runtime.loaders
_persistent_loader_workers = data_runtime.persistent_loader_workers
_transforms = data_runtime.transforms_for_imagenet100


@dataclass(frozen=True, slots=True)
class RunnerBindings:
    """Explicit experiment callbacks consumed by the shared training harness."""

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


def _parse_args(
    *,
    variants: tuple[str, ...] = VARIANTS,
    seeds: tuple[int, ...] = SEEDS,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--variants", choices=variants, nargs="+", default=list(variants))
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(seeds))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--initialize-only", action="store_true")
    return parser.parse_args()


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    if args.gradient_accumulation_steps < 1:
        message = "gradient_accumulation_steps must be positive"
        raise ValueError(message)
    model = ImageNetNanoConfig()
    loader_workers = _active_loader_workers(args.workers)
    loader_persistent_workers = _persistent_loader_workers(loader_workers)
    data_digest, train_count, validation_count = _dataset_digest(args.data_root)
    return {
        "schema": "lnet.alphabet2d.imagenet100_nano.v2",
        "evidence_status": "confirmatory G4 internal control",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "model": asdict(model),
        "recipe": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "optimizer": "AdamW",
            "learning_rate": 3.0e-3,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "schedule": "cosine",
            "label_smoothing": 0.1,
            "mixup_alpha": 0.8,
            "precision": args.precision,
            "loader_workers": loader_workers,
            "loader_persistent_workers": loader_persistent_workers,
            "validation_loader_persistent_workers": False,
            "loader_prefetch_factor": PREFETCH_FACTOR,
            "cpu_affinity": os.environ.get("LNET_CPU_AFFINITY_ACTIVE"),
            "device_prefetch_stream": True,
            "device_prefetch_scope": "copy_only",
            "fused_h2d_channels_last": False,
            "augmentation": ("RandomResizedCrop(224,bicubic)+HFlip+RandAugment(2,9)+RandomErasing"),
            "selection": "fixed final epoch; validation is not used for selection",
            "resume": "epoch-boundary RNG restore; worker RNG depends on loader persistence",
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
        "source_sha256": {
            "runner": _digest(Path(__file__)),
            "models": _digest(Path("src/lnet/alphabet2d_cifar.py")),
            "alphabet2d": _digest(Path("src/lnet/alphabet2d.py")),
        },
    }


def _initialize(root: Path, contract: dict[str, Any]) -> None:
    path = root / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            message = "existing ImageNet Nano root has a different immutable contract"
            raise RuntimeError(message)
    else:
        _atomic_json(path, contract)


def _evaluate(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    model.eval()
    runtime.eval()
    correct_terms: list[Tensor] = []
    count = 0
    loss_terms: list[Tensor] = []
    with torch.inference_mode():
        for inputs, targets in _device_batches(
            loader,
            device,
            channels_last=channels_last,
        ):
            _begin_cudagraph_step(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bfloat16",
            ):
                logits = runtime(inputs)
            loss_terms.append(functional.cross_entropy(logits, targets, reduction="sum"))
            correct_terms.append((logits.argmax(dim=-1) == targets).sum())
            count += targets.numel()
    loss_sum = float(torch.stack(loss_terms).double().sum())
    correct = int(torch.stack(correct_terms).sum())
    return {"accuracy": correct / count, "cross_entropy": loss_sum / count}


def _learning_rate_factor(epoch: int, epochs: int) -> float:
    if epoch < 5:
        return (epoch + 1) / 5.0
    progress = (epoch - 5) / max(1, epochs - 5)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _train_epoch(
    model: nn.Module,
    runtime: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    mixup_generator: np.random.Generator,
    mixup_alpha: float,
    precision: str,
    gradient_accumulation_steps: int = 1,
    channels_last: bool = False,
) -> dict[str, float]:
    if gradient_accumulation_steps < 1:
        message = "gradient accumulation steps must be positive"
        raise ValueError(message)
    model.train()
    runtime.train()
    correct_terms: list[Tensor] = []
    count = 0
    loss_terms: list[Tensor] = []
    batch_count = len(loader)
    batches = _device_batches(loader, device, channels_last=channels_last)
    for batch_index, (inputs, targets) in enumerate(batches):
        group_offset = batch_index % gradient_accumulation_steps
        if group_offset == 0:
            optimizer.zero_grad(set_to_none=True)
        group_size = min(
            gradient_accumulation_steps,
            batch_count - (batch_index - group_offset),
        )
        permutation = torch.randperm(targets.numel(), device=device)
        mixing = float(mixup_generator.beta(mixup_alpha, mixup_alpha))
        mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
        _begin_cudagraph_step(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            logits = runtime(mixed_inputs)
            loss = mixing * functional.cross_entropy(
                logits,
                targets,
                label_smoothing=0.1,
            ) + (1.0 - mixing) * functional.cross_entropy(
                logits,
                targets[permutation],
                label_smoothing=0.1,
            )
        (loss / group_size).backward()
        group_complete = group_offset + 1 == group_size or batch_index + 1 == batch_count
        if group_complete:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        loss_terms.append(loss.detach() * targets.numel())
        correct_terms.append((logits.argmax(dim=-1) == targets).sum())
        count += targets.numel()
    loss_sum = float(torch.stack(loss_terms).double().sum())
    correct = int(torch.stack(correct_terms).sum())
    return {"loss": loss_sum / count, "mixed_accuracy": correct / count}


def _prepare_model(model: nn.Module, recipe: dict[str, Any]) -> nn.Module:
    if bool(recipe.get("channels_last", False)):
        model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    return model


def _exclude_capture_hostile_submodules(
    model: nn.Module,
    *,
    exclude_classifier: bool = False,
) -> list[str]:
    """Keep parametrized submodules out of the compiled region.

    ``torch.nn.utils.parametrizations.orthogonal`` rebuilds its weight through
    ``matrix_exp``/``linalg.solve``/Householder on every access.  Those are
    cuSOLVER calls that synchronize with the host and allocate workspace, which
    CUDA stream capture forbids -- ``reduce-overhead`` aborts with
    ``operation not permitted when stream is capturing``.  Running them eagerly
    costs nothing measurable and lets everything else be captured.
    """
    excluded: list[str] = []
    for name, module in model.named_modules():
        if not parametrize.is_parametrized(module):
            continue
        module.forward = torch.compiler.disable(module.forward)
        excluded.append(name)
    classifier = getattr(model, "classifier", None)
    if exclude_classifier and isinstance(classifier, nn.Module):
        # A graph break immediately before the head can otherwise create a
        # separately captured classifier graph whose output storage is reused
        # before autograd consumes it.  The head is tiny relative to the pole
        # backbone, so keep it eager while retaining backbone graph replay.
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
    """Preserve mode tuning while using classic CUDA Graph replay."""
    options = {
        "triton.cudagraphs": True,
        "triton.cudagraph_trees": False,
    }
    if compile_mode == "max-autotune":
        options.update(
            {
                "coordinate_descent_tuning": True,
                "max_autotune": True,
            }
        )
    return options


def _build_runtime(model: nn.Module, recipe: dict[str, Any]) -> nn.Module:
    # The env override lets a measurement select a compile mode without editing
    # the recipe contract.  Unset, behaviour is exactly the recipe's.
    compile_mode = os.environ.get("LNET_COMPILE_MODE") or recipe.get("compile_mode")
    if compile_mode is None:
        return model
    requested_cudagraphs = str(compile_mode) in _CUDA_GRAPH_COMPILE_MODES
    repeated_stages = requested_cudagraphs and _has_replayed_cudagraph_stages(model)
    uses_cudagraphs = requested_cudagraphs
    os.environ["LNET_CUDAGRAPHS_ACTIVE"] = "1" if uses_cudagraphs else "0"
    _exclude_capture_hostile_submodules(
        model,
        exclude_classifier=uses_cudagraphs,
    )
    if repeated_stages:
        # CUDA Graph Trees split the large pole graph around opaque custom-op
        # launches and then reuse one memory pool for live saved tensors. A
        # classic graph keeps the same Inductor/Triton kernels and fixed-shape
        # replay without cross-segment Tree storage reuse.
        return cast(
            "nn.Module",
            torch.compile(
                model,
                options=_replayed_stage_compile_options(str(compile_mode)),
                fullgraph=False,
                dynamic=False,
            ),
        )
    return cast(
        "nn.Module",
        torch.compile(
            model,
            mode=str(compile_mode),
            fullgraph=False,
            dynamic=False,
        ),
    )


def _configure_compile_runtime(_root: Path, recipe: dict[str, Any]) -> str | None:
    """Publish the actual compile contract for launch-kernel tuning."""
    compile_mode = os.environ.get("LNET_COMPILE_MODE") or recipe.get("compile_mode")
    if compile_mode is None:
        return None
    profile = json.dumps(
        {
            "dynamic": False,
            "fullgraph": False,
            "mode": str(compile_mode),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    os.environ["LNET_COMPILE_PROFILE"] = profile
    # Launch records are already guarded by device, dtype, shape, compiler,
    # and kernel revision.  Keep them machine-local rather than run-local so a
    # measured winner is reused by later experiment roots automatically.
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
    """Create a deterministic, resume-safe W&B run when tracking is enabled."""
    project = os.environ.get("WANDB_PROJECT")
    if not project or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ModuleNotFoundError as error:
        message = "WANDB_PROJECT is set but the wandb package is not installed"
        raise RuntimeError(message) from error
    run_key = f"{root.resolve()}::{variant}::seed{seed}"
    run_id = os.environ.get("WANDB_RUN_ID") or hashlib.sha256(run_key.encode()).hexdigest()[:16]
    tracking_root = root / "wandb"
    tracking_root.mkdir(parents=True, exist_ok=True)
    variant_config = contract.get("variant_configs", {}).get(variant)
    return wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        group=os.environ.get("WANDB_GROUP", root.name),
        name=os.environ.get("WANDB_NAME", _compact_wandb_name(variant)),
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
            # The builder applies the variant transform to the model template.
            # Log the resulting architecture as `model` so W&B columns describe
            # the model that is actually trained, not its constructor template.
            "model": variant_config or contract["model"],
            "model_template": contract["model"],
            "variant_config": variant_config,
            "recipe": contract["recipe"],
            "schema": contract["schema"],
        },
    )


def _wandb_model_metrics(_model: nn.Module) -> dict[str, float]:
    """Return optional architecture-specific scalar diagnostics."""
    return {}


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
    model = bindings.build_model(
        variant,
        bindings.model_config(**contract["model"]),
    ).to(device)
    recipe = contract["recipe"]
    model = bindings.prepare_model(model, recipe)
    training_generator = torch.Generator().manual_seed(seed)
    mixup_generator = np.random.default_rng(seed)
    train_loader, validation_loader = _loaders(
        data_root,
        batch_size=recipe["batch_size"],
        workers=workers,
        training_generator=training_generator,
    )
    optimizer = bindings.build_optimizer(model, recipe)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _learning_rate_factor(epoch, recipe["epochs"]),
    )
    gradient_accumulation_steps = int(recipe.get("gradient_accumulation_steps", 1))
    steps_per_epoch = _optimizer_steps_per_epoch(
        train_loader,
        gradient_accumulation_steps,
    )
    checkpoint_path = root / "checkpoints" / f"{variant}__seed{seed}.pt"
    progress = {"global_step": 0}
    start_epoch, history, training_seconds = _restore_checkpoint(
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
    _restore_optimizer_runtime_options(optimizer, recipe)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    wandb_run = _initialize_wandb_run(
        root,
        contract,
        variant=variant,
        seed=seed,
        parameters=parameters,
    )
    runtime = _build_runtime(model, recipe)
    channels_last = bool(recipe.get("channels_last", False))
    for epoch in range(start_epoch, recipe["epochs"]):
        started = time.perf_counter()
        train, optimizer_steps = _train_epoch_with_step_count(
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
        _atomic_torch(
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
    _atomic_json(
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


def _summarize(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    paths = [
        root / "results" / f"{variant}__seed{seed}.json" for variant in VARIANTS for seed in SEEDS
    ]
    if not all(path.exists() for path in paths):
        return None
    rows = [json.loads(path.read_text()) for path in paths]
    means = {
        variant: sum(
            row["final_validation"]["accuracy"] for row in rows if row["variant"] == variant
        )
        / len(SEEDS)
        for variant in VARIANTS
    }
    paired = [
        next(
            row["final_validation"]["accuracy"]
            for row in rows
            if row["variant"] == "product_four" and row["seed"] == seed
        )
        - next(
            row["final_validation"]["accuracy"]
            for row in rows
            if row["variant"] == "pole_free" and row["seed"] == seed
        )
        for seed in SEEDS
    ]
    payload = {
        "schema": contract["schema"],
        "mean_final_validation_accuracy": means,
        "paired_product_minus_pole_free": paired,
        "mean_product_minus_pole_free_pp": 100.0 * sum(paired) / len(paired),
        "G4_product_beats_pole_free_1pp": sum(paired) / len(paired) >= 0.01,
        "decision": (
            "continue ALPHABET-2D"
            if sum(paired) / len(paired) >= 0.01
            else "kill: gain belongs to shared backbone"
        ),
        "parameter_counts": {
            variant: sorted({row["parameters"] for row in rows if row["variant"] == variant})
            for variant in VARIANTS
        },
    }
    _atomic_json(root / "summary.json", payload)
    return payload


def runner_bindings(
    *,
    variants: tuple[str, ...] = VARIANTS,
    seeds: tuple[int, ...] = SEEDS,
    model_config: Callable[..., Any] = ImageNetNanoConfig,
    build_model: Callable[..., nn.Module] = build_imagenet_nano,
    contract: Callable[[argparse.Namespace], dict[str, Any]] = _contract,
    build_optimizer: Callable[..., torch.optim.Optimizer] = _build_optimizer,
    prepare_model: Callable[[nn.Module, dict[str, Any]], nn.Module] = _prepare_model,
    train_epoch: Callable[..., dict[str, float]] = _train_epoch,
    evaluate: Callable[..., dict[str, float]] = _evaluate,
    wandb_model_metrics: Callable[[nn.Module], dict[str, float]] = _wandb_model_metrics,
    summarize: Callable[[Path, dict[str, Any]], dict[str, Any] | None] = _summarize,
) -> RunnerBindings:
    """Build immutable per-run callbacks without mutating harness globals."""
    return RunnerBindings(
        variants=variants,
        seeds=seeds,
        model_config=model_config,
        build_model=build_model,
        contract=contract,
        build_optimizer=build_optimizer,
        prepare_model=prepare_model,
        train_epoch=train_epoch,
        evaluate=evaluate,
        wandb_model_metrics=wandb_model_metrics,
        summarize=summarize,
    )


def main(bindings: RunnerBindings | None = None) -> None:
    active = bindings or runner_bindings()
    args = _parse_args(variants=active.variants, seeds=active.seeds)
    if not torch.cuda.is_available():
        message = "ImageNet Nano runner requires CUDA"
        raise RuntimeError(message)
    _configure_cpu_affinity()
    if not set(args.run_seeds) <= set(active.seeds):
        message = "run seeds fall outside the ImageNet Nano contract"
        raise ValueError(message)
    contract = active.contract(args)
    _configure_compile_runtime(args.root, contract["recipe"])
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
                bindings=active,
            )
    summary = active.summarize(args.root, contract)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
