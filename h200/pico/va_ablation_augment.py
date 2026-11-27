# ruff: noqa: ANN001, ANN201, ANN401, C901, PLC0415, SLF001
"""CutMix, timm RandAugment, and contract helpers for VA-Ablation runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import imagenet100_data_runtime as data_runtime
import numpy as np
import torch
from torch import Tensor, nn
from va_imagenet100_runtime import training_objective

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder


def mixing_description(recipe: dict[str, Any], spec: dict[str, Any]) -> str:
    return f"MixUp({recipe['mixup_alpha']}) / CutMix({spec.get('cutmix_alpha', 1.0)}) p=0.5"


def install(
    base: dict[str, Any],
    spec: dict[str, Any],
    *,
    contract: dict[str, Any],
    runtime_module: Any,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> None:
    """Install CutMix or timm RandAugment transforms if configured."""
    if spec.get("timm_ra"):
        from timm.data import create_transform
        from torchvision.transforms import v2

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_transform = create_transform(
            224,
            is_training=True,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=mean,
            std=std,
        )
        validation_transform = v2.Compose(
            [
                v2.Resize(256, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
                v2.CenterCrop(224),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=mean, std=std),
            ]
        )
        orig_loaders = data_runtime.loaders

        def timm_loaders(
            root: Any, *, batch_size: int, workers: int, training_generator: Any, **options: Any
        ) -> Any:
            tl, vl = orig_loaders(
                root,
                batch_size=batch_size,
                workers=workers,
                training_generator=training_generator,
                **options,
            )
            cast("ImageFolder", tl.dataset).transform = train_transform
            cast("ImageFolder", vl.dataset).transform = validation_transform
            return tl, vl

        data_runtime.loaders = timm_loaders  # type: ignore[assignment]
        contract["recipe"]["augmentation"] = (
            "timm RandomResizedCrop(224,bicubic)+HFlip+RandAug(rand-m9-mstd0.5-inc1)"
            "+RandomErasing(0.25,pixel)"
        )
    elif spec.get("cutmix"):
        contract["recipe"]["augmentation"] = (
            "matched A2D ImageNet-100 public recipe + "
            + mixing_description(contract["recipe"], spec)
        )
    _ = base, runtime_module, variants, seeds


def _cutmix_box(
    width: int,
    height: int,
    lam: float,
    generator: np.random.Generator,
) -> tuple[int, int, int, int]:
    cut = float(np.sqrt(max(0.0, 1.0 - lam)))
    cw = max(1, round(width * cut))
    ch = max(1, round(height * cut))
    cx = int(generator.integers(width))
    cy = int(generator.integers(height))
    x0 = max(0, cx - cw // 2)
    y0 = max(0, cy - ch // 2)
    x1 = min(width, cx + cw // 2)
    y1 = min(height, cy + ch // 2)
    return y0, y1, x0, x1


def mix_batch(
    inputs, targets, mixup_generator, mixup_alpha, cutmix_alpha, mix_prob, *, cutmix=True
):
    if mix_prob == 0.0 or (mix_prob < 1.0 and float(mixup_generator.random()) >= mix_prob):
        return inputs, targets, 1.0
    permutation = torch.randperm(targets.numel(), device=targets.device)
    permuted_targets = targets[permutation]
    if cutmix and float(mixup_generator.random()) < 0.5:
        lam = float(mixup_generator.beta(cutmix_alpha, cutmix_alpha))
        y0, y1, x0, x1 = _cutmix_box(
            inputs.shape[-1],
            inputs.shape[-2],
            lam,
            mixup_generator,
        )
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, y0:y1, x0:x1] = inputs[permutation, :, y0:y1, x0:x1]
        mixing = 1.0 - (y1 - y0) * (x1 - x0) / float(inputs.shape[-2] * inputs.shape[-1])
    else:
        mixing = float(mixup_generator.beta(mixup_alpha, mixup_alpha))
        mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
    return mixed_inputs, permuted_targets, mixing


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
    cutmix: bool = False,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 1.0,
) -> dict[str, float]:
    if not cutmix and mix_prob == 1.0:
        from va_imagenet100_runtime import train_epoch as base_train_epoch

        return base_train_epoch(
            model,
            runtime,
            loader,
            optimizer,
            device=device,
            mixup_generator=mixup_generator,
            mixup_alpha=mixup_alpha,
            label_smoothing=label_smoothing,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            channels_last=channels_last,
        )
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
            cutmix_alpha=cutmix_alpha,
        )
    from automatic_training_layout import training_microbatch

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    model.train()
    runtime.train()
    loss_terms: list[Tensor] = []
    correct_terms: list[Tensor] = []
    diagnostics: dict[str, list[Tensor]] = {}
    count = 0
    batch_count = len(loader)
    observe_batch = getattr(optimizer, "_va_observe_batch", None)
    gradient_norms: list[Tensor] = []
    track_gradients = getattr(optimizer, "_va_track_gradient_norms", False)
    batches = data_runtime.device_batches(loader, device, channels_last=channels_last)
    for batch_index, (inputs, targets) in enumerate(batches):
        group_offset = batch_index % gradient_accumulation_steps
        if group_offset == 0:
            optimizer.zero_grad(set_to_none=True)
        group_size = min(
            gradient_accumulation_steps,
            batch_count - (batch_index - group_offset),
        )
        mixed_inputs, permuted_targets, mixing = mix_batch(
            inputs,
            targets,
            mixup_generator,
            mixup_alpha,
            cutmix_alpha,
            mix_prob,
            cutmix=cutmix,
        )
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
            if group_complete:
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if track_gradients:
                    gradient_norms.append(gradient_norm.detach())
                optimizer.step()
        loss_terms.append(loss.detach() * targets.numel())
        correct_terms.append(logits.argmax(dim=-1).eq(targets).sum())
        for name, value in active_diagnostics.items():
            diagnostics.setdefault(name, []).append(value.detach() * targets.numel())
        count += targets.numel()
        if observe_batch is not None:
            observe_batch(batch_index + 1, loss_terms, count)
    result = {
        "loss": float(torch.stack(loss_terms).double().sum()) / count,
        "samples": count,
        "mixed_accuracy": int(torch.stack(correct_terms).sum()) / count,
    }
    if track_gradients:
        result["finite_gradients"] = bool(torch.stack(gradient_norms).isfinite().all())
        if not result["finite_gradients"]:
            raise RuntimeError("nonfinite gradient norm during training")
    cast("Any", model)._latest_training_diagnostics = {
        name: float(torch.stack(values).double().sum()) / count
        for name, values in diagnostics.items()
    }
    return result
