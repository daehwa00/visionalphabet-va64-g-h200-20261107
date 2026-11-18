"""Optional CutMix and timm RandAugment hooks for the frozen IN100 harness."""

from __future__ import annotations

# The frozen harness intentionally exposes these private callback hooks.
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import math
from typing import TYPE_CHECKING, Any, cast

import imagenet100_data_runtime as data_runtime
import run_a2d_qhead_e2e_imagenet100 as structured
import run_alphabet2d_imagenet100_nano as harness
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms

if TYPE_CHECKING:
    import numpy as np
    from torch.utils.data import DataLoader

_ORIGINAL_TRAIN_EPOCH = structured._train_epoch
_ORIGINAL_TRANSFORMS = data_runtime.transforms_for_imagenet100
TIMM_RAND_AUGMENT = "rand-m9-mstd0.5-inc1"
CUTMIX_ALPHA = 1.0
CUTMIX_SWITCH_PROBABILITY = 0.5


def cutmix_bbox(
    height: int, width: int, mixing: float, generator: np.random.Generator
) -> tuple[int, int, int, int]:
    """Return the clipped (top, bottom, left, right) CutMix rectangle."""
    ratio = math.sqrt(1.0 - mixing)
    cut_height, cut_width = int(height * ratio), int(width * ratio)
    center_y, center_x = int(generator.integers(height)), int(generator.integers(width))
    return (
        max(center_y - cut_height // 2, 0),
        min(center_y + cut_height // 2, height),
        max(center_x - cut_width // 2, 0),
        min(center_x + cut_width // 2, width),
    )


def mix_batch(
    inputs: Tensor,
    targets: Tensor,
    generator: np.random.Generator,
    mixup_alpha: float,
) -> tuple[Tensor, Tensor, float]:
    """Mix one batch, sharing the harness's checkpointed augmentation RNG."""
    if mixup_alpha <= 0:
        return inputs, targets, 1.0
    permutation = torch.randperm(targets.numel(), device=targets.device)
    permuted_targets = targets[permutation]
    if generator.random() < CUTMIX_SWITCH_PROBABILITY:
        mixing = float(generator.beta(CUTMIX_ALPHA, CUTMIX_ALPHA))
        height, width = inputs.shape[-2:]
        top, bottom, left, right = cutmix_bbox(height, width, mixing, generator)
        mixed_inputs = inputs.clone()
        mixed_inputs[..., top:bottom, left:right] = inputs[permutation, ..., top:bottom, left:right]
        mixing = 1.0 - (bottom - top) * (right - left) / (height * width)
    else:
        mixing = float(generator.beta(mixup_alpha, mixup_alpha))
        mixed_inputs = mixing * inputs + (1.0 - mixing) * inputs[permutation]
    return mixed_inputs, permuted_targets, mixing


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
    """Frozen structured loop, changing only the batch mixing operation."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    model.train()
    runtime.train()
    loss_terms: list[Tensor] = []
    correct_terms: list[Tensor] = []
    diagnostics: dict[str, list[Tensor]] = {}
    count = 0
    batch_count = len(loader)
    batches = harness._device_batches(loader, device, channels_last=channels_last)
    for batch_index, (inputs, targets) in enumerate(batches):
        group_offset = batch_index % gradient_accumulation_steps
        if group_offset == 0:
            optimizer.zero_grad(set_to_none=True)
        group_size = min(
            gradient_accumulation_steps,
            batch_count - (batch_index - group_offset),
        )
        mixed_inputs, permuted_targets, mixing = mix_batch(
            inputs, targets, mixup_generator, mixup_alpha
        )
        harness._begin_cudagraph_step(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            output = runtime(mixed_inputs)
            logits, loss, active_diagnostics = structured._training_objective(
                model, output, targets, permuted_targets, mixing
            )
        (loss / group_size).backward()
        structured._after_training_batch(model, output, targets, permuted_targets, mixing)
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


def _timm_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    from timm.data.auto_augment import rand_augment_transform

    train, evaluation = _ORIGINAL_TRANSFORMS()
    for index, transform in enumerate(train.transforms):
        if isinstance(transform, transforms.RandAugment):
            train.transforms[index] = rand_augment_transform(
                TIMM_RAND_AUGMENT,
                {
                    "translate_const": int(224 * 0.45),
                    "img_mean": tuple(round(255 * mean) for mean in (0.485, 0.456, 0.406)),
                    "interpolation": Image.Resampling.BICUBIC,
                },
            )
    return train, evaluation


def install(*, cutmix: bool = False, timm_ra: bool = False) -> None:
    """Call before building loaders/bindings; disabled flags restore the recipe.

    A noaugmentation runner can call ``install()`` and then install its own
    noaugmentation callbacks. No new NumPy RNG stream is created by these hooks.
    """
    structured._train_epoch = _train_epoch if cutmix else _ORIGINAL_TRAIN_EPOCH
    factory = _timm_transforms if timm_ra else _ORIGINAL_TRANSFORMS
    data_runtime.transforms_for_imagenet100 = factory
    harness._transforms = factory
