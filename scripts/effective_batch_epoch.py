"""Retain data ordering and augmentation while choosing activation batches independently."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np
    from effective_batch_runtime import Batch, EffectiveTrainingRuntime
    from torch.utils.data import DataLoader

from imagenet100_data_runtime import device_batches


def prepare(
    inputs: torch.Tensor, targets: torch.Tensor, generator: np.random.Generator,
    alpha: float, cutmix_alpha: float | None = None,
) -> Batch:
    perm = torch.randperm(len(targets), device=inputs.device)
    if cutmix_alpha is not None and float(generator.random()) < 0.5:
        from alphabet_augmentation import _cutmix_box

        lam = float(generator.beta(cutmix_alpha, cutmix_alpha))
        y0, y1, x0, x1 = _cutmix_box(inputs.shape[-1], inputs.shape[-2], lam, generator)
        mixed = inputs.clone()
        mixed[:, :, y0:y1, x0:x1] = inputs[perm, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (inputs.shape[-2] * inputs.shape[-1])
    else:
        lam = float(generator.beta(alpha, alpha))
        mixed = lam * inputs + (1 - lam) * inputs[perm]
    return mixed, targets, targets[perm], lam


def combine(pieces: list[Batch]) -> Batch:
    if len(pieces) == 1:
        return pieces[0]
    xs, ys, others, weights = zip(*pieces, strict=True)
    weight = torch.cat(
        [
            torch.full((len(y),), w, dtype=torch.float32, device=y.device)
            for y, w in zip(ys, weights, strict=True)
        ]
    )
    return torch.cat(xs), torch.cat(ys), torch.cat(others), weight


def train_epoch(
    model: torch.nn.Module,
    runtime: EffectiveTrainingRuntime,
    loader: DataLoader,
    *,
    device: torch.device,
    mixup_generator: np.random.Generator,
    mixup_alpha: float,
    group_size: int,
    channels_last: bool,
    cutmix_alpha: float | None = None,
) -> dict[str, float]:
    model.train()
    runtime.train()
    groups = []
    correct = []
    losses = []
    diagnostics = {}
    count = 0
    for i, (x, y) in enumerate(device_batches(loader, device, channels_last=channels_last)):
        groups.append(prepare(x, y, mixup_generator, mixup_alpha, cutmix_alpha))
        if len(groups) != group_size and i + 1 != len(loader):
            continue
        batch = combine(groups)
        groups.clear()
        result = runtime.logical_update(batch)
        n = result["samples"]
        losses.append(result["loss"] * n)
        correct.append(result["correct"])
        for key, value in result["diagnostics"].items():
            diagnostics.setdefault(key, []).append(value * n)
        count += n
    model._latest_training_diagnostics = {
        k: float(torch.stack(v).double().sum()) / count for k, v in diagnostics.items()
    }
    return {
        "loss": float(torch.stack(losses).double().sum()) / count,
        "mixed_accuracy": float(torch.stack(correct).sum()) / count,
    }
