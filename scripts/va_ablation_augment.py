"""CutMix training helpers for VA-Ablation runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import imagenet100_data_runtime as data_runtime
import numpy as np
import torch
from torch import Tensor, nn
from va_imagenet100_runtime import training_objective

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


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
    cutmix: bool = False,
    cutmix_alpha: float = 1.0,
) -> dict[str, float]:
    if not cutmix:
        from va_imagenet100_runtime import train_epoch as base_train_epoch

        return base_train_epoch(
            model,
            runtime,
            loader,
            optimizer,
            device=device,
            mixup_generator=mixup_generator,
            mixup_alpha=mixup_alpha,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            channels_last=channels_last,
        )
    if hasattr(runtime, "logical_update"):
        from effective_batch_epoch import train_epoch as effective_epoch
        return effective_epoch(model,runtime,loader,device=device,mixup_generator=mixup_generator,
                               mixup_alpha=mixup_alpha,group_size=gradient_accumulation_steps,
                               channels_last=channels_last,cutmix_alpha=cutmix_alpha)
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
        if float(mixup_generator.random()) < 0.5:
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
