"""Execute one logical optimizer update, independently of the activation microbatch size."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from alphabet_imagenet100_runtime import training_objective
from imagenet100_data_runtime import begin_cudagraph_step

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn

OUTPUT_NAMES = ("joint", "affine", "fusion", "lrq", "descriptor")


def run_update(
    model: nn.Module, runtime: nn.Module, optimizer: torch.optim.Optimizer,
    batch: tuple[Tensor, Tensor, Tensor, float | Tensor], microbatch: int, precision: str, *,
    capture: bool = False, metrics: bool = False,
    compiled_update: Callable[[], Tensor] | None = None,
) -> dict[str, Any]:
    x, targets, other, mixing = batch
    count = len(targets)
    if microbatch < 1 or count < 1 or len(x) != count or len(other) != count:
        raise ValueError("Invalid logical update or microbatch")
    if torch.is_tensor(mixing) and mixing.shape != targets.shape:
        raise ValueError("Per-example MixUp weights must match the logical batch")
    device = x.device
    if capture or compiled_update is None:
        optimizer.zero_grad(set_to_none=True)
    if capture:
        parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        prior = {n: p.detach().cpu().clone() for n, p in parameters.items()}
        head_values = {n: [] for n in OUTPUT_NAMES}
    diagnostics = {}
    loss_terms, correct_terms = [], []
    for start in range(0, count, microbatch):
        end = min(start + microbatch, count)
        sl = slice(start, end)
        lam = mixing[sl] if torch.is_tensor(mixing) else mixing
        begin_cudagraph_step(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
        ):
            output = runtime(x[sl])
            logits, loss, diag = training_objective(model, output, targets[sl], other[sl], lam)
        (loss * ((end - start) / count)).backward()
        if capture:
            for name, value in zip(OUTPUT_NAMES, output, strict=True):
                head_values[name].append(value.detach().cpu().clone())
        if metrics or capture:
            loss_terms.append(loss.detach() * (end - start))
            correct_terms.append(logits.detach().argmax(-1).eq(targets[sl]).sum())
            for name, value in diag.items():
                diagnostics.setdefault(name, []).append(value.detach() * (end - start))
        del output, logits, loss, diag
    if capture:
        missing = [n for n, p in parameters.items() if p.grad is None]
        if missing:
            raise RuntimeError(f"Missing accumulated gradients: {missing}")
        gradients = {n: p.grad.detach().cpu().clone() for n, p in parameters.items()}
    if compiled_update is None:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    else:
        norm = compiled_update()
    result = {
        "samples": count,
        "microbatches": (count + microbatch - 1) // microbatch,
        "optimizer_steps": 1,
        "clip_norm": norm.detach(),
    }
    if metrics or capture:
        result.update(
            loss=torch.stack(loss_terms).sum() / count,
            correct=torch.stack(correct_terms).sum(),
            diagnostics={k: torch.stack(v).sum() / count for k, v in diagnostics.items()},
        )
    if capture:
        checks = {
            "outputs": {k: torch.cat(v, dim=0) for k, v in head_values.items()},
            "grads": gradients,
            "updates": {n: p.detach().cpu() - prior[n] for n, p in parameters.items()},
        }
        if not all(
            torch.isfinite(t).all() for tensors in checks.values() for t in tensors.values()
        ):
            raise FloatingPointError("Nonfinite forward, accumulated gradient, or AdamW update")
        result["checks"] = checks
    return result
