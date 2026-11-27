# ruff: noqa: ANN001, ANN201, ANN401, PLC0415, PLR0915, PTH123, SLF001, TC003
"""Single-pass dual evaluation (stored EMA RMS vs clean dataset RMS) for PolewiseRunningRMS."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imagenet100_data_runtime as data
import torch
import va_imagenet100_runtime as runtime
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_CALIBRATION_SAMPLES = 2048
DEFAULT_CALIBRATION_BATCH_SIZE = 128
DEFAULT_SUBSET_SEED = 501
DEFAULT_LOADER_SEED = 10_501


def make_calibration_loader(
    root: Path,
    *,
    sample_count: int = DEFAULT_CALIBRATION_SAMPLES,
    batch_size: int = DEFAULT_CALIBRATION_BATCH_SIZE,
    workers: int | None = None,
    subset_seed: int = DEFAULT_SUBSET_SEED,
    loader_seed: int = DEFAULT_LOADER_SEED,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader[Any]:
    if sample_count <= 0:
        raise ValueError("calibration sample_count must be positive")
    if batch_size <= 0:
        raise ValueError("calibration batch_size must be positive")
    dataset = ImageFolder(root / "train", data.transforms_for_imagenet100()[1])
    if len(dataset) == 0:
        raise RuntimeError("calibration training dataset is empty")
    count = min(sample_count, len(dataset))
    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(subset_seed),
    )[:count]
    active_workers = min(4, data.active_loader_workers(4 if workers is None else workers))
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        num_workers=active_workers,
        prefetch_factor=prefetch_factor if active_workers else None,
        persistent_workers=persistent_workers and active_workers > 0,
        generator=torch.Generator().manual_seed(loader_seed),
        pin_memory=torch.cuda.is_available(),
    )


def make_evaluator(
    root: Path,
    *,
    sample_count: int = DEFAULT_CALIBRATION_SAMPLES,
    batch_size: int = DEFAULT_CALIBRATION_BATCH_SIZE,
    workers: int | None = None,
    subset_seed: int = DEFAULT_SUBSET_SEED,
    loader_seed: int = DEFAULT_LOADER_SEED,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> Callable[..., dict[str, float]]:
    loader = make_calibration_loader(
        root,
        sample_count=sample_count,
        batch_size=batch_size,
        workers=workers,
        subset_seed=subset_seed,
        loader_seed=loader_seed,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return partial(evaluate, calibration_loader=loader)


@torch.no_grad()
def calibrate_rms_moments(
    model: nn.Module,
    engine: nn.Module,
    calibration_loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> tuple[Tensor, Tensor]:
    model.eval()
    engine.eval()
    features: list[Tensor] = []
    with torch.inference_mode():
        for inputs, _ in data.device_batches(
            calibration_loader, device, channels_last=channels_last
        ):
            data.begin_cudagraph_step(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bfloat16",
            ):
                features.append(engine(inputs)[-1].float().clone())
    descriptors = torch.cat(features, dim=0)
    std = model.classifier.affine.standardizer
    mean = descriptors.mean(dim=0)
    centered = descriptors - mean if std.center_mean else descriptors
    if std.complex_modulus:
        re_c, im_c = centered.chunk(2, dim=-1)
        ms = (re_c.square() + im_c.square()).mean(dim=0)
    else:
        ms = centered.square().mean(dim=0)
    return mean, ms


@torch.inference_mode()
def freeze_full_dataset_rms(
    models: list[nn.Module],
    data_root: Path,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int | None = None,
    precision: str,
    channels_last: bool,
    compact_manifest: bool = False,
    clean_loader_workers: bool = False,
) -> tuple[Tensor, int, dict[str, float | int]]:
    """Replace terminal EMA moments with clean, full-training-split moments."""
    from alphabet.image_layers import LivePolewiseComplexRMS

    standards = [model.classifier.affine.standardizer for model in models]
    if not all(isinstance(std, LivePolewiseComplexRMS) for std in standards):
        raise TypeError("Expected live terminal RMS")
    transform = data.transforms_for_imagenet100()[1]
    if compact_manifest:
        from imagenet_compact_manifest import CompactImageFolder
        from torchvision.datasets.folder import pil_loader

        dataset = CompactImageFolder(
            data_root / "train", transform, pil_loader, mapped=clean_loader_workers
        )
    else:
        dataset = ImageFolder(data_root / "train", transform)
    active_workers = data.active_loader_workers(workers)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=active_workers,
        prefetch_factor=prefetch_factor if active_workers else None,
        pin_memory=device.type == "cuda",
        worker_init_fn=data.init_loader_worker if active_workers else None,
        multiprocessing_context=data.clean_worker_context()
        if active_workers and clean_loader_workers
        else None,
    )
    sums = [
        torch.zeros_like(std.running_ms, dtype=torch.float64, device=device) for std in standards
    ]
    sample_count = 0
    for model in models:
        model.eval()
    for inputs, _ in data.device_batches(loader, device, channels_last=channels_last):
        for model, total in zip(models, sums, strict=True):
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
            ):
                descriptors = model(inputs)[-1]
            re, im = descriptors.double().chunk(2, dim=-1)
            total.add_((re.square() + im.square()).sum(dim=0))
        sample_count += len(inputs)
    if sample_count == 0 or sample_count != len(dataset):
        raise RuntimeError("full-dataset RMS freeze did not process the entire train split")

    results = []
    for std, sum_sq in zip(standards, sums, strict=True):
        ms_full = (sum_sq / sample_count).float()
        rms = (ms_full + std.eps * std.eps).sqrt()
        stats = {
            "sample_count": sample_count,
            "poles": rms.numel(),
            "min": rms.min().item(),
            "median": rms.median().item(),
            "max": rms.max().item(),
            "frac_pole_rms_lt_eps": (ms_full.sqrt() < std.eps).float().mean().item(),
        }
        std.running_ms.copy_(ms_full)
        std.ema_weight.fill_(1.0)
        std.running_mean.zero_()
        results.append((ms_full, sample_count, stats))
    return results


@torch.inference_mode()
def _score_from_descriptors(
    classifier: Any,
    descriptors_gpu: Tensor,
    targets_cpu: Tensor,
) -> dict[str, float]:
    standardized = classifier.affine.standardizer(descriptors_gpu)
    logits = classifier.affine.linear(standardized).float().cpu()
    cross_entropy = float(functional.cross_entropy(logits, targets_cpu))
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(targets_cpu).float().mean()),
        "cross_entropy": cross_entropy,
        "nll": cross_entropy,
        "ece": runtime.expected_calibration_error(logits, targets_cpu),
        "affine_only_accuracy": float(logits.argmax(dim=-1).eq(targets_cpu).float().mean()),
        "affine_only_nll": cross_entropy,
        "affine_reconstruction_max_error": 0.0,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    engine: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    calibration_loader: DataLoader[Any],
    dual_eval: bool = True,
    precision: str,
    channels_last: bool = False,
) -> dict[str, float]:
    classifier = model.classifier
    std = classifier.affine.standardizer
    saved_mean = std.running_mean.clone()
    saved_ms = std.running_ms.clone()
    saved_w = std.ema_weight.clone()

    clean_mean, clean_ms = calibrate_rms_moments(
        model,
        engine,
        calibration_loader,
        device,
        precision=precision,
        channels_last=channels_last,
    )

    model.eval()
    engine.eval()
    val_descriptors: list[Tensor] = []
    val_joint: list[Tensor] = []
    val_affine: list[Tensor] = []
    val_labels: list[Tensor] = []
    try:
        std.running_mean.copy_(
            clean_mean.to(device=std.running_mean.device, dtype=std.running_mean.dtype)
        )
        std.running_ms.copy_(clean_ms.to(device=std.running_ms.device, dtype=std.running_ms.dtype))
        std.ema_weight.fill_(1.0)
        with torch.inference_mode():
            for inputs, targets in data.device_batches(loader, device, channels_last=channels_last):
                data.begin_cudagraph_step(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bfloat16",
                ):
                    output = engine(inputs)
                val_joint.append(output[0].detach().float().cpu())
                val_affine.append(output[1].detach().float().cpu())
                val_descriptors.append(output[-1].detach().float())
                val_labels.append(targets.cpu())

        descriptors_gpu = torch.cat(val_descriptors, dim=0)
        joint = torch.cat(val_joint, dim=0)
        affine = torch.cat(val_affine, dim=0)
        target = torch.cat(val_labels, dim=0)

        cross_entropy = float(functional.cross_entropy(joint, target))
        with torch.inference_mode():
            standardized = classifier.affine.standardizer(descriptors_gpu)
            reconstructed = classifier.affine.linear(standardized).float().cpu()
        calibrated = {
            "accuracy": float(joint.argmax(dim=-1).eq(target).float().mean()),
            "cross_entropy": cross_entropy,
            "nll": cross_entropy,
            "ece": runtime.expected_calibration_error(joint, target),
            "affine_only_accuracy": float(affine.argmax(dim=-1).eq(target).float().mean()),
            "affine_only_nll": float(functional.cross_entropy(affine.float(), target)),
            "affine_reconstruction_max_error": float((affine - reconstructed).abs().max()),
        }
    finally:
        std.running_mean.copy_(saved_mean)
        std.running_ms.copy_(saved_ms)
        std.ema_weight.copy_(saved_w)

    model._latest_evaluation_metrics = calibrated

    w_safe = float(saved_w.clamp_min(1e-12).item())
    stored_rms = (saved_ms / w_safe).clamp_min(0.0).sqrt()
    clean_rms = clean_ms.clamp_min(0.0).sqrt()
    term = model.backbone.terminal
    in_w = term.pole_input_projection.input_norm.weight.detach().float()
    in_rms = float(in_w.square().mean().sqrt().item())
    dx = term.damping_logits_x.detach().float()
    dy = term.damping_logits_y.detach().float()

    record = {
        "calibrated_rms_eval": {
            key: calibrated[key] for key in ("accuracy", "cross_entropy", "ece")
        },
        "affine_reconstruction_max_error": calibrated["affine_reconstruction_max_error"],
        "stored_pole_rms_median": float(stored_rms.median()),
        "clean_pole_rms_median": float(clean_rms.median()),
        "clean_pole_rms_min": float(clean_rms.min()),
        "clean_pole_rms_max": float(clean_rms.max()),
        "frac_pole_rms_lt_eps": float((clean_rms < std.eps).float().mean()),
        "input_norm_rms": in_rms,
        "dx_logit_mean": float(dx.mean()),
        "dy_logit_mean": float(dy.mean()),
    }
    if dual_eval:
        stored = _score_from_descriptors(classifier, descriptors_gpu, target)
        record["stored_rms_eval"] = {
            key: stored[key] for key in ("accuracy", "cross_entropy", "ece")
        }
    if os.environ.get("VA_DUAL_EVAL_LOG_SKIP") != "1":
        with open(
            os.environ.get("VA_DUAL_EVAL_LOG", "dual-eval.jsonl"), "a", encoding="utf-8"
        ) as log:
            log.write(json.dumps(record, allow_nan=False) + "\n")
    return calibrated


@torch.inference_mode()
def verify_folded_affine(model, engine, loader, device, *, precision, channels_last):
    """Score folded FP32 affine logits on the same BF16 validation descriptors."""
    from alphabet.image_layers import fold_terminal_rms_for_inference

    descriptors, labels = [], []
    for inputs, targets in data.device_batches(loader, device, channels_last=channels_last):
        data.begin_cudagraph_step(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
        ):
            descriptors.append(engine(inputs)[-1].float().clone())
        labels.append(targets.cpu())
    features = torch.cat(descriptors)
    targets = torch.cat(labels)
    affine = model.classifier.affine
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        frozen = affine(features).float()
        fold_terminal_rms_for_inference(model)
        folded = affine(features).float()
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    error = float((folded - frozen).abs().max())
    if not error < 1e-3:
        raise RuntimeError(f"frozen versus folded affine max error {error}")
    return {
        "folded_affine_reconstruction_max_error": error,
        "folded_affine_only_accuracy": float(
            folded.cpu().argmax(dim=-1).eq(targets).float().mean()
        ),
        "folded_affine_only_cross_entropy": float(functional.cross_entropy(folded.cpu(), targets)),
    }
