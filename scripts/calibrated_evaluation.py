"""Calibrate terminal moments from clean training images."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imagenet100_data_runtime as data
import torch
import va_imagenet100_runtime as runtime
from torch.nn import functional
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from lnet.classifier_heads import expected_calibration_error
from lnet.image_layers import LivePolewiseComplexRMS

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn

DEFAULT_CALIBRATION_SAMPLES = 2048
DEFAULT_CALIBRATION_BATCH_SIZE = 128
DEFAULT_SUBSET_SEED = 501
DEFAULT_LOADER_SEED = 502


def make_calibration_loader(
    root: Path,
    *,
    sample_count: int = DEFAULT_CALIBRATION_SAMPLES,
    batch_size: int = DEFAULT_CALIBRATION_BATCH_SIZE,
    workers: int | None = None,
    subset_seed: int = DEFAULT_SUBSET_SEED,
    loader_seed: int = DEFAULT_LOADER_SEED,
    persistent_workers: bool = False,
) -> DataLoader[Any]:
    """Build a deterministic clean-eval DataLoader over a fixed training subset."""
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
    active_workers = (
        min(2, data.active_loader_workers(2))
        if workers is None
        else data.active_loader_workers(workers)
    )
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        num_workers=active_workers,
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
) -> Callable[..., dict[str, float]]:
    """Create an ImageNet-100 evaluator that temporarily calibrates terminal moments."""
    loader = make_calibration_loader(
        root,
        sample_count=sample_count,
        batch_size=batch_size,
        workers=workers,
        subset_seed=subset_seed,
        loader_seed=loader_seed,
        persistent_workers=persistent_workers,
    )
    return partial(evaluate, calibration_loader=loader)


@torch.no_grad()
def calibrate_bn_moments(
    model: nn.Module,
    engine: nn.Module,
    calibration_loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute (mean, variance) of the terminal raw-G descriptor on clean training images."""
    model.eval()
    engine.eval()
    features: list[Tensor] = []
    for inputs, _ in data.device_batches(calibration_loader, device, channels_last=channels_last):
        data.begin_cudagraph_step(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            features.append(engine(inputs)[-1].float().clone())
    if not features:
        raise RuntimeError("calibration loader produced zero batches")
    descriptors = torch.cat(features, dim=0)
    variance, mean = torch.var_mean(descriptors, dim=0, unbiased=descriptors.shape[0] > 1)
    return mean, variance


@torch.inference_mode()
def calibrate_rms_moments(
    model: nn.Module,
    engine: nn.Module,
    calibration_loader: DataLoader[Any],
    device: torch.device,
    *,
    precision: str,
    channels_last: bool = False,
) -> tuple[Tensor, int]:
    """Accumulate clean complex-modulus second moments in float64."""
    std = model.classifier.affine.standardizer
    if not isinstance(std, LivePolewiseComplexRMS):
        raise TypeError("RMS calibration requires LivePolewiseComplexRMS")
    model.eval()
    engine.eval()
    sum_sq = torch.zeros_like(std.running_ms, dtype=torch.float64, device=device)
    sample_count = 0
    for inputs, _ in data.device_batches(calibration_loader, device, channels_last=channels_last):
        data.begin_cudagraph_step(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
        ):
            descriptors = engine(inputs)[-1]
        re, im = descriptors.double().chunk(2, dim=-1)
        sum_sq.add_((re.square() + im.square()).sum(dim=0))
        sample_count += re.shape[0]
    if sample_count == 0:
        raise RuntimeError("calibration loader produced zero batches")
    return (sum_sq / sample_count).float(), sample_count


@torch.inference_mode()
def freeze_full_dataset_rms(
    model: nn.Module,
    engine: nn.Module,
    data_root: Path,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
    precision: str,
    channels_last: bool,
) -> tuple[Tensor, int, dict[str, float | int]]:
    """Replace terminal EMA moments with clean, full-training-split moments."""
    std = model.classifier.affine.standardizer
    if not isinstance(std, LivePolewiseComplexRMS):
        raise TypeError("full-dataset RMS freeze requires LivePolewiseComplexRMS")
    dataset = ImageFolder(data_root / "train", data.transforms_for_imagenet100()[1])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=data.active_loader_workers(workers),
        pin_memory=device.type == "cuda",
    )
    ms_full, sample_count = calibrate_rms_moments(
        model, engine, loader, device, precision=precision, channels_last=channels_last
    )
    if sample_count != len(dataset):
        raise RuntimeError("full-dataset RMS freeze did not process the entire train split")

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
    return ms_full, sample_count, stats


@torch.inference_mode()
def _score_from_descriptors(
    classifier: Any,
    descriptors_gpu: Tensor,
    targets: Tensor,
    reference_affine: Tensor | None = None,
) -> dict[str, float]:
    """Score terminal descriptors with the classifier's current evaluation moments."""
    classify = getattr(classifier, "forward_q4", classifier)
    logits, affine, *_ = classify(descriptors_gpu)
    logits, affine = logits.float().cpu(), affine.float().cpu()
    cross_entropy = float(functional.cross_entropy(logits, targets))
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(targets).float().mean()),
        "cross_entropy": cross_entropy,
        "nll": cross_entropy,
        "ece": expected_calibration_error(logits, targets),
        "affine_only_accuracy": float(affine.argmax(dim=-1).eq(targets).float().mean()),
        "affine_only_nll": float(functional.cross_entropy(affine, targets)),
        "affine_reconstruction_max_error": (
            float((affine - reference_affine).abs().max()) if reference_affine is not None else 0.0
        ),
    }


def _log_dual_evaluation(
    model: nn.Module,
    std: LivePolewiseComplexRMS,
    stored: dict[str, float] | None,
    calibrated: dict[str, float],
    clean_ms: Tensor,
    sample_count: int,
) -> None:
    backbone = getattr(model, "backbone", None)
    terminal = getattr(backbone, "terminal", None)
    path = os.environ.get("VA_DUAL_EVAL_LOG")
    if os.environ.get("VA_DUAL_EVAL_LOG_SKIP") == "1" or (path is None and terminal is None):
        return
    clean_rms = clean_ms.sqrt()
    record = {
        "calibrated_rms_eval": calibrated,
        "calibration_samples": sample_count,
        "clean_pole_rms_min": float(clean_rms.min()),
        "clean_pole_rms_median": float(clean_rms.median()),
        "clean_pole_rms_max": float(clean_rms.max()),
        "frac_pole_rms_lt_eps": float((clean_rms < std.eps).float().mean()),
        "pole_count": clean_rms.numel(),
    }
    if stored is not None:
        record["stored_rms_eval"] = stored
    for field, stage in (
        ("input_norm_rms", terminal),
        ("input_norm_rms_s3", getattr(backbone, "terminal_s3", None)),
    ):
        projection = getattr(stage, "pole_input_projection", None)
        weight = getattr(getattr(projection, "input_norm", None), "weight", None)
        if weight is not None:
            record[field] = float(weight.detach().float().square().mean().sqrt())
    for axis in ("x", "y"):
        damping = getattr(terminal, f"damping_logits_{axis}", None)
        if damping is not None:
            record[f"d{axis}_logit_mean"] = float(damping.detach().float().mean())
    with Path(path or "dual-eval.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, allow_nan=False) + "\n")


@torch.inference_mode()
def _evaluate_live_rms(
    model: nn.Module,
    engine: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    calibration_loader: DataLoader[Any],
    precision: str,
    channels_last: bool,
    std: LivePolewiseComplexRMS,
    dual_eval: bool,
) -> dict[str, float]:
    buffers = (std.running_mean, std.running_ms, std.ema_weight)
    saved = tuple(buffer.clone() for buffer in buffers)
    try:
        clean_ms, sample_count = calibrate_rms_moments(
            model,
            engine,
            calibration_loader,
            device,
            precision=precision,
            channels_last=channels_last,
        )
        descriptors: list[Tensor] = []
        targets: list[Tensor] = []
        affine_outputs: list[Tensor] = []
        for inputs, labels in data.device_batches(loader, device, channels_last=channels_last):
            data.begin_cudagraph_step(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
            ):
                output = engine(inputs)
            descriptors.append(output[-1].float().clone())
            if dual_eval:
                affine_outputs.append(output[1].float().cpu())
            targets.append(labels.cpu())
        descriptor = torch.cat(descriptors)
        target = torch.cat(targets)
        classifier = model.classifier
        stored = (
            _score_from_descriptors(classifier, descriptor, target, torch.cat(affine_outputs))
            if dual_eval else None
        )
        std.running_mean.zero_()
        std.running_ms.copy_(clean_ms)
        std.ema_weight.fill_(1.0)
        calibrated = _score_from_descriptors(classifier, descriptor, target)
        _log_dual_evaluation(model, std, stored, calibrated, clean_ms, sample_count)
        model._latest_evaluation_metrics = calibrated
        return calibrated
    finally:
        for buffer, value in zip(buffers, saved, strict=True):
            buffer.copy_(value)


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
    """Evaluate using temporarily calibrated terminal moments, then restore."""
    bn = model.classifier.affine.standardizer
    if isinstance(bn, LivePolewiseComplexRMS):
        return _evaluate_live_rms(
            model,
            engine,
            loader,
            device,
            calibration_loader=calibration_loader,
            precision=precision,
            channels_last=channels_last,
            std=bn,
            dual_eval=dual_eval,
        )
    saved_mean = bn.running_mean.clone()
    saved_var = bn.running_var.clone()
    mean, variance = calibrate_bn_moments(
        model,
        engine,
        calibration_loader,
        device,
        precision=precision,
        channels_last=channels_last,
    )
    try:
        bn.running_mean.copy_(mean.to(device=bn.running_mean.device, dtype=bn.running_mean.dtype))
        bn.running_var.copy_(variance.to(device=bn.running_var.device, dtype=bn.running_var.dtype))
        return runtime.evaluate(
            model,
            engine,
            loader,
            device,
            precision=precision,
            channels_last=channels_last,
        )
    finally:
        bn.running_mean.copy_(saved_mean)
        bn.running_var.copy_(saved_var)
