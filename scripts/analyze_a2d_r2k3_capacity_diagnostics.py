#!/usr/bin/env python3
"""Run checkpoint-based capacity diagnostics for trained uniform R2K3 models.

The script is intentionally analysis-only: every intervention is temporary and
the checkpoint is never overwritten.  It measures excitation/pole utilization,
reader-rank dependence, PostFusion dependence and compressibility, frozen-Q
linear separability, and exact affine-head pole removal curves.
"""

from __future__ import annotations

# The campaign runners deliberately expose private construction hooks.
# pyright: reportArgumentType=false, reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import argparse
import json
import math
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_diagnostics as diagnostics
import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_uniform_prenorm_gated_postfusion_imagenet100 as runner
import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision import datasets

from lnet.pac_complex_layers import WidelyLinear
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


STAGE_NAMES = runtime.STAGE_NAMES
TRANSITION_NAMES = STAGE_NAMES[:3]
SCHEMA = "lnet.a2d.r2k3.capacity_diagnostics.v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="*", default=list(runner.VARIANTS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stats-batches", type=int, default=16)
    parser.add_argument("--rows-per-batch", type=int, default=1024)
    parser.add_argument("--probe-epochs", type=int, default=40)
    parser.add_argument("--probe-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _checkpoint(campaign_root: Path, variant: str) -> Path:
    return campaign_root / "runs" / variant / "checkpoints" / f"{variant}__seed501.pt"


def _build_model(
    campaign_root: Path,
    variant: str,
    device: torch.device,
) -> tuple[nn.Module, Path, str]:
    if variant not in runner.SPECS:
        raise ValueError(f"unknown uniform R2K3 variant: {variant}")
    config = runtime.model_config()
    model = runner._build(variant, config)
    checkpoint = _checkpoint(campaign_root, variant)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("variant") != variant or payload.get("epoch") != 100:
        raise ValueError(f"checkpoint identity mismatch: {checkpoint}")
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    if hasattr(model, "prepare_for_inference_"):
        model.prepare_for_inference_()
    return model, checkpoint, diagnostics.file_sha256(checkpoint)


def _loader(
    data_root: Path,
    split: str,
    batch_size: int,
    workers: int,
) -> DataLoader[Any]:
    _, evaluation_transform = harness._transforms()
    dataset = datasets.ImageFolder(data_root / split, evaluation_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=harness.PREFETCH_FACTOR if workers > 0 else None,
    )


@contextmanager
def _postfusion_scales(model: nn.Module, scales: dict[str, float]) -> Generator[None]:
    with ExitStack() as stack:
        for name, scale in scales.items():
            postfusion = getattr(model, name).augmented.post_fusion

            def hook(
                _module: nn.Module,
                inputs: tuple[Tensor, ...],
                output: tuple[Tensor, Tensor],
                *,
                active_scale: float = scale,
            ) -> tuple[Tensor, Tensor]:
                if active_scale == 1.0:
                    return output
                return (
                    inputs[0] + active_scale * (output[0] - inputs[0]),
                    inputs[1] + active_scale * (output[1] - inputs[1]),
                )

            stack.callback(postfusion.register_forward_hook(hook).remove)
        yield


@contextmanager
def _remove_rank2(model: nn.Module, stages: Sequence[str]) -> Generator[None]:
    backups: list[tuple[FactorizedComplexConv2dReader, Tensor, Tensor]] = []
    with torch.no_grad():
        for name in stages:
            reader = getattr(model, name).pole_input_projection
            if type(reader) is not FactorizedComplexConv2dReader or reader.rank != 2:
                raise TypeError(f"{name} is not a rank-2 factorized reader")
            real = reader.spatial_weight_real[:, 1].detach().clone()
            imag = reader.spatial_weight_imag[:, 1].detach().clone()
            backups.append((reader, real, imag))
            reader.spatial_weight_real[:, 1].zero_()
            reader.spatial_weight_imag[:, 1].zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for reader, real, imag in backups:
                reader.spatial_weight_real[:, 1].copy_(real)
                reader.spatial_weight_imag[:, 1].copy_(imag)


def _wl_matrix(layer: WidelyLinear) -> Tensor:
    a = layer.weight_real.detach().float()
    b = layer.weight_imag.detach().float()
    c = layer.conjugate_real.detach().float()
    d = layer.conjugate_imag.detach().float()
    return torch.cat(
        (
            torch.cat((a + c, -b + d), dim=1),
            torch.cat((b + d, a - c), dim=1),
        ),
        dim=0,
    )


def _set_wl_matrix(layer: WidelyLinear, matrix: Tensor) -> None:
    output_modes = layer.output_modes
    input_modes = layer.input_modes
    rr = matrix[:output_modes, :input_modes]
    ri = matrix[:output_modes, input_modes:]
    ir = matrix[output_modes:, :input_modes]
    ii = matrix[output_modes:, input_modes:]
    with torch.no_grad():
        layer.weight_real.copy_(0.5 * (rr + ii))
        layer.conjugate_real.copy_(0.5 * (rr - ii))
        layer.weight_imag.copy_(0.5 * (ir - ri))
        layer.conjugate_imag.copy_(0.5 * (ir + ri))


def _activation_aware_matrix(
    matrix: Tensor,
    samples: Tensor,
    fraction: float,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    weight = matrix.to(device=device, dtype=torch.float32)
    inputs = samples.to(device=device, dtype=torch.float32)
    outputs = inputs @ weight.T
    covariance = outputs.T @ outputs
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    maximum_rank = min(weight.shape)
    retained_rank = max(1, min(maximum_rank, round(fraction * maximum_rank)))
    basis = eigenvectors[:, -retained_rank:]
    reduced = basis @ (basis.T @ weight)
    energy = eigenvalues.clamp_min(0)
    retained = float(energy[-retained_rank:].sum() / energy.sum().clamp_min(1.0e-30))
    error = outputs - inputs @ reduced.T
    relative_error = float(error.square().sum() / outputs.square().sum().clamp_min(1.0e-30))
    return reduced, {
        "rank": retained_rank,
        "maximum_rank": maximum_rank,
        "retained_output_energy": retained,
        "relative_calibration_mse": relative_error,
    }


@contextmanager
def _activation_aware_postfusion(
    model: nn.Module,
    samples: dict[str, dict[str, Tensor]],
    fraction: float,
    device: torch.device,
) -> Generator[dict[str, Any]]:
    backups: dict[str, Tensor] = {}
    diagnostics: dict[str, Any] = {}
    for name in TRANSITION_NAMES:
        postfusion = getattr(model, name).augmented.post_fusion
        diagnostics[name] = {}
        for key, module in (
            ("value", postfusion.value),
            ("gate", postfusion.gate),
            ("out", postfusion.out),
        ):
            if isinstance(module, WidelyLinear):
                matrix = _wl_matrix(module)
                for parameter_name in (
                    "weight_real",
                    "weight_imag",
                    "conjugate_real",
                    "conjugate_imag",
                ):
                    parameter = getattr(module, parameter_name)
                    backups[f"{name}.{key}.{parameter_name}"] = parameter.detach().clone()
            elif isinstance(module, nn.Linear):
                matrix = module.weight.detach().float()
                backups[f"{name}.{key}.weight"] = module.weight.detach().clone()
            else:
                raise TypeError(f"unsupported PostFusion layer: {type(module)}")
            reduced, layer_diagnostics = _activation_aware_matrix(
                matrix,
                samples[name][key],
                fraction,
                device,
            )
            diagnostics[name][key] = layer_diagnostics
            if isinstance(module, WidelyLinear):
                _set_wl_matrix(module, reduced)
            else:
                with torch.no_grad():
                    module.weight.copy_(reduced)
    try:
        yield diagnostics
    finally:
        with torch.no_grad():
            for name in TRANSITION_NAMES:
                postfusion = getattr(model, name).augmented.post_fusion
                for key, module in (
                    ("value", postfusion.value),
                    ("gate", postfusion.gate),
                    ("out", postfusion.out),
                ):
                    if isinstance(module, WidelyLinear):
                        for parameter_name in (
                            "weight_real",
                            "weight_imag",
                            "conjugate_real",
                            "conjugate_imag",
                        ):
                            getattr(module, parameter_name).copy_(
                                backups[f"{name}.{key}.{parameter_name}"]
                            )
                    else:
                        module.weight.copy_(backups[f"{name}.{key}.weight"])


def _fit_frozen_probe(
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, float | int | str]:
    mean = train_features.double().mean(0).float()
    variance = train_features.double().var(0, unbiased=False).float()
    inverse_std = torch.rsqrt(variance + 1.0e-5)
    width = train_features.shape[1]
    torch.manual_seed(seed)
    head = nn.Linear(width, 100).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.2, momentum=0.9)
    generator = torch.Generator().manual_seed(seed)
    count = train_features.shape[0]
    head.train()
    for epoch in range(epochs):
        learning_rate = 0.2 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        order = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            indices = order[start : start + batch_size]
            features = train_features.index_select(0, indices)
            features = ((features - mean) * inverse_std).to(device, non_blocking=True)
            labels = train_labels.index_select(0, indices).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(head(features), labels)
            loss.backward()
            optimizer.step()

    @torch.inference_mode()
    def score(features: Tensor, labels: Tensor) -> tuple[float, float]:
        correct = 0
        loss = 0.0
        for start in range(0, features.shape[0], batch_size):
            active = ((features[start : start + batch_size] - mean) * inverse_std).to(device)
            target = labels[start : start + batch_size].to(device)
            logits = head(active)
            correct += int((logits.argmax(-1) == target).sum())
            loss += float(functional.cross_entropy(logits, target, reduction="sum"))
        return correct / labels.numel(), loss / labels.numel()

    train_accuracy, train_loss = score(train_features, train_labels)
    validation_accuracy, validation_loss = score(validation_features, validation_labels)
    return {
        "train_accuracy": train_accuracy,
        "train_cross_entropy": train_loss,
        "validation_accuracy": validation_accuracy,
        "validation_cross_entropy": validation_loss,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.2,
        "optimizer": "SGD momentum=0.9 cosine; zero-init affine; no weight decay",
    }


def _analyze_variant(
    args: argparse.Namespace, variant: str, device: torch.device
) -> dict[str, Any]:
    print(json.dumps({"event": "variant_start", "variant": variant}), flush=True)
    model, checkpoint, checkpoint_sha = _build_model(args.campaign_root, variant, device)
    spec = runner.SPECS[variant]
    maximum_batches = 2 if args.smoke else None
    stats_batches = 1 if args.smoke else args.stats_batches
    probe_epochs = 1 if args.smoke else args.probe_epochs
    branch_scales = (0.0, 1.0) if args.smoke else (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    low_rank_fractions = (0.5,) if args.smoke else (0.75, 0.5, 0.25)
    validation_loader = _loader(args.data_root, "val", args.batch_size, args.workers)
    stats, postfusion_samples = diagnostics.run_stats(
        model,
        validation_loader,
        device,
        stats_batches,
        args.rows_per_batch,
    )
    baseline = diagnostics.evaluate(
        model, validation_loader, device, maximum_batches=maximum_batches
    )

    postfusion_sweep: dict[str, Any] = {}
    for scale in branch_scales:
        with _postfusion_scales(model, dict.fromkeys(TRANSITION_NAMES, scale)):
            postfusion_sweep[str(scale)] = diagnostics.evaluate(
                model, validation_loader, device, maximum_batches=maximum_batches
            )
    stage_off: dict[str, Any] = {}
    for name in TRANSITION_NAMES:
        with _postfusion_scales(model, {name: 0.0}):
            stage_off[name] = diagnostics.evaluate(
                model, validation_loader, device, maximum_batches=maximum_batches
            )

    rank2_ablation: dict[str, Any] = {}
    for selection in [*STAGE_NAMES, "all"]:
        stages = STAGE_NAMES if selection == "all" else (selection,)
        with _remove_rank2(model, stages):
            rank2_ablation[selection] = diagnostics.evaluate(
                model, validation_loader, device, maximum_batches=maximum_batches
            )

    low_rank: dict[str, Any] = {}
    for fraction in low_rank_fractions:
        with _activation_aware_postfusion(
            model, postfusion_samples, fraction, device
        ) as approximation:
            low_rank[str(fraction)] = {
                "calibration": approximation,
                "validation": diagnostics.evaluate(
                    model, validation_loader, device, maximum_batches=maximum_batches
                ),
            }

    cache_root = args.output.parent / "cache" / variant
    validation_descriptor, validation_labels = diagnostics.descriptor_cache(
        model,
        validation_loader,
        device,
        cache_root / "val.pt",
        checkpoint_sha,
        "val",
        maximum_batches,
    )
    training_loader = _loader(args.data_root, "train", args.batch_size, args.workers)
    training_descriptor, training_labels = diagnostics.descriptor_cache(
        model,
        training_loader,
        device,
        cache_root / "train.pt",
        checkpoint_sha,
        "train",
        maximum_batches,
    )
    train_parts = diagnostics.descriptor_parts(training_descriptor, spec.pole_modes)
    validation_parts = diagnostics.descriptor_parts(validation_descriptor, spec.pole_modes)

    pole_activation = {
        name: diagnostics.pole_activation_metrics(q, poles)
        for name, q, poles in zip(STAGE_NAMES, validation_parts, spec.pole_modes, strict=True)
    }
    probes: dict[str, Any] = {}
    for index, (name, train_q, validation_q) in enumerate(
        zip(STAGE_NAMES, train_parts, validation_parts, strict=True)
    ):
        probes[name] = _fit_frozen_probe(
            train_q,
            training_labels,
            validation_q,
            validation_labels,
            device,
            epochs=probe_epochs,
            batch_size=args.probe_batch_size,
            seed=args.seed + index,
        )
    contribution = diagnostics.classifier_contribution(
        model,
        validation_parts[-1],
        validation_labels,
        spec.pole_modes[-1],
        args.seed,
    )
    payload = {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "excitation_modes": list(spec.excitation_modes),
        "pole_modes": list(spec.pole_modes),
        "baseline": baseline,
        **stats,
        "pole_activation": pole_activation,
        "rank2_component_removal": rank2_ablation,
        "postfusion_branch_scale_sweep": postfusion_sweep,
        "postfusion_stage_off": stage_off,
        "postfusion_activation_aware_low_rank": low_rank,
        "frozen_q_linear_probes": probes,
        "terminal_classifier_pole_contribution": contribution,
    }
    print(
        json.dumps(
            {
                "event": "variant_complete",
                "variant": variant,
                "baseline_accuracy": baseline["accuracy"],
            }
        ),
        flush=True,
    )
    return payload


def main() -> None:
    args = _arguments()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("capacity diagnostics require an available CUDA device")
    variants = list(args.variants)
    if args.smoke:
        variants = variants[:1]
    started = time.time()
    results = {variant: _analyze_variant(args, variant, device) for variant in variants}
    payload = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "smoke": bool(args.smoke),
        "protocol": {
            "activation_samples": "deterministic validation center-crop rows",
            "pole_correlation": "Pearson correlation of raw Q over image x direction",
            "rank2_ablation": "remove component 2 then recompute exact unit-row-energy kernel",
            "postfusion_scale": "input + alpha * (trained_output - input)",
            "low_rank": "output-subspace projection from calibration activation covariance",
            "linear_probe": "frozen deterministic train/val Q; standardized SGD affine probe",
            "pole_removal": "mean-imputed standardized Q coordinates in the trained affine head",
        },
        "results": results,
    }
    diagnostics.atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
