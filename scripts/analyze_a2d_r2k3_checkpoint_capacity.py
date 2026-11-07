#!/usr/bin/env python3
"""Measure K/P/rank utilization from a trained R2K3 checkpoint.

This is a read-only diagnostic. It reuses the shared covariance,
pole-correlation, and classifier-contribution math while supporting the later
same-resolution depth and factorial runners.
Temporary rank-2 interventions are restored after every validation pass.
"""

from __future__ import annotations

# The experiment runners intentionally expose private construction hooks.
# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_diagnostics as diagnostics
import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_same_resolution_depth_imagenet100 as depth_runner
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as factorial_runner
import run_a2d_r2k3_uniform_prenorm_gated_postfusion_imagenet100 as uniform_runner
import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


SCHEMA = "lnet.a2d.r2k3.checkpoint_capacity.v1"
STAGE_NAMES = runtime.STAGE_NAMES


@dataclass(frozen=True, slots=True)
class RunnerBinding:
    module: Any
    excitation_modes: tuple[int, int, int, int]
    pole_modes: tuple[int, int, int, int]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("uniform", "depth", "factorial"), required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--stats-batches", type=int, default=16)
    parser.add_argument("--rows-per-batch", type=int, default=1024)
    parser.add_argument(
        "--validation-per-class",
        type=int,
        default=10,
        help="Deterministic balanced samples per class; zero uses the full validation set.",
    )
    parser.add_argument(
        "--individual-readers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--skip-rank-ablation", action="store_true")
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def _binding(family: str, variant: str) -> RunnerBinding:
    if family == "uniform":
        module = uniform_runner
        try:
            spec = module.SPECS[variant]
        except KeyError as error:
            raise ValueError(f"unknown uniform variant: {variant}") from error
        return RunnerBinding(module, tuple(spec.excitation_modes), tuple(spec.pole_modes))
    if family == "depth":
        module = depth_runner
        if variant not in module.SPECS:
            raise ValueError(f"unknown depth variant: {variant}")
        modes = (int(module.MODES),) * 4
        return RunnerBinding(module, modes, modes)
    module = factorial_runner
    try:
        spec = module.SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unknown factorial variant: {variant}") from error
    modes = (int(spec.width),) * 4
    return RunnerBinding(module, modes, modes)


def _checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return cast("Path", args.checkpoint)
    return (
        args.campaign_root / "runs" / args.variant / "checkpoints" / f"{args.variant}__seed501.pt"
    )


def _build_model(
    binding: RunnerBinding,
    variant: str,
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, str]:
    model = binding.module._build(variant, runtime.model_config())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("variant") != variant or payload.get("epoch") != 100:
        raise ValueError(f"checkpoint identity mismatch: {checkpoint}")
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    if hasattr(model, "prepare_for_inference_"):
        model.prepare_for_inference_()
    return model, diagnostics.file_sha256(checkpoint)


def _balanced_indices(targets: Sequence[int], per_class: int) -> list[int]:
    if per_class < 0:
        raise ValueError("validation-per-class must be non-negative")
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)
    if per_class == 0:
        selections = [by_class[target] for target in sorted(by_class)]
        return [
            selected[sample]
            for sample in range(max(map(len, selections), default=0))
            for selected in selections
            if sample < len(selected)
        ]
    selections: list[list[int]] = []
    for target in sorted(by_class):
        candidates = by_class[target]
        if len(candidates) < per_class:
            raise ValueError(f"class {target} has fewer than {per_class} validation samples")
        selected: list[int] = []
        for sample in range(per_class):
            position = ((2 * sample + 1) * len(candidates)) // (2 * per_class)
            selected.append(candidates[min(position, len(candidates) - 1)])
        selections.append(selected)
    return [selected[sample] for sample in range(per_class) for selected in selections]


def _dataset_manifest(
    split_root: Path,
    class_to_idx: dict[str, int],
    samples: Sequence[tuple[str, int]],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(class_to_idx, sort_keys=True, separators=(",", ":")).encode())
    for filename, target in samples:
        path = Path(filename)
        metadata = path.stat()
        digest.update(path.relative_to(split_root).as_posix().encode())
        digest.update(f"\0{target}\0{metadata.st_size}\0{metadata.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _loader(
    data_root: Path,
    batch_size: int,
    workers: int,
    device: torch.device,
    per_class: int,
) -> tuple[DataLoader[Any], list[int], str]:
    _, evaluation_transform = harness._transforms()
    split_root = data_root / "val"
    dataset = datasets.ImageFolder(split_root, evaluation_transform)
    indices = _balanced_indices(dataset.targets, per_class)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=harness.PREFETCH_FACTOR if workers > 0 else None,
    )
    manifest = _dataset_manifest(split_root, dataset.class_to_idx, dataset.samples)
    return loader, indices, manifest


def _reader_map(model: nn.Module) -> dict[str, FactorizedComplexConv2dReader]:
    readers: dict[str, FactorizedComplexConv2dReader] = {}
    for name in STAGE_NAMES:
        reader = getattr(model, name).pole_input_projection
        if type(reader) is not FactorizedComplexConv2dReader:
            raise TypeError(f"{name} does not expose a factorized reader")
        readers[f"main.{name}"] = reader

    blocks = getattr(model, "same_resolution_blocks", None)
    if isinstance(blocks, nn.ModuleDict):
        for resolution, block in blocks.items():
            reader = getattr(block, "reader", None)
            if type(reader) is not FactorizedComplexConv2dReader:
                raise TypeError(f"same-resolution block {resolution} lost its reader")
            readers[f"same_resolution.{resolution}"] = reader
    for resolution in (14, 7):
        block = getattr(model, f"block{resolution}", None)
        if block is None:
            continue
        reader = getattr(block, "reader", None)
        if type(reader) is not FactorizedComplexConv2dReader:
            raise TypeError(f"depth block {resolution} lost its reader")
        readers[f"same_resolution.{resolution}"] = reader
    return readers


def _rank2_weight_metrics(reader: FactorizedComplexConv2dReader) -> dict[str, float | int]:
    if reader.rank != 2:
        raise ValueError("rank-2 weight metrics require a rank-2 reader")
    point = torch.complex(
        reader.point_weight_real.detach().double().cpu(),
        reader.point_weight_imag.detach().double().cpu(),
    )
    spatial = torch.complex(
        reader.spatial_weight_real.detach().double().cpu().flatten(2),
        reader.spatial_weight_imag.detach().double().cpu().flatten(2),
    )
    components = (point.unsqueeze(-1) * spatial.unsqueeze(-2)).flatten(2)
    energy = components.abs().square().sum(-1)
    inner = (components[:, 0].conj() * components[:, 1]).sum(-1)
    total = energy.sum(-1).clamp_min(1.0e-30)
    cosine = inner.abs() / (energy[:, 0] * energy[:, 1]).sqrt().clamp_min(1.0e-30)
    determinant = (energy[:, 0] * energy[:, 1] - inner.abs().square()).clamp_min(0)
    discriminant = (total.square() - 4 * determinant).clamp_min(0).sqrt()
    independent_second = (total - discriminant) / (2 * total)
    return {
        "outputs": int(energy.shape[0]),
        "component_2_energy_share_mean": float((energy[:, 1] / total).mean()),
        "minor_component_energy_share_median": float((energy.min(-1).values / total).median()),
        "component_cosine_median": float(cosine.median()),
        "component_cosine_p95": float(torch.quantile(cosine, 0.95)),
        "independent_second_fraction_mean": float(independent_second.mean()),
        "independent_second_fraction_median": float(independent_second.median()),
    }


@contextmanager
def _remove_rank2(
    readers: Sequence[FactorizedComplexConv2dReader],
) -> Generator[None]:
    backups: list[tuple[FactorizedComplexConv2dReader, Tensor, Tensor]] = []
    with torch.no_grad():
        for reader in readers:
            if reader.rank != 2:
                raise TypeError("rank-2 ablation received a non-rank-2 reader")
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


def _selection_map(
    readers: dict[str, FactorizedComplexConv2dReader],
    *,
    individual: bool,
) -> dict[str, tuple[str, ...]]:
    labels = tuple(readers)
    main = tuple(label for label in labels if label.startswith("main."))
    same_resolution = tuple(label for label in labels if label.startswith("same_resolution."))
    selections: dict[str, tuple[str, ...]] = {}
    if individual:
        selections.update({label: (label,) for label in labels})
    selections["main.all"] = main
    if same_resolution:
        selections["same_resolution.all"] = same_resolution
    selections["all"] = labels
    unique: dict[str, tuple[str, ...]] = {}
    seen: set[tuple[str, ...]] = set()
    for name, selection in selections.items():
        if selection and selection not in seen:
            unique[name] = selection
            seen.add(selection)
    return unique


def main() -> None:
    args = _arguments()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    binding = _binding(args.family, args.variant)
    checkpoint = _checkpoint(args)
    started = time.time()
    model, checkpoint_sha = _build_model(binding, args.variant, checkpoint, device)
    loader, validation_indices, dataset_manifest = _loader(
        args.data_root,
        args.batch_size,
        args.workers,
        device,
        args.validation_per_class,
    )
    validation_indices_sha = hashlib.sha256(
        json.dumps(validation_indices, separators=(",", ":")).encode()
    ).hexdigest()
    stats, _postfusion_samples = diagnostics.run_stats(
        model,
        loader,
        device,
        min(args.stats_batches, len(loader)),
        args.rows_per_batch,
    )
    repo = Path(__file__).resolve().parents[1]
    source_paths = runtime.source_dependency_paths(
        repo,
        (
            Path(__file__).stem,
            Path(diagnostics.__file__).stem,
            str(binding.module.__name__),
        ),
    )
    source_manifest = runtime.source_fingerprint(repo, source_paths)
    source_sha256 = {
        "checkpoint_analyzer": diagnostics.file_sha256(Path(__file__)),
        "diagnostics_core": diagnostics.file_sha256(Path(diagnostics.__file__)),
        "runner": diagnostics.file_sha256(Path(binding.module.__file__)),
        "runtime": diagnostics.file_sha256(Path(runtime.__file__)),
        "evaluation_harness": diagnostics.file_sha256(Path(harness.__file__)),
        "model_dependency_tree": source_manifest,
    }
    cache = (
        args.output.parent
        / "cache"
        / (
            f"{args.variant}-val-{validation_indices_sha[:12]}-"
            f"{dataset_manifest[:12]}-{source_manifest[:12]}.pt"
        )
    )
    descriptor, labels = diagnostics.descriptor_cache(
        model,
        loader,
        device,
        cache,
        checkpoint_sha,
        (
            f"val-balanced-{args.validation_per_class}-"
            f"data-{dataset_manifest}-source-{source_manifest}"
        ),
        None,
    )
    parts = diagnostics.descriptor_parts(descriptor, binding.pole_modes)
    readers = _reader_map(model)
    reader_weights = {name: _rank2_weight_metrics(reader) for name, reader in readers.items()}
    pole_activation = {
        name: diagnostics.pole_activation_metrics(part, poles)
        for name, part, poles in zip(STAGE_NAMES, parts, binding.pole_modes, strict=True)
    }
    contribution = diagnostics.classifier_contribution(
        model,
        parts[-1],
        labels,
        binding.pole_modes[-1],
        args.seed,
    )
    baseline = diagnostics.evaluate(model, loader, device)

    ablations: dict[str, Any] = {}
    if not args.skip_rank_ablation:
        for name, selection in _selection_map(
            readers,
            individual=args.individual_readers,
        ).items():
            print(json.dumps({"event": "rank2_ablation", "selection": name}), flush=True)
            with _remove_rank2([readers[label] for label in selection]):
                ablations[name] = diagnostics.evaluate(model, loader, device)

    payload = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "family": args.family,
        "variant": args.variant,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "source_sha256": source_sha256,
        "source_manifest_sha256": source_manifest,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "threads": args.threads,
        "excitation_modes": list(binding.excitation_modes),
        "pole_modes": list(binding.pole_modes),
        "validation": {
            "per_class": args.validation_per_class,
            "samples": len(validation_indices),
            "indices_sha256": validation_indices_sha,
            "dataset_manifest_sha256": dataset_manifest,
        },
        "baseline": baseline,
        "excitation_covariance": stats["excitation_covariance"],
        "pole_activation": pole_activation,
        "terminal_classifier_pole_contribution": contribution,
        "rank2_weight_space": reader_weights,
        "rank2_component_removal": ablations,
        "protocol": {
            "activation_rows": "deterministic class-interleaved rows from a balanced validation subset",
            "dataset_manifest": "class mapping plus relative path, target, size, and nanosecond mtime",
            "pole_correlation": "Pearson correlation of raw Q over image x direction",
            "rank2_weight_space": "two-component complex full-kernel Gram spectrum",
            "rank2_ablation": "remove component 2 and recompute the exact normalized reader kernel",
            "rank2_baseline": "fresh end-to-end evaluation on the same loader and BF16 autocast path",
            "checkpoint_mutation": "temporary in-memory intervention only; checkpoint is never written",
        },
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
