#!/usr/bin/env python3
"""Compute frozen-checkpoint ERF, integrated gradients, and occlusion maps."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from r2k3_frozen_study_common import (
    artifact_record,
    atomic_json,
    balanced_indices,
    branch_intervention,
    classifier_logits,
    coefficient_intervention,
    imagefolder,
    load_frozen_model,
    make_loader,
    model_state_sha256,
    path_intervention,
    path_modules,
    pole_mask_transform,
    scan_modules,
    sha256_file,
    write_family,
)
from torch import Tensor

mpl.use("Agg")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--ig-images", type=int, default=20)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--occlusion-images", type=int, default=20)
    parser.add_argument("--occlusion-grid", type=int, default=7)
    parser.add_argument("--max-new-conditions", type=int, default=0)
    return parser.parse_args()


def _study(root: Path) -> dict[str, Any]:
    return json.loads((root / "study.json").read_text())


def _upsert_artifact(study: dict[str, Any], record: dict[str, Any]) -> None:
    artifacts = cast("list[dict[str, Any]]", study.setdefault("artifacts", []))
    artifacts[:] = [
        item
        for item in artifacts
        if item.get("artifact_id") != record["artifact_id"]
        and item.get("path") != record["path"]
    ]
    artifacts.append(record)


def _map_metrics(value: Tensor) -> dict[str, float]:
    active = value.float().clamp_min(0)
    total = active.sum()
    if float(total) <= 0.0:
        return {
            "area50_fraction": 0.0,
            "area90_fraction": 0.0,
            "radius50_pixels": 0.0,
            "radius90_pixels": 0.0,
            "entropy": 0.0,
            "zero_mass": 1.0,
        }
    probability = active.flatten() / total
    ordered = probability.sort(descending=True).values.cumsum(0)
    count50 = int(torch.searchsorted(ordered, torch.tensor(0.5)).item() + 1)
    count90 = int(torch.searchsorted(ordered, torch.tensor(0.9)).item() + 1)
    height, width = active.shape
    yy, xx = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )
    radius = torch.sqrt(
        (yy - (height - 1) / 2.0).square() + (xx - (width - 1) / 2.0).square()
    ).flatten()
    order = radius.argsort()
    radial = probability[order].cumsum(0)
    radius50 = float(radius[order][torch.searchsorted(radial, torch.tensor(0.5))])
    radius90 = float(radius[order][torch.searchsorted(radial, torch.tensor(0.9))])
    return {
        "area50_fraction": count50 / probability.numel(),
        "area90_fraction": count90 / probability.numel(),
        "radius50_pixels": radius50,
        "radius90_pixels": radius90,
        "entropy": float(-(probability * probability.clamp_min(1.0e-30).log()).sum()),
        "zero_mass": 0.0,
    }


def _gradient_erf(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    context: contextlib.AbstractContextManager[Any],
) -> tuple[Tensor, list[dict[str, float]]]:
    maps: list[Tensor] = []
    metrics: list[dict[str, float]] = []
    with context:
        for images_cpu, labels_cpu, _indices in loader:
            images = images_cpu.to(device).requires_grad_(requires_grad=True)
            labels = labels_cpu.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = classifier_logits(model, images)
            selected = logits.gather(1, labels[:, None]).sum()
            gradient = torch.autograd.grad(selected, images, retain_graph=False)[0]
            saliency = gradient.detach().float().abs().sum(1)
            for value in saliency:
                normalized = value / value.sum().clamp_min(1.0e-30)
                maps.append(normalized.cpu())
                metrics.append(_map_metrics(normalized.cpu()))
    return torch.stack(maps), metrics


def _integrated_gradients(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    images_limit: int,
    steps: int,
) -> tuple[Tensor, list[dict[str, float]]]:
    maps: list[Tensor] = []
    metrics: list[dict[str, float]] = []
    seen = 0
    for images_cpu, labels_cpu, _indices in loader:
        for image_cpu, label_cpu in zip(images_cpu, labels_cpu, strict=True):
            if seen >= images_limit:
                return torch.stack(maps), metrics
            image = image_cpu[None].to(device)
            baseline = torch.zeros_like(image)
            alpha = torch.linspace(0, 1, steps, device=device).view(steps, 1, 1, 1)
            interpolated = (baseline + alpha * (image - baseline)).requires_grad_(
                requires_grad=True
            )
            labels = torch.full((steps,), int(label_cpu), device=device, dtype=torch.long)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = classifier_logits(model, interpolated)
            target = logits.gather(1, labels[:, None]).sum()
            gradient = torch.autograd.grad(target, interpolated)[0].float().mean(0, keepdim=True)
            attribution = ((image - baseline).float() * gradient).abs().sum(1)[0]
            attribution /= attribution.sum().clamp_min(1.0e-30)
            maps.append(attribution.cpu())
            metrics.append(_map_metrics(attribution.cpu()))
            seen += 1
    return torch.stack(maps), metrics


def _occlusion(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    images_limit: int,
    grid: int,
) -> tuple[Tensor, list[dict[str, float]]]:
    maps: list[Tensor] = []
    metrics: list[dict[str, float]] = []
    seen = 0
    with torch.inference_mode():
        for images_cpu, labels_cpu, _indices in loader:
            for image_cpu, label_cpu in zip(images_cpu, labels_cpu, strict=True):
                if seen >= images_limit:
                    return torch.stack(maps), metrics
                image = image_cpu[None].to(device)
                label = int(label_cpu)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    baseline_logit = float(classifier_logits(model, image)[0, label])
                side = 224 // grid
                variants = image.repeat(grid * grid, 1, 1, 1)
                for y in range(grid):
                    for x in range(grid):
                        variants[y * grid + x, :, y * side : (y + 1) * side, x * side : (x + 1) * side] = 0
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = classifier_logits(model, variants)[:, label].float()
                coarse = (baseline_logit - logits).clamp_min(0).reshape(1, 1, grid, grid)
                active = torch.nn.functional.interpolate(
                    coarse,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                active /= active.sum().clamp_min(1.0e-30)
                maps.append(active.cpu())
                metrics.append(_map_metrics(active.cpu()))
                seen += 1
    return torch.stack(maps), metrics


def _aggregate(name: str, maps: Tensor, metrics: list[dict[str, float]]) -> dict[str, Any]:
    keys = metrics[0]
    return {
        "condition_id": name,
        "status": "complete",
        "metrics": {
            "images": len(maps),
            "mean": {key: sum(row[key] for row in metrics) / len(metrics) for key in keys},
            "std": {
                key: float(torch.tensor([row[key] for row in metrics]).std(unbiased=False))
                for key in keys
            },
        },
    }


def _long_memory_selection(model: torch.nn.Module) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    for name, module in scan_modules(model).items():
        damping_x, damping_y = module._damping_fields()
        radial = torch.sqrt(damping_x.reshape(-1) * damping_y.reshape(-1))
        selected[name] = sorted(map(int, torch.argsort(radial)[: module.modes // 3].tolist()))
    return selected


def _figure(root: Path, maps: dict[str, Tensor]) -> tuple[Path, Path]:
    names = ("gradient_full", "gradient_memory_zero", "gradient_carry_zero")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, name in zip(axes, names, strict=True):
        image = axis.imshow(maps[name].mean(0), cmap="magma")
        axis.set_title(name)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    path = root / "figures" / "FIG05_full_memory_carry_erf.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    sidecar = path.with_suffix(".json")
    atomic_json(
        sidecar,
        {
            "figure_id": "FIG05_full_memory_carry_erf",
            "panels": [
                {"panel_id": name, "source": f"features/erf_maps.pt#/{name}"} for name in names
            ],
            "caption": "Class-logit effective receptive fields for full, memory-zero, and carry-zero inference.",
        },
    )
    return path, sidecar


def main() -> None:
    args = _args()
    study = _study(args.root)
    device = torch.device(args.device)
    frozen = load_frozen_model(args.checkpoint, device)
    dataset = imagefolder(args.data_root, "val")
    indices = balanced_indices(dataset.targets, args.per_class)
    loader = make_loader(dataset, batch_size=1, workers=0, indices=indices)
    scans = scan_modules(frozen.model)
    paths = path_modules(frozen.model)
    cache = args.root / "features" / "erf_maps.pt"
    family_path = args.root / "families" / "F10_erf_attribution.json"
    maps: dict[str, Tensor] = (
        torch.load(cache, map_location="cpu", weights_only=True) if cache.is_file() else {}
    )
    existing_conditions: dict[str, dict[str, Any]] = {}
    if family_path.is_file():
        existing_family = json.loads(family_path.read_text())
        existing_conditions = {
            str(condition["condition_id"]): condition
            for condition in existing_family.get("conditions", [])
        }
    new_conditions = 0
    contexts: tuple[tuple[str, contextlib.AbstractContextManager[Any]], ...] = (
        ("gradient_full", contextlib.nullcontext()),
        ("gradient_memory_zero", branch_intervention(frozen.model, zero_memory=True)),
        ("gradient_carry_zero", branch_intervention(frozen.model, zero_carry=True)),
        (
            "gradient_long_memory_off",
            coefficient_intervention(
                scans,
                pole_mask_transform(_long_memory_selection(frozen.model)),
            ),
        ),
        ("gradient_path0_off", path_intervention(paths, (0,))),
    )
    for name, context in contexts:
        if name not in maps:
            if args.max_new_conditions and new_conditions >= args.max_new_conditions:
                break
            active_maps, active_metrics = _gradient_erf(frozen.model, loader, device, context)
            maps[name] = active_maps
            existing_conditions[name] = _aggregate(name, active_maps, active_metrics)
            new_conditions += 1
            print(json.dumps({"event": "erf_condition_complete", "condition": name}), flush=True)
    if (
        "integrated_gradients_full" not in maps
        and (not args.max_new_conditions or new_conditions < args.max_new_conditions)
    ):
        ig_maps, ig_metrics = _integrated_gradients(
            frozen.model,
            loader,
            device,
            args.ig_images,
            args.ig_steps,
        )
        maps["integrated_gradients_full"] = ig_maps
        existing_conditions["integrated_gradients_full"] = _aggregate(
            "integrated_gradients_full", ig_maps, ig_metrics
        )
        new_conditions += 1
    if (
        "occlusion_full" not in maps
        and (not args.max_new_conditions or new_conditions < args.max_new_conditions)
    ):
        occlusion_maps, occlusion_metrics = _occlusion(
            frozen.model,
            loader,
            device,
            args.occlusion_images,
            args.occlusion_grid,
        )
        maps["occlusion_full"] = occlusion_maps
        existing_conditions["occlusion_full"] = _aggregate(
            "occlusion_full", occlusion_maps, occlusion_metrics
        )
        new_conditions += 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(maps, cache)
    required_ids = [
        "gradient_full",
        "gradient_memory_zero",
        "gradient_carry_zero",
        "gradient_long_memory_off",
        "gradient_path0_off",
        "integrated_gradients_full",
        "occlusion_full",
    ]
    observed_ids = sorted(existing_conditions)
    conditions = [existing_conditions[name] for name in observed_ids]
    complete = set(observed_ids) == set(required_ids)
    family = {
        "family_id": "F10_erf_attribution",
        "schema": "lnet.r2k3.study.family.v1",
        "status": "complete" if complete else "partial",
        "required_condition_ids": sorted(required_ids),
        "observed_condition_ids": observed_ids,
        "conditions": conditions,
        "protocol": {
            "images": len(indices),
            "selection": "deterministic balanced validation images",
            "ig_steps": args.ig_steps,
            "occlusion_grid": args.occlusion_grid,
        },
    }
    family_path = write_family(args.root, family)
    cast("dict[str, str]", study["families"])["F10_erf_attribution"] = str(
        family_path.relative_to(args.root)
    )
    for artifact_id, path, media_type in (
        ("F10_erf_attribution", family_path, "application/json"),
        ("A_erf_maps", cache, "application/x-pytorch"),
    ):
        _upsert_artifact(study, artifact_record(args.root, artifact_id, path, media_type))
    if complete:
        figure, sidecar = _figure(args.root, maps)
        figure_record = artifact_record(
            args.root, "FIG05_full_memory_carry_erf", figure, "image/png"
        )
        sidecar_record = artifact_record(
            args.root, "FIG05_full_memory_carry_erf_sidecar", sidecar, "application/json"
        )
        figures = cast("list[dict[str, Any]]", study["figures"])
        figures[:] = [
            item for item in figures if item.get("figure_id") != "FIG05_full_memory_carry_erf"
        ]
        figures.append(
            {
                "figure_id": "FIG05_full_memory_carry_erf",
                **{key: figure_record[key] for key in ("path", "sha256", "bytes")},
                "sidecar_path": sidecar_record["path"],
                "sidecar_sha256": sidecar_record["sha256"],
                "sidecar_bytes": sidecar_record["bytes"],
            }
        )
    study["provenance"]["no_training_guard"]["model_state_sha256_after"] = model_state_sha256(
        frozen.model
    )
    atomic_json(args.root / "study.json", study)
    if model_state_sha256(frozen.model) != frozen.initial_state_sha256:
        raise RuntimeError("ERF analysis changed model state")
    if sha256_file(args.checkpoint) != frozen.checkpoint_sha256:
        raise RuntimeError("ERF analysis changed checkpoint")
    print(
        json.dumps(
            {
                "event": "erf_complete" if complete else "checkpoint_yield",
                "conditions": observed_ids,
                "new_conditions": new_conditions,
            }
        )
    )


if __name__ == "__main__":
    main()
