#!/usr/bin/env python3
"""Extract frozen representations and run no-training geometry analyses."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from r2k3_frozen_study_common import (
    artifact_record,
    atomic_json,
    balanced_indices,
    capture_pooled_representations,
    flatten_captured,
    imagefolder,
    load_frozen_model,
    make_loader,
    model_state_sha256,
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-bank-per-class", type=int, default=10)
    parser.add_argument("--tsne-classes", type=int, default=20)
    parser.add_argument("--tsne-per-class", type=int, default=20)
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


def _extract(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    labels: list[Tensor] = []
    indices: list[Tensor] = []
    with capture_pooled_representations(model) as captured, torch.inference_mode():
        for images_cpu, labels_cpu, indices_cpu in loader:
            images = images_cpu.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                cast("Any", model)(images)
            labels.append(labels_cpu.long())
            indices.append(indices_cpu.long())
    return flatten_captured(captured), torch.cat(labels), torch.cat(indices)


def _normalize(values: Tensor) -> Tensor:
    return functional_normalize(values.float())


def functional_normalize(values: Tensor) -> Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _knn(
    bank: Tensor,
    bank_labels: Tensor,
    queries: Tensor,
    labels: Tensor,
    k: int,
) -> float:
    bank = _normalize(bank)
    queries = _normalize(queries)
    correct = 0
    for start in range(0, len(queries), 256):
        similarity = queries[start : start + 256] @ bank.T
        neighbors = similarity.topk(min(k, len(bank)), dim=-1).indices
        votes = torch.nn.functional.one_hot(bank_labels[neighbors], 100).sum(1)
        prediction = votes.argmax(-1)
        correct += int((prediction == labels[start : start + len(prediction)]).sum())
    return correct / len(labels)


def _centroid(
    bank: Tensor,
    bank_labels: Tensor,
    queries: Tensor,
    labels: Tensor,
) -> float:
    centroids = torch.stack([bank[bank_labels == label].mean(0) for label in range(100)])
    prediction = (_normalize(queries) @ _normalize(centroids).T).argmax(-1)
    return float((prediction == labels).float().mean())


def _spectrum(values: Tensor) -> dict[str, Any]:
    centered = values.double() - values.double().mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1.0e-30)
    cumulative = probability.cumsum(0)
    participation = probability.sum().square() / probability.square().sum().clamp_min(1.0e-30)
    entropy = torch.exp(-(probability * probability.clamp_min(1.0e-30).log()).sum())
    return {
        "dimensions": int(values.shape[1]),
        "samples": int(values.shape[0]),
        "participation_ratio": float(participation),
        "entropy_effective_rank": float(entropy),
        "rank_90": int(torch.searchsorted(cumulative, torch.tensor(0.9)).item() + 1),
        "rank_95": int(torch.searchsorted(cumulative, torch.tensor(0.95)).item() + 1),
        "top10_explained": probability[:10].tolist(),
    }


def _class_separation(values: Tensor, labels: Tensor) -> dict[str, float]:
    mean = values.mean(0)
    centroids = torch.stack([values[labels == label].mean(0) for label in range(100)])
    between = (centroids - mean).square().sum(1).mean()
    within = torch.stack(
        [
            (values[labels == label] - centroids[label]).square().sum(1).mean()
            for label in range(100)
        ]
    ).mean()
    return {
        "between_class_variance": float(between),
        "within_class_variance": float(within),
        "between_to_within_ratio": float(between / within.clamp_min(1.0e-30)),
    }


def _linear_cka(first: Tensor, second: Tensor) -> float:
    first = first.double() - first.double().mean(0, keepdim=True)
    second = second.double() - second.double().mean(0, keepdim=True)
    numerator = (first.T @ second).square().sum()
    denominator = torch.sqrt(
        (first.T @ first).square().sum() * (second.T @ second).square().sum()
    ).clamp_min(1.0e-30)
    return float(numerator / denominator)


def _phase_stats(values: Tensor) -> dict[str, float]:
    width = values.shape[1] // 2
    real, imag = values[:, :width], values[:, width:]
    magnitude = torch.hypot(real, imag)
    phase = torch.atan2(imag, real)
    resultant = torch.hypot(phase.cos().mean(), phase.sin().mean())
    return {
        "magnitude_mean": float(magnitude.mean()),
        "magnitude_std": float(magnitude.std(unbiased=False)),
        "phase_circular_resultant": float(resultant),
        "phase_sine_mean": float(phase.sin().mean()),
        "phase_cosine_mean": float(phase.cos().mean()),
    }


def _retrieval(values: Tensor, labels: Tensor, indices: Tensor) -> list[dict[str, Any]]:
    normalized = _normalize(values)
    query_positions = _tsne_selection(labels, classes=100, per_class=1)
    similarity = normalized[query_positions] @ normalized.T
    similarity[torch.arange(len(query_positions)), query_positions] = -torch.inf
    neighbors = similarity.topk(5, dim=-1).indices
    return [
        {
            "query_index": int(indices[position]),
            "query_label": int(labels[position]),
            "neighbor_indices": [int(indices[position]) for position in neighbors[index]],
            "neighbor_labels": [int(labels[position]) for position in neighbors[index]],
            "similarities": similarity[index, neighbors[index]].tolist(),
        }
        for index, position in enumerate(query_positions)
    ]


def _tsne_selection(labels: Tensor, classes: int, per_class: int) -> Tensor:
    selected: list[Tensor] = []
    for label in range(classes):
        positions = torch.nonzero(labels == label, as_tuple=False).flatten()[:per_class]
        selected.append(positions)
    return torch.cat(selected)


def _joint_probabilities(values: Tensor, perplexity: float = 30.0) -> Tensor:
    distances = torch.cdist(values.double(), values.double()).square()
    count = len(values)
    target_entropy = math.log(perplexity)
    conditional = torch.zeros(count, count, dtype=torch.float64)
    for index in range(count):
        lower, upper, beta = -math.inf, math.inf, 1.0
        mask = torch.arange(count) != index
        active_distances = distances[index, mask]
        probability = torch.empty_like(active_distances)
        for _ in range(50):
            probability = torch.exp(-active_distances * beta)
            probability /= probability.sum().clamp_min(1.0e-30)
            entropy = -float((probability * probability.clamp_min(1.0e-30).log()).sum())
            difference = entropy - target_entropy
            if abs(difference) < 1.0e-5:
                break
            if difference > 0:
                lower = beta
                beta = beta * 2.0 if math.isinf(upper) else 0.5 * (beta + upper)
            else:
                upper = beta
                beta = beta * 0.5 if math.isinf(lower) else 0.5 * (beta + lower)
        conditional[index, mask] = probability
    joint = (conditional + conditional.T) / (2.0 * count)
    joint.fill_diagonal_(0)
    return joint.clamp_min(1.0e-12)


def _torch_tsne(
    values: Tensor,
    *,
    perplexity: float = 30.0,
    iterations: int = 1000,
    seed: int = 20260901,
) -> Tensor:
    probability = _joint_probabilities(values, perplexity)
    generator = torch.Generator().manual_seed(seed)
    embedding = 1.0e-4 * torch.randn(len(values), 2, generator=generator, dtype=torch.float64)
    velocity = torch.zeros_like(embedding)
    for iteration in range(iterations):
        difference = embedding[:, None, :] - embedding[None, :, :]
        numerator = 1.0 / (1.0 + difference.square().sum(-1))
        numerator.fill_diagonal_(0)
        q = numerator / numerator.sum().clamp_min(1.0e-30)
        active_probability = probability * (12.0 if iteration < 250 else 1.0)
        gradient = 4.0 * (
            ((active_probability - q) * numerator)[:, :, None] * difference
        ).sum(1)
        momentum = 0.5 if iteration < 250 else 0.8
        velocity = momentum * velocity - 200.0 * gradient
        embedding += velocity
        embedding -= embedding.mean(0, keepdim=True)
    return embedding.float()


def _figures(
    root: Path,
    metrics: dict[str, Any],
    cka: Tensor,
    names: list[str],
    tsne_coordinates: Tensor,
    tsne_labels: Tensor,
) -> tuple[Path, Path]:
    depth_names = [name for name in names if name.startswith("r14/")]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(
        range(len(depth_names)),
        [metrics[name]["spectrum"]["participation_ratio"] for name in depth_names],
        marker="o",
        label="participation rank",
    )
    axes[0].plot(
        range(len(depth_names)),
        [100.0 * metrics[name]["knn_accuracy_k5"] for name in depth_names],
        marker="s",
        label="kNN-5 accuracy (%)",
    )
    axes[0].set_xticks(range(len(depth_names)), [name.split("/")[1] for name in depth_names], rotation=45)
    axes[0].legend()
    axes[0].set_title("14x14 depth geometry")
    image = axes[1].imshow(cka, vmin=0, vmax=1, cmap="viridis")
    axes[1].set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    axes[1].set_yticks(range(len(names)), names, fontsize=6)
    axes[1].set_title("Linear CKA")
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    scatter = axes[2].scatter(
        tsne_coordinates[:, 0],
        tsne_coordinates[:, 1],
        c=tsne_labels,
        s=8,
        cmap="tab20",
    )
    del scatter
    axes[2].set_title("t-SNE: selected classes")
    figure.tight_layout()
    path = root / "figures" / "FIG06_depth_rank_cka_knn.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    sidecar = path.with_suffix(".json")
    atomic_json(
        sidecar,
        {
            "figure_id": "FIG06_depth_rank_cka_knn",
            "panels": [
                {"panel_id": "depth", "source": "families/F09_representation_geometry.json#/conditions"},
                {"panel_id": "cka", "source": "families/F09_representation_geometry.json#/cka"},
                {"panel_id": "tsne", "source": "features/representation_cache.pt"},
            ],
            "caption": "Frozen representation rank, kNN readability, CKA, and t-SNE.",
        },
    )
    return path, sidecar


def main() -> None:
    args = _args()
    study = _study(args.root)
    device = torch.device(args.device)
    frozen = load_frozen_model(args.checkpoint, device)
    train_dataset = imagefolder(args.data_root, "train", evaluation_view=True)
    val_dataset = imagefolder(args.data_root, "val")
    train_indices = balanced_indices(train_dataset.targets, args.train_bank_per_class)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        indices=train_indices,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    train_features, train_labels, train_ids = _extract(frozen.model, train_loader, device)
    val_features, val_labels, val_ids = _extract(frozen.model, val_loader, device)
    names = sorted(set(train_features) & set(val_features))
    metrics: dict[str, Any] = {}
    conditions: list[dict[str, Any]] = []
    for name in names:
        values = val_features[name]
        active = {
            "spectrum": _spectrum(values),
            "class_separation": _class_separation(values, val_labels),
            "knn_accuracy_k1": _knn(train_features[name], train_labels, values, val_labels, 1),
            "knn_accuracy_k5": _knn(train_features[name], train_labels, values, val_labels, 5),
            "knn_accuracy_k20": _knn(train_features[name], train_labels, values, val_labels, 20),
            "nearest_centroid_accuracy": _centroid(
                train_features[name], train_labels, values, val_labels
            ),
            "complex_statistics": _phase_stats(values),
            "retrieval": _retrieval(values, val_labels, val_ids),
        }
        metrics[name] = active
        conditions.append({"condition_id": name, "status": "complete", "metrics": active})
    cka_indices = _tsne_selection(val_labels, classes=100, per_class=10)
    cka_sample = len(cka_indices)
    cka = torch.tensor(
        [
            [
                _linear_cka(
                    val_features[first][cka_indices],
                    val_features[second][cka_indices],
                )
                for second in names
            ]
            for first in names
        ]
    )
    selected = _tsne_selection(val_labels, args.tsne_classes, args.tsne_per_class)
    tsne_source_name = "r14/extra4/E_out" if "r14/extra4/E_out" in val_features else names[-1]
    tsne_source = val_features[tsne_source_name][selected]
    centered = tsne_source - tsne_source.mean(0, keepdim=True)
    _, _, basis = torch.pca_lowrank(centered, q=min(50, centered.shape[1], len(centered) - 1))
    pca50 = centered @ basis
    tsne_coordinates = _torch_tsne(
        pca50,
        perplexity=float(min(30, max(5, len(selected) // 10))),
    )
    cache = args.root / "features" / "representation_cache.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "train_indices": train_ids,
            "val_features": val_features,
            "val_labels": val_labels,
            "val_indices": val_ids,
            "cka": cka,
            "cka_indices": val_ids[cka_indices],
            "names": names,
            "tsne_source": tsne_source_name,
            "tsne_indices": val_ids[selected],
            "tsne_labels": val_labels[selected],
            "tsne_coordinates": tsne_coordinates,
        },
        cache,
    )
    family = {
        "family_id": "F09_representation_geometry",
        "schema": "lnet.r2k3.study.family.v1",
        "status": "complete",
        "required_condition_ids": names,
        "observed_condition_ids": names,
        "conditions": conditions,
        "cka": {"names": names, "matrix": cka.tolist(), "samples": cka_sample},
        "tsne": {
            "source": tsne_source_name,
            "samples": len(selected),
            "classes": args.tsne_classes,
            "per_class": args.tsne_per_class,
            "random_state": 20260901,
            "implementation": "deterministic exact torch t-SNE; 1000 iterations",
        },
        "train_bank": {
            "samples": len(train_labels),
            "per_class": args.train_bank_per_class,
            "view": "deterministic bicubic evaluation transform",
            "note": "No classifier/probe was fit; training images are a frozen kNN/centroid bank only.",
        },
    }
    family_path = write_family(args.root, family)
    cast("dict[str, str]", study["families"])["F09_representation_geometry"] = str(
        family_path.relative_to(args.root)
    )
    _upsert_artifact(
        study,
        artifact_record(args.root, "F09_representation_geometry", family_path, "application/json"),
    )
    _upsert_artifact(
        study,
        artifact_record(args.root, "A_representation_cache", cache, "application/x-pytorch"),
    )
    figure, sidecar = _figures(
        args.root,
        metrics,
        cka,
        names,
        tsne_coordinates,
        val_labels[selected],
    )
    figure_record = artifact_record(args.root, "FIG06_depth_rank_cka_knn", figure, "image/png")
    sidecar_record = artifact_record(
        args.root, "FIG06_depth_rank_cka_knn_sidecar", sidecar, "application/json"
    )
    figures = cast("list[dict[str, Any]]", study.setdefault("figures", []))
    figures[:] = [item for item in figures if item.get("figure_id") != "FIG06_depth_rank_cka_knn"]
    figures.append(
        {
            "figure_id": "FIG06_depth_rank_cka_knn",
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
        raise RuntimeError("representation extraction changed frozen model state")
    if sha256_file(args.checkpoint) != frozen.checkpoint_sha256:
        raise RuntimeError("representation extraction changed checkpoint")
    print(json.dumps({"event": "representations_complete", "layers": names}))


if __name__ == "__main__":
    main()
