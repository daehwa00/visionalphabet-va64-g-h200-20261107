"""Shared checkpoint diagnostics for R2K3 models."""

from __future__ import annotations

# Torch hook and experiment-model boundaries are intentionally structural.
# pyright: reportArgumentType=false, reportExplicitAny=false
import hashlib
import json
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from lnet.pac_gated_post_fusion import GatedComplexPostFusion

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.utils.data import DataLoader


STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
TRANSITION_NAMES = STAGE_NAMES[:3]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complex_sample(real: Tensor, imag: Tensor, retained: int) -> tuple[Tensor, Tensor]:
    rows_real = real.detach().reshape(-1, real.shape[-1])
    rows_imag = imag.detach().reshape_as(rows_real)
    total = rows_real.shape[0]
    if total > retained:
        indices = torch.linspace(0, total - 1, retained, device=real.device).long()
        rows_real = rows_real.index_select(0, indices)
        rows_imag = rows_imag.index_select(0, indices)
    return rows_real.float().cpu(), rows_imag.float().cpu()


def real_sample(value: Tensor, retained: int) -> Tensor:
    rows = value.detach().reshape(-1, value.shape[-1])
    if rows.shape[0] > retained:
        indices = torch.linspace(0, rows.shape[0] - 1, retained, device=value.device).long()
        rows = rows.index_select(0, indices)
    return rows.float().cpu()


def realify(field: tuple[Tensor, Tensor]) -> Tensor:
    return torch.cat(field, dim=-1)


@dataclass
class FieldMoments:
    energy: float = 0.0
    count: int = 0

    def add(self, field: tuple[Tensor, Tensor]) -> None:
        real, imag = field
        self.energy += float(
            real.detach().float().square().sum() + imag.detach().float().square().sum()
        )
        self.count += real.numel()

    def rms(self) -> float:
        return math.sqrt(self.energy / max(1, self.count))


@dataclass
class TransitionMoments:
    memory: FieldMoments = field(default_factory=FieldMoments)
    carry: FieldMoments = field(default_factory=FieldMoments)
    merged: FieldMoments = field(default_factory=FieldMoments)
    update: FieldMoments = field(default_factory=FieldMoments)
    output: FieldMoments = field(default_factory=FieldMoments)
    memory_carry_inner: float = 0.0
    memory_carry_count: int = 0

    def payload(self) -> dict[str, float]:
        memory_rms = self.memory.rms()
        carry_rms = self.carry.rms()
        merged_rms = self.merged.rms()
        update_rms = self.update.rms()
        denominator = math.sqrt(self.memory.energy * self.carry.energy)
        return {
            "memory_rms": memory_rms,
            "carry_rms": carry_rms,
            "carry_to_memory_rms": carry_rms / max(memory_rms, 1.0e-12),
            "memory_carry_cosine": self.memory_carry_inner / max(denominator, 1.0e-12),
            "merged_rms": merged_rms,
            "postfusion_update_rms": update_rms,
            "postfusion_update_to_merged_rms": update_rms / max(merged_rms, 1.0e-12),
            "output_rms": self.output.rms(),
        }


class ActivationCollector:
    def __init__(self, model: nn.Module, rows_per_batch: int) -> None:
        self.model = model
        self.rows_per_batch = rows_per_batch
        self.excitation: dict[str, list[tuple[Tensor, Tensor]]] = {name: [] for name in STAGE_NAMES}
        self.transition = {name: TransitionMoments() for name in TRANSITION_NAMES}
        self.postfusion_inputs: dict[str, dict[str, list[Tensor]]] = {
            name: {"value": [], "gate": [], "out": []} for name in TRANSITION_NAMES
        }
        self.stack = ExitStack()

    def __enter__(self) -> Self:
        for name in STAGE_NAMES:
            stage = getattr(self.model, name)

            def excitation_hook(
                _module: nn.Module,
                inputs: tuple[Tensor, ...],
                *,
                stage_name: str = name,
            ) -> None:
                self.excitation[stage_name].append(
                    complex_sample(inputs[0], inputs[1], self.rows_per_batch)
                )

            self.stack.callback(stage.register_forward_pre_hook(excitation_hook).remove)

        for name in TRANSITION_NAMES:
            transition = getattr(self.model, name).augmented
            if transition is None or not hasattr(transition, "post_fusion"):
                raise TypeError(f"{name} does not expose gated PostFusion")
            postfusion = transition.post_fusion
            if type(postfusion) is not GatedComplexPostFusion:
                raise TypeError(f"{name} PostFusion type changed")

            def transition_hook(
                module: Any,
                inputs: tuple[Tensor, ...],
                output: tuple[Tensor, Tensor],
                *,
                stage_name: str = name,
            ) -> None:
                memory_raw = inputs[0], inputs[1]
                carry_raw = module._carry(inputs[2], inputs[3])
                memory = module._project(module.memory_projection, memory_raw)
                carry = module._project(module.carry_projection, carry_raw)
                merged = memory[0] + carry[0], memory[1] + carry[1]
                update = output[0] - merged[0], output[1] - merged[1]
                moments = self.transition[stage_name]
                moments.memory.add(memory)
                moments.carry.add(carry)
                moments.merged.add(merged)
                moments.update.add(update)
                moments.output.add(output)
                moments.memory_carry_inner += float(
                    (memory[0].detach().float() * carry[0].detach().float()).sum()
                    + (memory[1].detach().float() * carry[1].detach().float()).sum()
                )
                moments.memory_carry_count += memory[0].numel()

            self.stack.callback(transition.register_forward_hook(transition_hook).remove)

            def complex_input_hook(
                _module: nn.Module,
                inputs: tuple[Tensor, ...],
                *,
                stage_name: str = name,
                key: str,
            ) -> None:
                sample = complex_sample(inputs[0], inputs[1], self.rows_per_batch)
                self.postfusion_inputs[stage_name][key].append(realify(sample))

            def real_input_hook(
                _module: nn.Module,
                inputs: tuple[Tensor, ...],
                *,
                stage_name: str = name,
            ) -> None:
                self.postfusion_inputs[stage_name]["gate"].append(
                    real_sample(inputs[0], self.rows_per_batch)
                )

            self.stack.callback(
                postfusion.value.register_forward_pre_hook(
                    lambda module, inputs, n=name: complex_input_hook(
                        module, inputs, stage_name=n, key="value"
                    )
                ).remove
            )
            self.stack.callback(
                postfusion.gate.register_forward_pre_hook(
                    lambda module, inputs, n=name: real_input_hook(module, inputs, stage_name=n)
                ).remove
            )
            self.stack.callback(
                postfusion.out.register_forward_pre_hook(
                    lambda module, inputs, n=name: complex_input_hook(
                        module, inputs, stage_name=n, key="out"
                    )
                ).remove
            )
        return self

    def __exit__(self, *_args: object) -> None:
        self.stack.close()

    def payload(self) -> tuple[dict[str, Any], dict[str, dict[str, Tensor]]]:
        excitation_payload: dict[str, Any] = {}
        for name, batches in self.excitation.items():
            real = torch.cat([batch[0] for batch in batches])
            imag = torch.cat([batch[1] for batch in batches])
            excitation_payload[name] = complex_covariance_metrics(real, imag)
        transition_payload = {name: moments.payload() for name, moments in self.transition.items()}
        inputs = {
            name: {key: torch.cat(values) for key, values in layers.items()}
            for name, layers in self.postfusion_inputs.items()
        }
        return {
            "excitation_covariance": excitation_payload,
            "transition_rms": transition_payload,
        }, inputs


def complex_covariance_metrics(real: Tensor, imag: Tensor) -> dict[str, Any]:
    if real.shape != imag.shape or real.ndim != 2 or real.shape[0] < 2:
        raise ValueError("complex covariance requires aligned sample matrices")
    z = torch.complex(real.double(), imag.double())
    z = z - z.mean(0, keepdim=True)
    covariance = z.mH @ z / (z.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).real.clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1.0e-30)
    probabilities = eigenvalues / total
    cumulative = probabilities.cumsum(0)

    def rank_at(fraction: float) -> int:
        return int(torch.searchsorted(cumulative, cumulative.new_tensor(fraction)).item() + 1)

    participation = float(total.square() / eigenvalues.square().sum().clamp_min(1.0e-30))
    entropy = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()))
    return {
        "samples": int(real.shape[0]),
        "width_k": int(real.shape[1]),
        "effective_rank_participation": participation,
        "effective_rank_fraction": participation / real.shape[1],
        "effective_rank_entropy": entropy,
        "rank_90_percent_energy": rank_at(0.90),
        "rank_95_percent_energy": rank_at(0.95),
        "rank_99_percent_energy": rank_at(0.99),
        "top_eigenvalue_fraction": float(probabilities[0]),
        "condition_95_to_top": float(
            eigenvalues[min(rank_at(0.95) - 1, eigenvalues.numel() - 1)]
            / eigenvalues[0].clamp_min(1.0e-30)
        ),
    }


def run_stats(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    batches: int,
    rows_per_batch: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Tensor]]]:
    with ActivationCollector(model, rows_per_batch) as collector:
        for index, (images, _targets) in enumerate(loader):
            if index >= batches:
                break
            device_images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                cast("Any", model).raw_descriptor(device_images)
    return collector.payload()


def classifier_logits(model: nn.Module, descriptor: Tensor) -> Tensor:
    output = cast("Any", model).classifier(descriptor)
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("unsupported classifier output")


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    *,
    maximum_batches: int | None = None,
) -> dict[str, float]:
    correct = 0
    count = 0
    loss = 0.0
    started = time.perf_counter()
    for index, (images, targets_cpu) in enumerate(loader):
        if maximum_batches is not None and index >= maximum_batches:
            break
        device_images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        targets = targets_cpu.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            descriptor = cast("Any", model).raw_descriptor(device_images)
        logits = classifier_logits(model, descriptor.float()).float()
        loss += float(functional.cross_entropy(logits, targets, reduction="sum"))
        correct += int((logits.argmax(-1) == targets).sum())
        count += targets.numel()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "accuracy": correct / max(1, count),
        "cross_entropy": loss / max(1, count),
        "samples": count,
        "seconds": seconds,
        "images_per_second": count / max(seconds, 1.0e-12),
    }


def descriptor_cache(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    path: Path,
    checkpointfile_sha256: str,
    split: str,
    maximum_batches: int | None,
) -> tuple[Tensor, Tensor]:
    if path.exists() and maximum_batches is None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("checkpointfile_sha256") != checkpointfile_sha256
            or payload.get("split") != split
        ):
            raise RuntimeError(f"stale descriptor cache: {path}")
        return payload["descriptor"].float(), payload["labels"].long()
    descriptors: list[Tensor] = []
    labels: list[Tensor] = []
    started = time.perf_counter()
    for index, (images, targets) in enumerate(loader):
        if maximum_batches is not None and index >= maximum_batches:
            break
        device_images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            descriptor = cast("Any", model).raw_descriptor(device_images)
        descriptors.append(descriptor.detach().cpu())
        labels.append(targets.long())
        if (index + 1) % 100 == 0:
            count = sum(batch.shape[0] for batch in labels)
            print(
                json.dumps(
                    {
                        "event": "descriptor_progress",
                        "split": split,
                        "samples": count,
                        "images_per_second": count / (time.perf_counter() - started),
                    }
                ),
                flush=True,
            )
    features = torch.cat(descriptors)
    targets = torch.cat(labels)
    if maximum_batches is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.pt")
        torch.save(
            {
                "descriptor": features,
                "labels": targets,
                "checkpointfile_sha256": checkpointfile_sha256,
                "split": split,
                "descriptor_dtype": str(features.dtype),
            },
            temporary,
        )
        temporary.replace(path)
    return features.float(), targets


def descriptor_parts(descriptor: Tensor, pole_modes: Sequence[int]) -> list[Tensor]:
    widths = [4 * int(modes) for modes in pole_modes]
    if descriptor.shape[-1] != sum(widths):
        raise ValueError("descriptor does not match the four-stage pole schedule")
    return list(descriptor.split(widths, dim=-1))


def union_find_clusters(correlation: Tensor, threshold: float) -> list[list[int]]:
    count = correlation.shape[0]
    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    pairs = torch.nonzero(torch.triu(correlation.abs() >= threshold, diagonal=1), as_tuple=False)
    for first, second in pairs.tolist():
        union(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(root(index), []).append(index)
    return sorted((group for group in groups.values() if len(group) > 1), key=len, reverse=True)


def pole_activation_metrics(q: Tensor, poles: int) -> dict[str, Any]:
    activation = q.reshape(q.shape[0], 4, poles).reshape(-1, poles).double()
    activation = activation - activation.mean(0, keepdim=True)
    scale = activation.square().mean(0).sqrt().clamp_min(1.0e-12)
    normalized = activation / scale
    correlation = normalized.T @ normalized / normalized.shape[0]
    correlation.fill_diagonal_(0)
    off_diagonal = correlation[torch.triu_indices(poles, poles, offset=1).unbind()]
    covariance = activation.T @ activation / max(1, activation.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    participation = float(
        eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1.0e-30)
    )
    clusters = {
        str(threshold): union_find_clusters(correlation, threshold)
        for threshold in (0.90, 0.95, 0.99)
    }
    return {
        "samples_times_directions": int(activation.shape[0]),
        "poles": poles,
        "activation_effective_rank": participation,
        "activation_effective_rank_fraction": participation / poles,
        "correlation_mean_abs": float(off_diagonal.abs().mean()),
        "correlation_p95_abs": float(off_diagonal.abs().quantile(0.95)),
        "correlation_max": float(off_diagonal.max()),
        "correlation_min": float(off_diagonal.min()),
        "duplicate_clusters": clusters,
        "clustered_poles_at_0.95": sum(len(group) for group in clusters["0.95"]),
        "largest_cluster_at_0.95": max((len(group) for group in clusters["0.95"]), default=1),
    }


def classifier_contribution(
    model: nn.Module,
    q4: Tensor,
    labels: Tensor,
    poles: int,
    seed: int,
) -> dict[str, Any]:
    classifier = cast("Any", model).classifier.affine
    standardizer = classifier.standardizer
    linear = classifier.linear
    mean = standardizer.running_mean.detach().cpu().float()
    variance = standardizer.running_var.detach().cpu().float()
    standardized = (q4.float() - mean) * torch.rsqrt(variance + standardizer.eps)
    weight = linear.weight.detach().cpu().float()
    bias = linear.bias.detach().cpu().float()
    logits = standardized @ weight.T + bias
    baseline = float((logits.argmax(-1) == labels).float().mean())
    contributions: list[Tensor] = []
    importance: list[float] = []
    for pole in range(poles):
        indices = torch.tensor([direction * poles + pole for direction in range(4)])
        contribution = standardized.index_select(1, indices) @ weight.index_select(1, indices).T
        contributions.append(contribution)
        importance.append(float(contribution.square().mean().sqrt()))
    importance_tensor = torch.tensor(importance)
    ascending = torch.argsort(importance_tensor)
    descending = ascending.flip(0)
    generator = torch.Generator().manual_seed(seed)
    fractions = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90)

    def accuracy_after(indices: Tensor) -> float:
        if indices.numel() == 0:
            active = logits
        else:
            removed = torch.stack([contributions[index] for index in indices.tolist()]).sum(0)
            active = logits - removed
        return float((active.argmax(-1) == labels).float().mean())

    def accuracy_with_only(indices: Tensor) -> float:
        active = bias.expand(len(labels), -1).clone()
        if indices.numel() > 0:
            active += torch.stack([contributions[index] for index in indices.tolist()]).sum(0)
        return float((active.argmax(-1) == labels).float().mean())

    curve: list[dict[str, Any]] = []
    for fraction in fractions:
        removed = min(poles, round(fraction * poles))
        bottom = accuracy_after(ascending[:removed])
        top = accuracy_after(descending[:removed])
        repeats = 1 if removed in {0, poles} else 20
        random_scores = [
            accuracy_after(torch.randperm(poles, generator=generator)[:removed])
            for _ in range(repeats)
        ]
        retained = min(poles, round(fraction * poles))
        sufficiency_top = accuracy_with_only(descending[:retained])
        sufficiency_repeats = 1 if retained in {0, poles} else 20
        sufficiency_random_scores = [
            accuracy_with_only(torch.randperm(poles, generator=generator)[:retained])
            for _ in range(sufficiency_repeats)
        ]
        random_tensor = torch.tensor(random_scores)
        sufficiency_random = torch.tensor(sufficiency_random_scores)
        curve.append(
            {
                "removed_fraction": fraction,
                "removed_poles": removed,
                "least_contributing_first_accuracy": bottom,
                "most_contributing_first_accuracy": top,
                "random_accuracy_mean": float(random_tensor.mean()),
                "random_accuracy_std": float(random_tensor.std(unbiased=False)),
                "retained_fraction": fraction,
                "retained_poles": retained,
                "most_contributing_only_accuracy": sufficiency_top,
                "random_only_accuracy_mean": float(sufficiency_random.mean()),
                "random_only_accuracy_std": float(sufficiency_random.std(unbiased=False)),
            }
        )
    normalized = importance_tensor / importance_tensor.sum().clamp_min(1.0e-30)
    effective_poles = float(normalized.sum().square() / normalized.square().sum())
    return {
        "baseline_accuracy_from_cached_q4": baseline,
        "importance_definition": "RMS class-logit contribution of the four Q directions per pole",
        "pole_importance_rms": importance,
        "effective_classifier_poles": effective_poles,
        "effective_classifier_pole_fraction": effective_poles / poles,
        "top_10_percent_importance_share": float(
            importance_tensor.topk(max(1, round(0.1 * poles))).values.sum()
            / importance_tensor.sum().clamp_min(1.0e-30)
        ),
        "removal_curve": curve,
    }
