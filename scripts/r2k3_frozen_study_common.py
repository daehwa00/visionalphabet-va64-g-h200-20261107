"""Shared no-training utilities for the frozen K128/D2262 checkpoint study."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import contextlib
import hashlib
import json
import math
import random
import types
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_k128_depth_p_overnight_imagenet100 as runner
import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

if TYPE_CHECKING:
    from lnet.complex_scan_stage import ComplexScanStage
    from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
    from lnet.pac_gated_post_fusion import GatedComplexPostFusion
    from lnet.pac_phase_gated_transition import PathOnlyCollapse
    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

VARIANT = runner.D2262_P128_TERM
EXPECTED_CHECKPOINT_SHA256 = (
    "0f5d65e667ddd49f9c68c16db5a16d873631549ab773ef7acf8d6db5dab94f46"
)
STAGE_NAMES = ("stage1", "stage2", "stage3", "terminal")
ComplexField = tuple[Tensor, Tensor]


class _TransformDataset(Protocol):
    samples: list[tuple[str, int]]
    targets: list[int]
    class_to_idx: dict[str, int]


@dataclass(frozen=True, slots=True)
class FrozenModel:
    model: nn.Module
    payload: dict[str, Any]
    checkpoint: Path
    checkpoint_sha256: str
    initial_state_sha256: str


@dataclass(frozen=True, slots=True)
class PredictionResult:
    indices: Tensor
    labels: Tensor
    logits: Tensor
    losses: Tensor
    margins: Tensor
    confidences: Tensor
    predictions: Tensor

    @property
    def count(self) -> int:
        return int(self.labels.numel())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_mapping_sha256(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        active = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(active.dtype).encode())
        digest.update(json.dumps(list(active.shape)).encode())
        digest.update(active.numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    return tensor_mapping_sha256(model.state_dict())


def load_frozen_model(checkpoint: Path, device: torch.device) -> FrozenModel:
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA256 mismatch: {checkpoint_sha256}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity = {key: payload.get(key) for key in ("variant", "seed", "epoch")}
    expected = {"variant": VARIANT, "seed": 501, "epoch": 100}
    if identity != expected:
        raise RuntimeError(f"checkpoint identity mismatch: {identity}")
    torch.manual_seed(501)
    model = runner._build(VARIANT, runtime.model_config())
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()
    return FrozenModel(
        model=model,
        payload=payload,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        initial_state_sha256=model_state_sha256(model),
    )


def dataset_manifest(dataset: _TransformDataset, split_root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(dataset.class_to_idx, sort_keys=True).encode())
    for filename, target in dataset.samples:
        path = Path(filename)
        metadata = path.stat()
        digest.update(path.relative_to(split_root).as_posix().encode())
        digest.update(f"\0{target}\0{metadata.st_size}\0{metadata.st_mtime_ns}\n".encode())
    return digest.hexdigest()


class IndexedDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(self, source: Dataset[tuple[Tensor, int]]) -> None:
        self.source = source

    def __len__(self) -> int:
        return len(cast("Any", self.source))

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        image, target = self.source[index]
        return image, int(target), index


def imagefolder(
    data_root: Path,
    split: str,
    *,
    evaluation_view: bool = False,
) -> datasets.ImageFolder:
    training_transform, evaluation_transform = harness._transforms()
    transform = (
        training_transform
        if split == "train" and not evaluation_view
        else evaluation_transform
    )
    return datasets.ImageFolder(data_root / split, transform=transform)


def balanced_indices(targets: Sequence[int], per_class: int) -> list[int]:
    if per_class <= 0:
        return list(range(len(targets)))
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)
    selected: list[int] = []
    for target in sorted(by_class):
        candidates = by_class[target]
        if len(candidates) < per_class:
            raise ValueError(f"class {target} has fewer than {per_class} samples")
        for sample in range(per_class):
            position = ((2 * sample + 1) * len(candidates)) // (2 * per_class)
            selected.append(candidates[min(position, len(candidates) - 1)])
    return selected


def make_loader(
    dataset: Dataset[tuple[Tensor, int]],
    *,
    batch_size: int,
    workers: int,
    indices: Sequence[int] | None = None,
) -> DataLoader[Any]:
    indexed: Dataset[Any] = IndexedDataset(dataset)
    source: Dataset[Any] = indexed
    if indices is not None:
        source = Subset(indexed, list(indices))
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


def classifier_logits(model: nn.Module, images: Tensor) -> Tensor:
    output = cast("Any", model)(images)
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("model emitted unsupported classifier output")


def collect_predictions(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    *,
    require_input_grad: bool = False,
) -> PredictionResult:
    indices: list[Tensor] = []
    labels: list[Tensor] = []
    logits: list[Tensor] = []
    context: contextlib.AbstractContextManager[None]
    context = contextlib.nullcontext() if require_input_grad else torch.inference_mode()
    with context:
        for images_cpu, targets_cpu, indices_cpu in loader:
            images = images_cpu.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                active_logits = classifier_logits(model, images)
            indices.append(indices_cpu.long())
            labels.append(targets_cpu.long())
            logits.append(active_logits.detach().float().cpu())
    active_indices = torch.cat(indices)
    active_labels = torch.cat(labels)
    active_logits = torch.cat(logits)
    losses = functional.cross_entropy(active_logits, active_labels, reduction="none")
    probabilities = active_logits.softmax(-1)
    confidences, predictions = probabilities.max(-1)
    true_logits = active_logits.gather(1, active_labels[:, None]).squeeze(1)
    masked = active_logits.clone()
    masked.scatter_(1, active_labels[:, None], -torch.inf)
    margins = true_logits - masked.max(-1).values
    return PredictionResult(
        active_indices,
        active_labels,
        active_logits,
        losses,
        margins,
        confidences,
        predictions,
    )


def ece(probabilities: Tensor, labels: Tensor, bins: int = 15) -> float:
    confidence, prediction = probabilities.max(-1)
    correct = prediction.eq(labels)
    total = labels.numel()
    value = 0.0
    for lower in torch.linspace(0, 1, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += float(selected.float().mean()) * abs(
                float(correct[selected].float().mean()) - float(confidence[selected].mean())
            )
    return value if total else 0.0


def prediction_metrics(result: PredictionResult) -> dict[str, Any]:
    probabilities = result.logits.softmax(-1)
    one_hot = functional.one_hot(result.labels, result.logits.shape[-1]).float()
    correct = result.predictions.eq(result.labels)
    per_class: dict[str, Any] = {}
    for label in result.labels.unique(sorted=True).tolist():
        selected = result.labels == int(label)
        class_correct = int(correct[selected].sum())
        class_samples = int(selected.sum())
        per_class[str(label)] = {
            "correct": class_correct,
            "samples": class_samples,
            "accuracy": class_correct / max(1, class_samples),
            "cross_entropy": float(result.losses[selected].mean()),
        }
    sorted_confidence, order = result.confidences.sort(descending=True)
    sorted_correct = correct[order].float()
    cumulative_risk = 1.0 - sorted_correct.cumsum(0) / torch.arange(
        1, result.count + 1, dtype=torch.float32
    )
    coverage_points = (0.25, 0.5, 0.75, 1.0)
    risk_coverage = {
        str(coverage): float(cumulative_risk[max(0, math.ceil(coverage * result.count) - 1)])
        for coverage in coverage_points
    }
    correct_count = int(correct.sum())
    return {
        "correct": correct_count,
        "samples": result.count,
        "accuracy": correct_count / max(1, result.count),
        "cross_entropy_sum": float(result.losses.sum()),
        "cross_entropy_mean": float(result.losses.mean()),
        "nll": float(result.losses.mean()),
        "brier": float((probabilities - one_hot).square().sum(-1).mean()),
        "ece_15": ece(probabilities, result.labels),
        "mean_confidence": float(result.confidences.mean()),
        "mean_margin": float(result.margins.mean()),
        "top5_accuracy": float(
            result.logits.topk(5, dim=-1).indices.eq(result.labels[:, None]).any(-1).float().mean()
        ),
        "risk_coverage": risk_coverage,
        "per_class": per_class,
        "confidence_order_checksum": hashlib.sha256(sorted_confidence.numpy().tobytes()).hexdigest(),
    }


def save_predictions(path: Path, result: PredictionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(
        {
            "indices": result.indices,
            "labels": result.labels,
            "logits": result.logits,
            "losses": result.losses,
            "margins": result.margins,
            "confidences": result.confidences,
            "predictions": result.predictions,
        },
        temporary,
    )
    temporary.replace(path)


def load_predictions(path: Path) -> PredictionResult:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return PredictionResult(**payload)


def bootstrap_mean_ci(
    values: Tensor,
    labels: Tensor,
    *,
    repeats: int = 10_000,
    seed: int = 20260901,
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    class_values = [values[labels == label].float() for label in labels.unique(sorted=True)]
    draws: list[Tensor] = []
    chunk = 200
    for start in range(0, repeats, chunk):
        active = min(chunk, repeats - start)
        class_draws = []
        for selected in class_values:
            indices = torch.randint(
                len(selected),
                (active, len(selected)),
                generator=generator,
            )
            class_draws.append(selected[indices].mean(1))
        draws.append(torch.stack(class_draws).mean(0))
    distribution = torch.cat(draws)
    return float(distribution.quantile(0.025)), float(distribution.quantile(0.975))


def paired_effect(
    baseline: PredictionResult,
    candidate: PredictionResult,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 20260901,
) -> dict[str, Any]:
    if not torch.equal(baseline.indices, candidate.indices) or not torch.equal(
        baseline.labels, candidate.labels
    ):
        raise RuntimeError("paired predictions do not share identical samples/order")
    baseline_correct = baseline.predictions.eq(baseline.labels).float()
    candidate_correct = candidate.predictions.eq(candidate.labels).float()
    accuracy_delta = candidate_correct - baseline_correct
    loss_delta = candidate.losses - baseline.losses
    margin_delta = candidate.margins - baseline.margins
    confidence_delta = candidate.confidences - baseline.confidences
    accuracy_ci = bootstrap_mean_ci(
        accuracy_delta,
        baseline.labels,
        repeats=bootstrap_repeats,
        seed=seed,
    )
    loss_ci = bootstrap_mean_ci(
        loss_delta,
        baseline.labels,
        repeats=bootstrap_repeats,
        seed=seed + 1,
    )
    return {
        "accuracy_delta_pp": 100.0 * float(accuracy_delta.mean()),
        "accuracy_delta_pp_ci95": [100.0 * accuracy_ci[0], 100.0 * accuracy_ci[1]],
        "cross_entropy_delta": float(loss_delta.mean()),
        "cross_entropy_delta_ci95": list(loss_ci),
        "margin_delta": float(margin_delta.mean()),
        "confidence_delta": float(confidence_delta.mean()),
        "prediction_flips": int((candidate.predictions != baseline.predictions).sum()),
        "correct_to_wrong": int(((baseline_correct == 1) & (candidate_correct == 0)).sum()),
        "wrong_to_correct": int(((baseline_correct == 0) & (candidate_correct == 1)).sum()),
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": seed,
    }


def scan_modules(model: nn.Module) -> dict[str, ComplexScanStage]:
    modules = {name: cast("ComplexScanStage", getattr(model, name)) for name in STAGE_NAMES}
    base = cast("Any", model).same_resolution_blocks
    for resolution, block in base.items():
        modules[f"same_resolution.{resolution}"] = block.pole_scan
    extras = getattr(model, "extra_same_resolution_blocks", {})
    if "14" in extras:
        for index, block in enumerate(extras["14"], 1):
            modules[f"extra14.{index}"] = block.pole_scan
    return modules


def same_resolution_blocks(model: nn.Module) -> dict[str, SameResolutionPoleScanBlock]:
    blocks: dict[str, SameResolutionPoleScanBlock] = {}
    for resolution, block in cast("Any", model).same_resolution_blocks.items():
        blocks[f"same_resolution.{resolution}"] = block
    extras = getattr(model, "extra_same_resolution_blocks", {})
    if "14" in extras:
        for index, block in enumerate(extras["14"], 1):
            blocks[f"extra14.{index}"] = block
    return blocks


def path_modules(model: nn.Module) -> dict[str, PathOnlyCollapse]:
    modules: dict[str, PathOnlyCollapse] = {}
    for name in ("stage1", "stage2", "stage3"):
        modules[f"main.{name}"] = getattr(model, name).quadrant_path_mode_combiner
    for name, block in same_resolution_blocks(model).items():
        modules[name] = block.path_collapse
    return modules


def reader_modules(model: nn.Module) -> dict[str, FactorizedComplexConv2dReader]:
    modules: dict[str, FactorizedComplexConv2dReader] = {}
    for name in STAGE_NAMES:
        modules[f"main.{name}"] = getattr(model, name).pole_input_projection
    for name, block in same_resolution_blocks(model).items():
        modules[name] = block.reader
    return modules


def postfusion_modules(model: nn.Module) -> dict[str, GatedComplexPostFusion]:
    modules: dict[str, GatedComplexPostFusion] = {}
    for name in ("stage1", "stage2", "stage3"):
        modules[f"main.{name}"] = getattr(model, name).augmented.post_fusion
    for name, block in same_resolution_blocks(model).items():
        modules[name] = block.post_fusion
    return modules


@contextlib.contextmanager
def temporary_parameter_values(changes: Mapping[nn.Parameter, Tensor]) -> Generator[None]:
    backups = {parameter: parameter.detach().clone() for parameter in changes}
    with torch.no_grad():
        for parameter, value in changes.items():
            parameter.copy_(value)
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in backups.items():
                parameter.copy_(value)


@contextlib.contextmanager
def phase_intervention(
    modules: Mapping[str, ComplexScanStage],
    *,
    kind: str,
    seed: int = 0,
) -> Generator[None]:
    changes: dict[nn.Parameter, Tensor] = {}
    generator = torch.Generator().manual_seed(seed)
    for module in modules.values():
        if kind == "omega_zero":
            changes[module.phase_x] = torch.zeros_like(module.phase_x)
            changes[module.phase_y] = torch.zeros_like(module.phase_y)
        elif kind == "omega_sign_flip":
            changes[module.phase_x] = -module.phase_x.detach()
            changes[module.phase_y] = -module.phase_y.detach()
        elif kind == "omega_shuffle":
            permutation = torch.randperm(module.modes, generator=generator).to(module.phase_x.device)
            changes[module.phase_x] = module.phase_x.detach()[permutation]
            changes[module.phase_y] = module.phase_y.detach()[permutation]
        elif kind == "xy_swap":
            changes[module.phase_x] = module.phase_y.detach().clone()
            changes[module.phase_y] = module.phase_x.detach().clone()
            changes[module.damping_logits_x] = module.damping_logits_y.detach().clone()
            changes[module.damping_logits_y] = module.damping_logits_x.detach().clone()
        else:
            raise ValueError(f"unsupported phase intervention: {kind}")
    with temporary_parameter_values(changes):
        yield


@contextlib.contextmanager
def damping_shuffle(
    modules: Mapping[str, ComplexScanStage],
    *,
    seed: int,
) -> Generator[None]:
    generator = torch.Generator().manual_seed(seed)
    changes: dict[nn.Parameter, Tensor] = {}
    for module in modules.values():
        permutation = torch.randperm(module.modes, generator=generator).to(
            module.damping_logits_x.device
        )
        changes[module.damping_logits_x] = module.damping_logits_x.detach()[permutation]
        changes[module.damping_logits_y] = module.damping_logits_y.detach()[permutation]
    with temporary_parameter_values(changes):
        yield


@contextlib.contextmanager
def coefficient_intervention(
    modules: Mapping[str, ComplexScanStage],
    transform: Callable[
        [str, ComplexScanStage, tuple[Tensor, Tensor, Tensor, Tensor], str],
        tuple[Tensor, Tensor, Tensor, Tensor],
    ],
) -> Generator[None]:
    backups: dict[ComplexScanStage, Callable[..., Any]] = {}
    for label, module in modules.items():
        original = module.pole_coefficients
        backups[module] = original

        def replacement(
            active: ComplexScanStage,
            shape: tuple[int, int, int, int],
            *,
            module_label: str = label,
            base: Callable[..., Any] = original,
        ) -> Any:
            pole_x, pole_y = base(shape)
            return (
                transform(module_label, active, pole_x, "x"),
                transform(module_label, active, pole_y, "y"),
            )

        module.pole_coefficients = types.MethodType(replacement, module)  # type: ignore[method-assign]
    try:
        yield
    finally:
        for module, original in backups.items():
            module.pole_coefficients = original  # type: ignore[method-assign]


def recurrence_off_transform(
    _label: str,
    _module: ComplexScanStage,
    pole: tuple[Tensor, Tensor, Tensor, Tensor],
    _axis: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    decay_real, decay_imag, gamma_real, gamma_imag = pole
    # A literal complex zero hits an undefined 0**0 corner in the readable
    # reference path.  1e-6 leaves at most one-millionth of the previous state
    # and is numerically equivalent to recurrence-off at model precision.
    return (
        torch.full_like(decay_real, 1.0e-6),
        torch.zeros_like(decay_imag),
        gamma_real,
        gamma_imag,
    )


def half_life_cap_transform(
    half_life: float,
) -> Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]]:
    maximum_magnitude = 0.5 ** (1.0 / half_life)

    def transform(
        _label: str,
        _module: ComplexScanStage,
        pole: tuple[Tensor, Tensor, Tensor, Tensor],
        _axis: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        decay_real, decay_imag, gamma_real, gamma_imag = pole
        magnitude = torch.sqrt(decay_real.square() + decay_imag.square()).clamp_min(1.0e-12)
        scale = torch.clamp_max(maximum_magnitude / magnitude, 1.0)
        return decay_real * scale, decay_imag * scale, gamma_real, gamma_imag

    return transform


def pole_mask_transform(
    selected: Mapping[str, Sequence[int]],
) -> Callable[..., tuple[Tensor, Tensor, Tensor, Tensor]]:
    def transform(
        label: str,
        _module: ComplexScanStage,
        pole: tuple[Tensor, Tensor, Tensor, Tensor],
        _axis: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        decay_real, decay_imag, gamma_real, gamma_imag = pole
        indices = selected.get(label, ())
        if not indices:
            return pole
        mask = torch.ones_like(gamma_real)
        mask[..., list(indices)] = 0
        return decay_real, decay_imag, gamma_real * mask, gamma_imag * mask

    return transform


@contextlib.contextmanager
def path_intervention(
    modules: Mapping[str, PathOnlyCollapse],
    paths: Sequence[int],
) -> Generator[None]:
    backups: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for module in modules.values():
            for parameter in (
                module.path_input.weight_real,
                module.path_input.weight_imag,
                module.path_input.conjugate_real,
                module.path_input.conjugate_imag,
            ):
                backups.append((parameter, parameter.detach().clone()))
                parameter[..., list(paths)] = 0
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in backups:
                parameter.copy_(value)


@contextlib.contextmanager
def path_permutation(
    modules: Mapping[str, PathOnlyCollapse],
    permutation: Sequence[int],
) -> Generator[None]:
    if sorted(permutation) != [0, 1, 2, 3]:
        raise ValueError("D4 path permutation must contain 0, 1, 2, 3 exactly once")
    backups: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for module in modules.values():
            for parameter in (
                module.path_input.weight_real,
                module.path_input.weight_imag,
                module.path_input.conjugate_real,
                module.path_input.conjugate_imag,
            ):
                backups.append((parameter, parameter.detach().clone()))
                parameter.copy_(parameter.detach()[..., list(permutation)])
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in backups:
                parameter.copy_(value)


@contextlib.contextmanager
def reader_component_two_off(
    readers: Mapping[str, FactorizedComplexConv2dReader],
) -> Generator[None]:
    backups: list[tuple[Tensor, Tensor]] = []
    with torch.no_grad():
        for reader in readers.values():
            for parameter in (reader.spatial_weight_real, reader.spatial_weight_imag):
                backups.append((parameter, parameter.detach().clone()))
                parameter[:, 1].zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in backups:
                parameter.copy_(value)


@contextlib.contextmanager
def postfusion_scale(
    modules: Mapping[str, GatedComplexPostFusion],
    scale: float,
) -> Generator[None]:
    with contextlib.ExitStack() as stack:
        for module in modules.values():
            def hook(
                _module: nn.Module,
                inputs: tuple[Tensor, ...],
                output: ComplexField,
                *,
                active_scale: float = scale,
            ) -> ComplexField:
                return (
                    inputs[0] + active_scale * (output[0] - inputs[0]),
                    inputs[1] + active_scale * (output[1] - inputs[1]),
                )

            stack.callback(module.register_forward_hook(hook).remove)
        yield


@contextlib.contextmanager
def main_transition_branch(
    model: nn.Module,
    *,
    zero_memory: bool = False,
    zero_carry: bool = False,
) -> Generator[None]:
    with contextlib.ExitStack() as stack:
        for name in ("stage1", "stage2", "stage3"):
            transition = getattr(model, name).augmented

            def prehook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
                values = list(inputs)
                if zero_memory:
                    values[0] = torch.zeros_like(values[0])
                    values[1] = torch.zeros_like(values[1])
                if zero_carry:
                    values[2] = torch.zeros_like(values[2])
                    values[3] = torch.zeros_like(values[3])
                return tuple(values)

            stack.callback(transition.register_forward_pre_hook(prehook).remove)
        yield


@contextlib.contextmanager
def same_resolution_branch(
    blocks: Mapping[str, SameResolutionPoleScanBlock],
    *,
    zero_memory: bool = False,
    zero_carry: bool = False,
) -> Generator[None]:
    backups: dict[SameResolutionPoleScanBlock, Callable[..., Any]] = {}
    for block in blocks.values():
        original = block.forward
        backups[block] = original

        def replacement(
            active: SameResolutionPoleScanBlock,
            real: Tensor,
            imag: Tensor,
        ) -> ComplexField:
            memory_real, memory_imag = active.collapsed_memory(real, imag)
            if zero_memory:
                memory_real, memory_imag = torch.zeros_like(memory_real), torch.zeros_like(memory_imag)
            if zero_carry:
                real, imag = torch.zeros_like(real), torch.zeros_like(imag)
            return active.post_fusion(real + memory_real, imag + memory_imag)

        block.forward = types.MethodType(replacement, block)  # type: ignore[method-assign]
    try:
        yield
    finally:
        for block, original in backups.items():
            block.forward = original  # type: ignore[method-assign]


@contextlib.contextmanager
def branch_intervention(
    model: nn.Module,
    *,
    zero_memory: bool = False,
    zero_carry: bool = False,
) -> Generator[None]:
    with main_transition_branch(
        model, zero_memory=zero_memory, zero_carry=zero_carry
    ), same_resolution_branch(
        same_resolution_blocks(model),
        zero_memory=zero_memory,
        zero_carry=zero_carry,
    ):
        yield


def module_scope(
    modules: Mapping[str, Any],
    selector: str,
) -> dict[str, Any]:
    if selector == "all":
        return dict(modules)
    if selector == "main":
        return {name: module for name, module in modules.items() if name in STAGE_NAMES or name.startswith("main.")}
    if selector == "terminal":
        return {name: module for name, module in modules.items() if name.endswith("terminal")}
    if selector == "extra14":
        return {name: module for name, module in modules.items() if name.startswith("extra14.")}
    if selector in modules:
        return {selector: modules[selector]}
    raise KeyError(f"unknown module selector: {selector}")


def random_matched_indices(
    modules: Mapping[str, ComplexScanStage],
    counts: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, list[int]]:
    generator = random.Random(seed)  # noqa: S311 - deterministic experiment control
    output: dict[str, list[int]] = {}
    for name, count in counts.items():
        modes = modules[name].modes
        output[name] = sorted(generator.sample(range(modes), min(count, modes)))
    return output


def bh_fdr(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    previous = 1.0
    for rank in range(count, 0, -1):
        name, value = ordered[rank - 1]
        current = min(previous, value * count / rank)
        adjusted[name] = current
        previous = current
    return adjusted


def iter_tensors(value: Any) -> Iterator[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_tensors(item)


def complex_output(value: Any) -> ComplexField | None:
    if isinstance(value, tuple) and len(value) >= 2:
        if isinstance(value[0], Tensor) and isinstance(value[1], Tensor):
            return value[0], value[1]
        if isinstance(value[0], tuple) and len(value[0]) == 2:
            first = value[0]
            if isinstance(first[0], Tensor) and isinstance(first[1], Tensor):
                return first[0], first[1]
    return None


def pooled_complex(field: ComplexField) -> Tensor:
    real, imag = field
    spatial_axes = tuple(range(1, real.ndim - 1))
    if spatial_axes:
        real = real.float().mean(spatial_axes)
        imag = imag.float().mean(spatial_axes)
    return torch.cat((real.float(), imag.float()), dim=-1)


def representation_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules: dict[str, nn.Module] = {}
    base = cast("Any", model).same_resolution_blocks
    for resolution in ("56", "28", "14", "7"):
        modules[f"r{resolution}/base/E_out"] = base[resolution]
    extras = getattr(model, "extra_same_resolution_blocks", {})
    if "14" in extras:
        for index, block in enumerate(extras["14"], 1):
            modules[f"r14/extra{index}/E_out"] = block
    for name in ("stage1", "stage2", "stage3"):
        modules[f"main/{name}/E_out"] = getattr(model, name)
    return modules


@contextlib.contextmanager
def capture_pooled_representations(
    model: nn.Module,
) -> Generator[dict[str, list[Tensor]]]:
    captured: dict[str, list[Tensor]] = {name: [] for name in representation_modules(model)}
    with contextlib.ExitStack() as stack:
        for name, module in representation_modules(model).items():
            def hook(
                _module: nn.Module,
                _inputs: tuple[Tensor, ...],
                output: Any,
                *,
                label: str = name,
            ) -> None:
                field = complex_output(output)
                if field is not None:
                    captured[label].append(pooled_complex(field).detach().cpu())

            stack.callback(module.register_forward_hook(hook).remove)
        yield captured


def flatten_captured(captured: Mapping[str, Sequence[Tensor]]) -> dict[str, Tensor]:
    return {name: torch.cat(list(values)).float() for name, values in captured.items() if values}


def artifact_record(root: Path, artifact_id: str, path: Path, media_type: str) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"artifact escapes study root: {path}")
    return {
        "artifact_id": artifact_id,
        "path": str(resolved.relative_to(resolved_root)),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "media_type": media_type,
    }


def write_family(root: Path, family: Mapping[str, Any]) -> Path:
    family_id = str(family["family_id"])
    path = root / "families" / f"{family_id}.json"
    atomic_json(path, dict(family))
    return path


__all__ = [
    "EXPECTED_CHECKPOINT_SHA256",
    "STAGE_NAMES",
    "VARIANT",
    "FrozenModel",
    "IndexedDataset",
    "PredictionResult",
    "artifact_record",
    "atomic_json",
    "balanced_indices",
    "bh_fdr",
    "bootstrap_mean_ci",
    "branch_intervention",
    "capture_pooled_representations",
    "coefficient_intervention",
    "collect_predictions",
    "complex_output",
    "damping_shuffle",
    "dataset_manifest",
    "flatten_captured",
    "half_life_cap_transform",
    "imagefolder",
    "load_frozen_model",
    "load_predictions",
    "make_loader",
    "model_state_sha256",
    "module_scope",
    "paired_effect",
    "path_intervention",
    "path_modules",
    "path_permutation",
    "phase_intervention",
    "pole_mask_transform",
    "pooled_complex",
    "postfusion_modules",
    "postfusion_scale",
    "prediction_metrics",
    "random_matched_indices",
    "reader_component_two_off",
    "reader_modules",
    "recurrence_off_transform",
    "representation_modules",
    "same_resolution_blocks",
    "save_predictions",
    "scan_modules",
    "sha256_file",
    "tensor_mapping_sha256",
    "write_family",
]
