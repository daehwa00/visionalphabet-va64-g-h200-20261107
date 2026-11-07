#!/usr/bin/env python3
"""Measure 90-degree input-rotation consistency of pre-collapse D4 paths.

This is a frozen-checkpoint follow-up for ``K128-P160-160-160-128-D2262``.
It performs no fitting and changes no parameter or buffer.  A deterministic,
class-balanced subset of ImageNet-100 validation images is evaluated in its
clean and counter-clockwise 90-degree-rotated forms.  The rotated internal
fields are spatially rotated back before all 24 D4-path permutations are
scored.

Output protocol (all writes are temporary-file + rename):

* ``families/P0-4_input_rotation_d4_path_consistency.json``: protocol,
  provenance, aggregate metrics, permutation ranks, and no-training guard.
* ``artifacts/P0-4_rotation_path_per_example.pt``: paired per-image metrics
  for every module, permutation, and phase convention.
* ``figures/FIG_P0-4_rotation_path_consistency.png`` and adjacent JSON:
  direct-phase permutation heatmap and expected-versus-best summary.

The primary phase convention is *no conjugation*.  The product-scan source
already conjugates pole coefficients for negative scan directions, so all
four paths are represented in one global complex carrier convention.  A
single global conjugation of the rotated response is also scored as a fixed
sensitivity analysis.  Per-path or per-module conjugation is deliberately not
searched because that would be a post-hoc fit.
"""

from __future__ import annotations

# The experiment intentionally observes private scan/collapse contracts that
# define the exact pre-collapse research quantity.
# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportPrivateUsage=false, reportImplicitRelativeImport=false
import argparse
import contextlib
import itertools
import statistics
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import matplotlib.pyplot as plt
import torch
from r2k3_frozen_study_common import (
    artifact_record,
    atomic_json,
    balanced_indices,
    classifier_logits,
    dataset_manifest,
    imagefolder,
    load_frozen_model,
    make_loader,
    model_state_sha256,
    path_modules,
    same_resolution_blocks,
    sha256_file,
)
from torch import Tensor, nn

from lnet.pac_product_scan_pipeline import run_product_scan_pipeline
from lnet.pac_product_scan_reference import (
    direction_relative_cells_to_full_resolution,
    product_scan_full4_reference,
)

if TYPE_CHECKING:
    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


SCHEMA = "lnet.a2d.r2k3.followup.rotation_paths.v1"
FAMILY_ID = "P0-4_input_rotation_d4_path_consistency"
FIGURE_ID = "FIG_P0-4_rotation_path_consistency"
PATH_DIRECTIONS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
PERMUTATIONS = tuple(itertools.permutations(range(4)))


def expected_ccw90_permutation() -> tuple[int, int, int, int]:
    """Map each rotated-frame path to its clean-frame path after alignment.

    A CCW input rotation maps a rotated-frame displacement ``(dx, dy)`` back
    to clean coordinates as ``(-dy, dx)``.  The returned convention is
    ``aligned_rotated[..., p, :]`` versus ``clean[..., permutation[p], :]``.
    """
    clean_index = {direction: index for index, direction in enumerate(PATH_DIRECTIONS)}
    return cast(
        "tuple[int, int, int, int]",
        tuple(clean_index[(-dy, dx)] for dx, dy in PATH_DIRECTIONS),
    )


EXPECTED_PERMUTATION = expected_ccw90_permutation()


def align_rotated_field(field: Tensor) -> Tensor:
    """Undo the input's CCW spatial rotation without changing path labels."""
    if field.ndim != 5 or field.shape[-2] != 4:
        raise ValueError("D4 activation must have shape [B,H,W,4,M]")
    return torch.rot90(field, k=-1, dims=(1, 2))


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _atomic_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    temporary.replace(path)


def _layer_group(name: str) -> str:
    if name.startswith("main."):
        return "main"
    if name.startswith("extra14."):
        return "extra14"
    if name.startswith("same_resolution."):
        return "base"
    raise ValueError(f"unrecognized path-module name: {name}")


def _full_resolution_paths(
    block: SameResolutionPoleScanBlock,
    real: Tensor,
    imag: Tensor,
) -> tuple[Tensor, Tensor]:
    """Replay only a same-resolution scan to expose its fused pre-collapse Q."""
    source, pole_x, pole_y = block._scan_contract(real, imag)
    height, width = source[0].shape[1:3]
    if height % 2 == 0 and width % 2 == 0:
        output = run_product_scan_pipeline(
            pole_x,
            pole_y,
            source,
            epilogue="full16",
            gain_normalization=block.pole_scan.product_gain_normalization,
            memory_policy="retain",
        )
        if not isinstance(output, tuple):
            raise TypeError("full16 path replay did not return a field")
        return (
            direction_relative_cells_to_full_resolution(output[0]),
            direction_relative_cells_to_full_resolution(output[1]),
        )
    # The production full16 packing requires even grids.  The 7x7 base block
    # uses the exact readable full-resolution oracle instead.
    return product_scan_full4_reference(
        pole_x,
        pole_y,
        source,
        gain_normalization=block.pole_scan.product_gain_normalization,
    )


class D4PathCapture:
    """Capture the exact input to every learned four-to-one path collapse."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.module_names = tuple(path_modules(model))
        self._captured: dict[str, tuple[Tensor, Tensor]] = {}
        self._stack = contextlib.ExitStack()
        self._patched: list[nn.Module] = []

    def _record(self, name: str, real: Tensor, imag: Tensor) -> None:
        if name in self._captured:
            raise RuntimeError(f"path module ran more than once in one model pass: {name}")
        if real.shape != imag.shape or real.ndim != 5 or real.shape[-2] != 4:
            raise RuntimeError(f"{name} emitted an invalid pre-collapse D4 field")
        self._captured[name] = (real.detach(), imag.detach())

    def __enter__(self) -> Self:
        modules = path_modules(self.model)
        # Main coarsening stages call forward_packed directly, bypassing the
        # normal nn.Module hook dispatcher, so wrap that method explicitly.
        for name in ("main.stage1", "main.stage2", "main.stage3"):
            module = modules[name]
            original = module.forward_packed

            def wrapped(
                _module: nn.Module,
                real: Tensor,
                imag: Tensor,
                *,
                label: str = name,
                active: Any = original,
            ) -> tuple[Tensor, Tensor]:
                self._record(label, real, imag)
                return cast("tuple[Tensor, Tensor]", active(real, imag))

            module.forward_packed = types.MethodType(wrapped, module)  # type: ignore[method-assign]
            self._patched.append(module)

        # Same-resolution production stages fuse scan + collapse, so expose
        # the otherwise-unmaterialized four paths with one semantically exact
        # no-grad scan replay at the block input.
        for name, block in same_resolution_blocks(self.model).items():
            def hook(
                _module: nn.Module,
                inputs: tuple[Tensor, ...],
                *,
                label: str = name,
                active: SameResolutionPoleScanBlock = block,
            ) -> None:
                paths = _full_resolution_paths(active, inputs[0], inputs[1])
                self._record(label, *paths)

            handle = block.register_forward_pre_hook(hook)
            self._stack.callback(handle.remove)
        return self

    def clear(self) -> None:
        self._captured.clear()

    def take(self) -> dict[str, tuple[Tensor, Tensor]]:
        missing = set(self.module_names) - set(self._captured)
        extra = set(self._captured) - set(self.module_names)
        if missing or extra:
            raise RuntimeError(
                f"incomplete D4 capture; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        captured = dict(self._captured)
        self.clear()
        return captured

    def __exit__(self, *_args: object) -> None:
        self._stack.close()
        # Remove instance overrides to reveal each class' original method.
        for module in self._patched:
            delattr(module, "forward_packed")
        self._patched.clear()


def _path_cross_moments(
    clean: tuple[Tensor, Tensor],
    rotated: tuple[Tensor, Tensor],
) -> dict[str, Tensor]:
    clean_real, clean_imag = (value.float() for value in clean)
    rotated_real, rotated_imag = (align_rotated_field(value).float() for value in rotated)
    if clean_real.shape != rotated_real.shape or clean_imag.shape != rotated_imag.shape:
        raise RuntimeError("clean and aligned-rotated D4 fields have different shapes")
    batch = clean_real.shape[0]
    # Keep the path axis last so BMM yields [batch, rotated_path, clean_path].
    cr = clean_real.movedim(-2, -1).reshape(batch, -1, 4)
    ci = clean_imag.movedim(-2, -1).reshape(batch, -1, 4)
    rr = rotated_real.movedim(-2, -1).reshape(batch, -1, 4)
    ri = rotated_imag.movedim(-2, -1).reshape(batch, -1, 4)
    direct_dot = torch.bmm(rr.transpose(1, 2), cr) + torch.bmm(ri.transpose(1, 2), ci)
    conjugate_dot = torch.bmm(rr.transpose(1, 2), cr) - torch.bmm(
        ri.transpose(1, 2), ci
    )
    clean_energy = cr.square().add(ci.square()).sum(1)
    rotated_energy = rr.square().add(ri.square()).sum(1)
    clean_magnitude = cr.square().add(ci.square()).sqrt()
    rotated_magnitude = rr.square().add(ri.square()).sqrt()
    magnitude_dot = torch.bmm(
        rotated_magnitude.transpose(1, 2), clean_magnitude
    )
    return {
        "direct_dot": direct_dot.double().cpu(),
        "conjugate_dot": conjugate_dot.double().cpu(),
        "clean_energy": clean_energy.double().cpu(),
        "rotated_energy": rotated_energy.double().cpu(),
        "magnitude_dot": magnitude_dot.double().cpu(),
    }


def _permutation_metrics(moments: Mapping[str, Tensor]) -> dict[str, Tensor]:
    direct_dot = moments["direct_dot"]
    conjugate_dot = moments["conjugate_dot"]
    clean_energy = moments["clean_energy"]
    rotated_energy = moments["rotated_energy"]
    magnitude_dot = moments["magnitude_dot"]
    batch = direct_dot.shape[0]
    path_index = torch.arange(4)
    permutation_index = torch.tensor(PERMUTATIONS, dtype=torch.long)
    direct_values: list[Tensor] = []
    conjugate_values: list[Tensor] = []
    direct_errors: list[Tensor] = []
    conjugate_errors: list[Tensor] = []
    magnitude_values: list[Tensor] = []
    clean_total = clean_energy.sum(1)
    rotated_total = rotated_energy.sum(1)
    denominator = (clean_total * rotated_total).sqrt().clamp_min(1.0e-30)
    symmetric_energy = (0.5 * (clean_total + rotated_total)).clamp_min(1.0e-30)
    for permutation in permutation_index:
        direct = direct_dot[:, path_index, permutation].sum(1)
        conjugate = conjugate_dot[:, path_index, permutation].sum(1)
        magnitude = magnitude_dot[:, path_index, permutation].sum(1)
        direct_values.append(direct / denominator)
        conjugate_values.append(conjugate / denominator)
        direct_errors.append(
            ((clean_total + rotated_total - 2.0 * direct).clamp_min(0.0) / symmetric_energy).sqrt()
        )
        conjugate_errors.append(
            (
                (clean_total + rotated_total - 2.0 * conjugate).clamp_min(0.0)
                / symmetric_energy
            ).sqrt()
        )
        magnitude_values.append(magnitude / denominator)
    return {
        "direct_complex_cosine": torch.stack(direct_values, dim=1).reshape(batch, -1),
        "conjugate_complex_cosine": torch.stack(conjugate_values, dim=1).reshape(batch, -1),
        "direct_symmetric_nrmse": torch.stack(direct_errors, dim=1).reshape(batch, -1),
        "conjugate_symmetric_nrmse": torch.stack(conjugate_errors, dim=1).reshape(batch, -1),
        "magnitude_cosine": torch.stack(magnitude_values, dim=1).reshape(batch, -1),
    }


@dataclass
class LayerAccumulator:
    """Accumulate sufficient statistics and paired per-image metrics."""

    total: dict[str, Tensor] = field(default_factory=dict)
    per_image: dict[str, list[Tensor]] = field(default_factory=dict)

    def add(
        self,
        clean: tuple[Tensor, Tensor],
        rotated: tuple[Tensor, Tensor],
    ) -> None:
        moments = _path_cross_moments(clean, rotated)
        if not self.total:
            self.total = {key: value.sum(0) for key, value in moments.items()}
        else:
            for key, value in moments.items():
                self.total[key] += value.sum(0)
        for key, value in _permutation_metrics(moments).items():
            self.per_image.setdefault(key, []).append(value.float())

    def stacked_per_image(self) -> dict[str, Tensor]:
        return {key: torch.cat(values) for key, values in self.per_image.items()}

    def aggregate_metrics(self) -> dict[str, Tensor]:
        batchless = {key: value.unsqueeze(0) for key, value in self.total.items()}
        return {key: value.squeeze(0) for key, value in _permutation_metrics(batchless).items()}


def _rank(value: Tensor, index: int, *, larger_is_better: bool) -> int:
    selected = float(value[index])
    if larger_is_better:
        return 1 + int((value > selected + 1.0e-12).sum())
    return 1 + int((value < selected - 1.0e-12).sum())


def _ci(values: Tensor, labels: Tensor, seed: int) -> list[float]:
    """Bootstrap class means so one-per-class runs still have honest uncertainty."""
    classes = labels.unique(sorted=True)
    class_means = torch.stack([values[labels == label].float().mean() for label in classes])
    generator = torch.Generator().manual_seed(seed)
    draws: list[Tensor] = []
    for _start in range(0, 2_000, 200):
        sampled = torch.randint(
            len(class_means),
            (200, len(class_means)),
            generator=generator,
        )
        draws.append(class_means[sampled].mean(1))
    distribution = torch.cat(draws)
    return [float(distribution.quantile(0.025)), float(distribution.quantile(0.975))]


def _module_payload(
    name: str,
    accumulator: LayerAccumulator,
    labels: Tensor,
    module_index: int,
) -> dict[str, Any]:
    aggregate = accumulator.aggregate_metrics()
    per_image = accumulator.stacked_per_image()
    expected_index = PERMUTATIONS.index(EXPECTED_PERMUTATION)
    direct = aggregate["direct_complex_cosine"]
    conjugate = aggregate["conjugate_complex_cosine"]
    direct_error = aggregate["direct_symmetric_nrmse"]
    conjugate_error = aggregate["conjugate_symmetric_nrmse"]
    magnitude = aggregate["magnitude_cosine"]
    best_direct_index = int(direct.argmax())
    best_conjugate_index = int(conjugate.argmax())
    expected_per_image = per_image["direct_complex_cosine"][:, expected_index]
    expected_error_per_image = per_image["direct_symmetric_nrmse"][:, expected_index]
    return {
        "module": name,
        "group": _layer_group(name),
        "samples": int(labels.numel()),
        "expected_permutation": list(EXPECTED_PERMUTATION),
        "expected_permutation_index": expected_index,
        "expected_direct_complex_cosine": float(direct[expected_index]),
        "expected_direct_complex_cosine_image_mean": float(expected_per_image.mean()),
        "expected_direct_complex_cosine_image_ci95": _ci(
            expected_per_image, labels, 20264000 + module_index
        ),
        "expected_direct_symmetric_nrmse": float(direct_error[expected_index]),
        "expected_direct_symmetric_nrmse_image_mean": float(expected_error_per_image.mean()),
        "expected_direct_symmetric_nrmse_image_ci95": _ci(
            expected_error_per_image, labels, 20265000 + module_index
        ),
        "expected_magnitude_cosine": float(magnitude[expected_index]),
        "expected_direct_rank_of_24": _rank(direct, expected_index, larger_is_better=True),
        "expected_direct_margin_over_permutation_mean": float(
            direct[expected_index] - direct.mean()
        ),
        "best_direct_permutation": list(PERMUTATIONS[best_direct_index]),
        "best_direct_complex_cosine": float(direct[best_direct_index]),
        "best_direct_symmetric_nrmse": float(direct_error[best_direct_index]),
        "expected_conjugate_complex_cosine": float(conjugate[expected_index]),
        "expected_conjugate_symmetric_nrmse": float(conjugate_error[expected_index]),
        "best_conjugate_permutation": list(PERMUTATIONS[best_conjugate_index]),
        "best_conjugate_complex_cosine": float(conjugate[best_conjugate_index]),
        "all_permutations": [
            {
                "permutation": list(permutation),
                "direct_complex_cosine": float(direct[index]),
                "direct_symmetric_nrmse": float(direct_error[index]),
                "conjugate_complex_cosine": float(conjugate[index]),
                "conjugate_symmetric_nrmse": float(conjugate_error[index]),
                "magnitude_cosine": float(magnitude[index]),
            }
            for index, permutation in enumerate(PERMUTATIONS)
        ],
    }


def _plot(path: Path, modules: Sequence[Mapping[str, Any]]) -> None:
    names = [str(module["module"]) for module in modules]
    expected_index = PERMUTATIONS.index(EXPECTED_PERMUTATION)
    matrix = torch.tensor(
        [
            [float(record["direct_complex_cosine"]) for record in module["all_permutations"]]
            for module in modules
        ]
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, max(5.2, 0.42 * len(names))),
        gridspec_kw={"width_ratios": (2.25, 1.0)},
        constrained_layout=True,
    )
    image = axes[0].imshow(matrix.numpy(), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_title("All 24 path permutations (direct complex cosine)")
    axes[0].set_xlabel("Permutation index")
    axes[0].set_ylabel("Pre-collapse module")
    axes[0].set_yticks(range(len(names)), names)
    axes[0].set_xticks(range(24), [str(index) for index in range(24)], fontsize=7)
    axes[0].add_patch(
        plt.Rectangle(
            (expected_index - 0.5, -0.5),
            1,
            len(names),
            fill=False,
            edgecolor="black",
            linewidth=2.0,
        )
    )
    figure.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)

    expected = [float(module["expected_direct_complex_cosine"]) for module in modules]
    best = [float(module["best_direct_complex_cosine"]) for module in modules]
    positions = torch.arange(len(names)).numpy()
    axes[1].barh(positions + 0.18, expected, height=0.34, label="expected")
    axes[1].barh(positions - 0.18, best, height=0.34, label="best of 24", alpha=0.72)
    axes[1].set_yticks(positions, names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1.0, 1.0)
    axes[1].axvline(0.0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Direct complex cosine")
    axes[1].set_title(f"Expected mapping {list(EXPECTED_PERMUTATION)}")
    axes[1].legend(loc="lower right")
    figure.suptitle("Frozen K128/D2262: input rotation ↔ D4 path response")
    _atomic_figure(path, figure)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.per_class <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("per-class/batch-size must be positive and workers nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    frozen = load_frozen_model(args.checkpoint, device)
    model = frozen.model
    validation = imagefolder(args.data_root, "val", evaluation_view=True)
    selected = balanced_indices(validation.targets, args.per_class)
    loader = make_loader(
        validation,
        batch_size=args.batch_size,
        workers=args.workers,
        indices=selected,
    )
    module_names = tuple(path_modules(model))
    accumulators = {name: LayerAccumulator() for name in module_names}
    all_indices: list[Tensor] = []
    all_labels: list[Tensor] = []

    with torch.inference_mode(), D4PathCapture(model) as capture:
        for images_cpu, labels_cpu, indices_cpu in loader:
            images = images_cpu.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            if images.shape[-2] != images.shape[-1]:
                raise RuntimeError("90-degree pairing requires square evaluation crops")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                capture.clear()
                _ = classifier_logits(model, images)
                clean = capture.take()
                _ = classifier_logits(model, torch.rot90(images, k=1, dims=(-2, -1)))
                rotated = capture.take()
            for name in module_names:
                accumulators[name].add(clean[name], rotated[name])
            all_indices.append(indices_cpu.long())
            all_labels.append(labels_cpu.long())

    indices = torch.cat(all_indices)
    labels = torch.cat(all_labels)
    expected_indices = torch.tensor(selected, dtype=torch.long)
    if not torch.equal(indices, expected_indices):
        raise RuntimeError("loader changed deterministic balanced sample order")
    if len(labels.unique()) != 100 or len(labels) != 100 * args.per_class:
        raise RuntimeError("validation subset is not balanced across all 100 classes")

    state_after = model_state_sha256(model)
    checkpoint_after = sha256_file(args.checkpoint)
    if state_after != frozen.initial_state_sha256:
        raise RuntimeError("model state changed during frozen path-response analysis")
    if checkpoint_after != frozen.checkpoint_sha256:
        raise RuntimeError("checkpoint file changed during analysis")

    modules = [
        _module_payload(name, accumulators[name], labels, module_index)
        for module_index, name in enumerate(module_names)
    ]
    expected_scores = [float(module["expected_direct_complex_cosine"]) for module in modules]
    expected_ranks = [int(module["expected_direct_rank_of_24"]) for module in modules]
    output_root: Path = args.output_root
    tensor_path = output_root / "artifacts" / "P0-4_rotation_path_per_example.pt"
    figure_path = output_root / "figures" / f"{FIGURE_ID}.png"
    sidecar_path = figure_path.with_suffix(".json")
    per_image = {
        key: torch.stack(
            [accumulators[name].stacked_per_image()[key] for name in module_names],
            dim=1,
        )
        for key in next(iter(accumulators.values())).stacked_per_image()
    }
    _atomic_torch(
        tensor_path,
        {
            "schema": SCHEMA,
            "indices": indices,
            "labels": labels,
            "module_names": list(module_names),
            "permutations": torch.tensor(PERMUTATIONS, dtype=torch.long),
            "expected_permutation": torch.tensor(EXPECTED_PERMUTATION, dtype=torch.long),
            **per_image,
        },
    )
    _plot(figure_path, modules)
    atomic_json(
        sidecar_path,
        {
            "figure_id": FIGURE_ID,
            "caption": (
                "Direct complex cosine for every D4 path permutation after undoing the "
                "input's 90-degree spatial rotation. The outlined heatmap column and blue "
                f"bars are the geometry-derived mapping {list(EXPECTED_PERMUTATION)}; orange "
                "bars are an explicitly labeled best-of-24 ceiling."
            ),
            "expected_permutation_index": PERMUTATIONS.index(EXPECTED_PERMUTATION),
            "permutation_index": [list(permutation) for permutation in PERMUTATIONS],
        },
    )

    family = {
        "schema": SCHEMA,
        "family_id": FAMILY_ID,
        "status": "complete",
        "question": (
            "Do pre-collapse D4 path responses follow the geometry-derived path permutation "
            "when the same input is rotated counter-clockwise by 90 degrees?"
        ),
        "protocol": {
            "input_pair": "clean vs torch.rot90(input, k=1, dims=(-2,-1))",
            "spatial_alignment": "torch.rot90(rotated_activation, k=-1, dims=(H,W))",
            "path_order": [
                {"index": index, "direction_xy": list(direction)}
                for index, direction in enumerate(PATH_DIRECTIONS)
            ],
            "permutation_convention": (
                "aligned_rotated[..., p, :] is compared with clean[..., permutation[p], :]"
            ),
            "expected_permutation": list(EXPECTED_PERMUTATION),
            "permutations_scored": len(PERMUTATIONS),
            "primary_phase_assumption": (
                "no conjugation: negative-direction recurrences already conjugate their pole "
                "coefficients, retaining a shared global complex carrier convention"
            ),
            "sensitivity_phase_assumption": (
                "one fixed global conjugation of all rotated responses; no per-path or "
                "per-module conjugation search"
            ),
            "main_capture": "direct input to learned PathOnlyCollapse.forward_packed",
            "same_resolution_capture": (
                "semantically exact scan replay before the fused PathOnlyCollapse; full16 "
                "direction-relative cells are losslessly restored, with the 7x7 exact oracle"
            ),
            "selection": "deterministic balanced_indices over ImageFolder validation targets",
            "per_class": args.per_class,
            "bootstrap": "2,000 resamples of the 100 class means",
            "training_or_fitting": "none",
        },
        "provenance": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256_before": frozen.checkpoint_sha256,
            "checkpoint_sha256_after": checkpoint_after,
            "variant": frozen.payload.get("variant"),
            "seed": frozen.payload.get("seed"),
            "epoch": frozen.payload.get("epoch"),
            "data_root": str(args.data_root.resolve()),
            "validation_manifest_sha256": dataset_manifest(
                validation, args.data_root / "val"
            ),
            "selected_indices_sha256": __import__("hashlib").sha256(
                indices.numpy().tobytes()
            ).hexdigest(),
            "device": str(device),
            "torch_version": torch.__version__,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "no_training_guard": {
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parameter_updates": 0,
            "model_training_flag": bool(model.training),
            "model_state_sha256_before": frozen.initial_state_sha256,
            "model_state_sha256_after": state_after,
            "identical": state_after == frozen.initial_state_sha256,
        },
        "sample_count": int(labels.numel()),
        "class_count": int(labels.unique().numel()),
        "module_count": len(modules),
        "summary": {
            "mean_expected_direct_complex_cosine": statistics.fmean(expected_scores),
            "median_expected_direct_rank_of_24": statistics.median(expected_ranks),
            "modules_expected_is_best": sum(rank == 1 for rank in expected_ranks),
            "modules_expected_top3": sum(rank <= 3 for rank in expected_ranks),
            "interpretation_guardrail": (
                "High expected-permutation consistency supports directional path organization "
                "in this checkpoint. Low consistency is a valid null: exact rotation "
                "equivariance is not architecturally guaranteed because readers, x/y poles, "
                "and preceding layers are learned independently."
            ),
        },
        "modules": modules,
        "artifacts": [
            artifact_record(output_root, "P0-4_PER_EXAMPLE", tensor_path, "application/x-pytorch"),
            artifact_record(output_root, FIGURE_ID, figure_path, "image/png"),
            artifact_record(output_root, f"{FIGURE_ID}_SIDECAR", sidecar_path, "application/json"),
        ],
    }
    family_path = output_root / "families" / f"{FAMILY_ID}.json"
    atomic_json(family_path, family)
    print(f"wrote {family_path}")
    summary_line = " ".join(
        (
            f"samples={len(labels)} modules={len(modules)}",
            f"mean_expected_cosine={statistics.fmean(expected_scores):.6f}",
            f"median_expected_rank={statistics.median(expected_ranks):.1f}",
        )
    )
    print(summary_line)


if __name__ == "__main__":
    main()
