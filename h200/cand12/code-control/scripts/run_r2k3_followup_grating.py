#!/usr/bin/env python3
"""Measure actual-input grating tuning in the frozen K128/D2262 model.

This follow-up deliberately feeds images through the complete model.  It
records both the learned reader drive and the response after the production
D4 scan.  No optimizer is constructed and no parameter is modified.
"""

from __future__ import annotations

# Experiment-model hooks and cached torch payloads are intentionally dynamic.
# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
import argparse
import contextlib
import hashlib
import json
import math
import platform
import types
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import matplotlib as mpl

mpl.use("Agg")

import torch
from matplotlib import pyplot as plt
from r2k3_frozen_study_common import (
    EXPECTED_CHECKPOINT_SHA256,
    VARIANT,
    artifact_record,
    atomic_json,
    bh_fdr,
    load_frozen_model,
    model_state_sha256,
    reader_modules,
    same_resolution_blocks,
    scan_modules,
    sha256_file,
)
from torch import Tensor, nn

if TYPE_CHECKING:
    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


SCHEMA = "lnet.a2d.r2k3.k128_d2262.actual_input_grating.v1"
FAMILY_ID = "P0_3_actual_input_grating"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_FREQUENCIES = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0, 56.0)
MODULE_RESOLUTION = {
    "stage1": 56,
    "stage2": 28,
    "stage3": 14,
    "terminal": 7,
    "same_resolution.56": 56,
    "same_resolution.28": 28,
    "same_resolution.14": 14,
    "same_resolution.7": 7,
    "extra14.1": 14,
    "extra14.2": 14,
    "extra14.3": 14,
    "extra14.4": 14,
}


@dataclass(frozen=True, slots=True)
class Stimulus:
    stimulus_id: str
    kind: str
    frequency_cycles_per_image: float | None
    wavevector_orientation_degrees: float | None
    phase_degrees: float | None
    contrast: float
    seed: int | None = None


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frequencies", default=",".join(str(value) for value in DEFAULT_FREQUENCIES))
    parser.add_argument("--orientations", type=int, default=12)
    parser.add_argument("--phases", type=int, default=4)
    parser.add_argument("--contrast", type=float, default=0.90)
    parser.add_argument("--edge-taper-fraction", type=float, default=0.10)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument(
        "--max-new-chunks",
        type=int,
        default=0,
        help="stop cleanly after this many uncached chunks; zero means finish",
    )
    return parser.parse_args()


def parse_frequencies(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value <= 0.0 or value > IMAGE_SIZE / 2 for value in values):
        raise ValueError("frequencies must be in (0, 112] cycles per 224-pixel image")
    if tuple(sorted(set(values))) != values:
        raise ValueError("frequencies must be unique and strictly increasing")
    return values


def stimulus_grid(
    frequencies: Sequence[float],
    orientations: int,
    phases: int,
    contrast: float,
) -> list[Stimulus]:
    if orientations < 2 or phases < 2:
        raise ValueError("grating grid requires at least two orientations and phases")
    if not 0.0 < contrast <= 1.0:
        raise ValueError("contrast must be in (0, 1]")
    stimuli: list[Stimulus] = []
    for frequency in frequencies:
        for orientation_index in range(orientations):
            orientation = 180.0 * orientation_index / orientations
            for phase_index in range(phases):
                phase = 360.0 * phase_index / phases
                stimuli.append(
                    Stimulus(
                        stimulus_id=(
                            f"grating_f{frequency:g}_o{orientation:g}_p{phase:g}"
                        ),
                        kind="grating",
                        frequency_cycles_per_image=float(frequency),
                        wavevector_orientation_degrees=orientation,
                        phase_degrees=phase,
                        contrast=contrast,
                    )
                )
    stimuli.extend(
        (
            Stimulus("control_gray", "gray", None, None, None, 0.0),
            Stimulus("control_black", "black", None, None, None, 0.0),
            Stimulus("control_white", "white", None, None, None, 0.0),
            Stimulus("control_white_noise", "white_noise", None, None, None, contrast, 20260904),
            Stimulus("control_checkerboard", "checkerboard", None, None, None, contrast),
        )
    )
    return stimuli


def _edge_taper(size: int, fraction: float) -> Tensor:
    if not 0.0 <= fraction < 0.5:
        raise ValueError("edge taper fraction must be in [0, 0.5)")
    if fraction == 0.0:
        return torch.ones(size, size)
    border = max(1, round(size * fraction))
    ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, math.pi, border))
    one_dimensional = torch.ones(size)
    one_dimensional[:border] = ramp
    one_dimensional[-border:] = ramp.flip(0)
    return one_dimensional[:, None] * one_dimensional[None, :]


def render_stimuli(
    stimuli: Sequence[Stimulus],
    *,
    size: int = IMAGE_SIZE,
    edge_taper_fraction: float = 0.10,
) -> Tensor:
    """Render RGB stimuli and apply the checkpoint's ImageNet normalization."""
    coordinate = torch.arange(size, dtype=torch.float32) + 0.5 - size / 2.0
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    taper = _edge_taper(size, edge_taper_fraction)
    images: list[Tensor] = []
    for stimulus in stimuli:
        if stimulus.kind == "grating":
            if (
                stimulus.frequency_cycles_per_image is None
                or stimulus.wavevector_orientation_degrees is None
                or stimulus.phase_degrees is None
            ):
                raise ValueError("grating metadata is incomplete")
            theta = math.radians(stimulus.wavevector_orientation_degrees)
            phase = math.radians(stimulus.phase_degrees)
            cycles_per_pixel = stimulus.frequency_cycles_per_image / size
            carrier = torch.cos(
                2.0
                * math.pi
                * cycles_per_pixel
                * (x * math.cos(theta) + y * math.sin(theta))
                + phase
            )
            intensity = 0.5 + 0.5 * stimulus.contrast * taper * carrier
        elif stimulus.kind == "gray":
            intensity = torch.full((size, size), 0.5)
        elif stimulus.kind == "black":
            intensity = torch.zeros(size, size)
        elif stimulus.kind == "white":
            intensity = torch.ones(size, size)
        elif stimulus.kind == "white_noise":
            generator = torch.Generator().manual_seed(cast("int", stimulus.seed))
            noise = 2.0 * torch.rand((size, size), generator=generator) - 1.0
            intensity = 0.5 + 0.5 * stimulus.contrast * taper * noise
        elif stimulus.kind == "checkerboard":
            checkerboard = ((x.floor() + y.floor()).remainder(2.0) * 2.0 - 1.0)
            intensity = 0.5 + 0.5 * stimulus.contrast * taper * checkerboard
        else:
            raise ValueError(f"unsupported stimulus kind: {stimulus.kind}")
        images.append(intensity.clamp(0.0, 1.0).expand(3, -1, -1))
    batch = torch.stack(images)
    mean = batch.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    standard_deviation = batch.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (batch - mean) / standard_deviation


def circular_orientation_error_degrees(first: Tensor, second: Tensor) -> Tensor:
    """Return unsigned orientation error, treating 0 and 180 degrees as equal."""
    difference = torch.remainder(first - second + 90.0, 180.0) - 90.0
    return difference.abs()


def _field_response(real: Tensor, imag: Tensor) -> Tensor:
    spatial = tuple(range(1, real.ndim - 1))
    power = real.detach().float().square() + imag.detach().float().square()
    if spatial:
        power = power.mean(spatial)
    return torch.log1p(power).cpu()


class ActualResponseCollector:
    """Capture one pole-aligned reader and post-scan response per module."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._stack = contextlib.ExitStack()
        self._current: dict[str, dict[str, Tensor]] = {}
        self._expected = tuple(scan_modules(model))

    def _record(self, label: str, kind: str, value: Tensor) -> None:
        responses = self._current.setdefault(label, {})
        if kind in responses:
            raise RuntimeError(f"duplicate {kind} response for {label}")
        responses[kind] = value

    def __enter__(self) -> Self:
        scans = scan_modules(self.model)
        for reader_label, reader in reader_modules(self.model).items():
            label = reader_label.removeprefix("main.")

            def reader_hook(
                _module: nn.Module,
                _inputs: tuple[Tensor, ...],
                output: tuple[Tensor, Tensor],
                *,
                active_label: str = label,
            ) -> None:
                self._record(active_label, "reader_drive", _field_response(*output))

            self._stack.callback(reader.register_forward_hook(reader_hook).remove)

        for label in ("stage1", "stage2", "stage3", "terminal"):
            stage = scans[label]

            def stage_hook(
                active: nn.Module,
                _inputs: tuple[Tensor, ...],
                output: tuple[Any, Tensor],
                *,
                active_label: str = label,
            ) -> None:
                modes = cast("Any", active).modes
                descriptor = output[1].detach().float()
                if descriptor.shape[-1] != 4 * modes:
                    raise RuntimeError(f"{active_label} descriptor lost its D4-pole layout")
                response = descriptor.reshape(descriptor.shape[0], 4, modes).mean(1).cpu()
                self._record(active_label, "scan_state", response)

            self._stack.callback(stage.register_forward_hook(stage_hook).remove)

        for label, block in same_resolution_blocks(self.model).items():
            projection = block.memory_projection
            if projection is not None:

                def projection_prehook(
                    _module: nn.Module,
                    inputs: tuple[Tensor, ...],
                    *,
                    active_label: str = label,
                ) -> None:
                    self._record(active_label, "scan_state", _field_response(inputs[0], inputs[1]))

                self._stack.callback(
                    projection.register_forward_pre_hook(projection_prehook).remove
                )
                continue

            original = block.collapsed_memory

            def collapsed_memory(
                active: SameResolutionPoleScanBlock,
                real: Tensor,
                imag: Tensor,
                *,
                active_label: str = label,
                active_original: Any = original,
            ) -> tuple[Tensor, Tensor]:
                del active
                output = cast("tuple[Tensor, Tensor]", active_original(real, imag))
                self._record(active_label, "scan_state", _field_response(*output))
                return output

            block.collapsed_memory = types.MethodType(collapsed_memory, block)  # type: ignore[method-assign]
            self._stack.callback(setattr, block, "collapsed_memory", original)
        return self

    def __exit__(self, *_args: object) -> None:
        self._stack.close()

    def begin_batch(self) -> None:
        self._current = {}

    def finish_batch(self, batch_size: int) -> dict[str, dict[str, Tensor]]:
        missing = [
            f"{label}/{kind}"
            for label in self._expected
            for kind in ("reader_drive", "scan_state")
            if kind not in self._current.get(label, {})
        ]
        if missing:
            raise RuntimeError(f"response collector missed: {missing}")
        for label, responses in self._current.items():
            expected_modes = scan_modules(self.model)[label].modes
            for kind, values in responses.items():
                if values.shape != (batch_size, expected_modes):
                    expected = (batch_size, expected_modes)
                    raise RuntimeError(
                        f"{label}/{kind} emitted {tuple(values.shape)}, expected {expected}"
                    )
        return self._current


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _protocol_hash(protocol: Mapping[str, Any], checkpoint_sha256: str) -> str:
    payload = {
        "protocol": dict(protocol),
        "checkpoint_sha256": checkpoint_sha256,
        "script_sha256": sha256_file(Path(__file__)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_chunk(
    path: Path,
    *,
    protocol_hash: str,
    start: int,
    stop: int,
    responses: Mapping[str, Mapping[str, Tensor]],
) -> None:
    _atomic_torch(
        path,
        {
            "schema": SCHEMA,
            "protocol_hash": protocol_hash,
            "start": start,
            "stop": stop,
            "responses": {
                label: {kind: value.cpu() for kind, value in kinds.items()}
                for label, kinds in responses.items()
            },
        },
    )


def _load_chunk(
    path: Path,
    *,
    protocol_hash: str,
    start: int,
    stop: int,
) -> dict[str, dict[str, Tensor]] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, RuntimeError):
        return None
    if (
        payload.get("schema") != SCHEMA
        or payload.get("protocol_hash") != protocol_hash
        or payload.get("start") != start
        or payload.get("stop") != stop
    ):
        return None
    return cast("dict[str, dict[str, Tensor]]", payload["responses"])


def _rankdata(values: Tensor) -> Tensor:
    values = values.double().flatten()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        stop = start + 1
        while stop < values.numel() and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(first: Tensor, second: Tensor) -> float | None:
    if first.numel() != second.numel() or first.numel() < 3:
        raise ValueError("Spearman inputs must have the same length of at least three")
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = torch.linalg.vector_norm(first_centered) * torch.linalg.vector_norm(
        second_centered
    )
    if float(denominator) <= 1.0e-15:
        return None
    return float((first_centered * second_centered).sum() / denominator)


def _permutation_p_value(
    first: Tensor,
    second: Tensor,
    observed: float | None,
    *,
    permutations: int,
    seed: int,
    statistic: str,
) -> float | None:
    if observed is None:
        return None
    generator = torch.Generator().manual_seed(seed)
    exceedances = 0
    for _ in range(permutations):
        shuffled = first[torch.randperm(first.numel(), generator=generator)]
        if statistic == "spearman":
            value = spearman_correlation(shuffled, second)
            active = 0.0 if value is None else abs(value)
            threshold = abs(observed)
        elif statistic == "orientation":
            difference = torch.deg2rad(2.0 * (shuffled - second))
            active = float(torch.cos(difference).mean())
            threshold = observed
        else:
            raise ValueError(f"unknown permutation statistic: {statistic}")
        exceedances += int(active >= threshold)
    return (exceedances + 1) / (permutations + 1)


def _module_summary(
    label: str,
    response_kind: str,
    responses: Tensor,
    stimuli: Sequence[Stimulus],
    frequencies: Sequence[float],
    orientations: int,
    phases: int,
    phase_x: Tensor,
    phase_y: Tensor,
    *,
    permutations: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Tensor]:
    modes = phase_x.numel()
    grating_count = len(frequencies) * orientations * phases
    if responses.shape != (len(stimuli), modes):
        raise ValueError(f"{label}/{response_kind} response grid has the wrong shape")
    grating = responses[:grating_count].reshape(
        len(frequencies), orientations, phases, modes
    )
    phase_mean = grating.mean(2)
    gray_index = next(index for index, stimulus in enumerate(stimuli) if stimulus.kind == "gray")
    gray = responses[gray_index]
    evoked = phase_mean - gray.view(1, 1, -1)
    flat = evoked.reshape(len(frequencies) * orientations, modes)
    preferred_flat = flat.argmax(0)
    preferred_frequency_index = preferred_flat // orientations
    preferred_orientation_index = preferred_flat % orientations
    frequency_tensor = torch.tensor(frequencies, dtype=torch.float64)
    observed_frequency = frequency_tensor[preferred_frequency_index]
    observed_orientation = preferred_orientation_index.double() * (180.0 / orientations)

    resolution = MODULE_RESOLUTION[label]
    phase_x = phase_x.detach().double().cpu().flatten()
    phase_y = phase_y.detach().double().cpu().flatten()
    learned_radial = torch.hypot(phase_x, phase_y)
    learned_frequency_per_cell = learned_radial / (2.0 * math.pi)
    predicted_cycles_per_image = learned_frequency_per_cell * resolution
    learned_orientation = torch.rad2deg(torch.atan2(phase_y, phase_x)).remainder(180.0)
    orientation_error = circular_orientation_error_degrees(
        learned_orientation, observed_orientation
    )
    observed_cycles_per_cell = observed_frequency / resolution

    frequency_spearman = spearman_correlation(
        learned_frequency_per_cell, observed_cycles_per_cell
    )
    orientation_alignment = float(
        torch.cos(torch.deg2rad(2.0 * (learned_orientation - observed_orientation))).mean()
    )
    seed_base = int(hashlib.sha256(f"{label}/{response_kind}".encode()).hexdigest()[:8], 16)
    frequency_p = _permutation_p_value(
        learned_frequency_per_cell,
        observed_cycles_per_cell,
        frequency_spearman,
        permutations=permutations,
        seed=seed_base,
        statistic="spearman",
    )
    orientation_p = _permutation_p_value(
        learned_orientation,
        observed_orientation,
        orientation_alignment,
        permutations=permutations,
        seed=seed_base + 1,
        statistic="orientation",
    )

    rows: list[dict[str, Any]] = []
    for pole in range(modes):
        frequency_index = int(preferred_frequency_index[pole])
        orientation_index = int(preferred_orientation_index[pole])
        orientation_profile = evoked[frequency_index, :, pole]
        nonnegative_orientation = orientation_profile - orientation_profile.min() + 1.0e-12
        orientation_angles = torch.arange(orientations).double() * math.pi / orientations
        orientation_vector = (
            nonnegative_orientation.double()
            * torch.exp(2j * orientation_angles)
        ).sum() / nonnegative_orientation.sum()
        frequency_profile = evoked[:, orientation_index, pole]
        selected_phases = grating[frequency_index, orientation_index, :, pole]
        predicted = float(predicted_cycles_per_image[pole])
        observed = float(observed_frequency[pole])
        rows.append(
            {
                "module": label,
                "response_kind": response_kind,
                "resolution": resolution,
                "pole": pole,
                "learned_omega_x_radians_per_cell": float(phase_x[pole]),
                "learned_omega_y_radians_per_cell": float(phase_y[pole]),
                "learned_radial_omega_radians_per_cell": float(learned_radial[pole]),
                "predicted_frequency_cycles_per_image": predicted,
                "learned_wavevector_orientation_degrees": float(learned_orientation[pole]),
                "observed_preferred_frequency_cycles_per_image": observed,
                "observed_preferred_frequency_cycles_per_pixel": observed / IMAGE_SIZE,
                "observed_preferred_frequency_cycles_per_module_cell": (
                    observed / resolution
                ),
                "observed_wavevector_orientation_degrees": float(observed_orientation[pole]),
                "orientation_error_degrees": float(orientation_error[pole]),
                "frequency_error_octaves": abs(
                    math.log2(max(observed, 1.0e-12) / max(predicted, 1.0e-12))
                ),
                "peak_evoked_response": float(flat[preferred_flat[pole], pole]),
                "gray_response": float(gray[pole]),
                "frequency_selectivity_range": float(
                    frequency_profile.max() - frequency_profile.min()
                ),
                "orientation_selectivity_index": float(abs(orientation_vector)),
                "phase_standard_deviation_at_peak": float(selected_phases.std(unbiased=False)),
                "frequency_grid_boundary_peak": frequency_index in (0, len(frequencies) - 1),
            }
        )

    controls = {
        stimulus.kind: {
            "mean_across_poles": float(responses[index].mean()),
            "rms_across_poles": float(responses[index].square().mean().sqrt()),
        }
        for index, stimulus in enumerate(stimuli)
        if stimulus.kind != "grating"
    }
    summary = {
        "module": label,
        "response_kind": response_kind,
        "resolution": resolution,
        "poles": modes,
        "frequency_spearman_rho": frequency_spearman,
        "frequency_permutation_p_two_sided": frequency_p,
        "orientation_pairing_alignment_mean_cos_2delta": orientation_alignment,
        "orientation_permutation_p_one_sided": orientation_p,
        "median_orientation_error_degrees": float(orientation_error.median()),
        "median_frequency_error_octaves": float(
            torch.tensor([row["frequency_error_octaves"] for row in rows]).median()
        ),
        "frequency_grid_boundary_peak_fraction": sum(
            int(row["frequency_grid_boundary_peak"]) for row in rows
        )
        / modes,
        "median_orientation_selectivity_index": float(
            torch.tensor([row["orientation_selectivity_index"] for row in rows]).median()
        ),
        "controls": controls,
    }
    return summary, rows, evoked


def _fdr_summaries(summaries: list[dict[str, Any]]) -> None:
    for test_name, p_key, q_key in (
        (
            "frequency",
            "frequency_permutation_p_two_sided",
            "frequency_permutation_q_bh",
        ),
        (
            "orientation",
            "orientation_permutation_p_one_sided",
            "orientation_permutation_q_bh",
        ),
    ):
        p_values = {
            f"{summary['module']}/{summary['response_kind']}/{test_name}": float(summary[p_key])
            for summary in summaries
            if summary[p_key] is not None
        }
        adjusted = bh_fdr(p_values)
        for summary in summaries:
            key = f"{summary['module']}/{summary['response_kind']}/{test_name}"
            summary[q_key] = adjusted.get(key)


def _aggregate_summary(rows: Sequence[Mapping[str, Any]], response_kind: str) -> dict[str, Any]:
    selected = [row for row in rows if row["response_kind"] == response_kind]
    learned = torch.tensor(
        [float(row["predicted_frequency_cycles_per_image"]) for row in selected]
    )
    observed = torch.tensor(
        [float(row["observed_preferred_frequency_cycles_per_image"]) for row in selected]
    )
    orientation_errors = torch.tensor(
        [float(row["orientation_error_degrees"]) for row in selected]
    )
    return {
        "response_kind": response_kind,
        "poles_across_modules": len(selected),
        "descriptive_frequency_spearman_rho": spearman_correlation(learned, observed),
        "median_orientation_error_degrees": float(orientation_errors.median()),
        "within_22_5_degrees_fraction": float((orientation_errors <= 22.5).float().mean()),
        "median_frequency_error_octaves": float(
            torch.tensor([float(row["frequency_error_octaves"]) for row in selected]).median()
        ),
        "note": "Descriptive pooled statistic; modules and poles are not treated as independent replicates.",
    }


def _atomic_figure(path: Path, figure: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, format="png", dpi=220, bbox_inches="tight")
    temporary.replace(path)


def _figure(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    evoked: Mapping[str, Mapping[str, Tensor]],
    frequencies: Sequence[float],
    orientations: int,
) -> tuple[Path, Path]:
    scan_rows = [row for row in rows if row["response_kind"] == "scan_state"]
    learned = [float(row["predicted_frequency_cycles_per_image"]) for row in scan_rows]
    observed = [float(row["observed_preferred_frequency_cycles_per_image"]) for row in scan_rows]
    errors = [float(row["orientation_error_degrees"]) for row in scan_rows]
    representative = "extra14.4" if "extra14.4" in evoked else "terminal"
    heatmap = evoked[representative]["scan_state"]
    centered = heatmap - heatmap.mean((0, 1), keepdim=True)
    scale = centered.square().mean((0, 1), keepdim=True).sqrt().clamp_min(1.0e-12)
    population = (centered / scale).mean(-1).T

    with mpl.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
        axes[0].scatter(learned, observed, s=5, alpha=0.28, linewidths=0)
        lower = min(*learned, *observed)
        upper = max(*learned, *observed)
        axes[0].plot((lower, upper), (lower, upper), color="black", lw=0.7, ls="--")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set(
            title="Learned $\\omega$ vs actual preference",
            xlabel="Predicted cycles / image",
            ylabel="Observed cycles / image",
        )
        axes[0].grid(alpha=0.18)

        axes[1].hist(errors, bins=torch.linspace(0, 90, 13).tolist(), color="#4477aa")
        axes[1].axvline(45.0, color="black", lw=0.7, ls="--", label="random median")
        axes[1].set(
            title="Pole-orientation agreement",
            xlabel="Orientation error (degrees)",
            ylabel="Pole count",
        )
        axes[1].legend(frameon=False)

        image = axes[2].imshow(
            population,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            extent=(
                -0.5,
                len(frequencies) - 0.5,
                -90.0 / orientations,
                180.0 - 90.0 / orientations,
            ),
        )
        tick_indices = list(range(0, len(frequencies), max(1, len(frequencies) // 5)))
        axes[2].set_xticks(tick_indices, [f"{frequencies[index]:g}" for index in tick_indices])
        axes[2].set(
            title=f"{representative}: population tuning",
            xlabel="Input cycles / image",
            ylabel="Wave-vector orientation (deg)",
        )
        figure.colorbar(image, ax=axes[2], shrink=0.8, label="Mean pole-wise z-score")
        path = root / "figures" / "P0_3_actual_input_grating.png"
        _atomic_figure(path, figure)
        plt.close(figure)

    sidecar = path.with_suffix(".json")
    atomic_json(
        sidecar,
        {
            "figure_id": "P0_3_actual_input_grating",
            "panels": [
                {"panel": "frequency", "source": "followups/P0_3_actual_input_grating.json#/pole_rows"},
                {"panel": "orientation", "source": "followups/P0_3_actual_input_grating.json#/pole_rows"},
                {
                    "panel": "population_heatmap",
                    "module": representative,
                    "source": "followups/P0_3_actual_input_grating_responses.pt",
                },
            ],
            "caption": (
                "Actual-input sinusoidal-grating tuning after reader, nonlinearities, "
                "and the production D4 scan. Orientation denotes the grating wave vector."
            ),
        },
    )
    return path, sidecar


def _chunks(
    model: nn.Module,
    stimuli: Sequence[Stimulus],
    *,
    root: Path,
    protocol_hash: str,
    device: torch.device,
    batch_size: int,
    edge_taper_fraction: float,
    max_new_chunks: int,
) -> tuple[dict[str, dict[str, Tensor]], int, int]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    cache_root = root / "cache" / "actual_input_grating" / protocol_hash[:16]
    collected: dict[str, dict[str, list[Tensor]]] = {}
    cache_hits = 0
    new_chunks = 0
    with ActualResponseCollector(model) as collector, torch.inference_mode():
        for start in range(0, len(stimuli), batch_size):
            stop = min(start + batch_size, len(stimuli))
            path = cache_root / f"chunk_{start:05d}_{stop:05d}.pt"
            active = _load_chunk(
                path,
                protocol_hash=protocol_hash,
                start=start,
                stop=stop,
            )
            if active is None:
                if max_new_chunks and new_chunks >= max_new_chunks:
                    raise RuntimeError(
                        f"checkpoint yield after {new_chunks} new chunks; rerun to resume"
                    )
                images = render_stimuli(
                    stimuli[start:stop],
                    edge_taper_fraction=edge_taper_fraction,
                ).to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
                collector.begin_batch()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type in {"cuda", "cpu"},
                ):
                    _ = model(images)
                active = collector.finish_batch(stop - start)
                _cache_chunk(
                    path,
                    protocol_hash=protocol_hash,
                    start=start,
                    stop=stop,
                    responses=active,
                )
                new_chunks += 1
            else:
                cache_hits += 1
            for label, kinds in active.items():
                target = collected.setdefault(label, {})
                for kind, values in kinds.items():
                    target.setdefault(kind, []).append(values)
    return (
        {
            label: {kind: torch.cat(values) for kind, values in kinds.items()}
            for label, kinds in collected.items()
        },
        cache_hits,
        new_chunks,
    )


def main() -> None:
    args = _args()
    frequencies = parse_frequencies(args.frequencies)
    if args.permutations < 100:
        raise ValueError("at least 100 permutations are required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    stimuli = stimulus_grid(
        frequencies,
        args.orientations,
        args.phases,
        args.contrast,
    )
    protocol = {
        "image_size": IMAGE_SIZE,
        "frequencies_cycles_per_image": list(frequencies),
        "orientations": args.orientations,
        "orientation_definition": "wave-vector orientation modulo 180 degrees; bars are perpendicular",
        "phases": args.phases,
        "contrast": args.contrast,
        "edge_taper_fraction": args.edge_taper_fraction,
        "controls": [stimulus.kind for stimulus in stimuli if stimulus.kind != "grating"],
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "response_definitions": {
            "reader_drive": "log1p spatial mean complex power at learned reader output",
            "scan_state_main": "mean of four production Raw-Q log-energy directions per pole",
            "scan_state_same_resolution": (
                "log1p spatial mean complex power after exact production D4 scan and "
                "nonlinear Path-SwiGLU collapse, before any cross-pole memory projection"
            ),
            "evoked": "phase-mean response minus uniform-gray response",
        },
        "permutations": args.permutations,
        "seed": 20260904,
    }
    frozen = load_frozen_model(args.checkpoint, device)
    if frozen.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("unexpected checkpoint")
    if any(parameter.grad is not None for parameter in frozen.model.parameters()):
        raise RuntimeError("frozen model unexpectedly has gradients before analysis")
    before_state = frozen.initial_state_sha256
    active_protocol_hash = _protocol_hash(protocol, frozen.checkpoint_sha256)
    responses, cache_hits, new_chunks = _chunks(
        frozen.model,
        stimuli,
        root=args.root,
        protocol_hash=active_protocol_hash,
        device=device,
        batch_size=args.batch_size,
        edge_taper_fraction=args.edge_taper_fraction,
        max_new_chunks=args.max_new_chunks,
    )

    scans = scan_modules(frozen.model)
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    evoked: dict[str, dict[str, Tensor]] = {}
    for label, kinds in responses.items():
        module = scans[label]
        evoked[label] = {}
        for response_kind, values in kinds.items():
            summary, active_rows, active_evoked = _module_summary(
                label,
                response_kind,
                values,
                stimuli,
                frequencies,
                args.orientations,
                args.phases,
                module.phase_x,
                module.phase_y,
                permutations=args.permutations,
            )
            summaries.append(summary)
            rows.extend(active_rows)
            evoked[label][response_kind] = active_evoked
    _fdr_summaries(summaries)

    response_path = args.root / "followups" / "P0_3_actual_input_grating_responses.pt"
    _atomic_torch(
        response_path,
        {
            "schema": SCHEMA,
            "protocol_hash": active_protocol_hash,
            "stimuli": [asdict(stimulus) for stimulus in stimuli],
            "responses": responses,
            "phase_averaged_gray_subtracted": evoked,
        },
    )
    figure_path, sidecar_path = _figure(
        args.root,
        rows,
        evoked,
        frequencies,
        args.orientations,
    )

    after_state = model_state_sha256(frozen.model)
    checkpoint_after = sha256_file(args.checkpoint)
    gradient_tensors = sum(parameter.grad is not None for parameter in frozen.model.parameters())
    if before_state != after_state or frozen.checkpoint_sha256 != checkpoint_after:
        raise RuntimeError("frozen checkpoint or model state changed during analysis")
    if gradient_tensors:
        raise RuntimeError("grating analysis unexpectedly populated parameter gradients")
    family = {
        "schema": SCHEMA,
        "family_id": FAMILY_ID,
        "status": "complete",
        "question": (
            "Do learned pole phases predict visual spatial-frequency and orientation "
            "preference after actual images pass through reader, nonlinearities, and scan?"
        ),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256_before": frozen.checkpoint_sha256,
            "sha256_after": checkpoint_after,
            "variant": VARIANT,
            "seed": frozen.payload.get("seed"),
            "epoch": frozen.payload.get("epoch"),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "precision": "bfloat16 autocast",
        },
        "protocol_hash": active_protocol_hash,
        "protocol": protocol,
        "execution": {
            "stimuli": len(stimuli),
            "gratings": sum(stimulus.kind == "grating" for stimulus in stimuli),
            "cache_hits": cache_hits,
            "new_chunks": new_chunks,
            "batch_size": args.batch_size,
        },
        "no_training_guard": {
            "optimizer_constructed": False,
            "optimizer_step_calls": 0,
            "backward_calls": 0,
            "parameter_update_calls": 0,
            "parameter_gradient_tensors_after": gradient_tensors,
            "model_state_sha256_before": before_state,
            "model_state_sha256_after": after_state,
            "checkpoint_sha256_before": frozen.checkpoint_sha256,
            "checkpoint_sha256_after": checkpoint_after,
        },
        "statistical_scope": {
            "unit": "pole within frozen checkpoint",
            "phase_handling": "four-phase mean",
            "null": "permutation of learned pole identity within module",
            "multiple_testing": "Benjamini-Hochberg separately for frequency and orientation",
            "inference_limit": (
                "Exploratory within-checkpoint mechanistic evidence; poles and modules "
                "are not independent model replicates."
            ),
        },
        "module_summaries": summaries,
        "aggregate_descriptive": [
            _aggregate_summary(rows, "reader_drive"),
            _aggregate_summary(rows, "scan_state"),
        ],
        "pole_rows": rows,
        "artifacts": [
            artifact_record(
                args.root,
                "P0_3_RESPONSE_TENSORS",
                response_path,
                "application/x-pytorch",
            ),
            artifact_record(args.root, "P0_3_FIGURE", figure_path, "image/png"),
            artifact_record(args.root, "P0_3_FIGURE_SIDECAR", sidecar_path, "application/json"),
        ],
        "interpretation_contract": {
            "support": (
                "Positive frequency rank association and orientation alignment after the "
                "scan support a link between learned phase and actual visual tuning."
            ),
            "null": (
                "Weak or absent alignment means the intrinsic pole-only transfer result "
                "does not survive the learned reader/nonlinear network path."
            ),
            "not_claimed": (
                "This experiment does not establish causal necessity, cross-seed "
                "generalization, or tuning on natural images."
            ),
        },
    }
    family_path = args.root / "followups" / f"{FAMILY_ID}.json"
    atomic_json(family_path, family)
    print(json.dumps({"family": str(family_path), "status": "complete"}, sort_keys=True))


if __name__ == "__main__":
    main()
