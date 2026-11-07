#!/usr/bin/env python3
"""Evaluate exact periodic state resets in the frozen D4 product scan."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
import argparse
import contextlib
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import torch
from matplotlib import pyplot as plt
from r2k3_frozen_study_common import (
    PredictionResult,
    atomic_json,
    balanced_indices,
    collect_predictions,
    imagefolder,
    load_frozen_model,
    load_predictions,
    make_loader,
    model_state_sha256,
    paired_effect,
    prediction_metrics,
    save_predictions,
    sha256_file,
)
from torch import Tensor

import lnet.complex_scan_stage as complex_scan_stage_module
import lnet.pac_same_resolution_depth as same_resolution_module
from lnet.pac_product_scan_contracts import (
    DEFAULT_EPSILON,
    ProductGainNormalization,
)
from lnet.pac_product_scan_reference import (
    _product_coarse4_reference,
    _product_full4_reference,
    _product_full16_reference,
    raw_product_descriptor_reference,
)
from lnet.pac_recurrence import recurrence_real2d_state_variance_directional
from lnet.pac_triton_grouped_path_cffn import (
    d4_grouped_cell_path_collapse_reference,
    d4_grouped_cell_path_swiglu_reference,
    d4_grouped_path_collapse_reference,
    d4_grouped_path_swiglu_reference,
)

mpl.use("Agg")

ComplexField = tuple[Tensor, Tensor]
Pole = tuple[Tensor, Tensor, Tensor, Tensor]
DirectionalState = tuple[Tensor, Tensor, Tensor]
ProductScans = tuple[DirectionalState, DirectionalState, DirectionalState, DirectionalState]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-study-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--max-new-conditions", type=int, default=1)
    return parser.parse_args()


def _reset_mask(
    batch: int,
    length: int,
    modes: int,
    interval: int,
    *,
    reverse: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if interval <= 0 or interval >= length:
        return torch.ones((batch, length, modes), device=device, dtype=dtype)
    positions = torch.arange(length, device=device)
    distance = length - 1 - positions if reverse else positions
    continuation = distance.remainder(interval).ne(0).to(dtype)
    return continuation.view(1, length, 1).expand(batch, length, modes)


def _axis_scan(
    pole: Pole,
    source: ComplexField,
    source_variance: Tensor,
    *,
    interval: int,
    reverse: bool,
) -> DirectionalState:
    decay_real, decay_imag, gamma_real, gamma_imag = (
        value.reshape(-1).float() for value in pole
    )
    source_real, source_imag = source
    batch, length, modes = source_real.shape
    mask = _reset_mask(
        batch,
        length,
        modes,
        interval,
        reverse=reverse,
        device=source_real.device,
        dtype=torch.float32,
    )
    active_decay_real = decay_real.view(1, 1, modes) * mask
    active_decay_imag = decay_imag.view(1, 1, modes) * mask
    input_real = (
        gamma_real.view(1, 1, modes) * source_real.float()
        - gamma_imag.view(1, 1, modes) * source_imag.float()
    )
    input_imag = (
        gamma_imag.view(1, 1, modes) * source_real.float()
        + gamma_real.view(1, 1, modes) * source_imag.float()
    )
    variance_decay = decay_real.square().add(decay_imag.square()).view(1, 1, modes) * mask
    variance_input = gamma_real.square().add(gamma_imag.square()).view(1, 1, modes)
    variance_input = variance_input * source_variance.float()
    state_real, state_imag, variance = recurrence_real2d_state_variance_directional(
        active_decay_real,
        active_decay_imag,
        input_real,
        input_imag,
        variance_decay,
        variance_input,
        "triton_fused",
        "backward" if reverse else "forward",
    )
    return (
        state_real.to(source_real.dtype),
        state_imag.to(source_imag.dtype),
        variance.to(source_real.dtype),
    )


def _horizontal(
    pole: Pole,
    source: ComplexField,
    *,
    interval: int,
) -> tuple[DirectionalState, DirectionalState]:
    source_real, source_imag = source
    batch, height, width, modes = source_real.shape
    flat_source = (
        source_real.reshape(batch * height, width, modes),
        source_imag.reshape(batch * height, width, modes),
    )
    unit_variance = source_real.new_ones((batch * height, width, modes))
    positive = _axis_scan(
        pole,
        flat_source,
        unit_variance,
        interval=interval,
        reverse=False,
    )
    negative_pole = (pole[0], -pole[1], pole[2], -pole[3])
    negative = _axis_scan(
        negative_pole,
        flat_source,
        unit_variance,
        interval=interval,
        reverse=True,
    )

    def restore(state: DirectionalState) -> DirectionalState:
        return cast(
            "DirectionalState",
            tuple(value.reshape(batch, height, width, modes) for value in state),
        )

    return restore(positive), restore(negative)


def _vertical(
    pole: Pole,
    source: DirectionalState,
    *,
    interval: int,
    reverse: bool,
) -> DirectionalState:
    source_real, source_imag, source_variance = source
    batch, height, width, modes = source_real.shape

    def flatten(value: Tensor) -> Tensor:
        return value.permute(0, 2, 1, 3).reshape(batch * width, height, modes)

    flat = _axis_scan(
        pole,
        (flatten(source_real), flatten(source_imag)),
        flatten(source_variance),
        interval=interval,
        reverse=reverse,
    )

    def restore(value: Tensor) -> Tensor:
        return value.reshape(batch, width, height, modes).permute(0, 2, 1, 3).contiguous()

    return cast("DirectionalState", tuple(restore(value) for value in flat))


def segmented_product_scans(
    pole_x: Pole,
    pole_y: Pole,
    source: ComplexField,
    *,
    interval: int,
) -> ProductScans:
    positive_x, negative_x = _horizontal(pole_x, source, interval=interval)
    negative_y = (pole_y[0], -pole_y[1], pole_y[2], -pole_y[3])
    return (
        _vertical(pole_y, positive_x, interval=interval, reverse=False),
        _vertical(pole_y, negative_x, interval=interval, reverse=False),
        _vertical(negative_y, positive_x, interval=interval, reverse=True),
        _vertical(negative_y, negative_x, interval=interval, reverse=True),
    )


def segmented_product_scan_pipeline(
    pole_x: Pole,
    pole_y: Pole,
    source: ComplexField,
    *,
    epilogue: str,
    gain_normalization: ProductGainNormalization,
    memory_policy: str = "retain",
    epsilon: float = DEFAULT_EPSILON,
    interval: int,
) -> tuple[Tensor, Tensor, Tensor] | Tensor:
    _ = memory_policy
    scans = segmented_product_scans(pole_x, pole_y, source, interval=interval)
    descriptor = raw_product_descriptor_reference(
        *scans,
        epsilon=epsilon,
        gain_normalization=gain_normalization,
    )
    if epilogue == "descriptor":
        return descriptor
    paths = (
        _product_full16_reference(
            scans,
            epsilon=epsilon,
            gain_normalization=gain_normalization,
        )
        if epilogue == "full16"
        else _product_coarse4_reference(
            scans,
            epsilon=epsilon,
            gain_normalization=gain_normalization,
        )
    )
    return paths[0], paths[1], descriptor


def segmented_path_collapse_pipeline(
    pole_x: Pole,
    pole_y: Pole,
    source: ComplexField,
    path_collapse: tuple[Tensor, Tensor, Tensor, Tensor],
    *,
    gain_normalization: ProductGainNormalization,
    memory_policy: str = "retain",
    path_swiglu: bool = False,
    epsilon: float = DEFAULT_EPSILON,
    interval: int,
) -> ComplexField:
    _ = memory_policy
    scans = segmented_product_scans(pole_x, pole_y, source, interval=interval)
    height, width = source[0].shape[1:3]
    if height % 2 == 0 and width % 2 == 0:
        paths = _product_full16_reference(
            scans,
            epsilon=epsilon,
            gain_normalization=gain_normalization,
        )
        operation = (
            d4_grouped_cell_path_swiglu_reference
            if path_swiglu
            else d4_grouped_cell_path_collapse_reference
        )
    else:
        paths = _product_full4_reference(
            scans,
            epsilon=epsilon,
            gain_normalization=gain_normalization,
        )
        operation = (
            d4_grouped_path_swiglu_reference
            if path_swiglu
            else d4_grouped_path_collapse_reference
        )
    return operation(paths[0], paths[1], *path_collapse)


@contextlib.contextmanager
def exact_reset_context(interval: int) -> Generator[None]:
    original_stage = complex_scan_stage_module.run_product_scan_pipeline
    original_same = same_resolution_module.run_product_scan_path_collapse_pipeline

    def stage(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor] | Tensor:
        return segmented_product_scan_pipeline(*args, **kwargs, interval=interval)

    def same(*args: Any, **kwargs: Any) -> ComplexField:
        return segmented_path_collapse_pipeline(*args, **kwargs, interval=interval)

    complex_scan_stage_module.run_product_scan_pipeline = stage  # type: ignore[assignment]
    same_resolution_module.run_product_scan_path_collapse_pipeline = same  # type: ignore[assignment]
    try:
        yield
    finally:
        complex_scan_stage_module.run_product_scan_pipeline = original_stage  # type: ignore[assignment]
        same_resolution_module.run_product_scan_path_collapse_pipeline = original_same  # type: ignore[assignment]


def _prediction_record(
    root: Path,
    condition_id: str,
    candidate: PredictionResult,
    production: PredictionResult,
    reference: PredictionResult,
    *,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    path = root / "predictions" / f"{condition_id}.pt"
    return {
        "condition_id": condition_id,
        "status": "complete",
        "metrics": prediction_metrics(candidate),
        "vs_production_baseline": paired_effect(
            production,
            candidate,
            bootstrap_repeats=bootstrap_repeats,
            seed=20260904,
        ),
        "vs_segmented_full_reference": paired_effect(
            reference,
            candidate,
            bootstrap_repeats=bootstrap_repeats,
            seed=20261904,
        ),
        "prediction_path": str(path.relative_to(root)),
        "prediction_sha256": sha256_file(path),
        "prediction_bytes": path.stat().st_size,
    }


def _slice_predictions(result: PredictionResult, selected_indices: list[int]) -> PredictionResult:
    positions = {int(index): position for position, index in enumerate(result.indices.tolist())}
    selection = torch.tensor([positions[index] for index in selected_indices], dtype=torch.long)
    return PredictionResult(
        indices=result.indices[selection],
        labels=result.labels[selection],
        logits=result.logits[selection],
        losses=result.losses[selection],
        margins=result.margins[selection],
        confidences=result.confidences[selection],
        predictions=result.predictions[selection],
    )


def _figure(root: Path, records: list[dict[str, Any]]) -> Path:
    ordered = sorted(
        records,
        key=lambda record: 10_000
        if record["condition_id"] == "reset_full"
        else int(str(record["condition_id"]).rsplit("_", 1)[-1]),
    )
    labels = ["full" if row["condition_id"] == "reset_full" else row["condition_id"].rsplit("_", 1)[-1] for row in ordered]
    values = [100.0 * float(row["metrics"]["accuracy"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(range(len(values)), values, marker="o")
    axis.set_xticks(range(len(labels)), labels)
    axis.set(xlabel="exact reset interval (spatial steps)", ylabel="Top-1 accuracy (%)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = root / "figures" / "state_reset_horizon.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    args = _args()
    args.root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    frozen = load_frozen_model(args.checkpoint, device)
    initial_state = model_state_sha256(frozen.model)
    base_study = json.loads((args.base_study_root / "study.json").read_text())
    production_full = load_predictions(
        args.base_study_root / str(base_study["baseline"]["prediction_path"])
    )
    dataset = imagefolder(args.data_root, "val")
    selected_indices = balanced_indices(dataset.targets, args.per_class)
    production = _slice_predictions(production_full, selected_indices)
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        indices=selected_indices,
    )
    conditions = [("reset_full", 0), *((f"reset_interval_{value}", value) for value in (32, 16, 8, 4, 2, 1))]
    prediction_paths = {
        condition_id: args.root / "predictions" / f"{condition_id}.pt"
        for condition_id, _interval in conditions
    }
    new_conditions = 0
    for condition_id, interval in conditions:
        path = prediction_paths[condition_id]
        if path.is_file():
            continue
        if args.max_new_conditions and new_conditions >= args.max_new_conditions:
            break
        with exact_reset_context(interval):
            candidate = collect_predictions(frozen.model, loader, device)
        save_predictions(path, candidate)
        new_conditions += 1
        if model_state_sha256(frozen.model) != initial_state:
            raise RuntimeError(f"model state changed after {condition_id}")
        print(
            json.dumps(
                {
                    "event": "state_reset_condition_complete",
                    "condition": condition_id,
                    "accuracy": prediction_metrics(candidate)["accuracy"],
                }
            ),
            flush=True,
        )
    if not prediction_paths["reset_full"].is_file():
        raise RuntimeError("segmented full reference must be evaluated first")
    reference = load_predictions(prediction_paths["reset_full"])
    records = [
        _prediction_record(
            args.root,
            condition_id,
            load_predictions(prediction_paths[condition_id]),
            production,
            reference,
            bootstrap_repeats=args.bootstrap_repeats,
        )
        for condition_id, _interval in conditions
        if prediction_paths[condition_id].is_file()
    ]
    complete = len(records) == len(conditions)
    figure_path = _figure(args.root, records)
    result = {
        "schema": "lnet.r2k3.followup.state_reset.v1",
        "status": "complete" if complete else "partial",
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "checkpoint_sha256_after": sha256_file(args.checkpoint),
        "model_state_sha256_before": initial_state,
        "model_state_sha256_after": model_state_sha256(frozen.model),
        "protocol": {
            "samples": len(selected_indices),
            "per_class": args.per_class,
            "intervals": ["full", 32, 16, 8, 4, 2, 1],
            "reset_definition": "direction-aware state reset before every interval-th horizontal and vertical recurrence step",
            "implementation": "position-dependent decay is zero at reset boundaries; fused Triton recurrence transports state and variance",
            "bootstrap_repeats": args.bootstrap_repeats,
            "no_training": True,
        },
        "conditions": records,
        "figure": {
            "path": str(figure_path.relative_to(args.root)),
            "sha256": sha256_file(figure_path),
            "bytes": figure_path.stat().st_size,
        },
    }
    atomic_json(args.root / "state_reset.json", result)
    print(json.dumps({"event": "state_reset_progress", "status": result["status"], "conditions": len(records)}))


if __name__ == "__main__":
    main()
