#!/usr/bin/env python3
"""Run baseline and frozen-checkpoint causal interventions for K128/D2262."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
import argparse
import contextlib
import json
import statistics
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import a2d_r2k3_diagnostics as diagnostics
import run_a2d_r2k3_k128_depth_p_overnight_imagenet100 as model_runner
import torch
from analyze_a2d_r2k3_capacity_diagnostics import (
    _activation_aware_matrix,
    _set_wl_matrix,
    _wl_matrix,
)
from r2k3_frozen_study_common import (
    PredictionResult,
    artifact_record,
    atomic_json,
    balanced_indices,
    bh_fdr,
    bootstrap_mean_ci,
    branch_intervention,
    coefficient_intervention,
    collect_predictions,
    damping_shuffle,
    half_life_cap_transform,
    imagefolder,
    load_frozen_model,
    load_predictions,
    make_loader,
    model_state_sha256,
    module_scope,
    paired_effect,
    path_intervention,
    path_modules,
    path_permutation,
    phase_intervention,
    pole_mask_transform,
    postfusion_modules,
    postfusion_scale,
    prediction_metrics,
    random_matched_indices,
    reader_component_two_off,
    reader_modules,
    recurrence_off_transform,
    save_predictions,
    scan_modules,
    sha256_file,
    write_family,
)
from scipy import stats
from torch import Tensor
from torch.nn import functional


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--intervention-per-class", type=int, default=50)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--max-new-conditions", type=int, default=0)
    parser.add_argument(
        "--group",
        choices=("baseline", "recurrence", "poles", "paths", "branches", "rank", "q", "all"),
        default="all",
    )
    return parser.parse_args()


class CheckpointYieldError(RuntimeError):
    """Signal a clean process restart after a bounded number of GPU conditions."""


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


def _register_family(root: Path, study: dict[str, Any], family: dict[str, Any]) -> None:
    path = write_family(root, family)
    family_id = str(family["family_id"])
    cast("dict[str, str]", study.setdefault("families", {}))[family_id] = str(
        path.relative_to(root)
    )
    _upsert_artifact(
        study,
        artifact_record(root, family_id, path, "application/json"),
    )


def _slice_predictions(result: PredictionResult, indices: Tensor) -> PredictionResult:
    positions = {int(index): position for position, index in enumerate(result.indices.tolist())}
    selection = torch.tensor([positions[int(index)] for index in indices.tolist()])
    return PredictionResult(
        indices=result.indices[selection],
        labels=result.labels[selection],
        logits=result.logits[selection],
        losses=result.losses[selection],
        margins=result.margins[selection],
        confidences=result.confidences[selection],
        predictions=result.predictions[selection],
    )


def _p_value(delta: Tensor) -> float:
    if delta.numel() < 2 or float(delta.std(unbiased=False)) == 0.0:
        return 1.0 if float(delta.abs().max()) == 0.0 else 0.0
    result = stats.ttest_1samp(delta.double().numpy(), popmean=0.0)
    return float(result.pvalue)


class ConditionRunner:
    def __init__(
        self,
        *,
        root: Path,
        model: torch.nn.Module,
        loader: Any,
        device: torch.device,
        baseline: PredictionResult,
        bootstrap_repeats: int,
        max_new_conditions: int = 0,
    ) -> None:
        self.root = root
        self.model = model
        self.loader = loader
        self.device = device
        self.baseline = baseline
        self.bootstrap_repeats = bootstrap_repeats
        self.max_new_conditions = max_new_conditions
        self.new_conditions = 0
        self.records: list[dict[str, Any]] = []

    def execute(
        self,
        family_id: str,
        condition_id: str,
        context: contextlib.AbstractContextManager[Any],
        *,
        factors: dict[str, Any] | None = None,
        matched_control: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = self.root / "predictions" / family_id / f"{condition_id}.pt"
        record_path = self.root / "condition_records" / family_id / f"{condition_id}.json"
        if path.is_file() and record_path.is_file():
            record = json.loads(record_path.read_text())
            record["prediction_source"] = "record_cache"
            self.records.append(record)
            print(
                json.dumps(
                    {
                        "event": "condition_complete",
                        "family": family_id,
                        "condition": condition_id,
                        "accuracy": cast("dict[str, Any]", record["metrics"])["accuracy"],
                        "accuracy_delta_pp": cast(
                            "dict[str, Any]", record["paired_effect"]
                        )["accuracy_delta_pp"],
                        "source": "record_cache",
                    }
                ),
                flush=True,
            )
            return record
        if path.is_file():
            candidate = load_predictions(path)
            source = "cache"
        else:
            if self.max_new_conditions and self.new_conditions >= self.max_new_conditions:
                raise CheckpointYieldError(
                    f"yield after {self.new_conditions} new GPU conditions for fresh context"
                )
            with context:
                candidate = collect_predictions(self.model, self.loader, self.device)
            save_predictions(path, candidate)
            source = "evaluation"
            self.new_conditions += 1
        effect = paired_effect(
            self.baseline,
            candidate,
            bootstrap_repeats=self.bootstrap_repeats,
            seed=20260901 + len(self.records),
        )
        effect["p_value"] = _p_value(candidate.losses - self.baseline.losses)
        record = {
            "condition_id": condition_id,
            "status": "complete",
            "factors": factors or {},
            "matched_control": matched_control or {},
            "metrics": prediction_metrics(candidate),
            "paired_effect": effect,
            "per_example_path": str(path.relative_to(self.root)),
            "per_example_sha256": sha256_file(path),
            "per_example_bytes": path.stat().st_size,
            "prediction_source": source,
        }
        self.records.append(record)
        atomic_json(record_path, record)
        print(
            json.dumps(
                {
                    "event": "condition_complete",
                    "family": family_id,
                    "condition": condition_id,
                    "accuracy": cast("dict[str, Any]", record["metrics"])["accuracy"],
                    "accuracy_delta_pp": effect["accuracy_delta_pp"],
                    "source": source,
                }
            ),
            flush=True,
        )
        return record

    @staticmethod
    def unsupported(condition_id: str, reason: str) -> dict[str, Any]:
        return {
            "condition_id": condition_id,
            "status": "unsupported",
            "reason": reason,
            "metrics": {},
        }


def _baseline(
    args: argparse.Namespace,
    frozen: Any,
    full_loader: Any,
    study: dict[str, Any],
) -> PredictionResult:
    path = args.root / "predictions" / "baseline_full_val.pt"
    if path.is_file():
        result = load_predictions(path)
    else:
        result = collect_predictions(frozen.model, full_loader, torch.device(args.device))
        save_predictions(path, result)
    metrics = prediction_metrics(result)
    study["baseline"] = {
        **metrics,
        "prediction_path": str(path.relative_to(args.root)),
        "prediction_sha256": sha256_file(path),
        "checkpoint_accuracy_record": 0.8830000162124634,
        "historical_best_accuracy": 0.883400022983551,
        "historical_best_epoch": 99,
    }
    _upsert_artifact(
        study,
        artifact_record(args.root, "A_baseline_predictions", path, "application/x-pytorch"),
    )
    atomic_json(args.root / "study.json", study)
    return result


def _band_indices(model: torch.nn.Module, kind: str) -> dict[str, dict[str, list[int]]]:
    output = {"low": {}, "mid": {}, "high": {}}
    for name, module in scan_modules(model).items():
        damping_x, damping_y = (value.reshape(-1).detach().cpu() for value in module._damping_fields())
        if kind == "memory":
            values = -torch.sqrt(damping_x * damping_y)
        elif kind == "frequency":
            values = torch.hypot(module.phase_x.detach().cpu(), module.phase_y.detach().cpu())
        else:
            raise ValueError(kind)
        ordered = torch.argsort(values)
        chunks = torch.tensor_split(ordered, 3)
        for label, chunk in zip(("low", "mid", "high"), chunks, strict=True):
            output[label][name] = sorted(map(int, chunk.tolist()))
    return output


def _family(
    family_id: str,
    conditions: list[dict[str, Any]],
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    condition_ids = [str(condition["condition_id"]) for condition in conditions]
    p_values = {
        str(condition["condition_id"]): float(condition["paired_effect"]["p_value"])
        for condition in conditions
        if condition.get("status") == "complete"
        and isinstance(condition.get("paired_effect"), dict)
    }
    adjusted = bh_fdr(p_values)
    for condition in conditions:
        if str(condition["condition_id"]) in adjusted:
            condition["paired_effect"]["q_value_bh"] = adjusted[str(condition["condition_id"])]
    return {
        "family_id": family_id,
        "schema": "lnet.r2k3.study.family.v1",
        "status": "complete",
        "required_condition_ids": sorted(required or condition_ids),
        "observed_condition_ids": sorted(condition_ids),
        "conditions": conditions,
        "statistics": {
            "paired_test": "two-sided one-sample t-test on per-image CE delta",
            "fdr": "Benjamini-Hochberg within family",
        },
    }


def _recurrence(
    runner: ConditionRunner,
    model: torch.nn.Module,
) -> dict[str, Any]:
    scans = scan_modules(model)
    conditions = [
        runner.execute(
            "F03_recurrence_causality",
            "recurrence_off",
            coefficient_intervention(scans, recurrence_off_transform),
        )
    ]
    for half_life in (2, 4, 8, 16):
        conditions.append(
            runner.execute(
                "F03_recurrence_causality",
                f"effective_half_life_cap_{half_life}",
                coefficient_intervention(scans, half_life_cap_transform(float(half_life))),
                factors={"half_life_cap_steps": half_life},
            )
        )
    reason = (
        "Exact periodic state reset requires position-dependent pole coefficients, while the frozen "
        "production scan accepts compact mode-static 1x1x1xM coefficients. The executed half-life caps "
        "are exact static-coefficient alternatives, not reset-horizon experiments."
    )
    for condition_id in (
        "periodic_reset_1",
        "periodic_reset_2",
        "periodic_reset_4",
        "periodic_reset_8",
        "periodic_reset_16",
        "distant_region_reset",
    ):
        conditions.append(runner.unsupported(condition_id, reason))
    return _family("F03_recurrence_causality", conditions)


def _poles(
    args: argparse.Namespace,
    runner: ConditionRunner,
    model: torch.nn.Module,
) -> dict[str, Any]:
    scans = scan_modules(model)
    conditions: list[dict[str, Any]] = []
    bands_by_kind = {kind: _band_indices(model, kind) for kind in ("memory", "frequency")}
    for kind, bands in bands_by_kind.items():
        for band, selected in bands.items():
            counts = {name: len(indices) for name, indices in selected.items()}
            target_id = f"{kind}_{band}_off"
            random_ids = [f"{target_id}_random_{seed:02d}" for seed in range(args.random_repeats)]
            conditions.append(
                runner.execute(
                    "F04_pole_targeted",
                    target_id,
                    coefficient_intervention(scans, pole_mask_transform(selected)),
                    factors={"kind": kind, "band": band, "removed_count_by_module": counts},
                    matched_control={
                        "condition_ids": random_ids,
                        "matching_keys": ["module", "removed_count"],
                        "repeats": args.random_repeats,
                    },
                )
            )
            for seed, random_id in enumerate(random_ids):
                selected_random = random_matched_indices(scans, counts, seed=91000 + seed)
                conditions.append(
                    runner.execute(
                        "F04_pole_targeted",
                        random_id,
                        coefficient_intervention(scans, pole_mask_transform(selected_random)),
                        factors={"control_for": target_id, "seed": seed},
                    )
                )
    for condition_id, context in (
        ("omega_zero", phase_intervention(scans, kind="omega_zero")),
        ("omega_sign_flip", phase_intervention(scans, kind="omega_sign_flip")),
        ("omega_shuffle", phase_intervention(scans, kind="omega_shuffle", seed=20260901)),
        ("xy_swap", phase_intervention(scans, kind="xy_swap")),
        ("alpha_shuffle", damping_shuffle(scans, seed=20260901)),
    ):
        conditions.append(runner.execute("F04_pole_targeted", condition_id, context))
    terminal = module_scope(scans, "terminal")
    terminal_selected = {name: list(range(module.modes)) for name, module in terminal.items()}
    conditions.append(
        runner.execute(
            "F04_pole_targeted",
            "terminal_all_poles_off",
            coefficient_intervention(scans, pole_mask_transform(terminal_selected)),
        )
    )
    extra = module_scope(scans, "extra14")
    extra_selected: dict[str, list[int]] = {}
    for name, module in extra.items():
        damping_x, damping_y = module._damping_fields()
        values = torch.sqrt(damping_x.reshape(-1) * damping_y.reshape(-1))
        extra_selected[name] = sorted(map(int, torch.argsort(values)[: module.modes // 3].tolist()))
    conditions.append(
        runner.execute(
            "F04_pole_targeted",
            "extra14_long_memory_off",
            coefficient_intervention(scans, pole_mask_transform(extra_selected)),
        )
    )
    family = _family("F04_pole_targeted", conditions)
    condition_by_id = {str(condition["condition_id"]): condition for condition in conditions}
    contrasts: list[dict[str, Any]] = []
    for target in conditions:
        matched = target.get("matched_control")
        if not isinstance(matched, dict) or "condition_ids" not in matched:
            continue
        target_prediction = load_predictions(runner.root / str(target["per_example_path"]))
        controls = [
            load_predictions(
                runner.root / str(condition_by_id[str(control_id)]["per_example_path"])
            )
            for control_id in matched["condition_ids"]
        ]
        baseline_correct = runner.baseline.predictions.eq(runner.baseline.labels).float()
        target_accuracy_change = target_prediction.predictions.eq(
            target_prediction.labels
        ).float() - baseline_correct
        control_accuracy_change = torch.stack(
            [control.predictions.eq(control.labels).float() - baseline_correct for control in controls]
        )
        accuracy_contrast = target_accuracy_change - control_accuracy_change.mean(0)
        target_loss_change = target_prediction.losses - runner.baseline.losses
        control_loss_change = torch.stack(
            [control.losses - runner.baseline.losses for control in controls]
        )
        loss_contrast = target_loss_change - control_loss_change.mean(0)
        accuracy_ci = bootstrap_mean_ci(
            accuracy_contrast,
            runner.baseline.labels,
            repeats=args.bootstrap_repeats,
            seed=20262901 + len(contrasts),
        )
        loss_ci = bootstrap_mean_ci(
            loss_contrast,
            runner.baseline.labels,
            repeats=args.bootstrap_repeats,
            seed=20263901 + len(contrasts),
        )
        target_delta = float(cast("dict[str, Any]", target["paired_effect"])["accuracy_delta_pp"])
        control_deltas = [
            float(cast("dict[str, Any]", condition_by_id[control_id]["paired_effect"])["accuracy_delta_pp"])
            for control_id in matched["condition_ids"]
        ]
        contrasts.append(
            {
                "target_condition_id": target["condition_id"],
                "control_condition_ids": matched["condition_ids"],
                "accuracy_delta_pp_vs_random_mean": 100.0 * float(accuracy_contrast.mean()),
                "accuracy_delta_pp_vs_random_mean_ci95": [
                    100.0 * accuracy_ci[0],
                    100.0 * accuracy_ci[1],
                ],
                "cross_entropy_delta_vs_random_mean": float(loss_contrast.mean()),
                "cross_entropy_delta_vs_random_mean_ci95": list(loss_ci),
                "random_accuracy_delta_pp_mean": statistics.fmean(control_deltas),
                "random_accuracy_delta_pp_range": [min(control_deltas), max(control_deltas)],
                "empirical_p_more_harmful": (
                    1 + sum(control_delta <= target_delta for control_delta in control_deltas)
                )
                / (1 + len(control_deltas)),
                "bootstrap_repeats": args.bootstrap_repeats,
            }
        )
    adjusted = bh_fdr(
        {
            str(contrast["target_condition_id"]): float(
                contrast["empirical_p_more_harmful"]
            )
            for contrast in contrasts
        }
    )
    for contrast in contrasts:
        contrast["empirical_q_more_harmful_bh"] = adjusted[
            str(contrast["target_condition_id"])
        ]
    family["matched_control_contrasts"] = contrasts
    return family


def _paths(
    runner: ConditionRunner,
    model: torch.nn.Module,
) -> dict[str, Any]:
    paths = path_modules(model)
    conditions: list[dict[str, Any]] = []
    for path in range(4):
        conditions.append(
            runner.execute(
                "F05_d4_paths",
                f"path_{path}_off",
                path_intervention(paths, (path,)),
                factors={"removed_paths": [path]},
            )
        )
    for name, removed in (
        ("paths_0_1_off", (0, 1)),
        ("paths_2_3_off", (2, 3)),
        ("paths_0_2_off", (0, 2)),
        ("paths_1_3_off", (1, 3)),
        ("all_paths_off", (0, 1, 2, 3)),
    ):
        conditions.append(
            runner.execute(
                "F05_d4_paths",
                name,
                path_intervention(paths, removed),
                factors={"removed_paths": list(removed)},
            )
        )
    for name, permutation in (
        ("path_reverse", (3, 2, 1, 0)),
        ("path_rotate", (2, 0, 3, 1)),
        ("path_swap_x", (1, 0, 3, 2)),
        ("path_swap_y", (2, 3, 0, 1)),
    ):
        conditions.append(
            runner.execute(
                "F05_d4_paths",
                name,
                path_permutation(paths, permutation),
                factors={"permutation": list(permutation)},
            )
        )
    conditions.append(
        runner.unsupported(
            "rotation_equivariance",
            "Executed in F11 with paired rotated inputs because it is an input transformation, not a parameter intervention.",
        )
    )
    return _family("F05_d4_paths", conditions)


def _branches(
    runner: ConditionRunner,
    model: torch.nn.Module,
) -> dict[str, Any]:
    conditions = [
        runner.execute(
            "F06_transition_decomposition",
            "memory_zero",
            branch_intervention(model, zero_memory=True),
        ),
        runner.execute(
            "F06_transition_decomposition",
            "carry_zero",
            branch_intervention(model, zero_carry=True),
        ),
        runner.execute(
            "F06_transition_decomposition",
            "memory_and_carry_zero",
            branch_intervention(model, zero_memory=True, zero_carry=True),
        ),
    ]
    postfusion = postfusion_modules(model)
    for scale in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
        conditions.append(
            runner.execute(
                "F06_transition_decomposition",
                f"postfusion_scale_{str(scale).replace('.', '_')}",
                postfusion_scale(postfusion, scale),
                factors={"scale": scale},
            )
        )
    for label in sorted(postfusion):
        conditions.append(
            runner.execute(
                "F06_transition_decomposition",
                f"postfusion_off_{label.replace('.', '_')}",
                postfusion_scale({label: postfusion[label]}, 0.0),
                factors={"module": label, "scale": 0.0},
            )
        )
    for condition_id in ("same_class_branch_swap", "different_class_branch_swap"):
        conditions.append(
            runner.unsupported(
                condition_id,
                "Branch swapping requires batch-coupled activation injection; deferred to F09 cached representation analysis to preserve sample identity.",
            )
        )
    return _family("F06_transition_decomposition", conditions)


def _collect_postfusion_samples(
    model: torch.nn.Module,
    modules: Mapping[str, torch.nn.Module],
    loader: Any,
    device: torch.device,
    *,
    batches: int,
    rows_per_batch: int,
) -> dict[str, dict[str, Tensor]]:
    collected: dict[str, dict[str, list[Tensor]]] = {
        name: {"value": [], "gate": [], "out": []} for name in modules
    }
    with contextlib.ExitStack() as stack:
        for name, module in modules.items():
            def hook(
                active: Any,
                inputs: tuple[Tensor, ...],
                *,
                label: str = name,
            ) -> None:
                real, imag = inputs
                unit_real, unit_imag = active.norm(real, imag)
                value_real, value_imag = active.value(unit_real, unit_imag)
                gate = active._gate_values(unit_real, unit_imag)
                out_real, out_imag = value_real * gate, value_imag * gate
                collected[label]["value"].append(
                    diagnostics.realify(
                        diagnostics.complex_sample(unit_real, unit_imag, rows_per_batch)
                    ).cpu()
                )
                collected[label]["gate"].append(
                    diagnostics.real_sample(
                        torch.cat((unit_real, unit_imag), dim=-1), rows_per_batch
                    ).cpu()
                )
                collected[label]["out"].append(
                    diagnostics.realify(
                        diagnostics.complex_sample(out_real, out_imag, rows_per_batch)
                    ).cpu()
                )

            stack.callback(module.register_forward_pre_hook(hook).remove)
        with torch.inference_mode():
            for batch_index, (images_cpu, _labels_cpu, _indices_cpu) in enumerate(loader):
                if batch_index >= batches:
                    break
                images = images_cpu.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    cast("Any", model).raw_descriptor(images)
    return {
        name: {key: torch.cat(values).float() for key, values in layers.items()}
        for name, layers in collected.items()
    }


@contextlib.contextmanager
def _activation_aware_postfusions(
    modules: Mapping[str, torch.nn.Module],
    samples: Mapping[str, Mapping[str, Tensor]],
    fraction: float,
    device: torch.device,
) -> Generator[dict[str, Any]]:
    backups: list[tuple[Any, str, Tensor]] = []
    active_diagnostics: dict[str, Any] = {}
    with torch.no_grad():
        for name, postfusion in modules.items():
            active_diagnostics[name] = {}
            for key, layer in (
                ("value", cast("Any", postfusion).value),
                ("gate", cast("Any", postfusion).gate),
                ("out", cast("Any", postfusion).out),
            ):
                if isinstance(layer, torch.nn.Linear):
                    matrix = layer.weight.detach().float()
                    backups.append((layer, "weight", layer.weight.detach().clone()))
                else:
                    matrix = _wl_matrix(layer)
                    for parameter_name in (
                        "weight_real",
                        "weight_imag",
                        "conjugate_real",
                        "conjugate_imag",
                    ):
                        parameter = cast("Tensor", getattr(layer, parameter_name))
                        backups.append((layer, parameter_name, parameter.detach().clone()))
                reduced, layer_diagnostics = _activation_aware_matrix(
                    matrix,
                    samples[name][key],
                    fraction,
                    device,
                )
                active_diagnostics[name][key] = layer_diagnostics
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.copy_(reduced)
                else:
                    _set_wl_matrix(layer, reduced)
    try:
        yield active_diagnostics
    finally:
        with torch.no_grad():
            for layer, parameter_name, value in backups:
                cast("Tensor", getattr(layer, parameter_name)).copy_(value)


def _rank_interventions(
    args: argparse.Namespace,
    runner: ConditionRunner,
    model: torch.nn.Module,
    existing: dict[str, Any],
) -> dict[str, Any]:
    conditions = list(existing.get("conditions", []))
    readers = reader_modules(model)
    scopes = {
        "reader_component2_main_off": module_scope(readers, "main"),
        "reader_component2_extra14_off": module_scope(readers, "extra14"),
        "reader_component2_all_off": readers,
    }
    for condition_id, selected in scopes.items():
        conditions.append(
            runner.execute(
                "F07_rank_capacity",
                condition_id,
                reader_component_two_off(selected),
            )
        )
    calibration_dataset = imagefolder(args.data_root, "train", evaluation_view=True)
    calibration_indices = balanced_indices(calibration_dataset.targets, 10)
    calibration_loader = make_loader(
        calibration_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        indices=calibration_indices,
    )
    postfusions = postfusion_modules(model)
    samples = _collect_postfusion_samples(
        model,
        postfusions,
        calibration_loader,
        runner.device,
        batches=8,
        rows_per_batch=1024,
    )
    calibration_samples_evaluated = min(
        len(calibration_indices),
        8 * args.batch_size,
    )
    for fraction in (0.75, 0.50, 0.25):
        condition_id = f"postfusion_all_activation_rank_{round(100 * fraction)}"
        conditions.append(
            runner.execute(
                "F07_rank_capacity",
                condition_id,
                _activation_aware_postfusions(postfusions, samples, fraction, runner.device),
                factors={
                    "retained_rank_fraction": fraction,
                    "scope": "all main/base/extra14 PostFusion value/gate/out",
                    "calibration_split": "train",
                    "calibration_bank_samples": len(calibration_indices),
                    "calibration_samples_evaluated": calibration_samples_evaluated,
                    "calibration_batches": 8,
                    "rows_per_batch": 1024,
                },
            )
        )
    scoped_postfusions = {
        "main": module_scope(postfusions, "main"),
        "base": {
            name: module
            for name, module in postfusions.items()
            if name.startswith("same_resolution.")
        },
        "extra14": module_scope(postfusions, "extra14"),
        **{
            f"main_{stage}": {f"main.{stage}": postfusions[f"main.{stage}"]}
            for stage in ("stage1", "stage2", "stage3")
        },
    }
    for scope, selected in scoped_postfusions.items():
        conditions.append(
            runner.execute(
                "F07_rank_capacity",
                f"postfusion_{scope}_activation_rank_50",
                _activation_aware_postfusions(selected, samples, 0.50, runner.device),
                factors={
                    "retained_rank_fraction": 0.50,
                    "scope": scope,
                    "calibration_split": "train",
                    "calibration_bank_samples": len(calibration_indices),
                    "calibration_samples_evaluated": calibration_samples_evaluated,
                    "calibration_batches": 8,
                    "rows_per_batch": 1024,
                },
            )
        )
    return _family("F07_rank_capacity", conditions)


def _descriptor_cache(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    descriptors: list[Tensor] = []
    labels: list[Tensor] = []
    indices: list[Tensor] = []
    with torch.inference_mode():
        for images_cpu, targets_cpu, indices_cpu in loader:
            images = images_cpu.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                descriptor = cast("Any", model).raw_descriptor(images)
            descriptors.append(descriptor.detach().float().cpu())
            labels.append(targets_cpu.long())
            indices.append(indices_cpu.long())
    return torch.cat(descriptors), torch.cat(labels), torch.cat(indices)


def _terminal_q(
    args: argparse.Namespace,
    model: torch.nn.Module,
    full_loader: Any,
    study: dict[str, Any],
) -> dict[str, Any]:
    descriptor, labels, indices = _descriptor_cache(model, full_loader, torch.device(args.device))
    pole_modes = model_runner.SPECS[model_runner.D2262_P128_TERM].pole_modes
    parts = diagnostics.descriptor_parts(descriptor, pole_modes)
    q4 = parts[-1]
    contribution = diagnostics.classifier_contribution(model, q4, labels, pole_modes[-1], 20260901)
    cache_path = args.root / "features" / "terminal_q_full_val.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"q4": q4, "labels": labels, "indices": indices}, cache_path)
    classifier = cast("Any", model).classifier.affine
    mean = classifier.standardizer.running_mean.detach().cpu().float()
    variance = classifier.standardizer.running_var.detach().cpu().float()
    standardized = (q4 - mean) * torch.rsqrt(variance + classifier.standardizer.eps)
    weight = classifier.linear.weight.detach().cpu().float()
    logits = standardized @ weight.T + classifier.linear.bias.detach().cpu().float()
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    runner_up = masked.argmax(-1)
    fingerprints = torch.zeros(100, pole_modes[-1])
    counts = torch.zeros(100)
    for pole in range(pole_modes[-1]):
        coordinates = torch.tensor([direction * pole_modes[-1] + pole for direction in range(4)])
        true_contribution = (
            standardized[:, coordinates]
            * weight[labels][:, coordinates]
        ).sum(1)
        runner_contribution = (
            standardized[:, coordinates]
            * weight[runner_up][:, coordinates]
        ).sum(1)
        margin_contribution = true_contribution - runner_contribution
        fingerprints.index_add_(0, labels, functional.one_hot(
            torch.full_like(labels, pole), pole_modes[-1]
        ).float() * margin_contribution[:, None])
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    fingerprints /= counts[:, None].clamp_min(1)
    conditions = [
        {
            "condition_id": "terminal_q_contribution",
            "status": "complete",
            "metrics": {
                **contribution,
                "mean_true_margin": float((true - masked.max(-1).values).mean()),
                "class_pole_fingerprint": fingerprints.tolist(),
            },
        }
    ]
    _upsert_artifact(
        study,
        artifact_record(args.root, "A_terminal_q", cache_path, "application/x-pytorch"),
    )
    return _family("F08_terminal_q", conditions)


def main() -> None:
    args = _args()
    study = _study(args.root)
    device = torch.device(args.device)
    frozen = load_frozen_model(args.checkpoint, device)
    dataset = imagefolder(args.data_root, "val")
    full_loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    baseline_full = _baseline(args, frozen, full_loader, study)
    selected_indices = balanced_indices(dataset.targets, args.intervention_per_class)
    intervention_loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        indices=selected_indices,
    )
    baseline = _slice_predictions(baseline_full, torch.tensor(selected_indices))
    runner_for_conditions = ConditionRunner(
        root=args.root,
        model=frozen.model,
        loader=intervention_loader,
        device=device,
        baseline=baseline,
        bootstrap_repeats=args.bootstrap_repeats,
        max_new_conditions=args.max_new_conditions,
    )
    groups = (
        ("recurrence", lambda: _recurrence(runner_for_conditions, frozen.model)),
        ("poles", lambda: _poles(args, runner_for_conditions, frozen.model)),
        ("paths", lambda: _paths(runner_for_conditions, frozen.model)),
        ("branches", lambda: _branches(runner_for_conditions, frozen.model)),
    )
    if args.group != "baseline":
        for name, function in groups:
            if args.group in {"all", name}:
                try:
                    family = function()
                except CheckpointYieldError as error:
                    if sha256_file(args.checkpoint) != frozen.checkpoint_sha256:
                        raise RuntimeError("checkpoint changed before checkpoint-yield") from error
                    if model_state_sha256(frozen.model) != frozen.initial_state_sha256:
                        raise RuntimeError("model state changed before checkpoint-yield") from error
                    print(
                        json.dumps(
                            {
                                "event": "checkpoint_yield",
                                "group": name,
                                "new_conditions": runner_for_conditions.new_conditions,
                                "reason": str(error),
                            }
                        ),
                        flush=True,
                    )
                    return
                _register_family(args.root, study, family)
                atomic_json(args.root / "study.json", study)
    if args.group in {"all", "rank"}:
        existing_path = args.root / "families" / "F07_rank_capacity.json"
        existing = json.loads(existing_path.read_text())
        try:
            rank_family = _rank_interventions(args, runner_for_conditions, frozen.model, existing)
        except CheckpointYieldError as error:
            if sha256_file(args.checkpoint) != frozen.checkpoint_sha256:
                raise RuntimeError("checkpoint changed before rank checkpoint-yield") from error
            if model_state_sha256(frozen.model) != frozen.initial_state_sha256:
                raise RuntimeError("model state changed before rank checkpoint-yield") from error
            print(
                json.dumps(
                    {
                        "event": "checkpoint_yield",
                        "group": "rank",
                        "new_conditions": runner_for_conditions.new_conditions,
                    }
                )
            )
            return
        _register_family(args.root, study, rank_family)
        atomic_json(args.root / "study.json", study)
    if args.group in {"all", "q"}:
        family8 = _terminal_q(args, frozen.model, full_loader, study)
        _register_family(args.root, study, family8)
        atomic_json(args.root / "study.json", study)
    checkpoint_after = sha256_file(args.checkpoint)
    state_after = model_state_sha256(frozen.model)
    guard = study["provenance"]["no_training_guard"]
    guard["checkpoint_sha256_after"] = checkpoint_after
    guard["model_state_sha256_after"] = state_after
    study["provenance"]["checkpoint"]["sha256_after"] = checkpoint_after
    atomic_json(args.root / "study.json", study)
    if checkpoint_after != frozen.checkpoint_sha256:
        raise RuntimeError("checkpoint changed during evaluation")
    if state_after != frozen.initial_state_sha256:
        active_state = frozen.model.state_dict()
        checkpoint_state = cast("dict[str, Tensor]", frozen.payload["model"])
        changed = [
            name
            for name, value in active_state.items()
            if name not in checkpoint_state
            or not torch.equal(value.detach().cpu(), checkpoint_state[name].detach().cpu())
        ]
        raise RuntimeError(
            "model parameters/buffers were not restored after interventions: "
            + ", ".join(changed[:20])
        )
    print(
        json.dumps(
            {
                "event": "evaluation_complete",
                "group": args.group,
                "baseline_accuracy": study["baseline"]["accuracy"],
                "state_restored": True,
            }
        )
    )


if __name__ == "__main__":
    main()
