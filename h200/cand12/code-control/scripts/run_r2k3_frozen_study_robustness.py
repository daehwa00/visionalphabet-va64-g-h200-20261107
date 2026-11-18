#!/usr/bin/env python3
"""Evaluate frozen-checkpoint robustness, invariance, and calibration."""

from __future__ import annotations

# pyright: reportAny=false, reportArgumentType=false, reportAssignmentType=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from r2k3_frozen_study_common import (
    artifact_record,
    atomic_json,
    bh_fdr,
    coefficient_intervention,
    collect_predictions,
    half_life_cap_transform,
    load_frozen_model,
    load_predictions,
    make_loader,
    paired_effect,
    prediction_metrics,
    save_predictions,
    scan_modules,
    sha256_file,
    write_family,
)

if TYPE_CHECKING:
    from PIL import Image
from scipy import stats
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--max-new-conditions", type=int, default=0)
    return parser.parse_args()


class PerturbedImageFolder(Dataset[tuple[Tensor, int]]):
    def __init__(self, root: Path, condition: str, severity: float = 0.0) -> None:
        self.source = datasets.ImageFolder(root)
        self.condition = condition
        self.severity = severity
        self.targets = self.source.targets
        self.samples = self.source.samples
        self.class_to_idx = self.source.class_to_idx

    def __len__(self) -> int:
        return len(self.source)

    def _base(self, image: Image.Image) -> Tensor:
        image = vision_functional.resize(
            image,
            256,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        image = vision_functional.center_crop(image, (224, 224))
        return vision_functional.to_tensor(image)

    def _resolution_view(self, image: Image.Image, size: int) -> Tensor:
        resize_size = round(256 * size / 224)
        image = vision_functional.resize(
            image,
            resize_size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        image = vision_functional.center_crop(image, (size, size))
        return vision_functional.to_tensor(image)

    def _patch_shuffle(self, value: Tensor, index: int) -> Tensor:
        patches = value.reshape(3, 4, 56, 4, 56).permute(1, 3, 0, 2, 4).reshape(16, 3, 56, 56)
        generator = torch.Generator().manual_seed(20260901 + index)
        patches = patches[torch.randperm(16, generator=generator)]
        return patches.reshape(4, 4, 3, 56, 56).permute(2, 0, 3, 1, 4).reshape(3, 224, 224)

    def _perturb(self, value: Tensor, index: int) -> Tensor:  # noqa: PLR0911
        kind, severity = self.condition, self.severity
        if kind == "clean":
            return value
        if kind == "noise":
            generator = torch.Generator().manual_seed(20260901 + index)
            return (value + severity * torch.randn(value.shape, generator=generator)).clamp(0, 1)
        if kind == "blur":
            return vision_functional.gaussian_blur(value, 17, [severity, severity])
        if kind == "contrast":
            return vision_functional.adjust_contrast(value, severity)
        if kind == "occlusion":
            side = max(1, round(224 * math.sqrt(severity)))
            start = (224 - side) // 2
            active = value.clone()
            active[:, start : start + side, start : start + side] = 0.5
            return active
        if kind == "resolution":
            raise RuntimeError("resolution views are constructed directly from the source image")
        if kind == "lowpass":
            return vision_functional.gaussian_blur(value, 21, [severity, severity])
        if kind == "highpass":
            low = vision_functional.gaussian_blur(value, 21, [severity, severity])
            return (0.5 + value - low).clamp(0, 1)
        if kind == "bandpass":
            low = vision_functional.gaussian_blur(value, 9, [1.5, 1.5])
            lower = vision_functional.gaussian_blur(value, 25, [severity, severity])
            return (0.5 + low - lower).clamp(0, 1)
        if kind == "rotate90":
            return torch.rot90(value, 1, (1, 2))
        if kind == "hflip":
            return value.flip(2)
        if kind == "vflip":
            return value.flip(1)
        if kind == "translate":
            return vision_functional.affine(
                value,
                angle=0,
                translate=[int(severity), int(severity // 2)],
                scale=1.0,
                shear=[0.0, 0.0],
                fill=0.5,
            )
        if kind == "crop_scale":
            side = round(224 * severity)
            start = (224 - side) // 2
            return vision_functional.resized_crop(
                value,
                start,
                start,
                side,
                side,
                [224, 224],
                antialias=True,
            )
        if kind == "patch_shuffle":
            return self._patch_shuffle(value, index)
        if kind == "grayscale":
            return vision_functional.rgb_to_grayscale(value, num_output_channels=3)
        raise ValueError(f"unknown perturbation: {kind}")

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, target = self.source.samples[index]
        image = self.source.loader(path).convert("RGB")
        value = (
            self._resolution_view(image, int(self.severity))
            if self.condition == "resolution"
            else self._perturb(self._base(image), index)
        )
        return vision_functional.normalize(value, MEAN, STD), int(target)


def _condition_grid() -> tuple[tuple[str, str, float], ...]:
    conditions = [
        *((f"noise_{value}", "noise", value) for value in (0.05, 0.10, 0.20)),
        *((f"blur_{value}", "blur", value) for value in (1.0, 2.0, 4.0)),
        *((f"contrast_{value}", "contrast", value) for value in (0.75, 0.50, 0.25)),
        *((f"occlusion_{value}", "occlusion", value) for value in (0.10, 0.20, 0.30, 0.40)),
        *((f"resolution_{value}", "resolution", float(value)) for value in (192, 256, 288)),
        ("lowpass_2", "lowpass", 2.0),
        ("lowpass_4", "lowpass", 4.0),
        ("highpass_2", "highpass", 2.0),
        ("highpass_4", "highpass", 4.0),
        ("bandpass", "bandpass", 4.0),
        ("rotate90", "rotate90", 90.0),
        ("hflip", "hflip", 0.0),
        ("vflip", "vflip", 0.0),
        ("translate_16", "translate", 16.0),
        ("crop_scale_0.75", "crop_scale", 0.75),
        ("patch_shuffle_4x4", "patch_shuffle", 4.0),
        ("grayscale", "grayscale", 0.0),
    ]
    return tuple(conditions)


def _cross_grid() -> tuple[tuple[str, str, float], ...]:
    """Representative corruptions crossed with a mild recurrence intervention."""
    return (
        ("noise_0.1", "noise", 0.1),
        ("blur_2.0", "blur", 2.0),
        ("occlusion_0.2", "occlusion", 0.2),
        ("lowpass_4", "lowpass", 4.0),
        ("highpass_4", "highpass", 4.0),
        ("rotate90", "rotate90", 90.0),
        ("grayscale", "grayscale", 0.0),
    )


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


def _p_value(delta: Tensor) -> float:
    if float(delta.std(unbiased=False)) == 0.0:
        return 1.0 if float(delta.abs().max()) == 0 else 0.0
    return float(stats.ttest_1samp(delta.double().numpy(), popmean=0.0).pvalue)


def _family(
    family_id: str,
    conditions: list[dict[str, Any]],
) -> dict[str, Any]:
    p_values = {
        str(condition["condition_id"]): float(condition["paired_effect"]["p_value"])
        for condition in conditions
    }
    adjusted = bh_fdr(p_values)
    for condition in conditions:
        condition["paired_effect"]["q_value_bh"] = adjusted[str(condition["condition_id"])]
    ids = [str(condition["condition_id"]) for condition in conditions]
    return {
        "family_id": family_id,
        "schema": "lnet.r2k3.study.family.v1",
        "status": "complete",
        "required_condition_ids": sorted(ids),
        "observed_condition_ids": sorted(ids),
        "conditions": conditions,
    }


def main() -> None:
    args = _args()
    study = _study(args.root)
    baseline_path = args.root / str(study["baseline"]["prediction_path"])
    baseline = load_predictions(baseline_path)
    frozen = load_frozen_model(args.checkpoint, torch.device(args.device))
    conditions: list[dict[str, Any]] = []
    new_conditions = 0
    for condition_id, kind, severity in _condition_grid():
        dataset = PerturbedImageFolder(args.data_root / "val", kind, severity)
        loader = make_loader(dataset, batch_size=args.batch_size, workers=args.workers)
        path = args.root / "predictions" / "F11_robustness_invariance" / f"{condition_id}.pt"
        record_path = (
            args.root / "condition_records" / "F11_robustness_invariance" / f"{condition_id}.json"
        )
        if path.is_file() and record_path.is_file():
            conditions.append(json.loads(record_path.read_text()))
            continue
        if path.is_file():
            candidate = load_predictions(path)
            source = "cache"
        else:
            if args.max_new_conditions and new_conditions >= args.max_new_conditions:
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_yield",
                            "group": "robustness",
                            "new_conditions": new_conditions,
                        }
                    )
                )
                return
            candidate = collect_predictions(frozen.model, loader, torch.device(args.device))
            save_predictions(path, candidate)
            source = "evaluation"
            new_conditions += 1
        effect = paired_effect(
            baseline,
            candidate,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=20260901 + len(conditions),
        )
        effect["p_value"] = _p_value(candidate.losses - baseline.losses)
        record = {
            "condition_id": condition_id,
            "status": "complete",
            "factors": {"kind": kind, "severity": severity},
            "metrics": prediction_metrics(candidate),
            "paired_effect": effect,
            "per_example_path": str(path.relative_to(args.root)),
            "per_example_sha256": sha256_file(path),
            "per_example_bytes": path.stat().st_size,
            "prediction_agreement": float((candidate.predictions == baseline.predictions).float().mean()),
            "prediction_source": source,
            "logit_kl_from_baseline": float(
                torch.nn.functional.kl_div(
                    candidate.logits.log_softmax(-1),
                    baseline.logits.softmax(-1),
                    reduction="batchmean",
                )
            ),
        }
        conditions.append(record)
        atomic_json(record_path, record)
        print(
            json.dumps(
                {
                    "event": "robustness_condition_complete",
                    "condition": condition_id,
                    "accuracy": cast("dict[str, Any]", record["metrics"])["accuracy"],
                }
            ),
            flush=True,
        )
    clean_intervention_path = (
        args.root
        / "predictions"
        / "F03_recurrence_causality"
        / "effective_half_life_cap_2.pt"
    )
    if not clean_intervention_path.is_file():
        raise RuntimeError("F03 effective_half_life_cap_2 predictions are required for F11 crosses")
    clean_intervention = load_predictions(clean_intervention_path)
    scans = scan_modules(frozen.model)
    cross_condition_ids: list[str] = []
    for base_condition_id, kind, severity in _cross_grid():
        condition_id = f"half_life_cap_2_x_{base_condition_id}"
        cross_condition_ids.append(condition_id)
        dataset = PerturbedImageFolder(args.data_root / "val", kind, severity)
        loader = make_loader(dataset, batch_size=args.batch_size, workers=args.workers)
        path = args.root / "predictions" / "F11_robustness_invariance" / f"{condition_id}.pt"
        record_path = (
            args.root
            / "condition_records"
            / "F11_robustness_invariance"
            / f"{condition_id}.json"
        )
        if path.is_file() and record_path.is_file():
            conditions.append(json.loads(record_path.read_text()))
            continue
        if path.is_file():
            candidate = load_predictions(path)
            source = "cache"
        else:
            if args.max_new_conditions and new_conditions >= args.max_new_conditions:
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_yield",
                            "group": "robustness_cross",
                            "new_conditions": new_conditions,
                        }
                    )
                )
                return
            with coefficient_intervention(scans, half_life_cap_transform(2.0)):
                candidate = collect_predictions(frozen.model, loader, torch.device(args.device))
            save_predictions(path, candidate)
            source = "evaluation"
            new_conditions += 1
        corruption_path = (
            args.root / "predictions" / "F11_robustness_invariance" / f"{base_condition_id}.pt"
        )
        corruption = load_predictions(corruption_path)
        effect = paired_effect(
            corruption,
            candidate,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=20261901 + len(conditions),
        )
        effect["p_value"] = _p_value(candidate.losses - corruption.losses)
        interaction_accuracy_delta_pp = 100.0 * (
            prediction_metrics(candidate)["accuracy"]
            - prediction_metrics(corruption)["accuracy"]
            - prediction_metrics(clean_intervention)["accuracy"]
            + prediction_metrics(baseline)["accuracy"]
        )
        record = {
            "condition_id": condition_id,
            "status": "complete",
            "factors": {
                "kind": kind,
                "severity": severity,
                "intervention": "effective_half_life_cap_2",
                "comparison_baseline": base_condition_id,
            },
            "metrics": prediction_metrics(candidate),
            "paired_effect": effect,
            "interaction_accuracy_delta_pp": interaction_accuracy_delta_pp,
            "per_example_path": str(path.relative_to(args.root)),
            "per_example_sha256": sha256_file(path),
            "per_example_bytes": path.stat().st_size,
            "prediction_agreement": float(
                (candidate.predictions == corruption.predictions).float().mean()
            ),
            "prediction_source": source,
            "logit_kl_from_comparison": float(
                torch.nn.functional.kl_div(
                    candidate.logits.log_softmax(-1),
                    corruption.logits.softmax(-1),
                    reduction="batchmean",
                )
            ),
        }
        conditions.append(record)
        atomic_json(record_path, record)
        print(
            json.dumps(
                {
                    "event": "robustness_cross_complete",
                    "condition": condition_id,
                    "accuracy": cast("dict[str, Any]", record["metrics"])["accuracy"],
                    "interaction_accuracy_delta_pp": interaction_accuracy_delta_pp,
                }
            ),
            flush=True,
        )
    family11 = _family("F11_robustness_invariance", conditions)
    family11["intervention_cross"] = {
        "status": "complete",
        "intervention": "effective_half_life_cap_2",
        "condition_ids": cross_condition_ids,
        "rationale": "The strongest finite non-off recurrence restriction (half-life cap 2) was crossed with seven representative perturbation families; recurrence-off was avoided because its near-floor clean accuracy would obscure interactions.",
        "interaction_definition": "100 * (accuracy[cross] - accuracy[corruption] - accuracy[cap2_clean] + accuracy[clean])",
    }
    path11 = write_family(args.root, family11)
    family12_conditions = [
        {
            "condition_id": "baseline_calibration",
            "status": "complete",
            "metrics": {
                key: study["baseline"][key]
                for key in (
                    "accuracy",
                    "cross_entropy_mean",
                    "nll",
                    "brier",
                    "ece_15",
                    "mean_confidence",
                    "mean_margin",
                    "risk_coverage",
                    "per_class",
                )
            },
        },
        {
            "condition_id": "robustness_calibration",
            "status": "complete",
            "metrics": {
                condition["condition_id"]: {
                    key: condition["metrics"][key]
                    for key in (
                        "accuracy",
                        "nll",
                        "brier",
                        "ece_15",
                        "mean_confidence",
                        "mean_margin",
                        "risk_coverage",
                    )
                }
                for condition in conditions
            },
        },
        {
            "condition_id": "robustness_error_transitions",
            "status": "complete",
            "metrics": {
                condition["condition_id"]: {
                    "correct_to_wrong": condition["paired_effect"]["correct_to_wrong"],
                    "wrong_to_correct": condition["paired_effect"]["wrong_to_correct"],
                    "prediction_flips": condition["paired_effect"]["prediction_flips"],
                    "accuracy_delta_pp_ci95": condition["paired_effect"]["accuracy_delta_pp_ci95"],
                    "cross_entropy_delta_ci95": condition["paired_effect"]["cross_entropy_delta_ci95"],
                }
                for condition in conditions
            },
        },
    ]
    family12 = {
        "family_id": "F12_calibration_error_statistics",
        "schema": "lnet.r2k3.study.family.v1",
        "status": "complete",
        "required_condition_ids": [
            "baseline_calibration",
            "robustness_calibration",
            "robustness_error_transitions",
        ],
        "observed_condition_ids": [
            "baseline_calibration",
            "robustness_calibration",
            "robustness_error_transitions",
        ],
        "conditions": family12_conditions,
        "statistics": {
            "bootstrap": "class-stratified paired image bootstrap",
            "bootstrap_repeats": args.bootstrap_repeats,
            "fdr": "Benjamini-Hochberg within F11",
        },
    }
    path12 = write_family(args.root, family12)
    for family_id, path in (
        ("F11_robustness_invariance", path11),
        ("F12_calibration_error_statistics", path12),
    ):
        cast("dict[str, str]", study["families"])[family_id] = str(path.relative_to(args.root))
        _upsert_artifact(
            study,
            artifact_record(args.root, family_id, path, "application/json"),
        )
    atomic_json(args.root / "study.json", study)
    if sha256_file(args.checkpoint) != frozen.checkpoint_sha256:
        raise RuntimeError("robustness evaluation changed checkpoint")
    print(json.dumps({"event": "robustness_complete", "conditions": len(conditions)}))


if __name__ == "__main__":
    main()
