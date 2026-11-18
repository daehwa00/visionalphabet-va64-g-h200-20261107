#!/usr/bin/env python3
"""Run provenance, pole-atlas, synthetic-transfer, and weight-rank studies."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportMissingImports=false, reportPrivateUsage=false
import argparse
import datetime as dt
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, cast

import a2d_r2k3_runtime as runtime
import matplotlib as mpl
import run_a2d_r2k3_k128_depth_p_overnight_imagenet100 as runner
import torch
from analyze_a2d_r2k3_capacity_diagnostics import _wl_matrix
from analyze_a2d_r2k3_checkpoint_capacity import _rank2_weight_metrics
from matplotlib import pyplot as plt
from r2k3_frozen_study_common import (
    EXPECTED_CHECKPOINT_SHA256,
    VARIANT,
    artifact_record,
    atomic_json,
    dataset_manifest,
    imagefolder,
    load_frozen_model,
    model_state_sha256,
    postfusion_modules,
    reader_modules,
    scan_modules,
    sha256_file,
    write_family,
)
from scipy import stats
from torch import Tensor, nn

mpl.use("Agg")

SCHEMA = "lnet.a2d.r2k3.k128_d2262_no_training_study.v1"
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


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _effective_rank(matrix: Tensor) -> dict[str, float | int]:
    active = matrix.double()
    if active.ndim == 1:
        active = active[:, None]
    active = active - active.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(active).square().clamp_min(0)
    probability = singular / singular.sum().clamp_min(1.0e-30)
    participation = probability.sum().square() / probability.square().sum().clamp_min(1.0e-30)
    entropy = torch.exp(-(probability * probability.clamp_min(1.0e-30).log()).sum())
    return {
        "dimensions": int(active.shape[-1]),
        "observations": int(active.shape[0]),
        "participation_ratio": float(participation),
        "entropy_effective_rank": float(entropy),
    }


def _source_records(repo: Path) -> list[dict[str, Any]]:
    paths = runtime.source_dependency_paths(
        repo,
        (
            Path(__file__).stem,
            "r2k3_frozen_study_common",
            runner.__name__,
        ),
    )
    return [
        {
            "path": str(path.relative_to(repo)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _directory_inventory(path: Path) -> str:
    records = [
        {
            "name": item.name,
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in sorted(path.iterdir())
        if item.is_file() and not item.is_symlink()
    ]
    return __import__("hashlib").sha256(
        json.dumps(records, sort_keys=True).encode()
    ).hexdigest()


def _provenance(
    args: argparse.Namespace,
    frozen: Any,
    checkpoint_after: str,
    inventory_before: str,
    inventory_after: str,
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    dataset = imagefolder(args.data_root, "val")
    source_records = _source_records(repo)
    dependency_digest = __import__("hashlib").sha256(
        json.dumps(source_records, sort_keys=True).encode()
    ).hexdigest()
    return {
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": frozen.checkpoint_sha256,
            "sha256_after": checkpoint_after,
            "bytes": args.checkpoint.stat().st_size,
            "payload": {
                key: frozen.payload.get(key)
                for key in ("variant", "seed", "epoch", "global_step")
            },
            "state_dict_sha256": frozen.initial_state_sha256,
            "parent_inventory_sha256_before": inventory_before,
            "parent_inventory_sha256_after": inventory_after,
        },
        "training_record": {
            "best_accuracy": 0.883400022983551,
            "best_epoch": 99,
            "final_accuracy": 0.8830000162124634,
            "final_epoch": 100,
        },
        "source": {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "dependency_tree_sha256": dependency_digest,
            "files": source_records,
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "precision": "bfloat16",
        },
        "data": {
            "split": "val",
            "samples": len(dataset),
            "classes": len(dataset.classes),
            "manifest_sha256": dataset_manifest(dataset, args.data_root / "val"),
            "class_mapping_sha256": __import__("hashlib").sha256(
                json.dumps(dataset.class_to_idx, sort_keys=True).encode()
            ).hexdigest(),
        },
        "no_training_guard": {
            "optimizer_step_calls": 0,
            "parameter_update_calls": 0,
            "linear_probe_runs": 0,
            "checkpoint_sha256_before": frozen.checkpoint_sha256,
            "checkpoint_sha256_after": checkpoint_after,
            "model_state_sha256_before": frozen.initial_state_sha256,
            "model_state_sha256_after": model_state_sha256(frozen.model),
        },
    }


def _pole_atlas(model: nn.Module, initial: nn.Module) -> dict[str, Any]:
    learned_modules = scan_modules(model)
    initial_modules = scan_modules(initial)
    rows: list[dict[str, Any]] = []
    modules: dict[str, Any] = {}
    for label, module in learned_modules.items():
        initial_module = initial_modules[label]
        damping_x, damping_y = (value.reshape(-1).detach().cpu() for value in module._damping_fields())
        initial_damping_x, initial_damping_y = (
            value.reshape(-1).detach().cpu() for value in initial_module._damping_fields()
        )
        phase_x = module.phase_x.detach().cpu()
        phase_y = module.phase_y.detach().cpu()
        initial_phase_x = initial_module.phase_x.detach().cpu()
        initial_phase_y = initial_module.phase_y.detach().cpu()
        features = torch.stack((damping_x, damping_y, phase_x, phase_y), dim=1)
        for index in range(module.modes):
            alpha_x = float(damping_x[index])
            alpha_y = float(damping_y[index])
            omega_x = float(phase_x[index])
            omega_y = float(phase_y[index])
            radial_alpha = math.sqrt(alpha_x * alpha_y)
            radial_omega = math.hypot(omega_x, omega_y)
            rows.append(
                {
                    "module": label,
                    "resolution": MODULE_RESOLUTION[label],
                    "pole": index,
                    "alpha_x": alpha_x,
                    "alpha_y": alpha_y,
                    "omega_x": omega_x,
                    "omega_y": omega_y,
                    "radial_alpha": radial_alpha,
                    "radial_omega": radial_omega,
                    "half_life_x_steps": math.log(2.0) / max(alpha_x, 1.0e-12),
                    "half_life_y_steps": math.log(2.0) / max(alpha_y, 1.0e-12),
                    "wavelength_x_steps": 2.0 * math.pi / max(abs(omega_x), 1.0e-12),
                    "wavelength_y_steps": 2.0 * math.pi / max(abs(omega_y), 1.0e-12),
                    "alpha_anisotropy_log_ratio": math.log(
                        max(alpha_x, 1.0e-12) / max(alpha_y, 1.0e-12)
                    ),
                    "omega_orientation_radians": math.atan2(omega_y, omega_x),
                    "damping_delta_l2": math.hypot(
                        alpha_x - float(initial_damping_x[index]),
                        alpha_y - float(initial_damping_y[index]),
                    ),
                    "phase_delta_l2": math.hypot(
                        omega_x - float(initial_phase_x[index]),
                        omega_y - float(initial_phase_y[index]),
                    ),
                }
            )
        modules[label] = {
            "modes": module.modes,
            "resolution": MODULE_RESOLUTION[label],
            "alpha_x": {
                "minimum": float(damping_x.min()),
                "median": float(damping_x.median()),
                "maximum": float(damping_x.max()),
            },
            "alpha_y": {
                "minimum": float(damping_y.min()),
                "median": float(damping_y.median()),
                "maximum": float(damping_y.max()),
            },
            "omega_abs": {
                "median": float(torch.hypot(phase_x, phase_y).median()),
                "maximum": float(torch.hypot(phase_x, phase_y).max()),
            },
            "feature_effective_rank": _effective_rank(features),
        }
    return {"rows": rows, "modules": modules}


def _complex(values: tuple[Tensor, Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    return torch.complex(values[0].reshape(-1), values[1].reshape(-1)), torch.complex(
        values[2].reshape(-1), values[3].reshape(-1)
    )


def _synthetic_identification(model: nn.Module) -> dict[str, Any]:
    frequencies = torch.linspace(0.0, 0.5, 65)
    orientations = torch.linspace(0.0, math.pi, 13)[:-1]
    modules: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    path_orientation = torch.zeros(4, len(orientations))
    for label, module in scan_modules(model).items():
        resolution = MODULE_RESOLUTION[label]
        pole_x, pole_y = module.pole_coefficients((1, resolution, resolution, module.modes))
        decay_x, gamma_x = (value.detach().cpu().to(torch.complex64) for value in _complex(pole_x))
        decay_y, gamma_y = (value.detach().cpu().to(torch.complex64) for value in _complex(pole_y))
        impulse_x = torch.stack(
            [gamma_x.abs() * decay_x.abs().pow(lag) for lag in range(128)], dim=0
        )
        impulse_y = torch.stack(
            [gamma_y.abs() * decay_y.abs().pow(lag) for lag in range(128)], dim=0
        )
        empirical_x = (impulse_x <= impulse_x[0:1] * 0.5).float().argmax(0)
        empirical_y = (impulse_y <= impulse_y[0:1] * 0.5).float().argmax(0)
        empirical_x = torch.where(empirical_x == 0, torch.full_like(empirical_x, 128), empirical_x)
        empirical_y = torch.where(empirical_y == 0, torch.full_like(empirical_y, 128), empirical_y)
        response = torch.zeros(module.modes, len(frequencies), len(orientations), 4)
        signs = ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))
        for frequency_index, frequency in enumerate(frequencies):
            for orientation_index, orientation in enumerate(orientations):
                omega_x = 2.0 * math.pi * frequency * torch.cos(orientation)
                omega_y = 2.0 * math.pi * frequency * torch.sin(orientation)
                for path_index, (sign_x, sign_y) in enumerate(signs):
                    hx = gamma_x.abs() / (
                        1.0 - decay_x * torch.exp(-1j * sign_x * omega_x)
                    ).abs().clamp_min(1.0e-12)
                    hy = gamma_y.abs() / (
                        1.0 - decay_y * torch.exp(-1j * sign_y * omega_y)
                    ).abs().clamp_min(1.0e-12)
                    response[:, frequency_index, orientation_index, path_index] = hx * hy
        reduced = response.amax(dim=(2, 3))
        preferred_index = reduced.argmax(1)
        preferred_frequency = frequencies[preferred_index]
        radial_omega = torch.hypot(module.phase_x.detach().cpu(), module.phase_y.detach().cpu())
        spearman = stats.spearmanr(
            radial_omega.numpy(), preferred_frequency.numpy(), nan_policy="omit"
        )
        theoretical_x = math.log(2.0) / module._damping_fields()[0].reshape(-1).detach().cpu()
        theoretical_y = math.log(2.0) / module._damping_fields()[1].reshape(-1).detach().cpu()
        half_life_x = stats.spearmanr(theoretical_x.numpy(), empirical_x.numpy())
        half_life_y = stats.spearmanr(theoretical_y.numpy(), empirical_y.numpy())
        modules[label] = {
            "resolution": resolution,
            "omega_preferred_frequency_spearman": {
                "rho": float(spearman.statistic),
                "p_value": float(spearman.pvalue),
            },
            "theoretical_empirical_half_life_spearman_x": {
                "rho": float(half_life_x.statistic),
                "p_value": float(half_life_x.pvalue),
            },
            "theoretical_empirical_half_life_spearman_y": {
                "rho": float(half_life_y.statistic),
                "p_value": float(half_life_y.pvalue),
            },
        }
        for pole in range(module.modes):
            rows.append(
                {
                    "module": label,
                    "pole": pole,
                    "empirical_half_life_x": int(empirical_x[pole]),
                    "empirical_half_life_y": int(empirical_y[pole]),
                    "preferred_frequency_cycles_per_pixel": float(preferred_frequency[pole]),
                    "radial_omega": float(radial_omega[pole]),
                }
            )
        path_orientation += response.mean((0, 1)).T
    path_orientation /= len(scan_modules(model))
    return {
        "protocol": {
            "impulse_lags": 128,
            "frequency_cycles_per_pixel": frequencies.tolist(),
            "orientations_radians": orientations.tolist(),
            "response": "intrinsic separable D4 pole transfer magnitude; reader excluded",
        },
        "modules": modules,
        "rows": rows,
        "path_orientation_mean_response": path_orientation.tolist(),
    }


def _rank_capacity(model: nn.Module) -> dict[str, Any]:
    readers = {
        name: _rank2_weight_metrics(reader) for name, reader in reader_modules(model).items()
    }
    postfusion: dict[str, Any] = {}
    for name, module in postfusion_modules(model).items():
        layers: dict[str, Any] = {}
        for layer_name, layer in (("value", module.value), ("gate", module.gate), ("out", module.out)):
            matrix = (
                _wl_matrix(cast("Any", layer))
                if not isinstance(layer, nn.Linear)
                else layer.weight.detach().float()
            )
            singular = torch.linalg.svdvals(matrix.detach().cpu().double())
            energy = singular.square()
            probability = energy / energy.sum().clamp_min(1.0e-30)
            layers[layer_name] = {
                "shape": list(matrix.shape),
                "spectral_norm": float(singular.max()),
                "frobenius_norm": float(matrix.norm()),
                "participation_ratio": float(
                    probability.sum().square() / probability.square().sum().clamp_min(1.0e-30)
                ),
                "rank_90": int(
                    torch.searchsorted(probability.cumsum(0), torch.tensor(0.9)).item() + 1
                ),
                "singular_values": singular[: min(32, len(singular))].tolist(),
            }
        postfusion[name] = layers
    return {"reader_rank2_weight_space": readers, "postfusion_weight_spectra": postfusion}


def _figure_alpha(root: Path, atlas: dict[str, Any]) -> tuple[Path, Path]:
    rows = atlas["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label in sorted({row["module"] for row in rows}):
        selected = [row for row in rows if row["module"] == label]
        axes[0].scatter(
            [row["radial_alpha"] for row in selected],
            [0.5 * (row["half_life_x_steps"] + row["half_life_y_steps"]) for row in selected],
            s=6,
            alpha=0.45,
            label=label,
        )
        axes[1].scatter(
            [row["omega_x"] for row in selected],
            [row["omega_y"] for row in selected],
            s=6,
            alpha=0.45,
            label=label,
        )
    axes[0].set(xlabel="radial damping alpha", ylabel="mean theoretical half-life (steps)")
    axes[0].set_yscale("log")
    axes[1].set(xlabel="ωx (rad/step)", ylabel="ωy (rad/step)")
    axes[0].legend(fontsize=5, ncol=2)
    figure.tight_layout()
    path = root / "figures" / "FIG01_alpha_half_life.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    sidecar = path.with_suffix(".json")
    atomic_json(
        sidecar,
        {
            "figure_id": "FIG01_alpha_half_life",
            "panels": [
                {"panel_id": "alpha_half_life", "source": "families/F01_pole_atlas.json#/rows"},
                {"panel_id": "omega_atlas", "source": "families/F01_pole_atlas.json#/rows"},
            ],
            "caption": "Learned damping-derived half-life and two-axis phase atlas.",
        },
    )
    return path, sidecar


def _figure_omega(root: Path, synthetic: dict[str, Any]) -> tuple[Path, Path]:
    rows = synthetic["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(
        [row["radial_omega"] for row in rows],
        [row["preferred_frequency_cycles_per_pixel"] for row in rows],
        s=7,
        alpha=0.4,
    )
    axes[0].set(xlabel="|ω| (rad/step)", ylabel="preferred frequency (cycles/pixel)")
    heatmap = torch.tensor(synthetic["path_orientation_mean_response"])
    axes[1].imshow(heatmap, aspect="auto", origin="lower")
    axes[1].set(
        xlabel="grating orientation index",
        ylabel="D4 path",
        title="Mean intrinsic response",
    )
    figure.tight_layout()
    path = root / "figures" / "FIG02_omega_preferred_frequency.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    sidecar = path.with_suffix(".json")
    atomic_json(
        sidecar,
        {
            "figure_id": "FIG02_omega_preferred_frequency",
            "panels": [
                {"panel_id": "omega_frequency", "source": "families/F02_synthetic_identification.json#/rows"},
                {"panel_id": "path_orientation", "source": "families/F02_synthetic_identification.json#/path_orientation_mean_response"},
            ],
            "caption": "Intrinsic pole transfer preferred frequency and D4 orientation response.",
        },
    )
    return path, sidecar


def main() -> None:
    args = _args()
    args.root.mkdir(parents=True, exist_ok=True)
    inventory_before = _directory_inventory(args.checkpoint.parent)
    device = torch.device(args.device)
    frozen = load_frozen_model(args.checkpoint, device)
    torch.manual_seed(501)
    initial = runner._build(VARIANT, runtime.model_config()).to(device).eval()
    atlas = _pole_atlas(frozen.model, initial)
    synthetic = _synthetic_identification(frozen.model)
    ranks = _rank_capacity(frozen.model)

    family1 = {
        "family_id": "F01_pole_atlas",
        "schema": "lnet.r2k3.study.family.v1",
        "status": "complete",
        "required_condition_ids": sorted(atlas["modules"]),
        "observed_condition_ids": sorted(atlas["modules"]),
        "conditions": [
            {"condition_id": name, "status": "complete", "metrics": metrics}
            for name, metrics in atlas["modules"].items()
        ],
        **atlas,
    }
    family2 = {
        "family_id": "F02_synthetic_identification",
        "schema": "lnet.r2k3.study.family.v1",
        "status": "complete",
        "required_condition_ids": sorted(synthetic["modules"]),
        "observed_condition_ids": sorted(synthetic["modules"]),
        "conditions": [
            {"condition_id": name, "status": "complete", "metrics": metrics}
            for name, metrics in synthetic["modules"].items()
        ],
        **synthetic,
    }
    family7 = {
        "family_id": "F07_rank_capacity",
        "schema": "lnet.r2k3.study.family.v1",
        "status": "partial",
        "required_condition_ids": ["weight_space", "component2_intervention", "postfusion_lowrank_intervention"],
        "observed_condition_ids": ["weight_space"],
        "conditions": [{"condition_id": "weight_space", "status": "complete", "metrics": ranks}],
        **ranks,
    }
    paths = [
        write_family(args.root, family1),
        write_family(args.root, family2),
        write_family(args.root, family7),
    ]
    figure1, sidecar1 = _figure_alpha(args.root, atlas)
    figure2, sidecar2 = _figure_omega(args.root, synthetic)
    checkpoint_after = sha256_file(args.checkpoint)
    inventory_after = _directory_inventory(args.checkpoint.parent)
    provenance = _provenance(
        args,
        frozen,
        checkpoint_after,
        inventory_before,
        inventory_after,
    )
    artifacts = [
        artifact_record(args.root, f"A{index:03d}", path, "application/json")
        for index, path in enumerate(paths, 1)
    ]
    figures = []
    for figure_id, path, sidecar in (
        ("FIG01_alpha_half_life", figure1, sidecar1),
        ("FIG02_omega_preferred_frequency", figure2, sidecar2),
    ):
        record = artifact_record(args.root, figure_id, path, "image/png")
        sidecar_record = artifact_record(args.root, figure_id + "_sidecar", sidecar, "application/json")
        figures.append(
            {
                "figure_id": figure_id,
                **{key: record[key] for key in ("path", "sha256", "bytes")},
                "sidecar_path": sidecar_record["path"],
                "sidecar_sha256": sidecar_record["sha256"],
                "sidecar_bytes": sidecar_record["bytes"],
            }
        )
    study = {
        "schema": SCHEMA,
        "study": {
            "study_id": "r2k3-k128-d2262-checkpoint-study-20260901",
            "variant": VARIANT,
            "status": "running",
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "completed_utc": None,
        },
        "provenance": provenance,
        "baseline": {},
        "families": {
            "F01_pole_atlas": str(paths[0].relative_to(args.root)),
            "F02_synthetic_identification": str(paths[1].relative_to(args.root)),
            "F07_rank_capacity": str(paths[2].relative_to(args.root)),
        },
        "artifacts": artifacts,
        "figures": figures,
        "report": {},
        "validation": {},
    }
    atomic_json(args.root / "study.json", study)
    if checkpoint_after != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint changed during observational study")
    if model_state_sha256(frozen.model) != frozen.initial_state_sha256:
        raise RuntimeError("model state changed during observational study")
    print(
        json.dumps(
            {
                "event": "observational_complete",
                "root": str(args.root),
                "families": sorted(cast("dict[str, str]", study["families"])),
            }
        )
    )


if __name__ == "__main__":
    main()
