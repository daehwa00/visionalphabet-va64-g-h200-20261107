#!/usr/bin/env python3
"""Fit direct 4-to-1 path collapses to a trained nonlinear teacher."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, cast

import a2d_r2k3_diagnostics as diagnostics
import a2d_r2k3_runtime as runtime
import analyze_a2d_r2k3_checkpoint_capacity as capacity_analysis
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as runner
import run_alphabet2d_imagenet100_nano as harness
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from lnet.pac_grouped_path_cffn import GroupedWidelyLinear
from lnet.pac_path_cffn import D4PathModeCombiner
from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_product_scan_reference import product_scan_full4_reference

ComplexField = tuple[Tensor, Tensor]


@dataclass
class RidgeMoments:
    xtx: Tensor
    xty: Tensor
    yty: Tensor
    ysum: Tensor
    count: int = 0

    @classmethod
    def create(cls, modes: int) -> RidgeMoments:
        return cls(
            xtx=torch.zeros(modes, 9, 9, dtype=torch.float64),
            xty=torch.zeros(modes, 9, 2, dtype=torch.float64),
            yty=torch.zeros(modes, dtype=torch.float64),
            ysum=torch.zeros(modes, 2, dtype=torch.float64),
        )

    def add(
        self,
        source_real: Tensor,
        source_imag: Tensor,
        output_real: Tensor,
        output_imag: Tensor,
        retained_rows: int,
    ) -> None:
        modes = source_real.shape[-1]
        real = source_real.detach().reshape(-1, 4, modes).permute(0, 2, 1)
        imag = source_imag.detach().reshape(-1, 4, modes).permute(0, 2, 1)
        target = torch.stack(
            (
                output_real.detach().reshape(-1, modes),
                output_imag.detach().reshape(-1, modes),
            ),
            dim=-1,
        )
        total = real.shape[0]
        if total > retained_rows:
            indices = torch.linspace(0, total - 1, retained_rows, device=real.device).long()
            real = real.index_select(0, indices)
            imag = imag.index_select(0, indices)
            target = target.index_select(0, indices)
        ones = torch.ones((*real.shape[:-1], 1), device=real.device, dtype=real.dtype)
        design = torch.cat((real, imag, ones), dim=-1).double().cpu()
        target = target.double().cpu()
        self.xtx += torch.einsum("tmi,tmj->mij", design, design)
        self.xty += torch.einsum("tmi,tmo->mio", design, target)
        self.yty += target.square().sum((0, 2))
        self.ysum += target.sum(0)
        self.count += design.shape[0]

    def solve(self, ridge: float) -> tuple[Tensor, Tensor, dict[str, float]]:
        if self.count == 0:
            raise RuntimeError("direct-path fit received no teacher observations")
        diagonal_scale = self.xtx.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0e-12)
        regularizer = torch.eye(9, dtype=torch.float64).expand_as(self.xtx).clone()
        regularizer[:, -1, -1] = 0.01
        system = self.xtx + ridge * diagonal_scale[:, None, None] * regularizer
        solution = torch.linalg.solve(system, self.xty)
        prediction_cross = torch.einsum("mio,mio->m", solution, self.xty)
        prediction_energy = torch.einsum("mio,mij,mjo->m", solution, self.xtx, solution)
        squared_error = (self.yty - 2.0 * prediction_cross + prediction_energy).clamp_min(0.0)
        centered_energy = (
            self.yty - self.ysum.square().sum(-1) / max(1, self.count)
        ).clamp_min(1.0e-12)
        metrics = {
            "samples_per_mode": self.count,
            "relative_rmse": float(
                torch.sqrt(squared_error.sum() / self.yty.sum().clamp_min(1.0e-12))
            ),
            "centered_r2": float(1.0 - squared_error.sum() / centered_energy.sum()),
            "mode_r2_median": float((1.0 - squared_error / centered_energy).median()),
            "mode_r2_p10": float(torch.quantile(1.0 - squared_error / centered_energy, 0.10)),
        }
        weight = solution[:, :8].transpose(1, 2).float()
        bias = solution[:, 8].float()
        return weight, bias, metrics


class DirectPathCollapse(D4PathModeCombiner):
    """General real-linear 4-to-1 collapse, independently for every mode."""

    collapses_product_paths = True

    def __init__(self, modes: int) -> None:
        super().__init__()
        self.modes = modes
        self.path_count = 4
        self.output_paths = 1
        self.input_modes = 4 * modes
        self.projection = GroupedWidelyLinear(modes, 4, 1, bias=True)

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        shape = (*real.shape[:-1], self.path_count, self.modes)
        return self.forward_packed(real.reshape(shape), imag.reshape(shape))

    def forward_packed(self, source_real: Tensor, source_imag: Tensor) -> ComplexField:
        return self.projection(source_real, source_imag)

    @torch.no_grad()
    def load_real_affine(self, weight: Tensor, bias: Tensor) -> None:
        top_left = weight[:, 0, :4]
        top_right = weight[:, 0, 4:]
        bottom_left = weight[:, 1, :4]
        bottom_right = weight[:, 1, 4:]
        self.projection.weight_real[:, 0].copy_(0.5 * (top_left + bottom_right))
        self.projection.conjugate_real[:, 0].copy_(0.5 * (top_left - bottom_right))
        self.projection.weight_imag[:, 0].copy_(0.5 * (bottom_left - top_right))
        self.projection.conjugate_imag[:, 0].copy_(0.5 * (bottom_left + top_right))
        if self.projection.bias_real is None or self.projection.bias_imag is None:
            raise RuntimeError("direct path projection unexpectedly lost its bias")
        self.projection.bias_real[:, 0].copy_(bias[:, 0])
        self.projection.bias_imag[:, 0].copy_(bias[:, 1])


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--calibration-per-class", type=int, default=5)
    parser.add_argument("--validation-per-class", type=int, default=0)
    parser.add_argument("--rows-per-batch", type=int, default=1024)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    return parser.parse_args()


def _balanced_loader(
    data_root: Path,
    split: str,
    per_class: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[DataLoader[Any], int]:
    _, evaluation_transform = harness._transforms()
    dataset = datasets.ImageFolder(data_root / split, evaluation_transform)
    indices = capacity_analysis._balanced_indices(dataset.targets, per_class)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=harness.PREFETCH_FACTOR if workers > 0 else None,
    )
    return loader, len(indices)


def _path_modules(model: nn.Module) -> dict[str, PathOnlyCollapse]:
    active_model = cast("Any", model)
    modules = {
        f"main.{name}": getattr(active_model, name).quadrant_path_mode_combiner
        for name in runner.STAGE_NAMES[:3]
    }
    modules.update(
        {
            f"same_resolution.{resolution}": block.path_collapse
            for resolution, block in active_model.same_resolution_blocks.items()
        }
    )
    if any(type(module) is not PathOnlyCollapse for module in modules.values()):
        raise TypeError("checkpoint model does not expose seven exact PathOnlyCollapse blocks")
    return cast("dict[str, PathOnlyCollapse]", modules)


def _use_observable_same_resolution_paths(model: nn.Module) -> None:
    def legacy_collapsed_memory(block: Any, real: Tensor, imag: Tensor) -> ComplexField:
        source = block.reader(real, imag)
        pole_x, pole_y = block.pole_scan.pole_coefficients(tuple(source[0].shape))
        paths = product_scan_full4_reference(
            pole_x,
            pole_y,
            source,
            gain_normalization=block.pole_scan.product_gain_normalization,
        )
        return block.collapse_paths(*paths)

    active_model = cast("Any", model)
    for block in active_model.same_resolution_blocks.values():
        block.collapsed_memory = MethodType(legacy_collapsed_memory, block)


def _collect_moments(
    model: nn.Module,
    modules: dict[str, PathOnlyCollapse],
    loader: DataLoader[Any],
    device: torch.device,
    rows_per_batch: int,
) -> dict[str, RidgeMoments]:
    moments = {name: RidgeMoments.create(module.modes) for name, module in modules.items()}
    patched_modules: list[PathOnlyCollapse] = []
    for name, module in modules.items():
        original_forward = module.forward_packed

        def observable_forward(
            _module: PathOnlyCollapse,
            source_real: Tensor,
            source_imag: Tensor,
            *,
            block_name: str = name,
            teacher_forward: Any = original_forward,
        ) -> ComplexField:
            output = teacher_forward(source_real, source_imag)
            moments[block_name].add(
                source_real,
                source_imag,
                output[0],
                output[1],
                rows_per_batch,
            )
            return output

        cast("Any", module).forward_packed = MethodType(observable_forward, module)
        patched_modules.append(module)
    try:
        with torch.no_grad():
            for input_images, _targets in loader:
                images = input_images.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    cast("Any", model).raw_descriptor(images)
    finally:
        for module in patched_modules:
            del cast("Any", module).forward_packed
    return moments


def _replace_paths(
    model: nn.Module,
    fitted: dict[str, tuple[Tensor, Tensor]],
    device: torch.device,
) -> None:
    active_model = cast("Any", model)
    for name in runner.STAGE_NAMES[:3]:
        key = f"main.{name}"
        direct = DirectPathCollapse(fitted[key][0].shape[0]).to(device)
        direct.load_real_affine(*fitted[key])
        getattr(active_model, name).quadrant_path_mode_combiner = direct
    for resolution, block in active_model.same_resolution_blocks.items():
        key = f"same_resolution.{resolution}"
        direct = DirectPathCollapse(fitted[key][0].shape[0]).to(device)
        direct.load_real_affine(*fitted[key])
        block.path_collapse = direct


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if min(
        args.batch_size,
        args.rows_per_batch,
        args.calibration_per_class,
    ) <= 0:
        raise ValueError("calibration dimensions must be positive")
    if args.validation_per_class < 0 or args.ridge < 0.0:
        raise ValueError("validation count and ridge must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostic requested without CUDA")
    torch.set_num_threads(max(1, min(24, torch.get_num_threads())))
    torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != runner.VARIANT or checkpoint.get("epoch") != 100:
        raise ValueError("direct-path diagnostic requires the completed CLPost checkpoint")
    model = runner._build(runner.VARIANT, runtime.model_config())
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    cast("Any", model).to(memory_format=torch.channels_last)
    model.eval()
    _use_observable_same_resolution_paths(model)
    modules = _path_modules(model)
    calibration_loader, calibration_samples = _balanced_loader(
        args.data_root,
        "train",
        args.calibration_per_class,
        args.batch_size,
        args.workers,
        device,
    )
    validation_loader, validation_samples = _balanced_loader(
        args.data_root,
        "val",
        args.validation_per_class,
        args.batch_size,
        args.workers,
        device,
    )

    with torch.inference_mode():
        original = diagnostics.evaluate(model, validation_loader, device)
    moments = _collect_moments(
        model,
        modules,
        calibration_loader,
        device,
        args.rows_per_batch,
    )
    fitted: dict[str, tuple[Tensor, Tensor]] = {}
    fit_metrics = {}
    for name, block_moments in moments.items():
        weight, bias, metrics = block_moments.solve(args.ridge)
        fitted[name] = weight.to(device), bias.to(device)
        fit_metrics[name] = metrics
    _replace_paths(model, fitted, device)
    with torch.inference_mode():
        direct = diagnostics.evaluate(model, validation_loader, device)

    payload = {
        "schema": "lnet.a2d.direct_path_collapse_distillation.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": diagnostics.file_sha256(args.checkpoint),
        "device": str(device),
        "calibration": {
            "samples": calibration_samples,
            "per_class": args.calibration_per_class,
            "rows_per_batch": args.rows_per_batch,
            "ridge": args.ridge,
        },
        "validation": {
            "samples": validation_samples,
            "per_class": args.validation_per_class,
        },
        "fit": fit_metrics,
        "original": original,
        "direct": direct,
        "accuracy_delta": direct["accuracy"] - original["accuracy"],
        "cross_entropy_delta": direct["cross_entropy"] - original["cross_entropy"],
        "status": "passed",
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
