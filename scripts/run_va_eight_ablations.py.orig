"""Run ImageNet-100 terminal readout ablations."""

from __future__ import annotations

import hashlib
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import va_model_builder as builder

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanConfig

SEEDS = (501,)
BASE: dict[str, Any] = {
    "channels": builder.DEFAULT_CHANNELS,
    "poles": builder.DEFAULT_POLES,
    "depth": builder.DEFAULT_DEPTH,
    "normalize": False,
    "cutmix": False,
    "timm_ra": False,
}
SPECS: dict[str, dict[str, Any]] = {
    "10-dw-terminal-rmsnorm": {
        "reader": "DW3x3 -> pointwise C-to-2P",
        "terminal_input_norm": "same as stage4 same-resolution Reader",
        "terminal_rms_matching": False,
        "factorized_reader": False,
        "multiscale": False,
        "parameters": builder.PARAMETERS,
    },
    "17-live-complex-rms-s03": {
        "reader": "factorized pointwise then spatial 3x3",
        "terminal_input_norm": "same as stage4 same-resolution Reader",
        "terminal_rms_matching": False,
        "factorized_reader": True,
        "multiscale": False,
        "parameters": 2_988_260,
    },
    "17-e2-multiscale-s3s4-live-rms-v2": {
        "reader": "factorized pointwise then spatial 3x3",
        "terminal_input_norm": "same as stage4 same-resolution Reader",
        "terminal_rms_matching": False,
        "factorized_reader": True,
        "multiscale": True,
        "parameters": 3_114_852,
    },
}
KEY = os.environ.get("VA_ABLATION", "10-dw-terminal-rmsnorm")
SPEC: dict[str, Any] = {**BASE, **SPECS[KEY]}
VARIANT = f"VA-Ablation-{KEY}"
VARIANTS = (VARIANT,)
PARAMETERS = SPEC["parameters"]
AblationModel = (
    builder.MultiScaleVisionAlphabetIN100 if SPEC["multiscale"] else builder.VisionAlphabetIN100
)


def _build(variant: str, config: ComplexScanConfig | None = None) -> builder.VisionAlphabetIN100:
    if variant != VARIANT:
        raise ValueError(f"unsupported variant: {variant}")
    model = builder.build_dw_terminal_rmsnorm(
        config,
        channels=SPEC["channels"],
        poles=SPEC["poles"],
        depth=SPEC["depth"],
        normalize=SPEC["normalize"],
        factorized_reader=SPEC["factorized_reader"],
    )
    if SPEC["multiscale"]:
        model = builder.add_multiscale_readout(model)
        if model.classifier.affine.linear.out_features == 100:
            measured = sum(parameter.numel() for parameter in model.parameters())
            if measured != PARAMETERS:
                raise RuntimeError(f"parameter count mismatch: expected {PARAMETERS}, got {measured}")
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    import va_imagenet100_runtime as runtime

    payload = runtime.base_contract(args, variants=VARIANTS, seeds=SEEDS)
    repo = Path(__file__).resolve().parents[1]
    source_paths = sorted(
        [
            *(repo / "src/lnet").glob("*.py"),
            *(repo / "scripts").glob("*.py"),
            repo / "pyproject.toml",
        ]
    )
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(path.read_bytes())

    descriptor_dim = 1536 if SPEC["multiscale"] else 1024
    architecture = AblationModel.architecture
    config = {
        "architecture": architecture,
        "head": (
            "LivePolewiseComplexRMS(momentum=0.05,eps=1e-12,output_scale=0.3) "
            f"Linear{descriptor_dim}-to-100"
        ),
        "readout": (
            "[mean Re(2s*conj(u)-abs(u)^2)4P; mean Im(2s*conj(u)-abs(u)^2)4P], "
            "u=gamma_y*horizontal_state"
        ),
        "correlation_axes": (
            "vertical conditional on horizontal scan; four existing direction orders"
        ),
        "correlation_normalization": "none before classifier RMS; no log on signed channels",
        "real_feature_channels": list(SPEC["channels"]),
        "pole_modes": list(SPEC["poles"]),
        "depth": list(SPEC["depth"]),
        "reader": SPEC.get("reader", "DW3x3 -> pointwise C-to-2P"),
        "terminal_input_norm": SPEC.get("terminal_input_norm", "none"),
        "terminal_rms_matching": SPEC.get("terminal_rms_matching", True),
    }
    if SPEC["multiscale"]:
        config["readout"] = "[Re(S3), Re(S4), Im(S3), Im(S4)] from mean(2s*conj(u)-abs(u)^2)"
        config["correlation_axes"] = "S3: V_fwd|H_fwd and V_fwd|H_bwd; S4: four A2 directions"
    payload.update(
        schema="lnet.dw_terminal_rmsnorm.in100.v1",
        evidence_status="scratch seed501; 100-epoch architecture comparison",
        initialization=(
            "scratch seed501; no trained weights; same-shaped reference tensors paired; "
            "imaginary head weights zero"
        ),
        variant_configs={VARIANT: config},
        parameter_counts={VARIANT: PARAMETERS},
        architecture={VARIANT: architecture},
        references={
            "baseline_variant": "VA-RealMemory-A-TerminalRawGOpt-IN100",
            "baseline_parameters": 2_988_004,
            "baseline_accuracy": 0.8666,
            "parameter_delta": PARAMETERS - 2_988_004,
        },
        ablation_recipe={
            "cutmix_alpha": 1.0 if SPEC["cutmix"] else None,
            "cutmix_switch_probability": 0.5 if SPEC["cutmix"] else 0,
            "timm_randaugment": "rand-m9-mstd0.5-inc1" if SPEC["timm_ra"] else None,
            "terminal_denominator": (
                "mean(|u|^2) across H,W; gradient detached" if SPEC["normalize"] else None
            ),
        },
        source_closure=[path.relative_to(repo).as_posix() for path in source_paths],
    )
    payload["recipe"]["head_objective"] = (
        "one MixUp cross-entropy on terminal-Q4 affine logits; no auxiliary term"
    )
    payload["source_sha256"].update(
        ablation_runner=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_closure=digest.hexdigest(),
    )
    return payload


def _build_optimizer(model: nn.Module, recipe: dict[str, Any]) -> torch.optim.Optimizer:
    import va_imagenet100_runtime as runtime

    return builder.build_optimizer(
        model,
        recipe,
        variant=VARIANT,
        seed=SEEDS[0],
        atomic_json=runtime.atomic_json,
    )


def main() -> None:
    import va_ablation_augment as augment
    import va_imagenet100_runtime as runtime

    from lnet.complex_scan_config import ComplexScanConfig

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    runtime.main(
        runtime.RunnerBindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=_build_optimizer,
            prepare_model=runtime.prepare_model,
            train_epoch=partial(augment.train_epoch, cutmix=bool(SPEC["cutmix"])),
            evaluate=runtime.evaluate,
            wandb_model_metrics=runtime.wandb_model_metrics,
            summarize=runtime.summarize,
        )
    )


if __name__ == "__main__":
    main()
