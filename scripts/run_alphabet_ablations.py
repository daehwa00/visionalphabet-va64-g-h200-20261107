"""Run ImageNet-100 terminal readout ablations."""

from __future__ import annotations

import hashlib
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import alphabet_model_builder as builder
import torch

from alphabet.capture_safe_orthogonal import prepare_capture_safe_orthogonal_

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from alphabet.complex_scan import ComplexScanConfig

SEEDS = (501,)
BASE: dict[str, Any] = {
    "channels": builder.DEFAULT_CHANNELS,
    "poles": builder.DEFAULT_POLES,
    "depth": builder.DEFAULT_DEPTH,
    "forward_vertical_only": False,
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
SPECS["17-a2-clean-512d-live-rms-v3"] = {
    **SPECS["17-live-complex-rms-s03"],
    "forward_vertical_only": True,
    "parameters": 2_937_060,
}
SPECS["17-p1-pole-unclamp-a2clean-v3"] = {
    **SPECS["17-a2-clean-512d-live-rms-v3"],
    "pole_unclamp": True,
}
SPECS["17-p1-a2-1024d-v3"] = {
    **SPECS["17-live-complex-rms-s03"],
    "pole_unclamp": True,
}
SPECS["17-p1-nr4-1024d-v3"] = {
    **SPECS["17-p1-a2-1024d-v3"],
    "non_redundant_four_directions": True,
}
KEY = os.environ.get("ALPHABET_ABLATION", "10-dw-terminal-rmsnorm")
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
    return build_variant(KEY, config)


def build_variant(key: str, config: ComplexScanConfig | None = None) -> builder.VisionAlphabetIN100:
    spec = {**BASE, **SPECS[key]}
    model = builder.build_dw_terminal_rmsnorm(
        config,
        channels=spec["channels"],
        poles=spec["poles"],
        depth=spec["depth"],
        normalize=spec["normalize"],
        factorized_reader=spec["factorized_reader"],
        forward_vertical_only=spec["forward_vertical_only"],
        non_redundant_four_directions=spec.get("non_redundant_four_directions", False),
    )
    if spec["multiscale"]:
        model = builder.add_multiscale_readout(model)
        if model.classifier.affine.linear.out_features == 100:
            measured = sum(parameter.numel() for parameter in model.parameters())
            if measured != spec["parameters"]:
                raise RuntimeError(f"parameter count mismatch: expected {spec['parameters']}, got {measured}")
    if spec.get("pole_unclamp"):
        for stage in model.modules():
            if isinstance(stage, builder.ComplexScanStage):
                stage.enable_pole_unclamp_()
    prepare_capture_safe_orthogonal_(model, auto=False)
    return model


def _contract(args: Namespace) -> dict[str, Any]:
    import alphabet_imagenet100_runtime as runtime

    payload = runtime.base_contract(args, variants=VARIANTS, seeds=SEEDS)
    repo = Path(__file__).resolve().parents[1]
    source_paths = sorted(
        [
            *(repo / "src/alphabet").glob("*.py"),
            *(repo / "scripts").glob("*.py"),
            repo / "pyproject.toml",
        ]
    )
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(path.read_bytes())

    clean = SPEC["forward_vertical_only"]
    descriptor_dim = 1536 if SPEC["multiscale"] else (512 if clean else 1024)
    directions = 2 if clean else 4
    architecture = AblationModel.architecture
    config = {
        "architecture": architecture,
        "descriptor_dim": descriptor_dim,
        "forward_vertical_only": clean,
        "pole_unclamp": SPEC.get("pole_unclamp", False),
        "non_redundant_four_directions": SPEC.get("non_redundant_four_directions", False),
        "head": (
            "LivePolewiseComplexRMS(momentum=0.05,eps=1e-12,output_scale=0.3) "
            f"Linear{descriptor_dim}-to-100"
        ),
        "readout": (
            f"[mean Re(2s*conj(u)-abs(u)^2){directions}P; "
            f"mean Im(2s*conj(u)-abs(u)^2){directions}P], "
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
    if clean:
        config["correlation_axes"] = (
            "vertical conditional on horizontal scan; forward vertical direction for each of two horizontal directions"
        )
    if SPEC.get("non_redundant_four_directions"):
        config["correlation_axes"] = ["→↑", "←↑", "↑→", "↓→"]
        config["readout"] = (
            "[mean Re(2s*conj(u)-abs(u)^2)4P; mean Im(2s*conj(u)-abs(u)^2)4P], "
            "u=second-axis gain times first-axis state"
        )
    if SPEC.get("pole_unclamp"):
        config["drop_path"] = 0.0
        config["pole_parameterization"] = {
            "damping_min": 0.01,
            "damping_max": 2.0,
            "phase": "pi * tanh(raw_phase / pi), strictly inside Nyquist",
            "initialization": f"preserve {'A2-clean' if clean else 'A2'} effective poles after model build",
        }
    if SPEC["multiscale"]:
        config["readout"] = "[Re(S3), Re(S4), Im(S3), Im(S4)] from mean(2s*conj(u)-abs(u)^2)"
        config["correlation_axes"] = "S3: V_fwd|H_fwd and V_fwd|H_bwd; S4: four A2 directions"
    payload.update(
        schema="alphabet.dw_terminal_rmsnorm.in100.v1",
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
    import alphabet_imagenet100_runtime as runtime

    return builder.build_optimizer(
        model,
        recipe,
        variant=VARIANT,
        seed=SEEDS[0],
        atomic_json=runtime.atomic_json,
    )


def main() -> None:
    import alphabet_augmentation as augment
    import alphabet_imagenet100_runtime as runtime

    from alphabet.complex_scan_config import ComplexScanConfig

    if os.environ.get("ALPHABET_AUTO_GPU_POLICY") == "1":
        from alphabet.gpu_policy import apply_gpu_policy

        apply_gpu_policy()
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = os.environ.get("ALPHABET_CUDNN_BENCHMARK", "1") != "0"
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
