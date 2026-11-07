#!/usr/bin/env python3
"""Compare two K schedules with projected identity shortcuts."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false, reportUnusedFunction=false
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_front_capacity_reallocation_d2262_imagenet100 as capacity
import run_a2d_r2k3_k32_64_96_128_p128_d2262_imagenet100 as widened
import run_a2d_r2k3_k32_p128_front_d2262_imagenet100 as compact
import run_a2d_r2k3_progressive_k_d2262_imagenet100 as progressive
import torch

from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    ResNetStyleGatedTransition,
    resnet_style_transition,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


WIDENED = "K32-64-96-128-P128x4-D2262-ResNetStyle-Gain1-WLPost15K-PathH4"
COMPACT = "K32-64-64-128-P128x4-D2262-ResNetStyle-Gain1-WLPost15K-PathH4"
VARIANTS = (WIDENED, COMPACT)
VARIANT = WIDENED
JOBS_BY_GPU = {0: (WIDENED,), 1: (COMPACT,)}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = capacity.RESOLUTIONS
STAGE_NAMES = capacity.STAGE_NAMES
SameResolutionFactorialBackbone = capacity.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = capacity.POST_HIDDEN_RATIO
SPECS = {
    WIDENED: widened.SPEC,
    COMPACT: compact.SPEC,
}
CONTROLS = {
    WIDENED: (widened, widened.VARIANT),
    COMPACT: (compact, compact.VARIANT),
}


def _install_resnet_style_transitions(
    model: ComplexScanBackbone,
) -> None:
    for name in STAGE_NAMES[:3]:
        stage = getattr(model, name)
        source = cast("GatedPoleExcitationS2DTransition", stage.augmented)
        stage.augmented = resnet_style_transition(source)


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported ResNet-style variant: {variant}") from error
    progressive._assert_model_for_spec(model, spec, variant)
    for name, input_modes, output_modes in zip(
        STAGE_NAMES[:3],
        spec.excitation_modes[:-1],
        spec.excitation_modes[1:],
        strict=True,
    ):
        transition = getattr(model, name).augmented
        if type(transition) is not ResNetStyleGatedTransition:
            raise RuntimeError(f"{variant}/{name} lost its ResNet-style transition")
        carry_projection = transition.carry_projection
        if (carry_projection is None) != (input_modes == output_modes):
            raise RuntimeError(f"{variant}/{name} changed its carry projection contract")
        if (transition.carry_gain is None) != (input_modes == output_modes):
            raise RuntimeError(f"{variant}/{name} changed its shortcut-gain contract")


def _build_candidate(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        control, control_variant = CONTROLS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported ResNet-style variant: {variant}") from error
    model = control._build(control_variant, config)
    _install_resnet_style_transitions(model)
    return model


@torch.no_grad()
def _pair_common_initialization_(
    target: ComplexScanBackbone,
    reference: ComplexScanBackbone,
) -> int:
    """Copy every semantically matching tensor that width did not reshape."""
    source_state = reference.state_dict()
    copied = 0
    for name, target_value in target.state_dict().items():
        source_value = source_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        target_value.copy_(source_value)
        copied += 1
    if copied == 0:
        raise RuntimeError("paired initialization found no common model state")
    return copied


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    initial_seed = torch.initial_seed()
    model = _build_candidate(variant, config)
    if variant == WIDENED:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initial_seed)
            reference = _build_candidate(COMPACT, config)
        _pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = capacity._variant_config_for_spec(variant, spec)
    payload["backbone"]["transition"]["merge"] = (
        "raw identity carry or unit-RMS projected carry plus unit-RMS memory"
    )
    payload["experiment"] = {
        "family": "resnet_style_complex_transition",
        "question": (
            "does K alone add useful capacity when equal-width carries remain exact "
            "identities and only changed-width carries use a normalized projection"
        ),
        "branch_gain_initial": {"carry": 1.0, "memory": 1.0},
        "paired_initialization": (
            "all same-name, same-shape state tensors match the compact control exactly"
        ),
        "excitation_modes": list(spec.excitation_modes),
        "pole_modes": list(spec.pole_modes),
        "controls": {
            "depth": "D2262",
            "post_hidden_ratio": POST_HIDDEN_RATIO,
            "path_hidden": 4,
            "head": "terminal raw-Q affine",
        },
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload = campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="resnet_style_transition_d2262_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.resnet_style_transition_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={list(SPECS[variant].excitation_modes)}, "
                "P=[128,128,128,128], D2262, projected identity shortcuts, "
                "WLPost1.5K, PathH4, terminal raw-Q affine head"
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={
            WIDENED: widened.VARIANT,
            COMPACT: compact.VARIANT,
            "compact_normalized_control_wandb": "5799d063ea995982",
        },
    )
    payload["jobs_by_gpu"] = {str(gpu): list(variants) for gpu, variants in JOBS_BY_GPU.items()}
    return payload


def main() -> None:
    runtime.run(
        variants=VARIANTS,
        seeds=SEEDS,
        build_model=_build,
        contract=_contract,
    )


if __name__ == "__main__":
    main()
