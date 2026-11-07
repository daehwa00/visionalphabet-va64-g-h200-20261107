#!/usr/bin/env python3
"""Compare K128 and K14=160 with scale-stable memory/carry adapters."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_d2262_p160_followup_imagenet100 as base
import run_a2d_r2k3_progressive_k_d2262_imagenet100 as progressive
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation

from lnet.pac_complex_layers import ComplexLinear, ScaleStableComplexLinear
from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
    from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


K128_CONTROL = "K128x4-P160x4-D2262-ScaleStableAdapter"
K14_160 = "K128-128-160-128-P160x4-D2262-ScaleStableAdapter"
VARIANTS = (K128_CONTROL, K14_160)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {0: (K128_CONTROL,), 1: (K14_160,)}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = progressive.RESOLUTIONS
STAGE_NAMES = progressive.STAGE_NAMES
SameResolutionFactorialBackbone = progressive.SameResolutionFactorialBackbone
SPECS = {
    K128_CONTROL: allocation.StageAllocationSpec(
        (128, 128, 128, 128),
        (160, 160, 160, 160),
        extra_blocks=(0, 0, 4, 0),
        family="scale_stable_adapter_control",
    ),
    K14_160: allocation.StageAllocationSpec(
        (128, 128, 160, 128),
        (160, 160, 160, 160),
        extra_blocks=(0, 0, 4, 0),
        family="scale_stable_adapter_control",
    ),
}


def _adapter_owners(
    model: ComplexScanBackbone,
) -> tuple[GatedPoleExcitationS2DTransition | SameResolutionPoleScanBlock, ...]:
    extra_groups = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    extras = cast("nn.ModuleList", extra_groups["14"])
    return (
        *(cast("GatedPoleExcitationS2DTransition", getattr(model, name).augmented) for name in STAGE_NAMES[:3]),
        *(cast("SameResolutionPoleScanBlock", block) for block in model.same_resolution_blocks.values()),
        *(cast("SameResolutionPoleScanBlock", block) for block in extras),
    )


def _install_scale_stable_adapters(model: ComplexScanBackbone) -> None:
    for owner in _adapter_owners(model):
        memory = owner.memory_projection
        if memory is None:
            progressive._materialize_square_memory_projection(owner)
            memory = owner.memory_projection
        if not isinstance(memory, ComplexLinear):
            raise TypeError("memory adapter must be a strict ComplexLinear")
        owner.memory_projection = ScaleStableComplexLinear.from_projection(memory)
        if isinstance(owner, GatedPoleExcitationS2DTransition):
            carry = owner.carry_projection
            if carry is not None:
                owner.carry_projection = ScaleStableComplexLinear.from_projection(carry)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported scale-stable adapter variant: {variant}") from error
    model = (
        base._build(base.VARIANT, config)
        if variant == K128_CONTROL
        else progressive._build_spec_model(spec, config)
    )
    _install_scale_stable_adapters(model)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    if variant == K128_CONTROL:
        base._assert_model(model, base.VARIANT)
    else:
        progressive._assert_model_for_spec(model, spec, variant)
    owners = _adapter_owners(model)
    if not all(type(owner.memory_projection) is ScaleStableComplexLinear for owner in owners):
        raise RuntimeError(f"{variant} lost a scale-stable memory adapter")
    for owner in owners:
        projection = cast("ScaleStableComplexLinear", owner.memory_projection)
        if not bool((projection.reference_row_norm > 0).all()):
            raise RuntimeError(f"{variant} has an invalid adapter reference norm")
        if isinstance(owner, GatedPoleExcitationS2DTransition):
            carry = owner.carry_projection
            if carry is not None and type(carry) is not ScaleStableComplexLinear:
                raise RuntimeError(f"{variant} lost a scale-stable carry adapter")


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = (
        deepcopy(base._variant_config())
        if variant == K128_CONTROL
        else progressive._variant_config_for_spec(variant, spec)
    )
    payload["backbone"]["name"] = f"A2D-{variant}"
    payload["backbone"]["excitation_schedule"] = list(spec.excitation_modes)
    payload["experiment"] = {
        "family": spec.family,
        "control": base.VARIANT,
        "adapter": (
            "learned strict-complex projection with each row norm fixed to its "
            "initialized identity/semi-orthogonal value"
        ),
        "question": "does stage-wise K generalize after removing adapter gain drift",
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
        runner_source_key="scale_stable_adapter_control_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.scale_stable_adapter_control.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required for both variants",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"P160/D2262 PathH4 backbone with K={list(SPECS[variant].excitation_modes)} "
                "and scale-stable memory/carry projection adapters."
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={
            "unconstrained_K128": {
                "variant": base.VARIANT,
                "parameters": 4_713_444,
                "best_validation_accuracy": 0.8828,
            }
        },
    )
    payload["jobs_by_gpu"] = {
        str(gpu): list(variants) for gpu, variants in JOBS_BY_GPU.items()
    }
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
