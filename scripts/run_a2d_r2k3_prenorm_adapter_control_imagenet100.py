#!/usr/bin/env python3
"""Compare K128 and K14=160 with pre-normalized projection branches."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
import math
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_d2262_p160_followup_imagenet100 as base
import run_a2d_r2k3_progressive_k_d2262_imagenet100 as progressive
import run_a2d_r2k3_scale_stable_adapter_control_imagenet100 as stable_control
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation

from lnet.pac_complex_layers import ComplexLinear
from lnet.pac_complex_projection_adapter import PreNormScaledComplexProjection
from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


K128_CONTROL = "K128x4-P160x4-D2262-PreNormAdapter"
K14_160 = "K128-128-160-128-P160x4-D2262-PreNormAdapter"
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
        family="prenorm_adapter_control",
    ),
    K14_160: allocation.StageAllocationSpec(
        (128, 128, 160, 128),
        (160, 160, 160, 160),
        extra_blocks=(0, 0, 4, 0),
        family="prenorm_adapter_control",
    ),
}


def _install_prenorm_adapters(model: ComplexScanBackbone) -> None:
    for owner in stable_control._adapter_owners(model):
        memory = owner.memory_projection
        if memory is None:
            progressive._materialize_square_memory_projection(owner)
            memory = owner.memory_projection
        if not isinstance(memory, ComplexLinear):
            raise TypeError("memory adapter must be a strict ComplexLinear")
        owner.memory_projection = PreNormScaledComplexProjection.from_projection(
            memory,
            initial_scale=1.0 / math.sqrt(memory.input_modes),
        )
        if isinstance(owner, GatedPoleExcitationS2DTransition):
            carry = owner.carry_projection
            if carry is not None:
                owner.carry_projection = PreNormScaledComplexProjection.from_projection(
                    carry,
                    initial_scale=1.0,
                )


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported pre-norm adapter variant: {variant}") from error
    model = (
        base._build(base.VARIANT, config)
        if variant == K128_CONTROL
        else progressive._build_spec_model(spec, config)
    )
    if variant == K128_CONTROL:
        base._assert_model(model, base.VARIANT)
    else:
        progressive._assert_model_for_spec(model, spec, variant)
    _install_prenorm_adapters(model)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    owners = stable_control._adapter_owners(model)
    if not all(
        type(owner.memory_projection) is PreNormScaledComplexProjection for owner in owners
    ):
        raise RuntimeError(f"{variant} lost a pre-normalized memory adapter")
    for owner in owners:
        memory = owner.memory_projection
        if not isinstance(memory, PreNormScaledComplexProjection):
            raise TypeError(f"{variant} lost a pre-normalized memory adapter")
        expected = 1.0 / math.sqrt(memory.input_modes)
        if not math.isclose(float(memory.scale.detach()), expected, rel_tol=1.0e-6):
            raise RuntimeError(f"{variant} changed its initial memory branch scale")
        if isinstance(owner, GatedPoleExcitationS2DTransition):
            carry = owner.carry_projection
            if carry is not None and (
                type(carry) is not PreNormScaledComplexProjection
                or not math.isclose(
                    float(carry.scale.detach()),
                    1.0,
                    rel_tol=1.0e-6,
                )
            ):
                raise RuntimeError(f"{variant} changed its carry branch scale")


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
        "memory_adapter": "CRMSNorm -> free strict ComplexLinear -> gamma",
        "memory_gamma_initial": "1/sqrt(P)",
        "carry_adapter": "identity if K is unchanged; otherwise CRMSNorm -> strict CL",
        "carry_gamma_initial": 1.0,
        "question": "does stage-wise K generalize with activation-scale-controlled adapters",
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
        runner_source_key="prenorm_adapter_control_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.prenorm_adapter_control.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required for both variants",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"P160/D2262 PathH4 backbone with K={list(SPECS[variant].excitation_modes)} "
                "and pre-normalized, branch-scaled memory/carry adapters."
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
