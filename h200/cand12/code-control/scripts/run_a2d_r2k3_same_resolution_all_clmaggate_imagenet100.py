#!/usr/bin/env python3
"""Train K128 all-resolution with strict projections and a magnitude gate."""

from __future__ import annotations

# The frozen campaign runners intentionally expose private construction hooks.
# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as clpost

from lnet.pac_complex_layers import PackedComplexLinear
from lnet.pac_gated_post_fusion import GatedStrictComplexMagnitudePostFusion

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


BASE_VARIANT = clpost.BASE_VARIANT
VARIANT = f"{BASE_VARIANT}-CLMagGate"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
SPECS = {VARIANT: clpost.base.SPECS[BASE_VARIANT]}
STAGE_NAMES = clpost.STAGE_NAMES
RESOLUTIONS = clpost.RESOLUTIONS
SameResolutionFactorialBackbone = clpost.SameResolutionFactorialBackbone


def _postfusions(model: nn.Module) -> tuple[Any, ...]:
    return clpost._postfusions(model)


def _install_magnitude_postfusion(model: nn.Module) -> None:
    for owner in _postfusions(model):
        source = owner.post_fusion
        owner.post_fusion = GatedStrictComplexMagnitudePostFusion(
            source.modes,
            source.hidden_modes,
        )


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return campaign.build_derived(
        requested=variant,
        variant=VARIANT,
        base_variant=BASE_VARIANT,
        config=config,
        parent_build=clpost.base._build,
        install=_install_magnitude_postfusion,
        assert_model=_assert_model,
        variants=VARIANTS,
        seeds=SEEDS,
    )


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported all-resolution CLMagGate variant: {variant}")
    owners = _postfusions(model)
    if len(owners) != 7:
        raise RuntimeError("all-resolution CLMagGate must expose seven PostFusion blocks")
    for owner in owners:
        postfusion = owner.post_fusion
        if (
            type(postfusion) is not GatedStrictComplexMagnitudePostFusion
            or type(postfusion.value) is not PackedComplexLinear
            or type(postfusion.out) is not PackedComplexLinear
            or postfusion.gate.in_features != 128
            or postfusion.gate.out_features != 192
            or postfusion.modes != 128
            or postfusion.hidden_modes != 192
        ):
            raise RuntimeError("all-resolution CLMagGate lost its phase-invariant contract")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(clpost.base._variant_config(BASE_VARIANT))
    payload["backbone"]["name"] = f"A2D-{VARIANT}"
    payload["backbone"]["transition"]["post_fusion"] = {
        "operator": "phase-invariant mean-one gated strict-complex PostFusion",
        "value_projection": "packed strict ComplexLinear 128-to-192",
        "gate_detector": "centered log1p complex energy 128-to-192",
        "gate_equation": "g0=1+0.5*tanh(a); g=g0/mean(g0)",
        "gate_initialization": "zero projection; exact unit gate",
        "output_projection": "packed strict ComplexLinear 192-to-128",
        "hidden_modes": 192,
        "hidden_ratio": 1.5,
        "residual": "Y = X + delta",
    }
    payload["backbone"]["same_resolution_factorial"]["post_fusion"] = (
        "strict-complex 128-to-192-to-128 with phase-invariant mean-one gate"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    return campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="clmaggate_runner",
        variant=VARIANT,
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.same_resolution_all_clmaggate.v1",
        evidence_status="CPU and compiled CUDA smoke required before training",
        variant_config=_variant_config(),
        parameter_count=sum(parameter.numel() for parameter in _build(VARIANT, config).parameters()),
        architecture="The K128 PF1.5K all-resolution CLPost control with only its phase-dependent "
        "real/imag gate replaced by a centered-magnitude, positive mean-one gate; "
        "all scans, strict value/output projections, carries, heads, and recipe settings "
        "are fixed.",
        references={
            "widely_linear_control": {
                "variant": BASE_VARIANT,
                "wandb_run": "40e7ef5278f97696",
                "final_validation_accuracy": 0.872,
                "parameters": 2_772_516,
            },
            "strict_real_gate_control": {
                "variant": clpost.VARIANT,
                "wandb_run": "d581f91bbd3359f8",
                "parameters": 2_084_388,
            },
        },
        extra_sources={
            "gated_post_fusion": Path("src/lnet/pac_gated_post_fusion.py"),
        },
    )


def main() -> None:
    runtime.run(
        variants=VARIANTS,
        seeds=SEEDS,
        build_model=_build,
        contract=_contract,
    )


if __name__ == "__main__":
    main()
