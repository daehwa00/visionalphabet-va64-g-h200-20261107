#!/usr/bin/env python3
"""Train the K128 all-resolution control with strict-complex Gated PostFusion."""

from __future__ import annotations

# The frozen campaign runner intentionally exposes private construction hooks.
# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as base
from torch import nn

from lnet.pac_complex_layers import PackedComplexLinear
from lnet.pac_gated_post_fusion import GatedStrictComplexPostFusion

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


BASE_VARIANT = base.variant_name(128, base.RESOLUTIONS)
VARIANT = f"{BASE_VARIANT}-CLPost"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
SPECS = {VARIANT: base.SPECS[BASE_VARIANT]}
STAGE_NAMES = base.STAGE_NAMES
RESOLUTIONS = base.RESOLUTIONS
SameResolutionFactorialBackbone = base.SameResolutionFactorialBackbone


def _postfusions(model: nn.Module) -> tuple[Any, ...]:
    main = tuple(getattr(model, name).augmented for name in base.STAGE_NAMES[:3])
    blocks = getattr(model, "same_resolution_blocks", nn.ModuleDict())
    return (*main, *tuple(blocks.values()))


def _install_strict_postfusion(model: nn.Module) -> None:
    for owner in _postfusions(model):
        source = owner.post_fusion
        owner.post_fusion = GatedStrictComplexPostFusion(
            source.modes,
            source.hidden_modes,
        )


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return campaign.build_derived(
        requested=variant,
        variant=VARIANT,
        base_variant=BASE_VARIANT,
        config=config,
        parent_build=base._build,
        install=_install_strict_postfusion,
        assert_model=_assert_model,
        variants=VARIANTS,
        seeds=SEEDS,
    )


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported all-resolution CLPost variant: {variant}")
    owners = _postfusions(model)
    if len(owners) != 7:
        raise RuntimeError("all-resolution CLPost must expose seven PostFusion blocks")
    for owner in owners:
        postfusion = owner.post_fusion
        if (
            type(postfusion) is not GatedStrictComplexPostFusion
            or type(postfusion.value) is not PackedComplexLinear
            or type(postfusion.out) is not PackedComplexLinear
            or postfusion.modes != 128
            or postfusion.hidden_modes != 192
        ):
            raise RuntimeError("all-resolution CLPost lost its strict-complex contract")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(base._variant_config(BASE_VARIANT))
    payload["backbone"]["name"] = f"A2D-{VARIANT}"
    payload["backbone"]["transition"]["post_fusion"] = {
        "operator": "real-gated strict-complex PostFusion",
        "value_projection": "packed strict ComplexLinear 128-to-192",
        "gate_projection": "real Linear 256-to-192 plus SiLU",
        "output_projection": "packed strict ComplexLinear 192-to-128",
        "hidden_modes": 192,
        "hidden_ratio": 1.5,
        "residual": "Y = X + delta",
    }
    payload["backbone"]["same_resolution_factorial"]["post_fusion"] = (
        "real-gated strict-complex 128-to-192-to-128"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    return campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="clpost_runner",
        variant=VARIANT,
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.same_resolution_all_clpost.v1",
        evidence_status="CPU and compiled CUDA smoke required before training",
        variant_config=_variant_config(),
        parameter_count=sum(
            parameter.numel() for parameter in _build(VARIANT, config).parameters()
        ),
        architecture=(
            "The completed K128 PF1.5K all-resolution model with all seven gated "
            "PostFusion value/output projections changed from WidelyLinear to strict "
            "ComplexLinear; every scan, carry, gate, head, and recipe setting is fixed."
        ),
        references={
            "widely_linear_control": {
                "variant": BASE_VARIANT,
                "run_root": (
                    "/home/qlab/experiments/alphabet/"
                    "r2k3-same-resolution-factorial-20260817/runs/"
                    f"{BASE_VARIANT}"
                ),
                "final_validation_accuracy": 0.872,
                "parameters": 2_772_516,
            }
        },
        extra_sources={"gated_post_fusion": Path("src/lnet/pac_gated_post_fusion.py")},
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
