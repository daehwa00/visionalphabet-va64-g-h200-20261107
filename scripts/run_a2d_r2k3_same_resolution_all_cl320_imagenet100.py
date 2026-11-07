#!/usr/bin/env python3
"""Train the parameter-matched K128 all-resolution CLPost control."""

from __future__ import annotations

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
from lnet.pac_gated_post_fusion import GatedStrictComplexPostFusion

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


BASE_VARIANT = clpost.BASE_VARIANT
VARIANT = "K128-PF25K-SR56-28-14-7-CLPost"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
SPECS = {VARIANT: clpost.base.SPECS[BASE_VARIANT]}
STAGE_NAMES = clpost.STAGE_NAMES
RESOLUTIONS = clpost.RESOLUTIONS
SameResolutionFactorialBackbone = clpost.SameResolutionFactorialBackbone
MODES = 128
HIDDEN_MODES = 320


def _postfusions(model: nn.Module) -> tuple[Any, ...]:
    return clpost._postfusions(model)


def _install_parameter_matched_postfusion(model: nn.Module) -> None:
    for owner in _postfusions(model):
        owner.post_fusion = GatedStrictComplexPostFusion(MODES, HIDDEN_MODES)
        if hasattr(owner, "post_hidden"):
            owner.post_hidden = HIDDEN_MODES


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return campaign.build_derived(
        requested=variant,
        variant=VARIANT,
        base_variant=BASE_VARIANT,
        config=config,
        parent_build=clpost.base._build,
        install=_install_parameter_matched_postfusion,
        assert_model=_assert_model,
        variants=VARIANTS,
        seeds=SEEDS,
    )


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported all-resolution CL320 variant: {variant}")
    owners = _postfusions(model)
    if len(owners) != 7:
        raise RuntimeError("all-resolution CL320 must expose seven PostFusion blocks")
    for owner in owners:
        postfusion = owner.post_fusion
        if (
            type(postfusion) is not GatedStrictComplexPostFusion
            or type(postfusion.value) is not PackedComplexLinear
            or type(postfusion.out) is not PackedComplexLinear
            or postfusion.gate.in_features != 2 * MODES
            or postfusion.gate.out_features != HIDDEN_MODES
            or postfusion.modes != MODES
            or postfusion.hidden_modes != HIDDEN_MODES
        ):
            raise RuntimeError("all-resolution CL320 lost its parameter-matched contract")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(clpost.base._variant_config(BASE_VARIANT))
    payload["backbone"]["name"] = f"A2D-{VARIANT}"
    payload["backbone"]["transition"]["post_fusion"] = {
        "operator": "parameter-matched real-gated strict-complex PostFusion",
        "value_projection": "packed strict ComplexLinear 128-to-320",
        "gate_projection": "real Linear 256-to-320 plus SiLU",
        "output_projection": "packed strict ComplexLinear 320-to-128",
        "hidden_modes": HIDDEN_MODES,
        "hidden_ratio": 2.5,
        "residual": "Y = X + delta",
    }
    payload["backbone"]["same_resolution_factorial"]["post_fusion"] = (
        "parameter-matched packed strict-complex 128-to-320-to-128"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    return campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="cl320_runner",
        variant=VARIANT,
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.same_resolution_all_cl320.v1",
        evidence_status="CPU and compiled CUDA smoke required before training",
        variant_config=_variant_config(),
        parameter_count=sum(parameter.numel() for parameter in _build(VARIANT, config).parameters()),
        architecture="The K128 all-resolution strict CLPost control widened from H=192 to H=320 "
        "so its seven PostFusion blocks exactly match the parameter and dense-real "
        "compute capacity of the H=192 WidelyLinear baseline; all other settings fixed.",
        references={
            "widely_linear_parameter_match": {
                "variant": BASE_VARIANT,
                "wandb_run": "40e7ef5278f97696",
                "hidden_modes": 192,
                "final_validation_accuracy": 0.872,
                "parameters": 2_772_516,
            },
            "narrow_strict_control": {
                "variant": clpost.VARIANT,
                "wandb_run": "d581f91bbd3359f8",
                "hidden_modes": 192,
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
