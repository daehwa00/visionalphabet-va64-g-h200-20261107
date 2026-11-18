#!/usr/bin/env python3
"""Train parameter-matched WidelyLinear PostFusion controls."""

from __future__ import annotations

# The frozen campaign runner intentionally exposes private construction hooks.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as base

from lnet.pac_complex_layers import WidelyLinear
from lnet.pac_gated_post_fusion import GatedComplexPostFusion

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


BASE_VARIANT = base.variant_name(128, base.RESOLUTIONS)
WL_H115 = "K128-PF-WL115-SR56-28-14-7"
WL_H154 = "K128-PF-WL154-SR56-28-14-7"
VARIANTS = (WL_H115, WL_H154)
VARIANT = VARIANTS[0]
SEEDS = runtime.DEFAULT_SEEDS
STAGE_NAMES = base.STAGE_NAMES
RESOLUTIONS = base.RESOLUTIONS
SameResolutionFactorialBackbone = base.SameResolutionFactorialBackbone
SPECS = dict.fromkeys(VARIANTS, base.SPECS[BASE_VARIANT])
HIDDEN_BY_VARIANT = {WL_H115: 115, WL_H154: 154}
JOBS_BY_GPU = {0: VARIANTS}
POSTFUSION_BLOCKS = 7


def _postfusions(model: nn.Module) -> tuple[Any, ...]:
    main = tuple(getattr(model, name).augmented for name in STAGE_NAMES[:3])
    same_resolution = tuple(model.same_resolution_blocks.values())
    return (*main, *same_resolution)


def _install_postfusion(model: nn.Module, hidden_modes: int) -> None:
    for owner in _postfusions(model):
        owner.post_fusion = GatedComplexPostFusion(128, hidden_modes)
        if hasattr(owner, "post_hidden"):
            owner.post_hidden = hidden_modes


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        hidden_modes = HIDDEN_BY_VARIANT[variant]
    except KeyError as error:
        raise ValueError(f"unsupported WL budget-match variant: {variant}") from error
    return campaign.build_derived(
        requested=variant,
        variant=variant,
        base_variant=BASE_VARIANT,
        config=config,
        parent_build=base._build,
        install=lambda model: _install_postfusion(model, hidden_modes),
        assert_model=lambda model: _assert_model(model, variant),
        variants=VARIANTS,
        seeds=SEEDS,
    )


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    try:
        hidden_modes = HIDDEN_BY_VARIANT[variant]
    except KeyError as error:
        raise ValueError(f"unsupported WL budget-match variant: {variant}") from error
    owners = _postfusions(model)
    if len(owners) != POSTFUSION_BLOCKS:
        raise RuntimeError("WL budget match must expose seven PostFusion blocks")
    for owner in owners:
        postfusion = owner.post_fusion
        if (
            type(postfusion) is not GatedComplexPostFusion
            or type(postfusion.value) is not WidelyLinear
            or type(postfusion.out) is not WidelyLinear
            or postfusion.modes != 128
            or postfusion.hidden_modes != hidden_modes
            or postfusion.gate.in_features != 256
            or postfusion.gate.out_features != hidden_modes
        ):
            raise RuntimeError("WL budget-match PostFusion contract changed")
    if model.descriptor_dim != 2048 or model.classifier.input_dim != 512:
        raise RuntimeError("WL budget match changed the established Raw-Q head")


def _variant_config(variant: str) -> dict[str, Any]:
    hidden_modes = HIDDEN_BY_VARIANT[variant]
    payload = deepcopy(base._variant_config(BASE_VARIANT))
    payload["backbone"]["name"] = f"A2D-{variant}"
    postfusion = {
        "operator": "real-gated WidelyLinear PostFusion",
        "value_projection": f"WidelyLinear 128-to-{hidden_modes}",
        "gate_projection": f"real Linear 256-to-{hidden_modes} plus SiLU",
        "output_projection": f"WidelyLinear {hidden_modes}-to-128",
        "hidden_modes": hidden_modes,
        "hidden_ratio": hidden_modes / 128,
        "residual": "Y = X + delta",
    }
    payload["backbone"]["transition"]["post_fusion"] = postfusion
    payload["backbone"]["same_resolution_factorial"]["post_fusion"] = postfusion
    payload["comparison"] = (
        "parameter-matched to strict CL H192"
        if variant == WL_H115
        else "parameter-matched to strict CL H256"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    return campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="wl_budget_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.wl_budget_match.v1",
        evidence_status="CPU and compiled CUDA smoke required before training",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        parameter_counts={
            variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
            for variant in VARIANTS
        },
        architectures={
            variant: (
                "Exact K=P=128 all-resolution WL control with only all seven PostFusion "
                f"hidden widths changed from 192 to {HIDDEN_BY_VARIANT[variant]}."
            )
            for variant in VARIANTS
        },
        references={
            "CL_H192": {"parameters": 2_084_388, "best_validation_accuracy": 0.8652},
            "CL_H256": {"parameters": 2_428_452, "best_validation_accuracy": 0.8700},
            "WL_H192": {"parameters": 2_772_516, "best_validation_accuracy": 0.8726},
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
