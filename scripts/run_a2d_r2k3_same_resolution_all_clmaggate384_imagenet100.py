#!/usr/bin/env python3
"""Train a parameter-matched phase-equivariant PostFusion control."""

from __future__ import annotations

# The frozen campaign runner intentionally exposes private construction hooks.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as clpost

from lnet.pac_complex_layers import PackedComplexLinear
from lnet.pac_gated_post_fusion import GatedStrictComplexMagnitudePostFusion

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


BASE_VARIANT = clpost.BASE_VARIANT
VARIANT = "K128-PF3K-SR56-28-14-7-CLMagEq"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
SPECS = {VARIANT: clpost.base.SPECS[BASE_VARIANT]}
STAGE_NAMES = clpost.STAGE_NAMES
RESOLUTIONS = clpost.RESOLUTIONS
SameResolutionFactorialBackbone = clpost.SameResolutionFactorialBackbone
MODES = 128
HIDDEN_MODES = 384
POSTFUSION_BLOCKS = 7
JOBS_BY_GPU = {0: VARIANTS}


def _postfusions(model: nn.Module) -> tuple[Any, ...]:
    return clpost._postfusions(model)


def _install_postfusion(model: nn.Module) -> None:
    for owner in _postfusions(model):
        owner.post_fusion = GatedStrictComplexMagnitudePostFusion(
            MODES,
            HIDDEN_MODES,
        )
        if hasattr(owner, "post_hidden"):
            owner.post_hidden = HIDDEN_MODES


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported CLMagEq H384 variant: {variant}")
    model = clpost.base._build(BASE_VARIANT, config)
    _install_postfusion(model)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model)
    return model


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported CLMagEq H384 variant: {variant}")
    owners = _postfusions(model)
    if len(owners) != POSTFUSION_BLOCKS:
        raise RuntimeError("CLMagEq H384 must expose seven PostFusion blocks")
    for owner in owners:
        postfusion = owner.post_fusion
        if (
            type(postfusion) is not GatedStrictComplexMagnitudePostFusion
            or type(postfusion.value) is not PackedComplexLinear
            or type(postfusion.out) is not PackedComplexLinear
            or postfusion.modes != MODES
            or postfusion.hidden_modes != HIDDEN_MODES
            or postfusion.gate.in_features != MODES
            or postfusion.gate.out_features != HIDDEN_MODES
        ):
            raise RuntimeError("CLMagEq H384 lost its phase-equivariant contract")
    if model.descriptor_dim != 2048 or model.classifier.input_dim != 512:
        raise RuntimeError("CLMagEq H384 changed the established Raw-Q head")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(clpost.base._variant_config(BASE_VARIANT))
    payload["backbone"]["name"] = f"A2D-{VARIANT}"
    postfusion = {
        "operator": "U(1)-equivariant mean-one gated strict-complex PostFusion",
        "value_projection": "packed strict ComplexLinear 128-to-384",
        "gate_detector": "centered log1p complex energy Linear 128-to-384",
        "gate_equation": "g0=1+0.5*tanh(a); g=g0/mean(g0)",
        "gate_initialization": "zero projection; exact unit gate",
        "output_projection": "packed strict ComplexLinear 384-to-128",
        "hidden_modes": HIDDEN_MODES,
        "hidden_ratio": 3.0,
        "parameter_match": "exactly matches WL H192 total model parameters",
        "residual": "Y = X + delta",
    }
    payload["backbone"]["transition"]["post_fusion"] = postfusion
    payload["backbone"]["same_resolution_factorial"]["post_fusion"] = postfusion
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = runtime.base_contract(args)
    config = runtime.model_config()
    payload["schema"] = "lnet.a2d.r2k3.same_resolution_all_clmaggate384.v1"
    payload["evidence_status"] = "CPU and compiled CUDA smoke required before training"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in _build(VARIANT, config).parameters())
    }
    payload["architecture"] = {
        VARIANT: (
            "The K=P=128 all-resolution WL-H192 control with only all seven PostFusion "
            "blocks replaced by parameter-matched strict CL H384 value/output projections "
            "and a phase-invariant centered-magnitude mean-one gate. GWL path "
            "canonicalization, scans, carries, Raw-Q head, and recipe are fixed."
        )
    }
    payload["references"] = {
        "WL_H192": {
            "parameters": 2_772_516,
            "best_validation_accuracy": 0.8726,
            "wandb_run": "40e7ef5278f97696",
        },
        "CL_H320_real_gate": {
            "parameters": 2_772_516,
            "wandb_run": "605eadf810e488ad",
        },
    }
    payload["source_sha256"]["gated_post_fusion"] = runtime.digest(
        Path("src/lnet/pac_gated_post_fusion.py")
    )
    payload["source_sha256"]["clmaggate384_runner"] = runtime.digest(Path(__file__))
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
