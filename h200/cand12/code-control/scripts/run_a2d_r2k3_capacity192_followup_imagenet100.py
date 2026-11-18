#!/usr/bin/env python3
"""Train the controlled K/P=192 capacity continuation."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
# pyright: reportUnusedFunction=false
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as parent

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


K192_P128 = "K192-P128x4-SR56-28-14-7-WLPostH192-PathH4"
K128_P192 = "K128-P192x4-SR56-28-14-7-WLPostH192-PathH4"
K192_P192 = "K192-P192x4-SR56-28-14-7-WLPostH192-PathH4"
VARIANTS = (K192_P128, K128_P192, K192_P192)
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = parent.RESOLUTIONS
WIDTHS = (128, 192)
STAGE_NAMES = runtime.STAGE_NAMES
SameResolutionFactorialBackbone = parent.SameResolutionFactorialBackbone
JOBS_BY_GPU = {
    0: (K192_P128, K192_P192),
    1: (K128_P192,),
}
SPECS = {
    K192_P128: parent.InsightSpec((192,) * 4, (128,) * 4),
    K128_P192: parent.InsightSpec((128,) * 4, (192,) * 4),
    K192_P192: parent.InsightSpec((192,) * 4, (192,) * 4),
}


def variant_name(width: int, resolutions: tuple[int, ...]) -> str:
    return parent.variant_name(width, resolutions)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported capacity-192 variant: {variant}") from error
    model = parent._build_spec_model(spec, config)
    runtime.configure(VARIANTS, SEEDS)
    parent._assert_model_for_spec(model, spec, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    parent._assert_model_for_spec(model, SPECS[variant], variant)


def _variant_config(variant: str) -> dict[str, Any]:
    return parent._variant_config_for_spec(variant, SPECS[variant])


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    payload = runtime.base_contract(args)
    payload["schema"] = "lnet.a2d.r2k3.capacity192_followup.v1"
    payload["evidence_status"] = "all variants require CPU and compiled CUDA full-batch smoke"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in VARIANTS}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload["architecture"] = {
        K192_P128: "Exact K/P capacity control with only excitation K widened 128-to-192.",
        K128_P192: "Exact K/P capacity control with only all-stage pole P widened 128-to-192.",
        K192_P192: "Exact K/P capacity control with both excitation K and pole P widened to 192.",
    }
    payload["references"] = {
        "K128_P128": {"parameters": 2_693_668, "best_validation_accuracy": 0.8694},
        "K160_P128": {"parameters": 3_671_780, "best_validation_accuracy": 0.8778},
        "K128_P160": {"parameters": 3_154_660, "best_validation_accuracy": 0.8798},
    }
    payload["source_sha256"]["capacity192_followup"] = runtime.digest(Path(__file__))
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
