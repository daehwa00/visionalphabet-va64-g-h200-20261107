#!/usr/bin/env python3
"""Train the K128, uniform-P160, D2262 interaction follow-up."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as capacity_insight
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as factorial
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
from torch import nn

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "K128-P160x4-D2262-FullSR14x4"
VARIANTS = (VARIANT,)
JOBS_BY_GPU = {1: VARIANTS}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = factorial.RESOLUTIONS
STAGE_NAMES = runtime.STAGE_NAMES
SameResolutionFactorialBackbone = factorial.SameResolutionFactorialBackbone
SPEC = allocation.StageAllocationSpec(
    (128,) * 4,
    (160,) * 4,
    extra_blocks=(0, 0, 4, 0),
    family="d2262_p160_interaction_followup",
)
SPECS = {VARIANT: SPEC}


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported D2262/P160 follow-up variant: {variant}")
    model = capacity_insight._build_all_resolution(SPEC.as_insight_spec(), config)
    model = allocation._append_repeated_blocks(model, SPEC)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported D2262/P160 follow-up variant: {variant}")
    extras = getattr(model, "extra_same_resolution_blocks", None)
    if extras is not None:
        delattr(model, "extra_same_resolution_blocks")
    try:
        capacity_insight._assert_model_for_spec(model, SPEC.as_insight_spec(), variant)
    finally:
        if extras is not None:
            model.extra_same_resolution_blocks = extras

    active_extras = cast("nn.ModuleDict", extras if extras is not None else nn.ModuleDict())
    if set(active_extras) != {"14"} or len(active_extras["14"]) != 4:
        raise RuntimeError(f"{variant} changed its D2262 repeated SR14 depth")
    for repeat_index, block in enumerate(active_extras["14"]):
        allocation._assert_block(
            block,
            excitation_modes=128,
            pole_modes=160,
            label=f"{variant}/ExtraSR14.{repeat_index}",
        )


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(
        capacity_insight._variant_config_for_spec(VARIANT, SPEC.as_insight_spec())
    )
    payload["experiment"] = {
        "family": SPEC.family,
        "depth_by_resolution": dict(zip(map(str, RESOLUTIONS), SPEC.depth, strict=True)),
        "extra_full_blocks_by_resolution": dict(
            zip(map(str, RESOLUTIONS), SPEC.extra_blocks, strict=True)
        ),
        "interaction": "uniform P160 on the completed D2262 depth anchor",
        "selection_role": "seed501 full-epoch follow-up",
    }
    payload["backbone"]["same_resolution_factorial"][
        "extra_repetitions_by_resolution"
    ] = dict(zip(map(str, RESOLUTIONS), SPEC.extra_blocks, strict=True))
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    parameters = sum(parameter.numel() for parameter in _build(VARIANT, config).parameters())
    payload = runtime.base_contract(args)
    payload["schema"] = "lnet.a2d.r2k3.d2262_p160_followup.v1"
    payload["evidence_status"] = (
        "single-cell seed501 D2262/P160 interaction; CPU and compiled CUDA full-batch smoke required"
    )
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["jobs_by_gpu"] = {"1": list(VARIANTS)}
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {VARIANT: parameters}
    payload["architecture"] = {
        VARIANT: (
            "K128-uniform, P160-uniform D2262 interaction follow-up with four extra "
            "full SR14 blocks and no other architecture or recipe change."
        )
    }
    payload["references"] = {
        "D2222_P160": {
            "variant": capacity_insight.K128_P160,
            "parameters": 3_154_660,
            "best_validation_accuracy": 0.8798,
        },
        "D2262_P128": {
            "variant": "K128-P128x4-D2262-FullSR14x4",
            "parameters": 4_006_436,
            "best_validation_accuracy": 0.8772,
            "final_validation_accuracy": 0.8758,
        },
    }
    payload["source_sha256"]["d2262_p160_followup_runner"] = runtime.digest(Path(__file__))
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
