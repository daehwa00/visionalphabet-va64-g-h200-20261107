#!/usr/bin/env python3
"""Run the approved K128 P/K=1.5 and D2242/D2262 Wave-1 screen."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as capacity_insight
import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as control
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as factorial
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
from torch import nn

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


D2242_P128 = "K128-P128x4-D2242-FullSR14x2"
D2242_P14_R15 = "K128-P128-128-192-128-D2242"
D2242_P28_14_R15 = "K128-P128-192-192-128-D2242"
D2242_PFRONT3_R15 = "K128-P192-192-192-128-D2242"
D2262_P128 = "K128-P128x4-D2262-FullSR14x4"
D2242_UNIFORM_R15 = "K128-P192x4-D2242"

GPU0_VARIANTS = (
    D2242_P128,
    D2242_P14_R15,
    D2242_P28_14_R15,
    D2242_PFRONT3_R15,
)
GPU1_VARIANTS = (D2262_P128, D2242_UNIFORM_R15)
VARIANTS = (*GPU0_VARIANTS, *GPU1_VARIANTS)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {0: GPU0_VARIANTS, 1: GPU1_VARIANTS}

SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = factorial.RESOLUTIONS
STAGE_NAMES = runtime.STAGE_NAMES
SameResolutionFactorialBackbone = factorial.SameResolutionFactorialBackbone
EXTRA_SR14_X2 = (0, 0, 2, 0)
EXTRA_SR14_X4 = (0, 0, 4, 0)
StageAllocationSpec = allocation.StageAllocationSpec

SPECS = {
    D2242_P128: StageAllocationSpec(
        (128,) * 4,
        (128,) * 4,
        extra_blocks=EXTRA_SR14_X2,
        family="wave1_depth_anchor",
    ),
    D2242_P14_R15: StageAllocationSpec(
        (128,) * 4,
        (128, 128, 192, 128),
        extra_blocks=EXTRA_SR14_X2,
        family="wave1_pole_allocation_r15",
    ),
    D2242_P28_14_R15: StageAllocationSpec(
        (128,) * 4,
        (128, 192, 192, 128),
        extra_blocks=EXTRA_SR14_X2,
        family="wave1_pole_allocation_r15",
    ),
    D2242_PFRONT3_R15: StageAllocationSpec(
        (128,) * 4,
        (192, 192, 192, 128),
        extra_blocks=EXTRA_SR14_X2,
        family="wave1_pole_allocation_r15",
    ),
    D2262_P128: StageAllocationSpec(
        (128,) * 4,
        (128,) * 4,
        extra_blocks=EXTRA_SR14_X4,
        family="wave1_depth_anchor",
    ),
    D2242_UNIFORM_R15: StageAllocationSpec(
        (128,) * 4,
        (192,) * 4,
        extra_blocks=EXTRA_SR14_X2,
        family="wave1_uniform_r15",
    ),
}


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported Wave-1 K/P/depth variant: {variant}") from error
    model = capacity_insight._build_all_resolution(spec.as_insight_spec(), config)
    model = allocation._append_repeated_blocks(model, spec)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    extras = getattr(model, "extra_same_resolution_blocks", None)
    if extras is not None:
        delattr(model, "extra_same_resolution_blocks")
    try:
        capacity_insight._assert_model_for_spec(model, spec.as_insight_spec(), variant)
    finally:
        if extras is not None:
            model.extra_same_resolution_blocks = extras

    active_extras = cast("nn.ModuleDict", extras if extras is not None else nn.ModuleDict())
    expected = {
        str(resolution): count
        for resolution, count in zip(RESOLUTIONS, spec.extra_blocks, strict=True)
        if count
    }
    if {name: len(blocks) for name, blocks in active_extras.items()} != expected:
        raise RuntimeError(f"{variant} changed its repeated same-resolution depth")
    for stage_index, resolution in enumerate(RESOLUTIONS):
        key = str(resolution)
        if key not in active_extras:
            continue
        for repeat_index, block in enumerate(active_extras[key]):
            allocation._assert_block(
                block,
                excitation_modes=spec.excitation_modes[stage_index],
                pole_modes=spec.pole_modes[stage_index],
                label=f"{variant}/ExtraSR{resolution}.{repeat_index}",
            )


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = deepcopy(
        capacity_insight._variant_config_for_spec(variant, spec.as_insight_spec())
    )
    payload["experiment"] = {
        "family": spec.family,
        "wave": 1,
        "depth_by_resolution": dict(zip(map(str, RESOLUTIONS), spec.depth, strict=True)),
        "extra_full_blocks_by_resolution": dict(
            zip(map(str, RESOLUTIONS), spec.extra_blocks, strict=True)
        ),
        "main_ratio_family": [1.0, 1.5, 2.0],
        "selection_role": "seed501 full-epoch Wave-1 screen",
    }
    payload["backbone"]["same_resolution_factorial"][
        "extra_repetitions_by_resolution"
    ] = dict(zip(map(str, RESOLUTIONS), spec.extra_blocks, strict=True))
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    payload = runtime.base_contract(args)
    payload["schema"] = "lnet.a2d.r2k3.wave1_kp_depth.v1"
    payload["evidence_status"] = (
        "six-cell seed501 Wave-1 campaign; CPU and compiled CUDA full-batch smoke required"
    )
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["jobs_by_gpu"] = {
        str(gpu): list(variants) for gpu, variants in JOBS_BY_GPU.items()
    }
    payload["variant_configs"] = {
        variant: _variant_config(variant) for variant in VARIANTS
    }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload["architecture"] = dict.fromkeys(
        VARIANTS,
        (
            "K128-uniform D2242/D2262 Wave-1 screen using only P/K 1.0 or 1.5; "
            "no P/K 1.25, P/K 1.75, D2232, D2223, or later-wave candidate is included."
        ),
    )
    payload["references"] = {
        "D2222_P128": {
            "variant": control.VARIANT,
            "parameters": 2_693_668,
            "best_validation_accuracy": 0.8694,
        },
        "D2233_P128": {
            "variant": capacity_insight.EXTRA_SR14_7,
            "parameters": 3_350_052,
            "best_validation_accuracy": 0.8720,
            "final_validation_accuracy": 0.8696,
        },
    }
    payload["source_sha256"]["wave1_kp_depth_runner"] = runtime.digest(Path(__file__))
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
