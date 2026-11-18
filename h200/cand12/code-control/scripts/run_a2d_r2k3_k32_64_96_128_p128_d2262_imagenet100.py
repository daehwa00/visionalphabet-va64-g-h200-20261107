#!/usr/bin/env python3
"""Train the stage-3 widened K32/P128 D2262 diagnostic."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false, reportUnusedFunction=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_front_capacity_reallocation_d2262_imagenet100 as capacity
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "K32-64-96-128-P128x4-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
VARIANTS = (VARIANT,)
JOBS_BY_GPU = {0: VARIANTS}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = capacity.RESOLUTIONS
STAGE_NAMES = capacity.STAGE_NAMES
SameResolutionFactorialBackbone = capacity.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = capacity.POST_HIDDEN_RATIO
MEMORY_SCALE_INITIAL = capacity.MEMORY_SCALE_INITIAL
EXTRA_BLOCKS = capacity.EXTRA_BLOCKS
SPEC = allocation.StageAllocationSpec(
    (32, 64, 96, 128),
    (128, 128, 128, 128),
    extra_blocks=EXTRA_BLOCKS,
    family="stage3_widened_excitation_uniform_p128",
)
SPECS = {VARIANT: SPEC}


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported stage-3 widened K32/P128 variant: {variant}")
    model = capacity._build_spec_model(SPEC, config)
    runtime.configure(VARIANTS, SEEDS)
    capacity._assert_model_for_spec(model, SPEC, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported stage-3 widened K32/P128 variant: {variant}")
    capacity._assert_model_for_spec(model, SPEC, variant)


def _variant_config() -> dict[str, Any]:
    payload = capacity._variant_config_for_spec(VARIANT, SPEC)
    payload["experiment"] = {
        "family": "stage3_widened_excitation_uniform_p128",
        "question": (
            "does widening only the six-block 14x14 stage from K64 to K96 "
            "remove the excitation bottleneck without adding high-resolution depth"
        ),
        "memory_scale_initial": MEMORY_SCALE_INITIAL,
        "controls": {
            "depth": "D2262",
            "post_hidden_ratio": POST_HIDDEN_RATIO,
            "path_hidden": 4,
            "head": "terminal raw-Q affine",
        },
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    parameter_count = sum(parameter.numel() for parameter in _build(VARIANT, config).parameters())
    payload = campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="k32_64_96_128_p128_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.k32_64_96_128_p128_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={VARIANT: _variant_config()},
        architectures={
            VARIANT: (
                "K=[32,64,96,128], P=[128,128,128,128], D2262, "
                "WLPost1.5K, PathH4, normalized Mem1 width shortcuts, "
                "terminal raw-Q affine head"
            )
        },
        parameter_counts={VARIANT: parameter_count},
        references={
            "accuracy_anchor": "K128-P160x4-D2262-FullSR14x4",
            "width_control": "K32-64-64-128-P128x4-D2262-NormShortcut-Mem1-WLPost15K-PathH4",
        },
    )
    payload["jobs_by_gpu"] = {"0": [VARIANT]}
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
