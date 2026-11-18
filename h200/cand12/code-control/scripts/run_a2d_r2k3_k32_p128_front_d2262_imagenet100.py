#!/usr/bin/env python3
"""Train the narrow-excitation, wide-memory K32/P128 diagnostic."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false, reportUnusedFunction=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_front_capacity_reallocation_d2262_imagenet100 as parent
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "K32-64-64-128-P128x4-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
VARIANTS = (VARIANT,)
JOBS_BY_GPU = {1: VARIANTS}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = parent.RESOLUTIONS
STAGE_NAMES = parent.STAGE_NAMES
SameResolutionFactorialBackbone = parent.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = parent.POST_HIDDEN_RATIO
MEMORY_SCALE_INITIAL = 1.0
EXTRA_BLOCKS = parent.EXTRA_BLOCKS
SPEC = allocation.StageAllocationSpec(
    excitation_modes=(32, 64, 64, 128),
    pole_modes=(128, 128, 128, 128),
    extra_blocks=EXTRA_BLOCKS,
    family="k32_p128_front_diagnostic",
)
SPECS = {VARIANT: SPEC}


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported K32/P128 variant: {variant}")
    model = parent._build_spec_model(SPEC, config)
    runtime.configure(VARIANTS, SEEDS)
    parent._assert_model_for_spec(model, SPEC, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported K32/P128 variant: {variant}")
    parent._assert_model_for_spec(model, SPEC, variant)


def _variant_config(variant: str) -> dict[str, Any]:
    if variant != VARIANT:
        raise ValueError(f"unsupported K32/P128 variant: {variant}")
    payload = parent._variant_config_for_spec(variant, SPEC)
    payload["experiment"] = {
        "family": "k32_p128_front_diagnostic",
        "question": (
            "can a K32 complex excitation use a P128 pole-memory bank to retain "
            "accuracy at nearly the same front-reader FLOPs as K64/P64"
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
        runner_source_key="k32_p128_front_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.k32_p128_front_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={VARIANT: _variant_config(VARIANT)},
        architectures={
            VARIANT: (
                "K=[32,64,64,128], P=[128,128,128,128], D2262, "
                "WLPost1.5K, PathH4, normalized Mem1 width shortcuts, "
                "terminal raw-Q affine head"
            )
        },
        parameter_counts={VARIANT: parameter_count},
        references={
            "matched_reader_compute_control": (
                "K64-128-128-128-P64-160-160-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
            )
        },
    )
    payload["jobs_by_gpu"] = {"1": [VARIANT]}
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
