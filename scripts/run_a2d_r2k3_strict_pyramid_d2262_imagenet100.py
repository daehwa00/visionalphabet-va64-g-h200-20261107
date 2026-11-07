#!/usr/bin/env python3
"""Train the compact strict-factorized K3 complex pyramid."""

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


VARIANT = "K32-64-128-128-P64-128-192-128-D2262-StrictR2K3-NormShortcut-Mem1-WLPost15K-PathH4"
VARIANTS = (VARIANT,)
JOBS_BY_GPU = {0: VARIANTS}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = parent.RESOLUTIONS
STAGE_NAMES = parent.STAGE_NAMES
SameResolutionFactorialBackbone = parent.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = parent.POST_HIDDEN_RATIO
MEMORY_SCALE_INITIAL = parent.MEMORY_SCALE_INITIAL
EXTRA_BLOCKS = parent.EXTRA_BLOCKS
SPEC = allocation.StageAllocationSpec(
    (32, 64, 128, 128),
    (64, 128, 192, 128),
    extra_blocks=EXTRA_BLOCKS,
    family="strict_complex_pyramid",
)
SPECS = {VARIANT: SPEC}


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported strict-pyramid variant: {variant}")
    model = parent._build_spec_model(SPEC, config)
    runtime.configure(VARIANTS, SEEDS)
    parent._assert_model_for_spec(model, SPEC, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported strict-pyramid variant: {variant}")
    parent._assert_model_for_spec(model, SPEC, variant)


def _variant_config() -> dict[str, Any]:
    payload = parent._variant_config_for_spec(VARIANT, SPEC)
    payload["experiment"] = {
        "family": "strict_complex_pyramid",
        "question": (
            "can a baseline-inspired complex pyramid retain accuracy with strict "
            "rank-2 factorized K3 readers"
        ),
        "reader": "strict-complex rank-2 factorized 3x3",
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
    parameters = sum(parameter.numel() for parameter in _build(VARIANT, config).parameters())
    return campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="strict_pyramid_d2262_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.strict_pyramid_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={VARIANT: _variant_config()},
        architectures={
            VARIANT: (
                "K=[32,64,128,128], P=[64,128,192,128], D2262, strict rank-2 "
                "factorized K3 readers, normalized Mem1 shortcuts, WLPost1.5K, "
                "PathH4, terminal raw-Q affine head"
            )
        },
        parameter_counts={VARIANT: parameters},
        references={
            "completed_best": "K128-P160x4-D2262-FullSR14x4",
            "design_basis": "compact hierarchical vision width schedule",
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
