#!/usr/bin/env python3
"""Prioritize K64 front-stage controls, then resume the pending capacity screen."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
# pyright: reportUnusedFunction=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_front_capacity_reallocation_d2262_imagenet100 as parent
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


K64_P96 = "K64-128-128-128-P96-160-160-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
K64_P64 = "K64-128-128-128-P64-160-160-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
K128_P96 = "K128x4-P96-160-160-160-D2262-WLPost15K-PathH4"
K96_P96 = "K96-128-128-128-P96-160-160-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
MID_POLE_REALLOCATION = "K96-128-128-128-P96-192-192-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
POLE14_REALLOCATION = "K96-128-128-128-P96-160-192-160-D2262-NormShortcut-Mem1-WLPost15K-PathH4"
VARIANTS = (
    K64_P96,
    K64_P64,
    K128_P96,
    K96_P96,
    MID_POLE_REALLOCATION,
    POLE14_REALLOCATION,
)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {
    0: (K64_P96, K128_P96, MID_POLE_REALLOCATION),
    1: (K64_P64, K96_P96, POLE14_REALLOCATION),
}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = parent.RESOLUTIONS
STAGE_NAMES = parent.STAGE_NAMES
SameResolutionFactorialBackbone = parent.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = parent.POST_HIDDEN_RATIO
MEMORY_SCALE_INITIAL = parent.MEMORY_SCALE_INITIAL
EXTRA_BLOCKS = parent.EXTRA_BLOCKS


def _spec(
    excitation_modes: tuple[int, int, int, int],
    pole_modes: tuple[int, int, int, int],
) -> allocation.StageAllocationSpec:
    return allocation.StageAllocationSpec(
        excitation_modes,
        pole_modes,
        extra_blocks=EXTRA_BLOCKS,
        family="front_k64_tail128_followup",
    )


SPECS = {
    K64_P96: _spec((64, 128, 128, 128), (96, 160, 160, 160)),
    K64_P64: _spec((64, 128, 128, 128), (64, 160, 160, 160)),
    K128_P96: _spec((128, 128, 128, 128), (96, 160, 160, 160)),
    K96_P96: _spec((96, 128, 128, 128), (96, 160, 160, 160)),
    MID_POLE_REALLOCATION: _spec((96, 128, 128, 128), (96, 192, 192, 160)),
    POLE14_REALLOCATION: _spec((96, 128, 128, 128), (96, 160, 192, 160)),
}


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported front-K64 variant: {variant}") from error
    model = parent._build_spec_model(spec, config)
    runtime.configure(VARIANTS, SEEDS)
    parent._assert_model_for_spec(model, spec, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported front-K64 variant: {variant}") from error
    parent._assert_model_for_spec(model, spec, variant)


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = parent._variant_config_for_spec(variant, spec)
    payload["experiment"]["family"] = "front_k64_tail128_followup"
    payload["experiment"]["question"] = (
        "can the 56x56 complex excitation shrink from K96 to K64 when this "
        "campaign uses K128 at later stages and allocates pole capacity independently"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    parameter_counts = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload = campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="front_k64_tail128_followup_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.front_k64_followup_d2262.v2",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={list(spec.excitation_modes)}, P={list(spec.pole_modes)}, "
                "D2262, WLPost1.5K, PathH4, normalized Mem1 width shortcuts, "
                "terminal raw-Q affine head"
            )
            for variant, spec in SPECS.items()
        },
        parameter_counts=parameter_counts,
        references={
            "abandoned_partial_k128_p128": parent.K128_P128,
            "abandoned_partial_k96_p128": parent.K96_P128,
            "abandoned_k160_followup": "front_k64_followup_mem1_d2262",
        },
    )
    payload["jobs_by_gpu"] = {str(gpu): list(variants) for gpu, variants in JOBS_BY_GPU.items()}
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
