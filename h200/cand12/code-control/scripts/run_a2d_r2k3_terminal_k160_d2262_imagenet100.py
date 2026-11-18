#!/usr/bin/env python3
"""Train the terminal-only K160 D2262/P160 factorial cell."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_progressive_k_d2262_imagenet100 as progressive
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "K128-128-128-160-P160x4-D2262-WLPost15K-PathH4"
VARIANTS = (VARIANT,)
JOBS_BY_GPU = {1: VARIANTS}
SEEDS = runtime.DEFAULT_SEEDS
STAGE_NAMES = progressive.STAGE_NAMES
RESOLUTIONS = progressive.RESOLUTIONS
SameResolutionFactorialBackbone = progressive.SameResolutionFactorialBackbone
SPEC = allocation.StageAllocationSpec(
    (128, 128, 128, 160),
    (160, 160, 160, 160),
    extra_blocks=(0, 0, 4, 0),
    family="terminal_k160_d2262",
)
SPECS = {VARIANT: SPEC}


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported terminal-K160 D2262 variant: {variant}")
    progressive._assert_model_for_spec(model, SPEC, variant)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported terminal-K160 D2262 variant: {variant}")
    model = progressive._build_spec_model(SPEC, config)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _variant_config() -> dict[str, Any]:
    payload = progressive._variant_config_for_spec(VARIANT, SPEC)
    payload["experiment"] = {
        "family": SPEC.family,
        "control": "K128-P160x4-D2262-FullSR14x4",
        "factorial_cell": {"K14_160": False, "K7_160": True},
        "question": "does retaining K160 only at 7x7 improve the D2262/P160 backbone",
        "projection_contract": (
            "established identity/semi-orthogonal strict ComplexLinear projections; "
            "no pre-normalization wrapper and no learned branch gamma"
        ),
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    parameters = sum(parameter.numel() for parameter in _build(VARIANT, config).parameters())
    payload = campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="terminal_k160_d2262_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.terminal_k160_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant=VARIANT,
        variant_config=_variant_config(),
        architecture=(
            "D2262/P160 PathH4 backbone with K=[128,128,128,160], stage-local "
            "Hpost=1.5K, and the established unwrapped projection path."
        ),
        parameter_count=parameters,
        references={
            "K14_128_K7_128": "K128-P160x4-D2262-FullSR14x4",
            "K14_160_K7_128": "K128-128-160-128-P160x4-D2262-WLPost15K-PathH4",
            "K14_160_K7_160": "K128-128-160-160-P160x4-D2262-WLPost15K-PathH4",
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
