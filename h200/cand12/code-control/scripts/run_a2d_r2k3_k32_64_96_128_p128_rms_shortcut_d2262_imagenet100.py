#!/usr/bin/env python3
"""Train K32-64-96-128 with RMS-matched projection shortcuts."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false, reportUnusedFunction=false
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_k32_64_96_128_p128_d2262_imagenet100 as control

from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    RMSMatchedShortcutGatedTransition,
    rms_matched_shortcut_transition,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "K32-64-96-128-P128x4-D2262-RMSShortcut-Mem1-WLPost15K-PathH4"
VARIANTS = (VARIANT,)
JOBS_BY_GPU = {0: VARIANTS}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = control.RESOLUTIONS
STAGE_NAMES = control.STAGE_NAMES
SameResolutionFactorialBackbone = control.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = control.POST_HIDDEN_RATIO
SPEC = control.SPEC
SPECS = {VARIANT: SPEC}


def _install_rms_matched_shortcuts(model: ComplexScanBackbone) -> None:
    for name, input_modes, output_modes in zip(
        STAGE_NAMES[:3],
        SPEC.excitation_modes[:-1],
        SPEC.excitation_modes[1:],
        strict=True,
    ):
        if input_modes == output_modes:
            continue
        stage = getattr(model, name)
        source = cast("GatedPoleExcitationS2DTransition", stage.augmented)
        stage.augmented = rms_matched_shortcut_transition(source)


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported RMS-matched K96 variant: {variant}")
    control.capacity.progressive._assert_model_for_spec(model, SPEC, variant)
    for name, input_modes, output_modes in zip(
        STAGE_NAMES[:3],
        SPEC.excitation_modes[:-1],
        SPEC.excitation_modes[1:],
        strict=True,
    ):
        transition = getattr(model, name).augmented
        if input_modes != output_modes:
            if type(transition) is not RMSMatchedShortcutGatedTransition:
                raise RuntimeError(f"{variant}/{name} lost its RMS-matched shortcut")
        elif isinstance(transition, RMSMatchedShortcutGatedTransition):
            raise RuntimeError(f"{variant}/{name} projected an equal-width shortcut")


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported RMS-matched K96 variant: {variant}")
    model = control._build(control.VARIANT, config)
    _install_rms_matched_shortcuts(model)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    backbone = payload["backbone"]
    backbone["name"] = f"A2D-{VARIANT}"
    backbone["transition"]["merge"] = (
        "RMS-matched semi-orthogonal carry projection plus unchanged raw memory merge"
    )
    payload["experiment"] = {
        "family": "stage3_widened_rms_matched_shortcut",
        "question": (
            "does K14=96 help when width changes only project and RMS-match the "
            "carry while retaining the equal-width memory merge contract"
        ),
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
        runner_source_key="k32_64_96_128_p128_rms_shortcut_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.k32_64_96_128_p128_rms_shortcut_d2262.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={VARIANT: _variant_config()},
        architectures={
            VARIANT: (
                "K=[32,64,96,128], P=[128,128,128,128], D2262, "
                "RMS-matched semi-orthogonal projection shortcuts, unchanged "
                "memory merge, WLPost1.5K, PathH4, terminal raw-Q affine head"
            )
        },
        parameter_counts={VARIANT: parameter_count},
        references={
            "accuracy_anchor": "K128-P160x4-D2262-FullSR14x4",
            "width_control": control.VARIANT,
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
