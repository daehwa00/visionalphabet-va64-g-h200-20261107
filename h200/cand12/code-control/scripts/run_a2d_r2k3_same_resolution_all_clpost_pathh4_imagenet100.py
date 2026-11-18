#!/usr/bin/env python3
"""Train the K128 all-resolution CLPost model with a four-path hidden width."""

from __future__ import annotations

# The frozen campaign runner intentionally exposes private construction hooks.
# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as clpost

from lnet.pac_gated_post_fusion import GatedStrictComplexPostFusion
from lnet.pac_phase_gated_transition import PathOnlyCollapse

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


BASE_VARIANT = clpost.VARIANT
VARIANT = "K128-PF15K-SR56-28-14-7-CLPost-GWL4to4to1"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
SPECS = {VARIANT: clpost.SPECS[BASE_VARIANT]}
STAGE_NAMES = clpost.STAGE_NAMES
RESOLUTIONS = clpost.RESOLUTIONS
SameResolutionFactorialBackbone = clpost.SameResolutionFactorialBackbone
PATH_HIDDEN = 4
PATH_BLOCKS = 7


def _path_collapses(model: Any) -> tuple[Any, ...]:
    main = tuple(
        getattr(model, name).quadrant_path_mode_combiner
        for name in STAGE_NAMES[:3]
    )
    same_resolution = tuple(
        block.path_collapse for block in model.same_resolution_blocks.values()
    )
    return (*main, *same_resolution)


def _install_path_hidden_four(model: Any) -> None:
    for name in STAGE_NAMES[:3]:
        stage = getattr(model, name)
        stage.quadrant_path_mode_combiner = PathOnlyCollapse(
            stage.modes,
            path_hidden=PATH_HIDDEN,
        )
    for block in model.same_resolution_blocks.values():
        block.path_collapse = PathOnlyCollapse(
            block.modes,
            path_hidden=PATH_HIDDEN,
        )


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return campaign.build_derived(
        requested=variant,
        variant=VARIANT,
        base_variant=BASE_VARIANT,
        config=config,
        parent_build=clpost._build,
        install=_install_path_hidden_four,
        assert_model=_assert_model,
        variants=VARIANTS,
        seeds=SEEDS,
    )


def _assert_model(model: Any, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported path-H4 CLPost variant: {variant}")
    collapses = _path_collapses(model)
    if len(collapses) != PATH_BLOCKS:
        raise RuntimeError("path-H4 CLPost must expose seven path collapses")
    if any(
        type(collapse) is not PathOnlyCollapse
        or collapse.modes != 128
        or collapse.path_count != 4
        or collapse.path_input.output_paths != PATH_HIDDEN
        or collapse.path_output.input_paths != PATH_HIDDEN
        or collapse.output_paths != 1
        for collapse in collapses
    ):
        raise RuntimeError("path-H4 CLPost lost its GWL 4-to-4-to-1 contract")
    postfusions = clpost._postfusions(model)
    if len(postfusions) != PATH_BLOCKS or any(
        type(owner.post_fusion) is not GatedStrictComplexPostFusion
        or owner.post_fusion.modes != 128
        or owner.post_fusion.hidden_modes != 192
        for owner in postfusions
    ):
        raise RuntimeError("path-H4 CLPost changed the established PostFusion")
    if model.descriptor_dim != 2048 or model.classifier.input_dim != 512:
        raise RuntimeError("path-H4 CLPost changed the established Raw-Q head")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(clpost._variant_config())
    payload["backbone"]["name"] = f"A2D-{VARIANT}"
    payload["backbone"]["transition"]["path_collapse"] = (
        "shared per-mode GWL 4-to-4-to-1 with Cartesian SiLU"
    )
    payload["backbone"]["same_resolution_factorial"]["collapse"] = (
        "shared per-mode fused GWL 4-to-4-to-1 Cartesian-SiLU"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    return campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="pathh4_runner",
        variant=VARIANT,
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.same_resolution_all_clpost_pathh4.v1",
        evidence_status="CPU and compiled CUDA smoke required before training",
        variant_config=_variant_config(),
        parameter_count=sum(parameter.numel() for parameter in _build(VARIANT, config).parameters()),
        architecture="The completed K128 PF1.5K all-resolution strict CLPost model with all "
        "seven per-mode path hidden widths reduced from eight to four; every "
        "reader, scan, pole, carry, PostFusion, Raw-Q head, optimizer, and "
        "recipe setting is fixed.",
        references={
            "path_h8_clpost": {
                "variant": BASE_VARIANT,
                "wandb_run": "d581f91bbd3359f8",
                "parameters": 2_084_388,
                "final_validation_accuracy": 0.8642,
            },
            "direct_gwl_h0": {
                "variant": "K128-PF15K-SR56-28-14-7-CLPost-GWL4to1",
                "wandb_run": "19622d30cc7ba6a0",
                "epoch6_validation_accuracy": 0.4438,
                "matched_baseline_epoch6_accuracy": 0.4740,
            },
        },
        extra_sources={
            "phase_gated_transition": Path("src/lnet/pac_phase_gated_transition.py"),
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
