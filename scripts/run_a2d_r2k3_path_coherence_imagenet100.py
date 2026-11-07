#!/usr/bin/env python3
"""Train the K128 PF1.5K coherence-gated D4 collapse candidate."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as base

from lnet.pac_gated_post_fusion import GatedStrictComplexPostFusion
from lnet.pac_phase_gated_transition import PathCoherenceGateCollapse

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "K128-PF15K-All-CoherenceGate-GWL4to1"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
SPECS = {VARIANT: base.SPECS[base.VARIANT]}
STAGE_NAMES = base.STAGE_NAMES
RESOLUTIONS = base.RESOLUTIONS
SameResolutionFactorialBackbone = base.SameResolutionFactorialBackbone
PATH_BLOCKS = 7


def _path_collapses(model: nn.Module) -> tuple[Any, ...]:
    main = tuple(
        getattr(model, name).quadrant_path_mode_combiner
        for name in STAGE_NAMES[:3]
    )
    same_resolution = tuple(
        block.path_collapse for block in model.same_resolution_blocks.values()
    )
    return (*main, *same_resolution)


def _install(model: nn.Module) -> None:
    for name in STAGE_NAMES[:3]:
        stage = getattr(model, name)
        stage.quadrant_path_mode_combiner = PathCoherenceGateCollapse(stage.modes)
    for block in model.same_resolution_blocks.values():
        block.path_collapse = PathCoherenceGateCollapse(block.pole_modes)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return campaign.build_derived(
        requested=variant,
        variant=VARIANT,
        base_variant=base.VARIANT,
        config=config,
        parent_build=base._build,
        install=_install,
        assert_model=_assert_model,
        variants=VARIANTS,
        seeds=SEEDS,
    )


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported coherence-gated variant: {variant}")
    collapses = _path_collapses(model)
    if len(collapses) != PATH_BLOCKS or any(
        type(collapse) is not PathCoherenceGateCollapse
        or collapse.modes != 128
        or collapse.path_count != 4
        or collapse.output_paths != 1
        for collapse in collapses
    ):
        raise RuntimeError("coherence candidate lost its seven D4-to-one collapses")
    owners = base._postfusions(model)
    if len(owners) != PATH_BLOCKS or any(
        type(owner.post_fusion) is not GatedStrictComplexPostFusion
        or owner.post_fusion.modes != 128
        or owner.post_fusion.hidden_modes != 192
        for owner in owners
    ):
        raise RuntimeError("coherence candidate changed the PF1.5K control")
    if model.descriptor_dim != 2048 or model.classifier.input_dim != 512:
        raise RuntimeError("coherence candidate changed the Raw-Q head")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(base._variant_config())
    payload["backbone"]["name"] = f"A2D-{VARIANT}"
    collapse = {
        "operator": "coherence-gated per-mode GWL 4-to-1",
        "score": "Re(z_d * conj(sum_e A_de z_e)) after path RMS normalization",
        "gate": "bounded exact-mean-one redistribution; A initialized to zero",
        "gram_materialization": False,
    }
    payload["backbone"]["transition"]["path_collapse"] = collapse
    payload["backbone"]["same_resolution_factorial"]["collapse"] = collapse
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    return campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="coherence_runner",
        variant=VARIANT,
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.path_coherence.v1",
        evidence_status="CPU and compiled full-batch CUDA smoke required",
        variant_config=_variant_config(),
        parameter_count=sum(parameter.numel() for parameter in _build(VARIANT, config).parameters()),
        architecture="Exact K128 PF1.5K all-resolution strict-CLPost Raw-Q control with only "
        "the seven D4 path collapses changed to a factorized pairwise-coherence "
        "mean-one gate followed by one per-mode GWL 4-to-1.",
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

