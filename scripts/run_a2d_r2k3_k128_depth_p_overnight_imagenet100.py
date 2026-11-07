#!/usr/bin/env python3
"""Run the K128 depth, terminal-pole, and stage-pole response surface."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as capacity_insight
import run_a2d_r2k3_d2262_p160_followup_imagenet100 as anchor
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import torch
from torch import nn

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


D2242_P160 = "K128-P160x4-D2242"
D2262_P128_TERM = "K128-P160-160-160-128-D2262"
D2282_P128_TERM = "K128-P160-160-160-128-D2282"
D2262_P1_128 = "K128-P128-160-160-96-D2262"
D2262_P3_128 = "K128-P160-160-128-96-D2262"
D2282_P160 = "K128-P160x4-D2282"
D2282_P96_TERM = "K128-P160-160-160-96-D2282"
D2262_P2_128 = "K128-P160-128-160-96-D2262"
D2262_P123_128 = "K128-P128-128-128-96-D2262"
VARIANTS = (
    D2242_P160,
    D2262_P128_TERM,
    D2282_P128_TERM,
    D2262_P1_128,
    D2262_P3_128,
    D2282_P160,
    D2282_P96_TERM,
    D2262_P2_128,
    D2262_P123_128,
)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {
    0: (D2242_P160, D2262_P128_TERM, D2282_P128_TERM, D2262_P1_128, D2262_P3_128),
    1: (D2282_P160, D2282_P96_TERM, D2262_P2_128, D2262_P123_128),
}
SMOKE_VARIANTS = (D2242_P160, D2282_P96_TERM)
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = anchor.RESOLUTIONS
STAGE_NAMES = anchor.STAGE_NAMES
SameResolutionFactorialBackbone = anchor.SameResolutionFactorialBackbone


def _spec(
    poles: tuple[int, int, int, int],
    *,
    stage3_depth: int,
) -> allocation.StageAllocationSpec:
    if stage3_depth not in (4, 6, 8):
        raise ValueError(f"unsupported Stage-3 depth: {stage3_depth}")
    return allocation.StageAllocationSpec(
        excitation_modes=(128,) * 4,
        pole_modes=poles,
        extra_blocks=(0, 0, stage3_depth - 2, 0),
        family="k128_depth_p_overnight",
    )


SPECS = {
    D2242_P160: _spec((160, 160, 160, 160), stage3_depth=4),
    D2262_P128_TERM: _spec((160, 160, 160, 128), stage3_depth=6),
    D2282_P128_TERM: _spec((160, 160, 160, 128), stage3_depth=8),
    D2262_P1_128: _spec((128, 160, 160, 96), stage3_depth=6),
    D2262_P3_128: _spec((160, 160, 128, 96), stage3_depth=6),
    D2282_P160: _spec((160, 160, 160, 160), stage3_depth=8),
    D2282_P96_TERM: _spec((160, 160, 160, 96), stage3_depth=8),
    D2262_P2_128: _spec((160, 128, 160, 96), stage3_depth=6),
    D2262_P123_128: _spec((128, 128, 128, 96), stage3_depth=6),
}
REFERENCE_SPEC = anchor.SPEC


def _build_spec(
    spec: allocation.StageAllocationSpec,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    model = capacity_insight._build_all_resolution(spec.as_insight_spec(), config)
    return allocation._append_repeated_blocks(model, spec)


@torch.no_grad()
def _pair_common_initialization_(
    target: ComplexScanBackbone,
    reference: ComplexScanBackbone,
) -> int:
    source = reference.state_dict()
    copied = 0
    for name, value in target.state_dict().items():
        source_value = source.get(name)
        if source_value is None or source_value.shape != value.shape:
            continue
        value.copy_(source_value)
        copied += 1
    if copied == 0:
        raise RuntimeError("K128 depth/P pairing found no common state")
    return copied


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported K128 depth/P variant: {variant}") from error
    initial_seed = torch.initial_seed()
    model = _build_spec(spec, config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initial_seed)
        reference = _build_spec(REFERENCE_SPEC, config)
    _pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    if set(blocks) != {"56", "28", "14", "7"}:
        raise RuntimeError(f"{variant} changed the terminal-depth-2 block set")
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    expected_extra = spec.extra_blocks[2]
    if set(extras) != {"14"} or len(extras["14"]) != expected_extra:
        raise RuntimeError(f"{variant} changed its Stage-3 depth")
    for name in STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if type(transition) is not GatedPoleExcitationS2DTransition:
            raise RuntimeError(f"{variant}/{name} changed the established transition")
        if transition.carry_projection is not None:
            raise RuntimeError(f"{variant}/{name} lost its K128 identity carry")
    terminal = model.terminal
    if terminal.modes != spec.pole_modes[-1]:
        raise RuntimeError(f"{variant} terminal poles changed")
    reader = terminal.pole_input_projection
    if reader is None or reader.output_modes != spec.pole_modes[-1]:
        raise RuntimeError(f"{variant} terminal reader width changed")


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = capacity_insight._variant_config_for_spec(variant, spec.as_insight_spec())
    payload["experiment"] = {
        "family": spec.family,
        "depth": list(spec.depth),
        "pole_modes": list(spec.pole_modes),
        "control": anchor.VARIANT,
        "paired_initialization": "same-name, same-shape tensors match K128/P160/D2262",
        "only_axes": ["Stage-3 depth", "stage pole allocation", "terminal poles"],
    }
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
        runner_source_key="k128_depth_p_overnight_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.k128_depth_p_overnight.v1",
        evidence_status="all CPU contracts and representative compiled CUDA full-batch smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: f"K128, P={list(SPECS[variant].pole_modes)}, D={list(SPECS[variant].depth)}"
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={"accuracy_anchor": anchor.VARIANT, "anchor_best": 0.8828},
    )
    payload["jobs_by_gpu"] = {str(gpu): list(jobs) for gpu, jobs in JOBS_BY_GPU.items()}
    payload["smoke_variants"] = list(SMOKE_VARIANTS)
    return payload


def main() -> None:
    runtime.run(variants=VARIANTS, seeds=SEEDS, build_model=_build, contract=_contract)


if __name__ == "__main__":
    main()
