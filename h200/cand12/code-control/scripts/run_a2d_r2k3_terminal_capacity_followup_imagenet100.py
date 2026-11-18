#!/usr/bin/env python3
"""Isolate terminal depth and pole capacity on the K128/P160/D2262 anchor."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_capacity_factory as capacity_factory
import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as capacity_insight
import run_a2d_r2k3_d2262_p160_followup_imagenet100 as anchor
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import torch
from torch import nn

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


D2261 = "K128-P160x4-D2261-TermDepth1"
D2281 = "K128-P160x4-D2281-TermDepth1"
TERM_P96 = "K128-P160-160-160-96-D2262-TermP96"
VARIANTS = (D2261, D2281, TERM_P96)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {0: (D2261, TERM_P96), 1: (D2281,)}
SMOKE_VARIANTS = (D2261, TERM_P96)
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = anchor.RESOLUTIONS
STAGE_NAMES = anchor.STAGE_NAMES
SameResolutionFactorialBackbone = anchor.SameResolutionFactorialBackbone
SPECS = {
    D2261: allocation.StageAllocationSpec(
        (128,) * 4, (160,) * 4, extra_blocks=(0, 0, 4, 0), family="terminal_capacity"
    ),
    D2281: allocation.StageAllocationSpec(
        (128,) * 4, (160,) * 4, extra_blocks=(0, 0, 6, 0), family="terminal_capacity"
    ),
    TERM_P96: allocation.StageAllocationSpec(
        (128,) * 4,
        (160, 160, 160, 96),
        extra_blocks=(0, 0, 4, 0),
        family="terminal_capacity",
    ),
}
REFERENCE_SPEC = anchor.SPEC


def _build_repeated(
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
        raise RuntimeError("terminal-capacity pairing found no common state")
    return copied


def _remove_sr7_(model: ComplexScanBackbone) -> None:
    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    if "7" not in blocks:
        raise RuntimeError("terminal depth-1 transform found no SR7 block")
    del blocks["7"]


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant not in SPECS:
        raise ValueError(f"unsupported terminal-capacity variant: {variant}")
    initial_seed = torch.initial_seed()
    if variant == D2281:
        model = _build_repeated(SPECS[variant], config)
        _remove_sr7_(model)
    else:
        model = _build_repeated(REFERENCE_SPEC, config)
        if variant == D2261:
            _remove_sr7_(model)
        else:
            capacity_factory.resize_terminal_poles_(model, target_poles=96)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initial_seed)
        reference = _build_repeated(REFERENCE_SPEC, config)
    _pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    expected_blocks = {"56", "28", "14"} if variant in (D2261, D2281) else {
        "56", "28", "14", "7"
    }
    if set(blocks) != expected_blocks:
        raise RuntimeError(f"{variant} same-resolution blocks changed: {set(blocks)}")
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    expected_extra = 6 if variant == D2281 else 4
    if set(extras) != {"14"} or len(extras["14"]) != expected_extra:
        raise RuntimeError(f"{variant} changed its SR14 depth")
    expected_terminal = 96 if variant == TERM_P96 else 160
    terminal = model.terminal
    if terminal.modes != expected_terminal:
        raise RuntimeError(f"{variant} terminal poles changed: {terminal.modes}")
    reader = terminal.pole_input_projection
    if reader is None or reader.output_modes != expected_terminal:
        raise RuntimeError(f"{variant} terminal reader width changed")


def _depth(variant: str) -> tuple[int, int, int, int]:
    return (2, 2, 8, 1) if variant == D2281 else ((2, 2, 6, 1) if variant == D2261 else (2, 2, 6, 2))


def _variant_config(variant: str) -> dict[str, Any]:
    payload = deepcopy(
        capacity_insight._variant_config_for_spec(variant, SPECS[variant].as_insight_spec())
    )
    payload["experiment"] = {
        "family": "terminal_capacity_followup",
        "depth": list(_depth(variant)),
        "pole_modes": list(SPECS[variant].pole_modes),
        "control": anchor.VARIANT,
        "paired_initialization": "all same-name, same-shape tensors match D2262/P160",
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
        runner_source_key="terminal_capacity_followup_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.terminal_capacity_followup.v1",
        evidence_status="paired CPU structure checks and representative compiled CUDA smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: f"K128, P={list(SPECS[variant].pole_modes)}, D={list(_depth(variant))}"
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
