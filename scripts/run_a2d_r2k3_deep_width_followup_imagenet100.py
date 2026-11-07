#!/usr/bin/env python3
"""Compare deep K96/K128 and balanced K128 block allocation."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_k_family_wave_a_imagenet100 as family
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import torch
from torch import nn

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


K128_D22102 = "K128-P160x4-D22102"
K128_D22122 = "K128-P160x4-D22122"
K128_D3393 = "K128-P160x4-D3393"
K96_D22102 = "K96-P128x4-D22102"
K96_D22122 = "K96-P128x4-D22122"
VARIANTS = (K128_D22102, K128_D22122, K128_D3393, K96_D22102, K96_D22122)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {
    0: (K128_D22102, K128_D22122),
    1: (K128_D3393, K96_D22102, K96_D22122),
}
SMOKE_VARIANTS = (K128_D3393, K96_D22122)
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = family.RESOLUTIONS
STAGE_NAMES = runtime.STAGE_NAMES
SameResolutionFactorialBackbone = family.SameResolutionFactorialBackbone


def _spec(
    width: int,
    poles: int,
    extra_blocks: tuple[int, int, int, int],
) -> allocation.StageAllocationSpec:
    return allocation.StageAllocationSpec(
        excitation_modes=(width,) * 4,
        pole_modes=(poles,) * 4,
        extra_blocks=extra_blocks,
        family="deep_width_followup",
    )


SPECS = {
    K128_D22102: _spec(128, 160, (0, 0, 8, 0)),
    K128_D22122: _spec(128, 160, (0, 0, 10, 0)),
    K128_D3393: _spec(128, 160, (1, 1, 7, 1)),
    K96_D22102: _spec(96, 128, (0, 0, 8, 0)),
    K96_D22122: _spec(96, 128, (0, 0, 10, 0)),
}
REFERENCE_SPECS = {
    (128, 160): _spec(128, 160, (0, 0, 4, 0)),
    (96, 128): _spec(96, 128, (0, 0, 4, 0)),
}


def _build_spec(
    spec: allocation.StageAllocationSpec,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    return family._build_spec(spec, config)


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
        raise RuntimeError("deep-width pairing found no common state")
    return copied


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported deep-width variant: {variant}") from error
    width, poles = spec.excitation_modes[0], spec.pole_modes[0]
    initial_seed = torch.initial_seed()
    model = _build_spec(spec, config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initial_seed)
        reference = _build_spec(REFERENCE_SPECS[(width, poles)], config)
    _pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    if set(blocks) != {"56", "28", "14", "7"}:
        raise RuntimeError(f"{variant} changed its canonical block set")
    extras = cast("nn.ModuleDict", model.extra_same_resolution_blocks)
    expected = {
        str(resolution): count
        for resolution, count in zip(RESOLUTIONS, spec.extra_blocks, strict=True)
        if count
    }
    if set(extras) != set(expected):
        raise RuntimeError(f"{variant} changed its extra-block placement")
    if any(len(extras[key]) != count for key, count in expected.items()):
        raise RuntimeError(f"{variant} changed its extra-block count")
    for name in STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if type(transition) is not GatedPoleExcitationS2DTransition:
            raise RuntimeError(f"{variant}/{name} changed the established transition")
        if transition.carry_projection is not None:
            raise RuntimeError(f"{variant}/{name} lost its uniform-K identity carry")
    if len(set(spec.pole_modes)) != 1:
        raise RuntimeError(f"{variant} lost uniform P")


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = family.capacity._variant_config_for_spec(variant, spec)
    payload["experiment"] = {
        "family": spec.family,
        "excitation_modes": list(spec.excitation_modes),
        "pole_modes": list(spec.pole_modes),
        "depth": list(spec.depth),
        "paired_reference_depth": [2, 2, 6, 2],
        "only_controlled_axes": ["uniform width", "uniform poles", "block placement"],
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
        runner_source_key="deep_width_followup_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.deep_width_followup.v1",
        evidence_status="all CPU contracts and representative compiled CUDA full-batch smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={list(SPECS[variant].excitation_modes)}, "
                f"P={list(SPECS[variant].pole_modes)}, D={list(SPECS[variant].depth)}"
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={
            "K128_anchor": "K128-P160x4-D2262-FullSR14x4",
            "K96_anchor": "XL-K96-U125",
        },
    )
    payload["jobs_by_gpu"] = {str(gpu): list(jobs) for gpu, jobs in JOBS_BY_GPU.items()}
    payload["smoke_variants"] = list(SMOKE_VARIANTS)
    return payload


def main() -> None:
    runtime.run(variants=VARIANTS, seeds=SEEDS, build_model=_build, contract=_contract)


if __name__ == "__main__":
    main()
