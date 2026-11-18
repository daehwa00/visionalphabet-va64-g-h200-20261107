#!/usr/bin/env python3
"""Complete the K96/K128 pole-ratio screen on the two qlab GPUs."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_k_family_wave_a_imagenet100 as family
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import torch

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


POLICIES: dict[str, tuple[int, tuple[int, int, int, int], str]] = {
    "XL-K96-U1": (96, (96, 96, 96, 96), "U1.0"),
    "XL-K96-U125": (96, (128, 128, 128, 128), "U1.25"),
    "XL-K96-Shaped": (96, (144, 192, 144, 144), "Shaped"),
    "XL-K96-Rich": (96, (144, 192, 192, 144), "Shaped-Rich"),
    "XXL-K128-U1": (128, (128, 128, 128, 128), "U1.0"),
    "XXL-K128-U125": (128, (160, 160, 160, 160), "U1.25"),
}
VARIANTS = tuple(POLICIES)
VARIANT = VARIANTS[0]
JOBS_BY_GPU = {
    0: ("XL-K96-U1", "XL-K96-U125", "XXL-K128-U1"),
    1: ("XL-K96-Shaped", "XL-K96-Rich", "XXL-K128-U125"),
}
SMOKE_VARIANTS = ("XL-K96-U1", "XXL-K128-U125")
SEEDS = family.SEEDS
EXTRA_BLOCKS = family.EXTRA_BLOCKS
RESOLUTIONS = family.RESOLUTIONS
STAGE_NAMES = family.STAGE_NAMES
SameResolutionFactorialBackbone = family.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = family.POST_HIDDEN_RATIO
SPECS = {
    variant: allocation.StageAllocationSpec(
        excitation_modes=(width,) * 4,
        pole_modes=poles,
        extra_blocks=EXTRA_BLOCKS,
        family="uniform_k_family_wave_b_qlab",
    )
    for variant, (width, poles, _policy) in POLICIES.items()
}
REFERENCE_SPECS = {
    width: allocation.StageAllocationSpec(
        excitation_modes=(width,) * 4,
        pole_modes=(width,) * 4,
        extra_blocks=EXTRA_BLOCKS,
        family="uniform_k_family_reference",
    )
    for width in (96, 128)
}


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    family.capacity._assert_model_for_spec(model, SPECS[variant], variant)
    for name in family.STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if type(transition) is not family.GatedPoleExcitationS2DTransition:
            raise RuntimeError(f"{variant}/{name} changed the established raw merge")
        if transition.carry_projection is not None:
            raise RuntimeError(f"{variant}/{name} lost its uniform-K identity carry")


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        width, _poles, _policy = POLICIES[variant]
    except KeyError as error:
        raise ValueError(f"unsupported K-family Wave-B variant: {variant}") from error
    initial_seed = torch.initial_seed()
    model = family._build_spec(SPECS[variant], config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initial_seed)
        reference = family._build_spec(REFERENCE_SPECS[width], config)
    family._pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    width, poles, policy = POLICIES[variant]
    payload = family.capacity._variant_config_for_spec(variant, SPECS[variant])
    payload["experiment"] = {
        "family": "uniform_k_family_wave_b_qlab",
        "policy": policy,
        "excitation_modes": [width] * 4,
        "pole_modes": list(poles),
        "depth": [2, 2, 6, 2],
        "paired_initialization": "same-shape tensors match the uniform-P=K reference",
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
        runner_source_key="uniform_k_family_wave_b_qlab_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.uniform_k_family_wave_b_qlab.v1",
        evidence_status="representative compiled CUDA smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={[POLICIES[variant][0]] * 4}, P={list(POLICIES[variant][1])}, "
                "D2262, raw identity-carry merge, WLPost1.5K, PathH4, Q4 affine"
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={"predecessor": "uniform K-family Wave-A on qlab"},
    )
    payload["jobs_by_gpu"] = {str(gpu): list(variants) for gpu, variants in JOBS_BY_GPU.items()}
    payload["smoke_variants"] = list(SMOKE_VARIANTS)
    return payload


def main() -> None:
    runtime.run(variants=VARIANTS, seeds=SEEDS, build_model=_build, contract=_contract)


if __name__ == "__main__":
    main()
