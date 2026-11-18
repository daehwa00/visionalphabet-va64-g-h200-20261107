#!/usr/bin/env python3
"""Train one-stage-at-a-time depth variants of the K64 P-small anchor."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false, reportUnusedFunction=false
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_front_capacity_reallocation_d2262_imagenet100 as capacity
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import torch

from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


D2242 = "K64-P96-128-128-96-D2242"
D2222 = "K64-P96-128-128-96-D2222"
D2282 = "K64-P96-128-128-96-D2282"
D2362 = "K64-P96-128-128-96-D2362"
D3262 = "K64-P96-128-128-96-D3262"
D3362 = "K64-P96-128-128-96-D3362"
VARIANTS = (D2242, D2222, D2282, D2362, D3262, D3362)
VARIANT = D2242
JOBS_BY_GPU = {
    0: (D2242, D2282, D3262),
    1: (D2222, D2362, D3362),
}
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = capacity.RESOLUTIONS
STAGE_NAMES = capacity.STAGE_NAMES
SameResolutionFactorialBackbone = capacity.SameResolutionFactorialBackbone
POST_HIDDEN_RATIO = capacity.POST_HIDDEN_RATIO
EXTRA_BLOCKS = {
    D2242: (0, 0, 2, 0),
    D2222: (0, 0, 0, 0),
    D2282: (0, 0, 6, 0),
    D2362: (0, 1, 4, 0),
    D3262: (1, 0, 4, 0),
    D3362: (1, 1, 4, 0),
}
REFERENCE_EXTRA_BLOCKS = (0, 0, 4, 0)
EXCITATION_MODES = (64, 64, 64, 64)
POLE_MODES = (96, 128, 128, 96)
SPECS = {
    variant: allocation.StageAllocationSpec(
        excitation_modes=EXCITATION_MODES,
        pole_modes=POLE_MODES,
        extra_blocks=extra_blocks,
        family="k64_p_small_depth_factorial",
    )
    for variant, extra_blocks in EXTRA_BLOCKS.items()
}
REFERENCE_SPEC = allocation.StageAllocationSpec(
    excitation_modes=EXCITATION_MODES,
    pole_modes=POLE_MODES,
    extra_blocks=REFERENCE_EXTRA_BLOCKS,
    family="k64_p_small_depth_reference",
)


def _build_spec(
    spec: allocation.StageAllocationSpec,
    config: ComplexScanConfig,
) -> ComplexScanBackbone:
    return capacity._build_spec_model(spec, config)


@torch.no_grad()
def _pair_common_initialization_(
    target: ComplexScanBackbone,
    reference: ComplexScanBackbone,
) -> int:
    """Pair all role- and shape-matched tensors with the D2262 reference."""
    source_state = reference.state_dict()
    copied = 0
    for name, target_value in target.state_dict().items():
        source_value = source_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            continue
        target_value.copy_(source_value)
        copied += 1
    if copied == 0:
        raise RuntimeError("depth-factorial pairing found no common model state")
    return copied


def _assert_model(model: ComplexScanBackbone, variant: str) -> None:
    spec = SPECS[variant]
    capacity._assert_model_for_spec(model, spec, variant)
    for name in STAGE_NAMES[:3]:
        transition = getattr(model, name).augmented
        if type(transition) is not GatedPoleExcitationS2DTransition:
            raise RuntimeError(f"{variant}/{name} changed the established raw merge")
        if transition.carry_projection is not None:
            raise RuntimeError(f"{variant}/{name} lost its K64 identity carry")


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported K64 depth variant: {variant}") from error
    initial_seed = torch.initial_seed()
    model = _build_spec(spec, config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initial_seed)
        reference = _build_spec(REFERENCE_SPEC, config)
    _pair_common_initialization_(model, reference)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = capacity._variant_config_for_spec(variant, spec)
    payload["backbone"]["transition"]["merge"] = (
        "established raw projected memory plus K64 identity S2D carry"
    )
    payload["experiment"] = {
        "family": "k64_p_small_depth_factorial",
        "question": "which resolution benefits from depth once K and P are fixed",
        "excitation_modes": list(EXCITATION_MODES),
        "pole_modes": list(POLE_MODES),
        "depth": list(spec.depth),
        "paired_initialization": (
            "all same-name, same-shape state tensors match the D2262 reference exactly"
        ),
        "controls": {
            "reference_depth": "D2262",
            "post_hidden_ratio": POST_HIDDEN_RATIO,
            "path_hidden": 4,
            "head": "terminal raw-Q affine",
        },
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
        runner_source_key="k64_p_small_depth_factorial_runner",
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.k64_p_small_depth_factorial.v1",
        evidence_status="CPU and compiled CUDA batch-128 smoke required",
        variant_configs={variant: _variant_config(variant) for variant in VARIANTS},
        architectures={
            variant: (
                f"K={list(EXCITATION_MODES)}, P={list(POLE_MODES)}, "
                f"D={list(SPECS[variant].depth)}, raw identity-carry merge, "
                "WLPost1.5K, PathH4, terminal raw-Q affine head"
            )
            for variant in VARIANTS
        },
        parameter_counts=parameter_counts,
        references={
            "depth_anchor": "K64-P96-128-128-96-D2262",
            "accuracy_anchor": "K128-P160x4-D2262-FullSR14x4",
            "prior_D3362": "K32-64-64-128-P128x4-D3362-NormShortcut",
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
