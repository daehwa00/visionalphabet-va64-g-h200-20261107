#!/usr/bin/env python3
"""Train the all-resolution K128/P256 strict-CLPost PF2K candidate."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_capacity_factory as capacity
import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as clpost
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as factorial
import run_a2d_r2k3_uniform_prenorm_gated_postfusion_imagenet100 as uniform
import torch
from torch import nn

from lnet.pac_complex_layers import ComplexLinear, PackedComplexLinear
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    GatedStrictComplexPostFusion,
)
from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


VARIANT = "K128-P256-PF2K-SR56-28-14-7-CLPost"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
STAGE_NAMES = runtime.STAGE_NAMES
RESOLUTIONS = factorial.RESOLUTIONS
SameResolutionFactorialBackbone = factorial.SameResolutionFactorialBackbone
EXCITATION_MODES = 128
POLE_MODES = 256
POST_HIDDEN = 256
SPEC = capacity.CapacitySpec(
    excitation_modes=(EXCITATION_MODES,) * 4,
    pole_modes=(POLE_MODES,) * 4,
    post_hidden_ratio=2.0,
)


@dataclass(frozen=True, slots=True)
class SmokeSpec:
    width: int
    excitation_width: int
    descriptor_dim: int
    resolutions: tuple[int, ...]


SPECS = {
    VARIANT: SmokeSpec(
        width=EXCITATION_MODES,
        excitation_width=EXCITATION_MODES,
        descriptor_dim=SPEC.descriptor_dim,
        resolutions=RESOLUTIONS,
    )
}


def _make_block(
    *,
    resolution: int,
    pole_template: Any,
) -> SameResolutionPoleScanBlock:
    initial_seed = torch.initial_seed()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((initial_seed + resolution * 1_001) % (2**63 - 1))
        return SameResolutionPoleScanBlock(
            EXCITATION_MODES,
            pole_modes=POLE_MODES,
            reader_rank=capacity.READER_RANK,
            kernel_size=capacity.KERNEL_SIZE,
            pole_template=pole_template,
            post_hidden=POST_HIDDEN,
        )


def _postfusions(model: nn.Module) -> tuple[Any, ...]:
    return clpost._postfusions(model)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported K128/P256 PF2K variant: {variant}")
    model = capacity._build_spec(SPEC, config)
    uniform._install_combined_transition(model)
    templates = {
        56: model.stage1,
        28: model.stage2,
        14: cast("Any", model).stage3,
        7: model.terminal,
    }
    blocks = {
        resolution: _make_block(
            resolution=resolution,
            pole_template=templates[resolution],
        )
        for resolution in RESOLUTIONS
    }
    model = SameResolutionFactorialBackbone(model, blocks)
    clpost._install_strict_postfusion(model)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model)
    return model


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported K128/P256 PF2K variant: {variant}")
    if model.descriptor_dim != SPEC.descriptor_dim:
        raise RuntimeError("K128/P256 model changed its 4096-D Raw-Q descriptor")
    if len(_postfusions(model)) != 7:
        raise RuntimeError("K128/P256 model must expose seven PostFusion blocks")
    for name in STAGE_NAMES:
        stage = getattr(model, name)
        reader = stage.pole_input_projection
        if (
            stage.input_modes != EXCITATION_MODES
            or stage.modes != POLE_MODES
            or type(reader) is not FactorizedComplexConv2dReader
            or reader.input_modes != EXCITATION_MODES
            or reader.output_modes != POLE_MODES
        ):
            raise RuntimeError(f"{name} lost its K128-to-P256 reader contract")
        if name == "terminal":
            if reader.normalize_input or not reader.match_input_rms:
                raise RuntimeError("terminal lost its RMS-matched reader contract")
            continue
        if not reader.normalize_input or reader.match_input_rms:
            raise RuntimeError(f"{name} lost its pre-normalized reader contract")
        transition = stage.augmented
        if (
            type(transition) is not GatedPoleExcitationS2DTransition
            or transition.input_modes != POLE_MODES
            or transition.excitation_modes != EXCITATION_MODES
            or transition.output_modes != EXCITATION_MODES
            or type(transition.memory_projection) is not ComplexLinear
            or type(transition.post_fusion) is not GatedStrictComplexPostFusion
            or transition.post_fusion.hidden_modes != POST_HIDDEN
        ):
            raise RuntimeError(f"{name} lost its P256-to-K128 PF2K transition")

    blocks = cast("nn.ModuleDict", model.same_resolution_blocks)
    if set(blocks) != {str(resolution) for resolution in RESOLUTIONS}:
        raise RuntimeError("K128/P256 model lost a same-resolution block")
    for block in blocks.values():
        if (
            type(block) is not SameResolutionPoleScanBlock
            or block.modes != EXCITATION_MODES
            or block.pole_modes != POLE_MODES
            or block.pole_scan.modes != POLE_MODES
            or type(block.memory_projection) is not ComplexLinear
            or type(block.path_collapse) is not PathOnlyCollapse
            or block.path_collapse.modes != POLE_MODES
            or type(block.post_fusion) is not GatedStrictComplexPostFusion
            or type(block.post_fusion.value) is not PackedComplexLinear
            or block.post_fusion.hidden_modes != POST_HIDDEN
        ):
            raise RuntimeError("same-resolution block lost its K128/P256 PF2K contract")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(capacity._variant_config_for_spec(VARIANT, SPEC))
    payload["backbone"]["same_resolution"] = {
        "resolutions": list(RESOLUTIONS),
        "reader": "pre-CRMSNorm rank-2 strict-complex K3 128-to-256",
        "scan": "fused associative D4 scan-to-path-collapse at P=256",
        "memory_adapter": "strict ComplexLinear 256-to-128",
        "carry": "identity K=128 excitation",
    }
    payload["backbone"]["transition"]["post_fusion"] = {
        "operator": "packed strict-complex PF2K",
        "modes": EXCITATION_MODES,
        "hidden_modes": POST_HIDDEN,
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    return campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="k128_p256_runner",
        variant=VARIANT,
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.k128_p256_pf2k_all.v1",
        evidence_status="CPU and compiled CUDA smoke required before training",
        variant_config=_variant_config(),
        parameter_count=sum(
            parameter.numel() for parameter in _build(VARIANT, config).parameters()
        ),
        architecture=(
            "The optimized PF2K control with K=128 excitation and P=256 poles at all "
            "four stages; all spatial depths, readers ranks, carries, heads, and recipe "
            "settings are fixed relative to the P=128 control."
        ),
        references={
            "k128_p128_pf2k": {
                "variant": "K128-PF2K-SR56-28-14-7-CLPost",
                "wandb_run": "f4a208d7e0977d58",
                "parameters": 2_428_452,
            }
        },
        extra_sources={
            "same_resolution_stage": Path("src/lnet/pac_same_resolution_depth.py"),
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
