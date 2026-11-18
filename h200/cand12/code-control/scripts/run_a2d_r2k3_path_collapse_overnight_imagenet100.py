#!/usr/bin/env python3
"""Train the controlled K128 path-collapse overnight candidates."""

from __future__ import annotations

# The frozen campaign runner intentionally exposes private construction hooks.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_same_resolution_all_cl256_imagenet100 as pf2k
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as pf15k

from lnet.pac_gated_post_fusion import GatedStrictComplexPostFusion
from lnet.pac_phase_gated_transition import (
    DirectMagnitudePathCollapse,
    PathOnlyCollapse,
    PathPhaseMagnitudeCollapse,
    PathSelfMagnitudeCollapse,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


PF15K_PATH_H4 = "K128-PF15K-All-PathH4-CSiLU"
PF15K_PATH_PG_H4 = "K128-PF15K-All-PathPGH4-Mean1"
PF15K_PATH_MAG_H4 = "K128-PF15K-All-PathMagH4-Mean1"
PF2K_PATH_H4 = "K128-PF2K-All-PathH4-CSiLU"
PF2K_PATH_PG_H4 = "K128-PF2K-All-PathPGH4-Mean1"
PF15K_DIRECT_MAG = "K128-PF15K-All-MagGateDirect4to1"

VARIANTS = (
    PF15K_PATH_H4,
    PF15K_PATH_PG_H4,
    PF15K_PATH_MAG_H4,
    PF2K_PATH_H4,
    PF2K_PATH_PG_H4,
    PF15K_DIRECT_MAG,
)
VARIANT = VARIANTS[0]
SEEDS = runtime.DEFAULT_SEEDS
STAGE_NAMES = pf15k.STAGE_NAMES
RESOLUTIONS = pf15k.RESOLUTIONS
SameResolutionFactorialBackbone = pf15k.SameResolutionFactorialBackbone
PATH_HIDDEN = 4
PATH_BLOCKS = 7
SPECS = dict.fromkeys(VARIANTS, pf15k.SPECS[pf15k.VARIANT])

JOBS_BY_GPU = {
    0: (
        PF2K_PATH_H4,
        PF2K_PATH_PG_H4,
        PF15K_DIRECT_MAG,
    ),
    1: (
        PF15K_PATH_H4,
        PF15K_PATH_PG_H4,
        PF15K_PATH_MAG_H4,
    ),
}


def _path_collapses(model: nn.Module) -> tuple[Any, ...]:
    main = tuple(
        getattr(model, name).quadrant_path_mode_combiner
        for name in STAGE_NAMES[:3]
    )
    same_resolution = tuple(
        block.path_collapse for block in model.same_resolution_blocks.values()
    )
    return (*main, *same_resolution)


def _make_collapse(variant: str, modes: int) -> nn.Module:
    if variant in (PF15K_PATH_H4, PF2K_PATH_H4):
        return PathOnlyCollapse(modes, path_hidden=PATH_HIDDEN)
    if variant in (PF15K_PATH_PG_H4, PF2K_PATH_PG_H4):
        return PathPhaseMagnitudeCollapse(modes, path_hidden=PATH_HIDDEN)
    if variant == PF15K_PATH_MAG_H4:
        return PathSelfMagnitudeCollapse(modes, path_hidden=PATH_HIDDEN)
    if variant == PF15K_DIRECT_MAG:
        return DirectMagnitudePathCollapse(modes)
    raise ValueError(f"unsupported path-collapse variant: {variant}")


def _install_collapses(model: nn.Module, variant: str) -> None:
    for name in STAGE_NAMES[:3]:
        stage = getattr(model, name)
        stage.quadrant_path_mode_combiner = _make_collapse(variant, stage.modes)
    for block in model.same_resolution_blocks.values():
        block.path_collapse = _make_collapse(variant, block.pole_modes)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported overnight path-collapse variant: {variant}")
    base = pf2k._build(pf2k.VARIANT, config) if variant.startswith("K128-PF2K-") else pf15k._build(
        pf15k.VARIANT,
        config,
    )
    _install_collapses(base, variant)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(base, variant)
    return base


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported overnight path-collapse variant: {variant}")
    collapses = _path_collapses(model)
    if len(collapses) != PATH_BLOCKS:
        raise RuntimeError("overnight candidate must expose seven path collapses")
    expected_type: type[nn.Module]
    if variant in (PF15K_PATH_H4, PF2K_PATH_H4):
        expected_type = PathOnlyCollapse
    elif variant in (PF15K_PATH_PG_H4, PF2K_PATH_PG_H4):
        expected_type = PathPhaseMagnitudeCollapse
    elif variant == PF15K_PATH_MAG_H4:
        expected_type = PathSelfMagnitudeCollapse
    else:
        expected_type = DirectMagnitudePathCollapse
    if any(type(collapse) is not expected_type for collapse in collapses):
        raise RuntimeError("overnight candidate changed its declared collapse operator")
    if any(
        collapse.modes != 128
        or collapse.path_count != 4
        or collapse.output_paths != 1
        for collapse in collapses
    ):
        raise RuntimeError("overnight candidate changed its D4-to-one contract")
    for collapse in collapses:
        if hasattr(collapse, "path_hidden") and collapse.path_hidden != PATH_HIDDEN:
            raise RuntimeError("overnight candidate changed its H4 contract")

    expected_post_hidden = 256 if variant.startswith("K128-PF2K-") else 192
    owners = pf15k._postfusions(model)
    if len(owners) != PATH_BLOCKS or any(
        type(owner.post_fusion) is not GatedStrictComplexPostFusion
        or owner.post_fusion.modes != 128
        or owner.post_fusion.hidden_modes != expected_post_hidden
        for owner in owners
    ):
        raise RuntimeError("overnight candidate changed its PostFusion control")
    if model.descriptor_dim != 2048 or model.classifier.input_dim != 512:
        raise RuntimeError("overnight candidate changed the Raw-Q head")


def _collapse_config(variant: str) -> dict[str, Any]:
    if variant in (PF15K_PATH_H4, PF2K_PATH_H4):
        return {
            "operator": "per-mode GWL 4-to-4-to-1",
            "activation": "Cartesian SiLU",
            "gate": None,
        }
    if variant in (PF15K_PATH_PG_H4, PF2K_PATH_PG_H4):
        return {
            "operator": "per-mode GWL 4-to-8 split value4/detector4, then GWL 4-to-1",
            "activation": None,
            "gate": "detector magnitude; exact mean one over hidden paths",
        }
    if variant == PF15K_PATH_MAG_H4:
        return {
            "operator": "per-mode GWL 4-to-4-to-1",
            "activation": None,
            "gate": "value magnitude; exact mean one over hidden paths",
        }
    return {
        "operator": "raw-path magnitude gate then per-mode GWL 4-to-1",
        "activation": None,
        "gate": "exact mean one over the four input paths",
    }


def _variant_config(variant: str) -> dict[str, Any]:
    source = pf2k._variant_config() if variant.startswith("K128-PF2K-") else pf15k._variant_config()
    payload = deepcopy(source)
    payload["backbone"]["name"] = f"A2D-{variant}"
    payload["backbone"]["transition"]["path_collapse"] = _collapse_config(variant)
    payload["backbone"]["same_resolution_factorial"]["collapse"] = _collapse_config(
        variant
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = runtime.base_contract(args)
    config = runtime.model_config()
    payload["schema"] = "lnet.a2d.r2k3.path_collapse_overnight.v1"
    payload["evidence_status"] = "CPU and compiled full-batch CUDA smoke required"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {
        variant: _variant_config(variant) for variant in VARIANTS
    }
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload["architecture"] = {
        variant: (
            "K=P=128 all-resolution Raw-Q control; only the seven D4 path collapses "
            f"use {_collapse_config(variant)}. PostFusion is "
            f"{'PF2K/H256' if variant.startswith('K128-PF2K-') else 'PF1.5K/H192'}."
        )
        for variant in VARIANTS
    }
    payload["source_sha256"]["phase_gated_transition"] = runtime.digest(
        Path("src/lnet/pac_phase_gated_transition.py")
    )
    payload["source_sha256"]["same_resolution_depth"] = runtime.digest(
        Path("src/lnet/pac_same_resolution_depth.py")
    )
    payload["source_sha256"]["overnight_runner"] = runtime.digest(Path(__file__))
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
