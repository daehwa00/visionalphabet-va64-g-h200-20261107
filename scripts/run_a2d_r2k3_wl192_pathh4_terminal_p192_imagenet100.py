#!/usr/bin/env python3
"""Train WL-H192 PathH4 with a jointly initialized terminal P192 reader."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false, reportIndexIssue=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_capacity_factory as capacity
import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as control

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_gated_post_fusion import GatedComplexPostFusion
from lnet.pac_phase_gated_transition import PathOnlyCollapse

if TYPE_CHECKING:
    from argparse import Namespace

    from torch import nn

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


BASE_VARIANT = control.VARIANT
VARIANT = "K128-PF15K-SR56-28-14-7-WLPost-PathH4-TermP192"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
STAGE_NAMES = control.STAGE_NAMES
RESOLUTIONS = control.RESOLUTIONS
SameResolutionFactorialBackbone = control.SameResolutionFactorialBackbone
EXCITATION_MODES = 128
TERMINAL_POLES = 192
DESCRIPTOR_DIM = 3 * 4 * EXCITATION_MODES + 4 * TERMINAL_POLES
Q4_DIM = 4 * TERMINAL_POLES
JOBS_BY_GPU = {1: VARIANTS}


@dataclass(frozen=True, slots=True)
class TerminalP192Spec:
    width: int = EXCITATION_MODES
    excitation_width: int = EXCITATION_MODES
    descriptor_dim: int = DESCRIPTOR_DIM
    resolutions: tuple[int, ...] = RESOLUTIONS


SPECS = {VARIANT: TerminalP192Spec()}


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    if variant != VARIANT:
        raise ValueError(f"unsupported terminal-P192 variant: {variant}")
    model = control._build(BASE_VARIANT, config)
    capacity.resize_terminal_poles_(model, target_poles=TERMINAL_POLES)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model)
    return model


def _assert_model(model: nn.Module, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported terminal-P192 variant: {variant}")
    collapses = control._path_collapses(model)
    if len(collapses) != 7 or any(
        type(collapse) is not PathOnlyCollapse
        or collapse.path_input.output_paths != control.PATH_HIDDEN
        for collapse in collapses
    ):
        raise RuntimeError("terminal-P192 changed the established PathH4 blocks")
    postfusions = control._postfusions(model)
    if len(postfusions) != 7 or any(
        type(owner.post_fusion) is not GatedComplexPostFusion
        or owner.post_fusion.hidden_modes != control.POST_HIDDEN
        for owner in postfusions
    ):
        raise RuntimeError("terminal-P192 changed the established WL-H192 PostFusion")

    terminal = model.terminal
    reader = terminal.pole_input_projection
    if (
        terminal.input_modes != EXCITATION_MODES
        or terminal.modes != TERMINAL_POLES
        or terminal.output_modes is not None
        or type(reader) is not FactorizedComplexConv2dReader
        or reader.input_modes != EXCITATION_MODES
        or reader.output_modes != TERMINAL_POLES
        or reader.rms_reference_modes != TERMINAL_POLES
    ):
        raise RuntimeError("terminal-P192 reader/scan contract changed")
    classifier = model.classifier
    if (
        type(classifier) is not capacity.CapacityQ4OnlyAffineClassifier
        or classifier.full_descriptor_dim != DESCRIPTOR_DIM
        or classifier.q4_dim != Q4_DIM
        or classifier.input_dim != Q4_DIM
    ):
        raise RuntimeError("terminal-P192 Q4 head contract changed")
    if model.descriptor_dim != DESCRIPTOR_DIM:
        raise RuntimeError("terminal-P192 descriptor width changed")
    if model.same_resolution_blocks["7"].pole_modes != EXCITATION_MODES:
        raise RuntimeError("terminal-P192 changed the nonterminal SR7 block")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = f"A2D-{VARIANT}"
    payload["backbone"]["pole_schedule"] = [128, 128, 128, TERMINAL_POLES]
    payload["backbone"]["descriptor_dim"] = DESCRIPTOR_DIM
    payload["backbone"]["descriptor"]["shape_by_stage"] = [512, 512, 512, Q4_DIM]
    payload["backbone"]["pole_input"]["shape_by_stage"] = [
        "128-to-128",
        "128-to-128",
        "128-to-128",
        f"128-to-{TERMINAL_POLES}",
    ]
    payload["backbone"]["terminal_initialization"] = {
        "policy": "joint fresh initialization",
        "reader": "single semi-orthogonal 128-to-192 frame",
        "poles": "single calibrated 192-pole atlas",
        "rms_reference_modes": TERMINAL_POLES,
        "head": "standard initialization over all Q4 coordinates",
    }
    payload["head"] = {
        "descriptor_source": f"terminal raw Q4 only; final {Q4_DIM} coordinates",
        "operator": f"BatchNorm{Q4_DIM}-affine-false-Linear100",
        "auxiliary": False,
    }
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    payload = runtime.base_contract(args)
    config = runtime.model_config()
    payload["schema"] = "lnet.a2d.r2k3.wl192_pathh4_terminal_p192.v1"
    payload["evidence_status"] = "CPU and compiled CUDA smoke required before training"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {VARIANT: _variant_config()}
    payload["parameter_counts"] = {
        VARIANT: sum(parameter.numel() for parameter in _build(VARIANT, config).parameters())
    }
    payload["references"] = {
        "terminal_P128_control": {"variant": BASE_VARIANT, "parameters": 2_693_668}
    }
    payload["architecture"] = {
        VARIANT: (
            "WL-H192 PathH4 with terminal K128-to-P192 reader, pole bank, RMS "
            "calibration, and Q4 head jointly initialized from scratch."
        )
    }
    payload["source_sha256"]["terminal_pole_resize"] = runtime.digest(
        Path("scripts/a2d_r2k3_capacity_factory.py")
    )
    payload["source_sha256"]["terminal_p192_runner"] = runtime.digest(Path(__file__))
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
