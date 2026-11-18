#!/usr/bin/env python3
"""Train terminal-reader and PathH controls selected by checkpoint analysis."""

from __future__ import annotations

# The campaign deliberately composes established private builder hooks.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportCallIssue=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_capacity_factory as capacity
import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_same_resolution_all_wl192_pathh4_imagenet100 as control
import run_a2d_r2k3_same_resolution_factorial_imagenet100 as base
import torch
from torch import nn

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_gated_post_fusion import GatedComplexPostFusion
from lnet.pac_phase_gated_transition import PathOnlyCollapse

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


P128_RANDOM_SQUARE = (
    "K128-P128x4-SR56-28-14-7-WLPostH192-PathH4-TermRandomSquare"
)
P96_PATH_H6 = "K128-PF15K-SR56-28-14-7-WLPost-PathH6-TermP96"
P96_PATH_H8 = "K128-PF15K-SR56-28-14-7-WLPost-PathH8-TermP96"
VARIANTS = (P128_RANDOM_SQUARE, P96_PATH_H6, P96_PATH_H8)
SEEDS = runtime.DEFAULT_SEEDS
RESOLUTIONS = base.RESOLUTIONS
WIDTHS = (128,)
STAGE_NAMES = runtime.STAGE_NAMES
SameResolutionFactorialBackbone = base.SameResolutionFactorialBackbone
BASE_VARIANT = base.variant_name(128, RESOLUTIONS)
POST_HIDDEN = control.POST_HIDDEN
JOBS_BY_GPU = {
    0: (P128_RANDOM_SQUARE,),
    1: (P96_PATH_H6, P96_PATH_H8),
}


@dataclass(frozen=True, slots=True)
class FollowupSpec:
    terminal_poles: int
    path_hidden: int
    random_square_reader: bool = False

    width: int = 128
    excitation_width: int = 128
    resolutions: tuple[int, ...] = RESOLUTIONS

    @property
    def descriptor_dim(self) -> int:
        return 3 * 4 * 128 + 4 * self.terminal_poles

    @property
    def q4_dim(self) -> int:
        return 4 * self.terminal_poles


SPECS = {
    P128_RANDOM_SQUARE: FollowupSpec(128, 4, random_square_reader=True),
    P96_PATH_H6: FollowupSpec(96, 6),
    P96_PATH_H8: FollowupSpec(96, 8),
}


def variant_name(width: int, resolutions: tuple[int, ...]) -> str:
    return base.variant_name(width, resolutions)


def _randomize_square_terminal_reader_(model: ComplexScanBackbone) -> None:
    reader = model.terminal.pole_input_projection
    if (
        type(reader) is not FactorizedComplexConv2dReader
        or reader.input_modes != 128
        or reader.output_modes != 128
    ):
        raise TypeError("random-square control requires a square K128 terminal reader")
    with torch.random.fork_rng(devices=[]):
        reader.initialize_semi_orthogonal_(identity_when_square=False)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    try:
        spec = SPECS[variant]
    except KeyError as error:
        raise ValueError(f"unsupported terminal/path follow-up variant: {variant}") from error

    if spec.random_square_reader:
        model = control._build(control.VARIANT, config)
        _randomize_square_terminal_reader_(model)
    else:
        model = base._build(BASE_VARIANT, config)
        if spec.path_hidden != 8:
            control._install_path_hidden(model, spec.path_hidden)
        capacity.resize_terminal_poles_(model, target_poles=spec.terminal_poles)
    runtime.configure(VARIANTS, SEEDS)
    _assert_model(model, variant)
    return model


def _assert_path(collapse: nn.Module, hidden: int, label: str) -> None:
    if (
        type(collapse) is not PathOnlyCollapse
        or collapse.modes != 128
        or collapse.path_count != 4
        or collapse.path_input.output_paths != hidden
        or collapse.path_output.input_paths != hidden
        or collapse.output_paths != 1
    ):
        raise RuntimeError(f"{label} changed the PathH{hidden} contract")


def _assert_model(model: nn.Module, variant: str) -> None:
    spec = SPECS[variant]
    collapses = control._path_collapses(model)
    if len(collapses) != control.PATH_BLOCKS:
        raise RuntimeError(f"{variant} changed the number of path blocks")
    for index, collapse in enumerate(collapses):
        _assert_path(collapse, spec.path_hidden, f"{variant}/path{index}")

    postfusions = control._postfusions(model)
    if len(postfusions) != control.PATH_BLOCKS or any(
        type(owner.post_fusion) is not GatedComplexPostFusion
        or owner.post_fusion.modes != 128
        or owner.post_fusion.hidden_modes != POST_HIDDEN
        for owner in postfusions
    ):
        raise RuntimeError(f"{variant} changed the WL-H192 PostFusion contract")

    terminal = model.terminal
    reader = terminal.pole_input_projection
    if (
        terminal.input_modes != 128
        or terminal.modes != spec.terminal_poles
        or terminal.output_modes is not None
        or type(reader) is not FactorizedComplexConv2dReader
        or reader.input_modes != 128
        or reader.output_modes != spec.terminal_poles
        or reader.rms_reference_modes != spec.terminal_poles
        or model.descriptor_dim != spec.descriptor_dim
        or model.classifier.input_dim != spec.q4_dim
    ):
        raise RuntimeError(f"{variant} changed its terminal Q contract")

    if spec.random_square_reader:
        primary = reader.point_weight_real[:, 0]
        primary_cpu = primary.detach().double().cpu()
        identity = torch.eye(128, dtype=primary_cpu.dtype)
        if torch.equal(primary_cpu, identity):
            raise RuntimeError("random-square reader retained the identity basis")
        torch.testing.assert_close(
            primary_cpu.T @ primary_cpu,
            identity,
            rtol=1.0e-5,
            atol=1.0e-5,
        )


def _variant_config(variant: str) -> dict[str, Any]:
    spec = SPECS[variant]
    payload = deepcopy(control._variant_config())
    payload["backbone"]["name"] = f"A2D-{variant}"
    payload["backbone"]["pole_schedule"] = [128, 128, 128, spec.terminal_poles]
    payload["backbone"]["descriptor_dim"] = spec.descriptor_dim
    payload["backbone"]["pole_input"]["shape_by_stage"][-1] = (
        f"128-to-{spec.terminal_poles}"
    )
    payload["backbone"]["transition"]["path_collapse"] = (
        f"shared per-mode GWL 4-to-{spec.path_hidden}-to-1 with Cartesian SiLU"
    )
    payload["backbone"]["same_resolution_factorial"]["collapse"] = (
        f"shared per-mode GWL 4-to-{spec.path_hidden}-to-1 with Cartesian SiLU"
    )
    payload["backbone"]["descriptor"]["shape_by_stage"][-1] = spec.q4_dim
    payload["backbone"]["pole_initialization"]["modes_per_stage"][-1] = (
        spec.terminal_poles
    )
    payload["backbone"]["pole_initialization"]["radial_levels"][-1] = (
        spec.terminal_poles // 8
    )
    payload["head"] = {
        "descriptor_source": f"terminal raw Q only; final {spec.q4_dim} coordinates",
        "operator": f"BatchNorm{spec.q4_dim}-affine-false-Linear100",
        "auxiliary": False,
    }
    payload["terminal_reader_initialization"] = (
        "joint random orthogonal square basis"
        if spec.random_square_reader
        else "joint fresh semi-orthogonal rectangular basis"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    payload = runtime.base_contract(args)
    payload["schema"] = "lnet.a2d.r2k3.terminal_path_followup.v1"
    payload["evidence_status"] = "all variants require CPU and compiled CUDA full-batch smoke"
    payload["variants"] = list(VARIANTS)
    payload["seeds"] = list(SEEDS)
    payload["variant_configs"] = {variant: _variant_config(variant) for variant in VARIANTS}
    payload["parameter_counts"] = {
        variant: sum(parameter.numel() for parameter in _build(variant, config).parameters())
        for variant in VARIANTS
    }
    payload["architecture"] = {
        P128_RANDOM_SQUARE: (
            "Exact P128 PathH4 control with only the terminal square reader primary "
            "basis changed from identity-centered to one random orthogonal frame."
        ),
        P96_PATH_H6: "Exact terminal-P96 control with all seven path hidden widths set to six.",
        P96_PATH_H8: "Exact terminal-P96 control with all seven path hidden widths set to eight.",
    }
    payload["references"] = {
        "P128_PathH4": {"parameters": 2_693_668, "best_validation_accuracy": 0.8694},
        "P96_PathH4": {"parameters": 2_663_204, "best_validation_accuracy": 0.8746},
        "P128_PathH8": {"parameters": 2_772_516, "best_validation_accuracy": 0.8726},
    }
    payload["source_sha256"]["factorized_reader"] = runtime.digest(
        Path("src/lnet/pac_factorized_complex_scan_reader.py")
    )
    payload["source_sha256"]["terminal_path_followup"] = runtime.digest(Path(__file__))
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
