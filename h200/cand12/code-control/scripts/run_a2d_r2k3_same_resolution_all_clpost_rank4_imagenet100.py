#!/usr/bin/env python3
"""Train the K128 all-resolution CLPost control with rank-4 K3 readers."""

from __future__ import annotations

# The frozen campaign runner intentionally exposes private construction hooks.
# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import a2d_r2k3_runtime as runtime
import r2k3_campaign as campaign
import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as clpost

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader

if TYPE_CHECKING:
    from argparse import Namespace

    from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig


BASE_VARIANT = clpost.VARIANT
VARIANT = "K128-R4K3-PF15K-SR56-28-14-7-CLPost"
VARIANTS = (VARIANT,)
SEEDS = runtime.DEFAULT_SEEDS
SPECS = {VARIANT: clpost.SPECS[BASE_VARIANT]}
STAGE_NAMES = clpost.STAGE_NAMES
RESOLUTIONS = clpost.RESOLUTIONS
SameResolutionFactorialBackbone = clpost.SameResolutionFactorialBackbone
READER_RANK = 4
READER_COUNT = 8


def _postfusions(model: Any) -> tuple[Any, ...]:
    return clpost._postfusions(model)


def _rank4_reader(source: FactorizedComplexConv2dReader) -> FactorizedComplexConv2dReader:
    reader = FactorizedComplexConv2dReader(
        source.input_modes,
        source.output_modes,
        rank=READER_RANK,
        kernel_size=source.kernel_size,
        variance_epsilon=source.variance_epsilon,
        normalize_input=source.normalize_input,
        match_input_rms=source.match_input_rms,
    )
    reader.initialize_orthogonal_()
    return reader


def _install_rank4_readers(model: Any) -> None:
    for name in STAGE_NAMES:
        stage = getattr(model, name)
        source = stage.pole_input_projection
        if type(source) is not FactorizedComplexConv2dReader:
            raise TypeError(f"{name} lost the established factorized K3 reader")
        stage.pole_input_projection = _rank4_reader(source)

    blocks = model.same_resolution_blocks
    for resolution, block in blocks.items():
        source = block.reader
        if type(source) is not FactorizedComplexConv2dReader:
            raise TypeError(f"same-resolution block {resolution} lost its K3 reader")
        block.reader = _rank4_reader(source)


def _readers(model: Any) -> tuple[FactorizedComplexConv2dReader, ...]:
    stages = tuple(getattr(model, name).pole_input_projection for name in STAGE_NAMES)
    blocks = tuple(block.reader for block in model.same_resolution_blocks.values())
    return (*stages, *blocks)


def _build(variant: str, config: ComplexScanConfig) -> ComplexScanBackbone:
    return campaign.build_derived(
        requested=variant,
        variant=VARIANT,
        base_variant=BASE_VARIANT,
        config=config,
        parent_build=clpost._build,
        install=_install_rank4_readers,
        assert_model=_assert_model,
        variants=VARIANTS,
        seeds=SEEDS,
    )


def _assert_model(model: Any, variant: str = VARIANT) -> None:
    if variant != VARIANT:
        raise ValueError(f"unsupported rank-4 CLPost variant: {variant}")
    clpost._assert_model(model, BASE_VARIANT)
    if len(_postfusions(model)) != 7:
        raise RuntimeError("rank-4 CLPost changed the seven PostFusion blocks")
    readers = _readers(model)
    if len(readers) != READER_COUNT:
        raise RuntimeError("rank-4 CLPost must expose eight K3 readers")
    for reader in readers:
        if (
            type(reader) is not FactorizedComplexConv2dReader
            or reader.input_modes != 128
            or reader.output_modes != 128
            or reader.rank != READER_RANK
            or reader.kernel_size != 3
        ):
            raise RuntimeError("rank-4 CLPost lost its reader contract")


def _variant_config() -> dict[str, Any]:
    payload = deepcopy(clpost._variant_config())
    payload["backbone"]["name"] = f"A2D-{VARIANT}"
    payload["backbone"]["pole_input"].update(
        operator="rank-4 strict-complex K3 excitation-to-pole reader",
        initialization="identity plus three mutually orthogonal local components",
    )
    payload["backbone"]["same_resolution_factorial"]["reader"] = (
        "pre-CRMSNorm orthogonal rank-4 strict-complex K3"
    )
    return payload


def _contract(args: Namespace) -> dict[str, Any]:
    config = runtime.model_config()
    return campaign.campaign_contract(
        args,
        runner_file=__file__,
        runner_source_key="rank4_runner",
        variant=VARIANT,
        variants=VARIANTS,
        seeds=SEEDS,
        schema="lnet.a2d.r2k3.same_resolution_all_clpost_rank4.v1",
        evidence_status="CPU and compiled CUDA smoke required before training",
        variant_config=_variant_config(),
        parameter_count=sum(parameter.numel() for parameter in _build(VARIANT, config).parameters()),
        architecture="The K128 all-resolution strict CLPost model with all eight factorized "
        "strict-complex K3 readers widened from rank 2 to rank 4; every scan, "
        "pole, transition, PostFusion, head, and training setting is fixed.",
        references={
            "rank2_clpost": {
                "variant": BASE_VARIANT,
                "wandb_run": "d581f91bbd3359f8",
                "parameters": 2_084_388,
            }
        },
        extra_sources={
            "factorized_complex_scan_reader": Path("src/lnet/pac_factorized_complex_scan_reader.py"),
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
