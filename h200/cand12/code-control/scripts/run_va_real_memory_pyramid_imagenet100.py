"""Scratch IN100 experiment with real feature widths 32/64/128/256."""

from __future__ import annotations

# Reuse the established model builders and frozen IN100 training callbacks.
# pyright: reportPrivateUsage=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportExplicitAny=false
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as capacity
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import run_va128_real_memory_imagenet100 as baseline
import torch
from torch import nn

from lnet.complex_scan import ComplexScanConfig
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_gated_post_fusion import GatedComplexPostFusion, GatedPoleExcitationS2DTransition

if TYPE_CHECKING:
    from argparse import Namespace

CHANNELS = (32, 64, 128, 256)
POLES = (160, 160, 160, 128)
DEPTH = (2, 2, 6, 2)
HIDDEN = (48, 96, 192, 384)
VARIANT = "VA-RealMemory-v3-C32-64-128-256-IN100"
VARIANTS = (VARIANT,)
SEEDS = (501,)
PARAMETERS = 2_736_964


class PyramidRealMemoryIN100(baseline.RealMemoryIN100):
    architecture = "va-real-memory-v3-pyramid-c32-64-128-256"


def _build(variant: str, config: ComplexScanConfig) -> PyramidRealMemoryIN100:
    if variant != VARIANT or config.output_dim != 100:
        raise ValueError("expected the C32/64/128/256 real-memory IN100 variant")
    seed = torch.initial_seed()
    spec = allocation.StageAllocationSpec(
        cast("tuple[int, int, int, int]", tuple(c // 2 for c in CHANNELS)),
        POLES,
        (0, 0, 4, 0),
        family="real_memory_pyramid",
    )
    model = capacity._build_all_resolution(spec.as_insight_spec(), config)
    model = allocation._append_repeated_blocks(model, spec)
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, GatedComplexPostFusion):
                hidden = round(1.5 * child.modes)
                setattr(parent, name, GatedComplexPostFusion(child.modes, hidden))
                if isinstance(parent, GatedPoleExcitationS2DTransition):
                    parent.post_hidden = hidden
    # Preserve common-shaped initialization, including pole dynamics and the
    # fixed terminal head, without consuming the candidate's subsequent RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = baseline.anchor._build(baseline.anchor.D2262_P128_TERM, config)
    baseline.anchor._pair_common_initialization_(model, reference)
    for module in model.modules():
        if isinstance(module, FactorizedComplexConv2dReader):
            module.execution_mode = "factorized_triton"
    baseline.enable_packed_va128_(model)
    result = PyramidRealMemoryIN100(model)
    if sum(p.numel() for p in result.parameters()) != PARAMETERS:
        raise RuntimeError("pyramid parameter inventory changed")
    baseline.runtime.configure(VARIANTS, SEEDS)
    return result


def _contract(args: Namespace) -> dict[str, Any]:
    payload = baseline.runtime.base_contract(args)
    root = Path(__file__).resolve().parents[1]
    paths = baseline.runtime.source_dependency_paths(root, (Path(__file__).stem,))
    payload.update(
        schema="lnet.real_memory_v3.pyramid_c32_64_128_256.in100.v1",
        evidence_status="scratch seed501 width-pyramid comparison; final epoch100",
        variants=list(VARIANTS),
        priority=list(VARIANTS),
        seeds=list(SEEDS),
        num_classes=100,
        initialization="scratch seed501; common-shaped canonical tensors paired before v3 conversion",
        variant_configs={
            VARIANT: {
                "architecture": PyramidRealMemoryIN100.architecture,
                "real_feature_channels": list(CHANNELS),
                "real_swiglu_hidden_channels": list(HIDDEN),
                "excitation_modes": [c // 2 for c in CHANNELS],
                "pole_modes": list(POLES),
                "depth": list(DEPTH),
                "reader_real_latent": 4,
                "reader_kernel": 3,
                "reader_execution_mode": "factorized_triton",
                "free_memory_io": True,
                "carry": "channel-wise weighted 2x2 followed by bias-free real 1x1 projection",
                "stem": "3-to-32 stride2, 32-to-32 stride2; original real input mixer at width32",
                "head": "unchanged terminal-Q4 descriptor512 to IN100 affine head",
                "objective": "same terminal-Q4 MixUp cross-entropy; no auxiliary loss",
            }
        },
        parameter_counts={VARIANT: PARAMETERS},
        architecture={
            VARIANT: "Real-memory v3 with channel pyramid32/64/128/256, fixed poles/depth"
        },
        references={
            "fixed_width_real_memory": baseline.VARIANT,
            "fixed_width_parameters": baseline.REAL_MEMORY_PARAMETERS,
            "original_complex_final_accuracy": baseline.BASELINE_FINAL_ACCURACY,
            "parameter_matched": False,
        },
        source_closure=[p.relative_to(root).as_posix() for p in paths],
    )
    payload["source_sha256"].update(
        pyramid_runner=baseline.runtime.digest(__file__),
        source_closure=baseline.runtime.source_fingerprint(root, paths),
    )
    return payload


def _build_optimizer(model: nn.Module, recipe: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer = baseline.runtime.build_optimizer(model, recipe)
    destination = os.environ.get("VA_IN100_PROGRESS_PATH")
    if not destination:
        return optimizer
    steps = 0

    def record(current: torch.optim.Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        nonlocal steps
        steps += 1
        if steps != 1 and steps % 100:
            return
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        finite = bool(gradients) and bool(
            torch.stack([g.isfinite().all() for g in gradients]).all()
        )
        baseline.harness._atomic_json(
            Path(destination),
            {
                "variant": VARIANT,
                "seed": SEEDS[0],
                "session_optimizer_steps": steps,
                "finite_gradients": finite,
                "current_learning_rates": [g["lr"] for g in current.param_groups],
                "unix_time": time.time(),
            },
        )
        if not finite:
            raise RuntimeError("nonfinite or missing gradients in pyramid training")

    optimizer.register_step_post_hook(record)
    return optimizer


def main() -> None:
    baseline.runtime.configure(VARIANTS, SEEDS)
    baseline.heads.VARIANTS, baseline.heads.SEEDS = VARIANTS, SEEDS
    baseline.structured._training_objective = baseline.heads._training_objective
    baseline.structured._after_training_batch = baseline.heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    baseline.harness.main(
        baseline.harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=_build_optimizer,
            prepare_model=baseline.prepare._prepare_model,
            train_epoch=baseline.structured._train_epoch,
            evaluate=baseline.heads._evaluate,
            wandb_model_metrics=baseline._wandb_model_metrics,
            summarize=baseline.heads._summarize,
        )
    )


if __name__ == "__main__":
    main()
