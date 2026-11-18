"""Train Real-memory v3 from scratch with the matched VA128 ImageNet-100 recipe.

The frozen ImageNet-100 harness owns arguments, data, optimization, evaluation,
and checkpoint continuation. This runner changes model construction and adapts
the real-memory backbone to that harness's five-output affine-head contract.
"""

from __future__ import annotations

# The frozen campaign exposes its construction and training callbacks privately.
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateUsage=false
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import a2d_r2k3_runtime as runtime
import run_a2d_affine_qhead_imagenet100 as heads
import run_a2d_deep4_calibrated_uniform_p96_phase_gated_imagenet100 as metrics
import run_a2d_qhead_e2e_imagenet100 as structured
import run_a2d_r2k3_k128_depth_p_overnight_imagenet100 as anchor
import run_a2d_resaux1_imagenet100 as prepare
import run_alphabet2d_imagenet100_nano as harness
import torch
from a2d_r2k3_capacity_factory import CapacityQ4OnlyAffineClassifier
from torch import Tensor, nn
from va128_packed_backbone import PackedVA128Backbone, enable_packed_va128_

from lnet.complex_scan import ComplexScanConfig
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.real_memory_backbone import RealMemoryBackbone

if TYPE_CHECKING:
    from argparse import Namespace


VARIANT = "VA128-RealMemory-v3-IN100"
VARIANTS = (VARIANT,)
SEEDS = (501,)
NUM_CLASSES = 100
BASELINE_PARAMETERS = 4_621_476
REAL_MEMORY_PARAMETERS = 6_593_188
BASELINE_FINAL_ACCURACY = 0.8830000162124634
BASELINE_BEST_ACCURACY = 0.883400022983551
SCHEMA = "lnet.va128.real_memory_swiglu_v3.imagenet100.v1"


class RealMemoryIN100(RealMemoryBackbone):
    """Expose the unchanged affine objective and five-output evaluation API."""

    @property
    def classifier(self) -> CapacityQ4OnlyAffineClassifier:
        classifier = self.backbone.classifier
        if not isinstance(classifier, CapacityQ4OnlyAffineClassifier):
            raise TypeError(
                "ImageNet-100 real-memory training requires the original Q4 affine head"
            )
        return classifier

    def prepare_for_compiled_training_(self) -> RealMemoryIN100:
        cast("PackedVA128Backbone", self.backbone).prepare_for_compiled_training_()
        return self

    # The inference-only parent returns logits; the frozen IN100 objective needs
    # all five tensors from the exact same classifier call.
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        descriptor = cast("PackedVA128Backbone", self.backbone)._packed_descriptor(
            inputs,
            full=False,
        )
        with torch.autocast(device_type=descriptor.device.type, enabled=False):
            return self.classifier(descriptor.float(), terminal_only=True)


def _build(variant: str, config: ComplexScanConfig) -> RealMemoryIN100:
    if variant != VARIANT:
        raise ValueError(f"unsupported real-memory ImageNet-100 variant: {variant}")
    if config.output_dim != NUM_CLASSES:
        raise ValueError("the matched ImageNet-100 experiment requires exactly 100 classes")
    # The anchor's seeded scratch construction is part of the baseline recipe.
    # No pretrained checkpoint is loaded or used for initialization.
    backbone = anchor._build(anchor.D2262_P128_TERM, config)
    if sum(parameter.numel() for parameter in backbone.parameters()) != BASELINE_PARAMETERS:
        raise RuntimeError("the original ImageNet-100 VA128 parameter count changed")
    for module in backbone.modules():
        if isinstance(module, FactorizedComplexConv2dReader):
            module.execution_mode = "factorized_triton"
    enable_packed_va128_(backbone)
    model = RealMemoryIN100(backbone)
    if sum(parameter.numel() for parameter in model.parameters()) != REAL_MEMORY_PARAMETERS:
        raise RuntimeError("the ImageNet-100 real-memory v3 parameter count changed")
    runtime.configure(VARIANTS, SEEDS)
    return model


def _source_paths() -> tuple[Path, ...]:
    repo = Path(__file__).resolve().parents[1]
    return runtime.source_dependency_paths(repo, (Path(__file__).stem,))


def _contract(args: Namespace) -> dict[str, Any]:
    # Preserve every frozen recipe value, including optimizer grouping, AMP,
    # augmentations, loader behavior, schedule, and exact resume semantics.
    payload = runtime.base_contract(args)
    repo = Path(__file__).resolve().parents[1]
    paths = _source_paths()
    payload.update(
        {
            "schema": SCHEMA,
            "evidence_status": "scratch one-seed matched ImageNet-100 real-memory v3 experiment",
            "variants": list(VARIANTS),
            "priority": list(VARIANTS),
            "seeds": list(SEEDS),
            "num_classes": NUM_CLASSES,
            "initialization": "scratch; seed 501 anchor initialization followed by v3 conversion",
            "variant_configs": {
                VARIANT: {
                    "architecture": RealMemoryBackbone.architecture,
                    "baseline_variant": anchor.D2262_P128_TERM,
                    "excitation_modes": [128, 128, 128, 128],
                    "pole_modes": [160, 160, 160, 128],
                    "depth": [2, 2, 6, 2],
                    "real_feature_channels": 256,
                    "real_swiglu_hidden_channels": 384,
                    "reader_execution_mode": "factorized_triton",
                    "free_memory_io": True,
                    "head": "original terminal-Q4 BatchNorm512-affine-false Linear512-to-100",
                    "objective": "unchanged affine-head MixUp CE; no auxiliary loss",
                },
            },
            "parameter_counts": {VARIANT: REAL_MEMORY_PARAMETERS},
            "architecture": {
                VARIANT: (
                    "VA128 D2262/P160-160-160-128 with independent real Reader/read maps, "
                    "feature norms and carry channels; eleven real SwiGLU feature blocks; "
                    "original complex memory recurrence, stem, path topology and IN100 head."
                ),
            },
            "references": {
                "variant": anchor.D2262_P128_TERM,
                "num_classes": NUM_CLASSES,
                "seed": SEEDS[0],
                "epochs": 100,
                "parameters": BASELINE_PARAMETERS,
                "final_accuracy": BASELINE_FINAL_ACCURACY,
                "best_accuracy": BASELINE_BEST_ACCURACY,
                "selection": "fixed final epoch 100; best is a historical reference only",
            },
            "source_closure": [path.relative_to(repo).as_posix() for path in paths],
        },
    )
    if _training_optimizations_enabled():
        payload["recipe"]["real_memory_training_optimized"] = True
    payload["source_sha256"].update(
        {
            "real_memory_imagenet100_runner": runtime.digest(__file__),
            "real_memory_backbone": runtime.digest(repo / "src/lnet/real_memory_backbone.py"),
            "packed_va128_backbone": runtime.digest(repo / "scripts/va128_packed_backbone.py"),
            "in100_anchor_builder": runtime.digest(anchor.__file__),
            "source_closure": runtime.source_fingerprint(repo, paths),
        },
    )
    return payload


def _wandb_model_metrics(model: nn.Module) -> dict[str, float]:
    if not isinstance(model, RealMemoryIN100):
        raise TypeError("real-memory metrics require the runner's training adapter")
    # The legacy structural metrics inspect stage1/2/3 directly. Unwrap only
    # that callback; objective/evaluation diagnostics are stored on the adapter.
    result = metrics._wandb_model_metrics(model.backbone)
    result.update(heads._wandb_model_metrics(model))
    return result


def _training_optimizations_enabled() -> bool:
    value = os.environ.get("VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS", "0")
    if value not in ("0", "1"):
        raise ValueError("VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS must be 0 or 1")
    return value == "1"


def _prepare_model(model: nn.Module, recipe: dict[str, Any]) -> nn.Module:
    prepared = prepare._prepare_model(model, recipe)
    if _training_optimizations_enabled():
        from lnet.real_memory_training import enable_real_memory_training_

        enable_real_memory_training_(prepared)
    return prepared


def _build_optimizer(model: nn.Module, recipe: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer = runtime.build_optimizer(model, recipe)
    destination = os.environ.get("VA_IN100_PROGRESS_PATH")
    if not destination:
        return optimizer
    path = Path(destination)
    steps = 0

    def record_step(
        current: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        nonlocal steps
        steps += 1
        if steps != 1 and steps % 100:
            return
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        finite = bool(gradients) and bool(
            torch.stack([torch.isfinite(grad).all() for grad in gradients]).all(),
        )
        harness._atomic_json(
            path,
            {
                "schema": "lnet.va128.real_memory.imagenet100.optimizer_progress.v1",
                "variant": VARIANT,
                "seed": SEEDS[0],
                "session_optimizer_steps": steps,
                "finite_gradients": finite,
                "current_learning_rates": [float(group["lr"]) for group in current.param_groups],
                "unix_time": time.time(),
            },
        )
        if not finite:
            raise RuntimeError(
                "nonfinite or missing gradients in the observed IN100 optimizer step"
            )

    optimizer.register_step_post_hook(record_step)
    return optimizer


def main() -> None:
    runtime.configure(VARIANTS, SEEDS)
    heads.VARIANTS = VARIANTS
    heads.SEEDS = SEEDS
    structured._training_objective = heads._training_objective
    structured._after_training_batch = heads._after_training_batch
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    harness.main(
        harness.runner_bindings(
            variants=VARIANTS,
            seeds=SEEDS,
            model_config=ComplexScanConfig,
            build_model=_build,
            contract=_contract,
            build_optimizer=_build_optimizer,
            prepare_model=(
                _prepare_model if _training_optimizations_enabled() else prepare._prepare_model
            ),
            train_epoch=structured._train_epoch,
            evaluate=heads._evaluate,
            wandb_model_metrics=_wandb_model_metrics,
            summarize=heads._summarize,
        ),
    )


if __name__ == "__main__":
    main()
