"""Cache-preserving wrapper for VA-Ablation-10 with eval-time terminal moment calibration.

Shares the model, optimizer, compile, and workload-calibration identity of
``run_alphabet_ablations.py``. The evaluation binding temporarily calibrates
terminal moments on 2,048 fixed clean training images.
"""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import alphabet_augmentation as augment
import alphabet_imagenet100_runtime as runtime
import calibrated_evaluation
import imagenet100_data_runtime as data_runtime
import run_alphabet_ablations as base
import torch

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable


def _eval_batch_size(args: Namespace) -> int:
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    return min(128, args.batch_size) if effective_batch == 1024 else args.batch_size


def build_bindings() -> runtime.RunnerBindings:
    """Build RunnerBindings identical to run_alphabet_ablations except calibrated eval."""
    from alphabet.complex_scan_config import ComplexScanConfig

    evaluator: Callable[..., dict[str, float]] | None = None

    def _contract(args: Namespace) -> dict[str, Any]:
        nonlocal evaluator
        payload = base._contract(args)
        if not args.initialize_only:
            evaluator = calibrated_evaluation.make_evaluator(
                args.data_root,
                batch_size=_eval_batch_size(args),
                workers=payload["recipe"]["validation_loader_workers"],
                persistent_workers=payload["recipe"]["validation_loader_persistent_workers"],
            )
        payload["recipe"]["terminal_bn_calibration"] = (
            f"{calibrated_evaluation.DEFAULT_CALIBRATION_SAMPLES} fixed clean training images; "
            "temporary eval moments"
        )
        payload["source_sha256"].update(
            calibrated_wrapper=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            calibrated_evaluation=hashlib.sha256(
                Path(calibrated_evaluation.__file__).read_bytes()
            ).hexdigest(),
        )
        return payload

    def _evaluate(*args: Any, **kwargs: Any) -> dict[str, float]:
        if evaluator is None:
            raise RuntimeError("calibrated evaluator was called before contract initialization")
        return evaluator(*args, **kwargs)

    def _close_evaluator() -> None:
        if isinstance(evaluator, partial):
            data_runtime.close_persistent_loader(evaluator.keywords["calibration_loader"])

    return runtime.RunnerBindings(
        variants=base.VARIANTS,
        seeds=base.SEEDS,
        model_config=ComplexScanConfig,
        build_model=base._build,
        contract=_contract,
        build_optimizer=base._build_optimizer,
        prepare_model=runtime.prepare_model,
        train_epoch=partial(augment.train_epoch, cutmix=bool(base.SPEC["cutmix"])),
        evaluate=_evaluate,
        wandb_model_metrics=runtime.wandb_model_metrics,
        summarize=runtime.summarize,
        calibrated_validation=True,
        close_evaluator=_close_evaluator,
    )


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    runtime.main(build_bindings())


if __name__ == "__main__":
    main()
