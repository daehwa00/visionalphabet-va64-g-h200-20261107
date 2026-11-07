#!/usr/bin/env python3
"""Train the frozen LNet K128/P160-160-160-128/D2262 on ImageNet-1K."""

from __future__ import annotations

# pyright: reportAny=false, reportExplicitAny=false, reportMissingImports=false
# pyright: reportImplicitRelativeImport=false, reportPrivateLocalImportUsage=false
# pyright: reportPrivateUsage=false
import argparse
import json
from pathlib import Path

import a2d_r2k3_runtime as runtime
import imagenet1k_runtime as imagenet_runtime
import run_a2d_r2k3_k128_depth_p_overnight_imagenet100 as anchor
from torch import Tensor, nn

MODEL_KEY = "lnet_k128_p160_160_160_128_d2262"
DISPLAY_NAME = "LNet-K128-P160-160-160-128-D2262"
VARIANT = anchor.D2262_P128_TERM
SEED = 501
LEARNING_RATE = 3.0e-3
EPOCHS = 100
# The ImageNet-100 checkpoint has 4,621,476 parameters; replacing its
# 100-class Q4 affine head with 1,000 classes adds 461,700 parameters.
EXPECTED_PARAMETERS = 5_083_176
ImageNet1KTask = imagenet_runtime.BaselineTask


class PrimaryLogitsAdapter(nn.Module):
    """Expose the canonical joint Q4 logits to the matched baseline worker."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.model(inputs)
        if not isinstance(output, tuple) or len(output) != 5:
            raise RuntimeError("LNet ImageNet-1K model lost its five-output Q-head contract")
        joint = output[0]
        if not isinstance(joint, Tensor):
            raise TypeError("LNet ImageNet-1K joint logits are not a tensor")
        return joint


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wandb-mode", choices=("disabled", "online"), default="online")
    return parser.parse_args()


def _build_model(
    key: str,
    _source_root: str | Path | None,
    num_classes: int,
) -> nn.Module:
    if key != MODEL_KEY or num_classes != 1000:
        raise ValueError("LNet ImageNet-1K builder received an incompatible task")
    model = anchor._build(VARIANT, runtime.model_config(output_dim=num_classes))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"LNet ImageNet-1K parameter count changed: {parameters} != {EXPECTED_PARAMETERS}"
        )
    prepared = model.prepare_for_compiled_training_()
    return PrimaryLogitsAdapter(prepared)


def _source_fingerprint() -> str:
    repo = Path(__file__).resolve().parents[1]
    paths = runtime.source_dependency_paths(repo, (Path(__file__).stem,))
    return runtime.source_fingerprint(repo, paths)


def _task(args: argparse.Namespace) -> ImageNet1KTask:
    output = args.output_root.resolve() / MODEL_KEY / f"seed_{SEED}"
    checkpoint = output / "checkpoint.pt"
    return ImageNet1KTask(
        phase="full",
        model_key=MODEL_KEY,
        seed=SEED,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        data_root=args.data_root.resolve(),
        output_dir=output,
        result_path=output / "result.json",
        checkpoint_path=checkpoint,
        source_root=Path(__file__).resolve().parents[1],
        batch_size=args.batch_size,
        workers=args.workers,
        wandb_mode=args.wandb_mode,
        resume=checkpoint.is_file(),
        source_sha256={
            "lnet_internal_closure": _source_fingerprint(),
            "imagenet1k_runner": runtime.digest(__file__),
        },
    )


def main() -> None:
    args = _arguments()
    result = imagenet_runtime.run_task(_task(args), model_builder=_build_model)
    print("LNET_IMAGENET1K_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
