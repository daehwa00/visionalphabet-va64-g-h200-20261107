#!/usr/bin/env python3
"""Compare the original and fully fused CLPost training paths on one GPU."""

from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false
# pyright: reportExplicitAny=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_r2k3_same_resolution_all_clpost_imagenet100 as runner
import smoke_a2d_r2k3_same_resolution_factorial as smoke
import torch

from lnet.pac_gated_post_fusion import GatedStrictComplexPostFusion, _GatedPostFusionBase
from lnet.pac_product_scan_reference import product_scan_full4_reference
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock

if TYPE_CHECKING:
    from torch import Tensor


class _ReferenceSameResolutionBlock(SameResolutionPoleScanBlock):
    @torch.compiler.disable
    def collapsed_memory(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source = self.reader(real, imag)
        shape = cast("tuple[int, int, int, int]", tuple(source[0].shape))
        pole_x, pole_y = self.pole_scan.pole_coefficients(shape)
        paths = product_scan_full4_reference(
            pole_x,
            pole_y,
            source,
            gain_normalization=self.pole_scan.product_gain_normalization,
        )
        return self.collapse_paths(*paths)


class _ReferenceStrictPostFusion(GatedStrictComplexPostFusion):
    def forward(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return _GatedPostFusionBase.forward(self, real, imag)


def _reference_build(variant: str, config: Any) -> Any:
    model = _CANDIDATE_BUILD(variant, config)
    for block in model.same_resolution_blocks.values():
        block.__class__ = _ReferenceSameResolutionBlock
    for owner in runner._postfusions(model):
        owner.post_fusion.__class__ = _ReferenceStrictPostFusion
    return model


_CANDIDATE_BUILD = runner._build


def _measure(build: Any, batch_size: int, steps: int) -> dict[str, float]:
    runner._build = build
    smoke.runner = runner  # type: ignore[reportConstantRedefinition]
    return smoke._full_batch_cuda_step(runner.VARIANT, batch_size, steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=1.1)
    args = parser.parse_args()
    if min(args.batch_size, args.steps) <= 0:
        raise ValueError("full-model benchmark dimensions must be positive")
    torch.set_float32_matmul_precision("high")
    reference = _measure(_reference_build, args.batch_size, args.steps)
    candidate = _measure(_CANDIDATE_BUILD, args.batch_size, args.steps)
    speedup = reference["median_step_ms"] / candidate["median_step_ms"]
    memory_ratio = candidate["peak_bytes"] / reference["peak_bytes"]
    passed = (
        math.isfinite(speedup)
        and speedup >= args.minimum_speedup
        and memory_ratio <= 1.0
    )
    payload = {
        "schema": "lnet.a2d.r2k3.full_optimization_benchmark.v1",
        "reference": reference,
        "candidate": candidate,
        "speedup": speedup,
        "memory_ratio": memory_ratio,
        "status": "passed" if passed else "failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
