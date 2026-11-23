"""Run the Wave 17B Dynamo and inference-fold checks on capture-safe models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liverms_v2_opt"))
import bench_h200

from lnet.capture_safe_orthogonal import prepare_capture_safe_orthogonal_

build_model = bench_h200.build_model


def build_capture_safe_model(variant: str) -> torch.nn.Module:
    model = build_model(variant)
    prepare_capture_safe_orthogonal_(model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=["dynamo", "fold"], required=True)
    parser.add_argument("--variant", choices=bench_h200.VARIANTS)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with patch.object(bench_h200, "build_model", build_capture_safe_model):
        if args.gate == "dynamo":
            result = bench_h200.dynamo_check(args)
        else:
            if args.variant is None:
                parser.error("--gate fold requires --variant")
            result = bench_h200.infer_fold(args)
    bench_h200.write_json(args.output, result)
