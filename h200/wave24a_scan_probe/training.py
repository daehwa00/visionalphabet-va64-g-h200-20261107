"""Run the established P1/NR4 whole-step harness with controlled cuDNN settings."""

import argparse
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import torch

P1 = "17-p1-nr4-1024d-v3"
SUITE = "h200/liverms_v2_opt/bench_h200.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("off", "auto", "on"), required=True)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    os.environ["LNET_SCAN_W24A"] = args.arm
    harness = runpy.run_path(SUITE)
    train = harness["train_arm"]
    setup = harness["setup"]

    def controlled_setup(batch: int) -> torch.Tensor:
        inputs = setup(batch)
        torch.backends.cudnn.benchmark = args.benchmark
        return inputs

    train.__globals__["setup"] = controlled_setup
    config = SimpleNamespace(variant=P1, batch=128, out=args.out, profile=True)
    result = train(config)
    result.update(
        status="ok", scan_policy=args.arm, cudnn_benchmark=args.benchmark,
        peak_allocated_bytes=round(result["peak_allocated_gib"] * 2**30),
        peak_reserved_bytes=round(result["peak_reserved_gib"] * 2**30),
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
