"""Run the production training benchmark with a selectable cuDNN control."""

# ruff: noqa: T201
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liverms_v2_opt"))
import bench_h200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-autotune", action="store_true")
    args = parser.parse_args()
    setup = bench_h200.setup

    def controlled_setup(batch: int) -> torch.Tensor:
        inputs = setup(batch)
        torch.backends.cudnn.benchmark = not args.no_autotune
        return inputs

    bench_h200.setup = controlled_setup
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as output:
        result = bench_h200.train_arm(argparse.Namespace(
            batch=128, variant="17-p1-nr4-1024d-v3", profile=True,
            out=Path(output) / "train.json",
        ))
    for field in ("allocated", "reserved"):
        result[f"peak_{field}_bytes"] = round(result[f"peak_{field}_gib"] * 2**30)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
