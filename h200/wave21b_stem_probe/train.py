"""Run the Wave 17B training probe with optional deterministic cuDNN dispatch."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liverms_v2_opt"))
import bench_h200


def setup_controlled(batch: int) -> torch.Tensor:

    inputs = setup(batch)
    torch.backends.cudnn.benchmark = False
    return inputs


if __name__ == "__main__":
    setup = bench_h200.setup
    if sys.argv[1] == "controlled":
        bench_h200.setup = setup_controlled
    del sys.argv[1]
    bench_h200.main()
