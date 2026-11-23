"""Run the existing compiled training and regression harness with a cuDNN toggle."""

import os
import runpy

import torch


def main() -> None:
    bench = runpy.run_path("h200/liverms_v2_opt/bench_h200.py")
    entry = bench["main"]
    namespace = entry.__globals__
    setup = namespace["setup"]

    def setup_with_policy(batch: int) -> torch.Tensor:
        inputs = setup(batch)
        torch.backends.cudnn.benchmark = os.environ.get("VA_T82_CUDNN_BENCHMARK", "1") == "1"
        return inputs

    namespace["setup"] = setup_with_policy
    entry()


if __name__ == "__main__":
    main()
