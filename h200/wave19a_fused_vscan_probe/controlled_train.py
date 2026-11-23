"""Control cuDNN algorithm selection while comparing the two fused paths."""

import runpy

import torch


def main() -> None:
    entry = runpy.run_path("h200/liverms_v2_opt/bench_h200.py")["main"]
    namespace = entry.__globals__
    setup_with_autotune = namespace["setup"]

    def setup(batch: int) -> torch.Tensor:
        inputs = setup_with_autotune(batch)
        torch.backends.cudnn.benchmark = False
        return inputs

    namespace["setup"] = setup
    entry()


if __name__ == "__main__":
    main()
