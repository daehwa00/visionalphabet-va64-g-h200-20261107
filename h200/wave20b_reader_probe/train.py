"""Run the existing matched training benchmark with selectable cuDNN autotuning."""

import argparse
import runpy
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    args, benchmark_args = parser.parse_known_args()
    path = "h200/liverms_v2_opt/bench_h200.py"
    benchmark = runpy.run_path(path)["main"]
    namespace = benchmark.__globals__
    setup = namespace["setup"]

    def configure(batch: int) -> torch.Tensor:
        inputs = setup(batch)
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
        return inputs

    namespace["setup"] = configure
    sys.argv = [path, *benchmark_args]
    benchmark()


if __name__ == "__main__":
    main()
