"""Paired Wave 14A and staged fused CUDA Graph timings."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wave14_vscan_probe"))
from bench import measure, sample

from lnet import general_cuda_scans as scans

CASES = (
    (56, 56, 160, False), (56, 56, 160, True),
    (28, 28, 320, False), (28, 28, 320, True),
    (14, 14, 640, False), (14, 14, 640, True),
    (7, 7, 640, False), (64, 96, 160, False), (37, 53, 192, False),
)


def probe(batch: int, dtype: torch.dtype, shape: tuple) -> dict:
    h, w, m, coarse = shape
    torch.manual_seed(731)
    scale = 1 + coarse
    states = sample((4, batch, h, w, m), dtype, 0.4)
    pole = [sample((m,), torch.float32, 0.12) for _ in range(4)]
    aux = [torch.ones((2, length, m), device="cuda") for length in (w, h)]
    path = [sample(size, torch.float32, 0.17)
            for size in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    dy = [sample((batch, h // scale, w // scale, m), dtype, 0.1) for _ in range(2)]

    def forward() -> tuple:
        return scans.vertical_forward_packed(pole, states, aux, path, coarse, 1e-6, True).unbind(0)  # noqa: FBT003

    def backward() -> list:
        return scans.vertical_backward(pole, states, aux, path, *dy, coarse, compact=coarse)

    results = {}
    for name, operation in (("forward", forward), ("backward", backward)):
        expected = None
        for policy in ("off", "on"):
            os.environ["LNET_VSCAN_FUSED_OPT"] = policy
            result, output = measure(operation, 9, ())
            results[f"{name}_{policy}"] = result
            if expected is None:
                expected = output
            else:
                for actual, reference in zip(output, expected, strict=True):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    return {"batch": batch, "dtype": str(dtype), "shape": shape, **results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--index", type=int, nargs="+", default=range(len(CASES)))
    args = parser.parse_args()
    os.environ["LNET_VSCAN_DECOUPLED"] = "off"
    with args.output.open("a") as out:
        for dtype in (torch.bfloat16, torch.float16):
            for index in args.index:
                print(json.dumps(probe(args.batch, dtype, CASES[index])), file=out, flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
