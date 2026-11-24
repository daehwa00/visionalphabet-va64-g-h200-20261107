"""Screen all Wave 28B presets against Wave 26 and FP32 convolution."""

# ruff: noqa: ANN001, ANN201, SLF001
import argparse
import itertools
import json
import os
from pathlib import Path

import torch
from bench import STANDARD, compare, compute, inputs, load_baseline
from screen import GENERAL, REAL

from lnet import factorized_reader_kernels as reader


def parity(shape, dtype, strict, baseline):
    # These batches enter specialized dispatch for all three modified resolutions.
    batch = 96 if shape in REAL[:3] and not strict else 1 if shape[0] >= 512 else 2
    tensors = inputs(batch, shape, dtype, strict)
    p, g, real, imag = (tensor.float() for tensor in tensors)
    expected = (
        reader._spatial_reference(p, real, imag, strict_complex=strict),
        *reader._spatial_reference_backward(g, p, real, imag, strict_complex=strict),
    )
    wave26 = compute(baseline, *tensors, strict)
    arms = {}
    for preset in ("off", "auto", *reader._WAVE28B_PRESETS):
        os.environ["LNET_READER_W28B"] = preset
        actual = compute(reader, *tensors, strict)
        arms[preset] = {
            "fp32": compare(actual, expected, dtype),
            "wave26": compare(actual, wave26, dtype),
        }
    return {"shape": (batch, *shape), "dtype": str(dtype), "strict": strict, "arms": arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    baseline = load_baseline(args.baseline)
    torch.manual_seed(99)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.update(LNET_READER_W25B="auto", LNET_READER_W27B="off", LNET_READER_VEC="auto")
    shapes = tuple(dict.fromkeys((*REAL, *GENERAL, *STANDARD)))
    passed = True
    with args.out.open("w") as out:
        for shape, dtype, strict in itertools.product(
            shapes, (torch.bfloat16, torch.float16), (False, True),
        ):
            row = parity(shape, dtype, strict, baseline)
            print(json.dumps(row), file=out, flush=True)
            passed &= all(
                metrics["passed"]
                for arm in row["arms"].values()
                for comparison in arm.values()
                for metrics in comparison.values()
            )
    raise SystemExit(int(not passed))


if __name__ == "__main__":
    main()
