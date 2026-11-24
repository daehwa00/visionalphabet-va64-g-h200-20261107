"""Check BF16/FP16 parity and time isolated dispatches on the installed GPU."""

# ruff: noqa: ANN001, ANN201, SLF001
import argparse
import itertools
import json
import os
from pathlib import Path

import torch
from bench import STANDARD, compare, compute, inputs, record_launches
from screen import GENERAL, REAL, capture, measure

from lnet import factorized_reader_kernels as reader

ARMS = ("off", "auto", *reader._WAVE27B_PRESETS)
DTYPES = (torch.bfloat16, torch.float16)


def parity(shape, dtype, strict):
    # Activate all modified tiles at modest batch size and include unspecialized cases.
    batch = 96 if shape in REAL[:3] and not strict else 1 if shape[0] >= 512 else 2
    tensors = inputs(batch, shape, dtype, strict)
    p, g, real, imag = (tensor.float() for tensor in tensors)
    reference = (
        reader._spatial_reference(p, real, imag, strict_complex=strict),
        *reader._spatial_reference_backward(g, p, real, imag, strict_complex=strict),
    )
    os.environ["LNET_READER_W27B"] = "off"
    baseline = compute(reader, *tensors, strict)
    arms = {}
    for arm in ARMS:
        os.environ["LNET_READER_W27B"] = arm
        actual = compute(reader, *tensors, strict)
        arms[arm] = {
            "fp32": compare(actual, reference, dtype),
            "baseline": compare(actual, baseline, dtype),
        }
    return {"shape": (batch, *shape), "dtype": str(dtype), "strict": strict, "arms": arms}


def bench(shape, dtype):
    batch = 128 if shape in REAL else 16
    tensors = inputs(batch, shape, dtype)
    graphs = {}
    call_buffers = {}  # CUDA Graphs retain addresses, so keep their allocations live.
    memory = {}
    for arm in ARMS:
        os.environ["LNET_READER_W27B"] = arm
        torch.cuda.reset_peak_memory_stats()
        starting_bytes = torch.cuda.memory_allocated()
        calls = call_buffers[arm] = record_launches(reader, *tensors)
        partial = calls[1][0][3][-1]
        memory[arm] = {
            "peak_increment_bytes": torch.cuda.max_memory_allocated() - starting_bytes,
            "partial_bytes": partial.numel() * partial.element_size(),
        }
        groups = {"forward": calls[:1], "dweight+reduce": calls[1:-1], "dinput": calls[-1:]}
        graphs[arm] = {op: capture(reader, launches) for op, launches in groups.items()}
    results = {
        op: measure({arm: operations[op] for arm, operations in graphs.items()})
        for op in graphs["off"]
    }
    return {"shape": (batch, *shape), "dtype": str(dtype), "operations": results, "memory": memory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parity", "bench"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=4)
    args = parser.parse_args()
    torch.manual_seed(95)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["LNET_READER_W25B"] = "auto"
    os.environ["LNET_READER_VEC"] = "auto"
    passed = True
    with args.out.open("w") as out:
        if args.mode == "parity":
            shapes = (tuple(args.shape),) if args.shape else tuple(
                dict.fromkeys((*REAL, *GENERAL, *STANDARD)),
            )
            for shape, dtype, strict in itertools.product(shapes, DTYPES, (False, True)):
                row = parity(shape, dtype, strict)
                print(json.dumps(row), file=out, flush=True)
                passed &= all(
                    metrics["passed"]
                    for arm in row["arms"].values()
                    for comparison in arm.values()
                    for metrics in comparison.values()
                )
        else:
            shapes = (tuple(args.shape),) if args.shape else (*REAL, *GENERAL[-2:])
            for shape, dtype in itertools.product(shapes, DTYPES):
                print(json.dumps(bench(shape, dtype)), file=out, flush=True)
    raise SystemExit(int(not passed))


if __name__ == "__main__":
    main()
