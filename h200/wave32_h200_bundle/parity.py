"""Compare the selected presets with the same tree's off path and FP32 reference."""

# ruff: noqa: SLF001
import json
import math
import os
import sys
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch
from bench import compute, inputs
from test_general_cuda_scans import _inputs, _reference_vscan

from lnet import factorized_reader_kernels as reader
from lnet import general_cuda_scans as scans

TensorGroup = tuple[torch.Tensor, ...]


def scan_case(shape: tuple[int, int, int, bool], dtype: torch.dtype) -> tuple[TensorGroup, ...]:
    h, w, modes, coarse = shape
    px, py, real, imag, aux, path = _inputs(h, w, modes, 8, dtype, "cuda", batch=2)
    states = scans.horizontal_forward(px, real, imag).detach().requires_grad_()
    targets = (*py, states, *path)
    with patch.dict(os.environ, {"LNET_SCAN_W32A": "off"}):
        off = scans.vertical_forward(py, states, aux, path, coarse=coarse)
    upstream = tuple(torch.randn_like(y) for y in off)
    off_values = (*off, *torch.autograd.grad(off, targets, upstream))
    fp32 = _reference_vscan(py, states.float(), aux, path, coarse)
    fp32_values = (*fp32, *torch.autograd.grad(fp32, targets, upstream))
    actual = scans.vertical_forward(py, states, aux, path, coarse=coarse)
    return (*actual, *torch.autograd.grad(actual, targets, upstream)), off_values, fp32_values


def reader_case(shape: tuple[int, int, int, int], dtype: torch.dtype) -> tuple[TensorGroup, ...]:
    h, w, _, _ = shape
    batch = max(33, (reader._SPECIALIZE_MIN_PIXELS + h * w - 1) // (h * w))
    p, g, real, imag = inputs(batch, shape, dtype, strict=False)
    with patch.dict(os.environ, {"LNET_READER_W32B": "off"}):
        off = compute(reader, p, g, real, imag, strict=False)
    pf, gf, rf, imf = (tensor.float() for tensor in (p, g, real, imag))
    fp32 = (
        reader._spatial_reference(pf, rf, imf, strict_complex=False),
        *reader._spatial_reference_backward(gf, pf, rf, imf, strict_complex=False),
    )
    return compute(reader, p, g, real, imag, strict=False), off, fp32


def relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    ref = expected.float()
    return float((actual.float() - ref).norm() / ref.norm().clamp_min(0.02))


def main() -> None:
    track, path = sys.argv[1:]
    torch.manual_seed(117)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if track == "a":
        case = scan_case
        shapes = [
            *((size, size, 160, coarse) for size, coarse in product((56, 28, 14), (False, True))),
            (7, 7, 128, False), (8, 8, 128, True), (13, 19, 63, False),
        ]
    else:
        case = reader_case
        shapes = [
            (56, 56, 160, 4), (28, 28, 160, 4), (14, 14, 160, 4), (7, 7, 128, 4),
            (9, 11, 34, 4), (13, 19, 63, 3),
        ]
    rows = []
    expected = 2 * len(shapes)
    for shape, dtype in product(shapes, (torch.bfloat16, torch.float16)):
        row = {"shape": shape, "dtype": str(dtype), "passed": False}
        try:
            actual, off, fp32 = case(shape, dtype)
            row["off_relative_l2"] = [relative(x, y) for x, y in zip(actual, off, strict=True)]
            row["fp32_relative_l2"] = [relative(x, y) for x, y in zip(actual, fp32, strict=True)]
            limit = 0.02 if dtype == torch.bfloat16 else 0.005
            row["passed"] = all(
                math.isfinite(value) and value <= limit
                for value in (*row["off_relative_l2"], *row["fp32_relative_l2"])
            )
        except (ValueError, RuntimeError) as exc:
            row["error"] = str(exc)[:512]
        rows.append(row)
        passed = len(rows) == expected and all(item["passed"] for item in rows)
        output = Path(path)
        temporary = output.with_suffix(".tmp")
        maxima = {
            key: max((value for item in rows for value in item.get(key, ())), default=None)
            for key in ("off_relative_l2", "fp32_relative_l2")
        }
        temporary.write_text(json.dumps({
            "expected": expected, "passed": passed, "max_relative_l2": maxima, "cases": rows,
        }))
        temporary.replace(output)
    raise SystemExit(int(not passed))


if __name__ == "__main__":
    main()
