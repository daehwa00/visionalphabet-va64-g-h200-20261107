"""Measure alternative weight-gradient row and rank tiles on the real reader calls."""

# ruff: noqa: ANN001, ANN201, ANN202, SLF001
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from bench import inputs, record_launches
from screen import REAL, capture, measure

from lnet import factorized_reader_kernels as reader

VARIANTS = (
    (1, 2, 4, 4), (1, 4, 4, 4), (1, 4, 4, 2), (1, 2, 4, 2),
    (1, 4, 8, 2), (2, 4, 4, 2), (2, 2, 4, 2), (2, 2, 4, 1), (2, 4, 2, 2),
    (2, 2, 2, 4), (2, 4, 2, 4), (2, 2, 4, 4),
)


def weight_arm(vec, rows, warps, tile):
    original = reader._operation_vector_plan

    def select(projected, real, grad, op, *, strict):
        if op != "dweight":
            return original(projected, real, grad, op, strict=strict)
        modes, rank, kernel, _ = real.shape
        columns = reader._reader_vector_columns(projected.shape[2], vec)
        types = (projected.dtype, real.dtype, grad.dtype)
        module = reader._reader_cuda_module(
            projected.device.index or 0, reader._reader_compile_shape(projected, modes),
            columns, rank, kernel, tuple(reader._CUDA_TYPES[dtype] for dtype in types),
            strict=False, weight_geometry=(tile, rows, warps), weight_vec=vec,
            weight_read_only=vec != 2,
        )
        if not reader._reader_resident(module, op, warps):
            raise ValueError("weight variant uses local memory or cannot run")
        return reader._VectorLaunch(vec, columns, rows, warps, tile), module

    return patch.object(reader, "_operation_vector_plan", select)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(644)
    with args.out.open("w") as output:
        for shape in REAL:
            tensors = inputs(128, shape, torch.bfloat16)
            graphs = {}
            arms = {}
            retained = []
            for config in VARIANTS:
                with weight_arm(*config):
                    try:
                        launches = record_launches(reader, *tensors)
                    except ValueError as error:
                        arms[str(config)] = {"error": str(error)}
                        continue
                    weight_launches = launches[1:-1]
                    retained.extend(weight_launches)
                    graphs[str(config)] = capture(reader, weight_launches)
                    del launches
            arms.update(measure(graphs))
            print(json.dumps({"shape": shape, "arms": arms}), file=output, flush=True)


if __name__ == "__main__":
    main()
