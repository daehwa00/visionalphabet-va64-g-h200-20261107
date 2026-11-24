"""Assemble candidate reader tiles on SM 9.0 without a CUDA device."""

# ruff: noqa: ANN001, ANN201, S603, SLF001, T201
import argparse
import itertools
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from offline import OfflineModule
from replay import resources

from lnet import factorized_reader_kernels as reader

PTXAS = "/usr/local/cuda-12.8/bin/ptxas"


def compile_candidate(job):
    op, height, dtype, vec, rows, columns, warps, rank_tile, stream = job
    modes = 128 if height == 7 else 160
    weight = op == "dweight"
    with (
        patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        patch.object(reader, "_CudaModule", OfflineModule),
        TemporaryDirectory() as scratch,
    ):
        path = Path(scratch)
        module = reader._reader_cuda_module(
            0, (1024, height, height, modes), columns, 4, 3, (dtype,) * 3,
            strict=False, spatial_geometry=(rows, warps) if not weight else (2, 1),
            weight_geometry=(rank_tile, rows, warps) if weight else None,
            spatial_vec=1 if weight else vec, weight_vec=vec if weight else 1,
            register_filters=0, keep_real=0, updated_spatial=True,
            weight_read_only=vec != 2, stream_forward=stream,
        )
        reader._reader_cuda_module.cache_clear()
        ptx = path / "reader.ptx"
        ptx.write_bytes(module.image)
        log = subprocess.run(
            [PTXAS, "-v", "-arch=sm_90", str(ptx), "-o", str(path / "reader.cubin")],
            check=True, capture_output=True, text=True,
        )
        return [*job, resources(log.stderr)[op]]


def jobs():
    for dtype in ("__nv_bfloat16", "__half"):
        for height in (7, 14, 28, 56):
            for rows in ((7, 8) if height == 56 else (7,)):
                for stream in (False, True):
                    yield "forward", height, dtype, 2, rows, 8 if height == 7 else 14, (
                        1 if height <= 14 else 2
                    ), 4, stream
            for rows, columns, warps in itertools.product((4, 7, 8), (7, 14), (2, 4)):
                if rows <= height:
                    yield "dinput", height, dtype, 1, rows, columns, warps, 4, False
            for vec, rows, warps, tile in itertools.product((1, 2), (4, 7, 8), (2, 4), (1, 2)):
                if rows <= height:
                    yield "dweight", height, dtype, vec, rows, 7 * vec, warps, tile, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinput-rank-loop", action="store_true")
    args = parser.parse_args()
    selected = jobs()
    if args.dinput_rank_loop:
        # Keep the trial out of production. Dinput indexes its window by input component,
        # so a runtime output-rank loop can leave the row windows in registers.
        source = reader._READER_CUDA
        start = source.index("        V ar[Rows] = {}, ai[Rows] = {};")
        reader._READER_CUDA = source[:start] + source[start:].replace(
            "#pragma unroll", "#pragma unroll (Backward ? 1 : R)", 1,
        )
        selected = (job for job in selected if job[0] == "dinput" and job[1] == 56 and job[4] == 8)
    with ProcessPoolExecutor(max_workers=12, mp_context=get_context("fork")) as pool:
        for result in pool.map(compile_candidate, selected):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
