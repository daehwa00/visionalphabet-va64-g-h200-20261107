"""Assemble the weight-window ablations for the three CUDA targets."""

# ruff: noqa: ANN001, S603, SLF001
import argparse
import itertools
import json
import subprocess
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from offline import OfflineModule
from replay import SM, resources
from screen import REAL
from weight_window import compile_weight

from lnet import factorized_reader_kernels as reader


def occupancy(meta: dict, target: str, warps: int) -> int:
    _, sm_warps, max_blocks, smem, _ = SM[target]
    registers = reader._ceildiv(meta["regs"] * 32, 256) * 256
    cta_shared = reader._ceildiv(meta["smem"] + 1024, 256) * 256
    return min(max_blocks, sm_warps // warps, (65536 // (registers * 4) * 4) // warps,
               smem * 1024 // cta_shared)


def assemble(module: OfflineModule, ptx: Path, cubin: Path, ptxas: str, target: str) -> dict:
    ptx.write_bytes(module.image)
    result = subprocess.run(
        [ptxas, "-v", f"-arch=sm_{target}", ptx, "-o", cubin],
        capture_output=True, text=True, check=True,
    )
    return resources(result.stderr)


def report(output, target: str, shape: tuple, dtype: str, op: str, vec: int, rows: int,
           cols: int, warps: int, grid: tuple, meta: dict, partial_bytes: int = 0) -> None:
    sms = SM[target][0]
    resident = occupancy(meta, target, warps)
    slots = resident * sms
    blocks = prod(grid)
    waves = reader._ceildiv(blocks, slots)
    record = dict(sm=target, shape=shape, dtype=dtype, op=op, vec=vec, rows=rows, cols=cols,
                  warps=warps, grid=grid, resident=resident, waves=waves,
                  wave_fill=blocks / (waves * slots), partial_bytes=partial_bytes, **meta)
    print(json.dumps(record), file=output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ptxas", default="/usr/local/cuda/bin/ptxas")
    parser.add_argument("--sm", choices=SM, nargs="+", default=tuple(SM))
    args = parser.parse_args()
    configurations = (
        (4, 4, False, False), (7, 4, False, False), (7, 8, False, False),
        (8, 4, False, False), (14, 4, False, False), (14, 8, False, False),
    )
    with (
        TemporaryDirectory() as tmp,
        args.out.open("w") as output,
        patch.object(reader, "_CudaModule", OfflineModule),
    ):
        ptx, cubin = Path(tmp) / "reader.ptx", Path(tmp) / "reader.cubin"
        for target in args.sm:
            capability = tuple(map(int, target))
            batch = 1024 if target == "90" else 128
            reader._reader_cuda_module.cache_clear()
            with patch.object(torch.cuda, "get_device_capability", return_value=capability):
                cases = itertools.product(REAL, ("__nv_bfloat16", "__half"), configurations)
                for (height, width, modes, rank), dtype, (rows, warps, readonly, convert) in cases:
                    shape = batch, height, width, modes
                    module = compile_weight(
                        0, shape, reader._reader_columns(width), rank, 3, (dtype,) * 3,
                        strict=False, weight_geometry=(2, rows, warps), weight_vec=2,
                        weight_read_only=readonly, convert=convert,
                    )
                    sms = SM[target][0]
                    modes_per_grid = reader._ceildiv(modes, 64)
                    workers = min(reader._ceildiv(batch * reader._ceildiv(height, rows), warps),
                                  reader._ceildiv(5 * sms, modes_per_grid))
                    grid = (modes_per_grid * reader._ceildiv(rank, 2), workers, 1)
                    metadata = assemble(module, ptx, cubin, args.ptxas, target)
                    report(output, target, shape, dtype, "dweight", 2, rows,
                           reader._reader_columns(width), warps, grid, metadata["dweight"],
                           workers * 2 * rank * 9 * modes * 4)
                for dtype in ("__nv_bfloat16", "__half"):
                    shape, rows, warps, rank = (batch, 14, 14, 160), 7, 4, 4
                    columns = reader._reader_columns(shape[2])
                    workers = min(reader._ceildiv(batch * 2, warps),
                                  reader._ceildiv(5 * SM[target][0], 5))
                    module = reader._reader_cuda_module(
                        0, shape, columns, rank, 3, (dtype,) * 3, strict=False,
                        weight_geometry=(rank, rows, warps),
                    )
                    metadata = assemble(module, ptx, cubin, args.ptxas, target)
                    report(output, target, shape, dtype, "dweight", 1, rows, columns, warps,
                           (5, workers, 1), metadata["dweight"], workers * 2 * rank * 9 * 160 * 4)
                for dtype, (vec, cols, warps) in itertools.product(
                    ("__nv_bfloat16", "__half"), ((1, 7, 2), (2, 8, 2), (2, 14, 1)),
                ):
                    shape, rows, rank = (batch, 14, 14, 160), 7, 4
                    module = reader._reader_cuda_module(
                        0, shape, cols, rank, 3, (dtype,) * 3, strict=False,
                        spatial_geometry=(rows, warps), spatial_vec=vec, updated_spatial=True,
                        register_filters=0, keep_real=0,
                    )
                    metadata = assemble(module, ptx, cubin, args.ptxas, target)
                    grid = reader._reader_spatial_grid(*shape, rows, warps, vec, cols)
                    for op in ("forward", "dinput"):
                        report(output, target, shape, dtype, op, vec, rows, cols, warps, grid,
                               metadata[op])



if __name__ == "__main__":
    main()
