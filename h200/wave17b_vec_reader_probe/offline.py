"""Compile the reader through its NVRTC path and assemble/disassemble offline."""

# Offline assembly deliberately uses the reader's private NVRTC path and CUDA binaries.
# ruff: noqa: ANN001, ANN201, ANN204, S603, SLF001, T201
import argparse
import itertools
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from lnet import factorized_reader_kernels as reader


class OfflineModule:
    def __init__(self, image, _device):
        self.image = image.value

    def load_function(self, _name, _smem):
        pass


def compile_case(job):
    target, geometry, dtype, rank, strict, vec, out, cuda = job
    shape, columns, spatial, weight, readonly, keep_real = geometry
    tag = "-".join(map(str, (target, dtype, rank, int(strict), vec, *shape, columns,
                            *spatial, *weight, int(readonly), keep_real)))
    directory = out / tag
    directory.mkdir(exist_ok=True)
    torch.cuda.get_device_capability = lambda _: tuple(map(int, target))
    reader._CudaModule = OfflineModule
    reader._reader_cuda_module.cache_clear()
    try:
        module = reader._reader_cuda_module(
            0, shape, columns, rank, 3, (dtype,) * 3, strict=strict,
            spatial_geometry=spatial, weight_geometry=weight, weight_vec=vec, spatial_vec=vec,
            register_filters=0, keep_real=keep_real, spatial_read_only=readonly,
            weight_read_only=readonly,
        )
        ptx = directory / "reader.ptx"
        elf = directory / "reader.cubin"
        ptx.write_bytes(module.image)
        with (directory / "ptxas.log").open("w") as log:
            subprocess.run(
                [cuda / "ptxas", "-v", f"-arch=sm_{target}", ptx, "-o", elf],
                stderr=log, check=True,
            )
        if vec > 1:
            with (directory / "sass.log").open("w") as log:
                subprocess.run([cuda / "cuobjdump", "--dump-sass", elf], stdout=log, check=True)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        (directory / "error.log").write_text(str(error))
        if dtype == "float" and vec > 1 and "ModePair" in str(error):
            return tag, "unsupported_fp32_vector"
        return tag, str(error)
    return tag, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda/bin"))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    geometries = [
        ((0, 0, 0, 0), 8, (1, 4), (1, 1, 4), False),
        ((128, 56, 56, 160), 7, (1, 4), (1, 1, 4), False),
    ]
    if args.ablate:
        geometries.extend(
            ((128, 56, 56, 160), col, (rows, warps), (tile, wr, ww), ro)
            for col, rows, warps, tile, wr, ww, ro in (
                (4, 1, 2, 1, 1, 4, False), (7, 2, 4, 1, 2, 4, True),
                (8, 1, 8, 1, 4, 8, True), (14, 2, 2, 2, 1, 4, False),
                (8, 2, 8, 2, 2, 8, False), (14, 1, 4, 2, 4, 8, True),
            )
        )
    if args.tuned:
        geometries = [
            (shape, columns, spatial, weight, False)
            for shape, (columns, spatial, weight) in itertools.product(
                ((0, 0, 0, 0), (128, 56, 56, 160), (128, 28, 28, 320)),
                (
                    (14, (4, 2), (2, 4, 4)), (8, (4, 2), (2, 4, 4)),
                    (7, (4, 4), (1, 4, 4)), (7, (2, 4), (1, 2, 4)),
                ),
            )
        ]
    geometries = [(*geometry, 0) for geometry in geometries]
    if args.matrix:
        geometries = [
            ((128, 56, 56, 160), columns, (rows, warps), (tile, rows, warps), False, keep)
            for columns, rows, warps, tile, keep in itertools.product(
                (7, 8, 14), (2, 4), (2, 4), (1, 2), (0, 1),
            )
        ]
    jobs = [
        (target, geometry, dtype, rank, strict, vec, args.output, args.cuda)
        for target, geometry, dtype, rank, strict, vec in itertools.product(
            ("86", "89", "90"), geometries, ("__half", "__nv_bfloat16", "float"),
            (1, 2, 4), (False, True), (1, 2, 4),
        )
        if args.matrix or vec == 1 or (dtype != "float" and not strict)
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(compile_case, jobs))
    unsupported = sum(error == "unsupported_fp32_vector" for _, error in results)
    failures = [(tag, error) for tag, error in results
                if error and error != "unsupported_fp32_vector"]
    for tag, error in failures:
        print(tag, error)
    compiled = len(results) - len(failures) - unsupported
    print(f"{compiled}/{len(results)} compiled; {unsupported} unsupported FP32 vectors; "
          f"{len(failures)} unexpected failures")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
