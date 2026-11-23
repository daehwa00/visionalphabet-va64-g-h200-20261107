"""Compile and disassemble the projection and pure vertical planner variants on CPU."""

import argparse
import itertools
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, S603


def variants(arch: int) -> dict[scans.Kernel, int]:
    cases = {}
    types = (torch.float16, torch.bfloat16)
    for dtype, backward, warps, index32, real, imag, packed in itertools.product(
        types, (False, True), (4, 8), *((False, True),) * 4,
    ):
        key = scans.Kernel(
            dtype, warps, backward, projection=True, projection_tc=True, hidden8=True,
            index32=index32, real_interleaved=real, imag_interleaved=imag, packed_features=packed,
        )
        cases[key] = scans._tc_smem(warps, backward)
    limit = {86: 99, 89: 99, 90: 227}[arch] * 1024
    for dtype, coarse, backward in itertools.product(
        (*types, torch.float32), (False, True), (False, True),
    ):
        for height in (4, 8, 16, 24, 32, 56, 128, 512, 8192):
            for plan in scans._scan_plans(
                height, limit, vertical=True, backward=backward, coarse=coarse,
                hidden=0, itemsize=dtype.itemsize, pure_vertical=True,
            ):
                layouts = (False, True) if dtype in types else (False,)
                options = itertools.product((False, True), (False, True), layouts)
                for index32, option, packed in options:
                    key = scans.Kernel(
                        dtype, plan.warps, backward, vertical=True, coarse=coarse,
                        index32=index32, pure_vertical=True, cell_major=not packed,
                        packed_features=packed,
                        chunk=0 if plan.full_trajectory else plan.chunk,
                        compact=option and coarse and backward, precise=option and not backward,
                        cache_line=plan.cache_line, cache_source=plan.cache_source,
                        global_checkpoints=plan.global_checkpoints,
                    )
                    cases[key] = max(cases.get(key, 0), plan.smem)
    return cases


def compile_case(
    arch: int, key: scans.Kernel, smem: int, folder: Path, *, cuda_bin: Path,
) -> dict:
    source = folder / "kernel.ptx"
    cubin = source.with_suffix(".cubin")
    source.write_bytes(scans._compile_ptx(key, divmod(arch, 10)).value)
    result = subprocess.run(
        [str(cuda_bin / "ptxas"), "-v", f"-arch=sm_{arch}", str(source), "-o", str(cubin)],
        capture_output=True, text=True, check=True,
    )
    log = result.stderr
    stack, spill_store, spill_load = map(int, re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log,
    ).groups())
    sass = []
    if key.projection_tc:
        result = subprocess.run(
            [str(cuda_bin / "cuobjdump"), "--dump-sass", str(cubin)],
            capture_output=True, text=True, check=True,
        )
        sass = [line.strip() for line in result.stdout.splitlines() if "HMMA.168" in line]
    return {
        "arch": arch, "key": str(key), "projection": key.projection_tc,
        "dtype": str(key.dtype), "backward": key.backward, "warps": key.warps,
        "registers": int(re.search(r"Used (\d+) registers", log)[1]), "launch_smem": smem,
        "stack": stack, "spill_store": spill_store, "spill_load": spill_load,
        "ptxas": log, "sass": sass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda-bin", type=Path, default=Path("/usr/local/cuda/bin"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    def run(item: tuple[int, scans.Kernel, int]) -> dict:
        with TemporaryDirectory(prefix="wave17a-ptxas-") as folder:
            return compile_case(*item, Path(folder), cuda_bin=args.cuda_bin)

    tasks = [(arch, key, smem) for arch in (86, 89, 90) for key, smem in variants(arch).items()]
    failed = 0
    with ThreadPoolExecutor(args.jobs) as pool, args.out.open("w", encoding="utf-8") as log:
        for result in pool.map(run, tasks):
            print(json.dumps(result), file=log, flush=True)
            failed += bool(
                result["stack"] or result["spill_store"] or result["spill_load"]
                or (result["projection"] and not result["sass"])
            )
    print(f"{len(tasks)} variants, {failed} failed resource or HMMA checks")  # noqa: T201
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
