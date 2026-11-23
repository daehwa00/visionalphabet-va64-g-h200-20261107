"""Compile representative scalar and vector geometries without a CUDA device."""
# ruff: noqa: S603

from __future__ import annotations

import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import torch

from lnet.fused_rmsnorm_ops import _compile_ptx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-bin", type=Path, default=Path("/usr/local/cuda/bin"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Cover every register pack count and lane count, plus masked scalar tails.
    widths = sorted({1, 2, 3, 4, 7, 8, 12, 16, 24, 33, 47, 1023, *range(32, 1025, 32)})
    cases = []
    for sm, dtype, channels in product(
        ((8, 6), (8, 9), (9, 0)), (torch.bfloat16, torch.float16, torch.float32), widths
    ):
        cases.append((sm, dtype, channels, 1))
        vector = 4 if dtype == torch.float32 else 8
        if channels % vector == 0:
            cases.append((sm, dtype, channels, vector))


    def compile_case(case: tuple[tuple[int, int], torch.dtype, int, int]) -> tuple[str, bool]:
        sm, dtype, channels, vector = case
        arch = f"sm_{sm[0]}{sm[1]}"
        with tempfile.TemporaryDirectory() as temp:
            ptx = Path(temp) / "rms.ptx"
            cubin = ptx.with_suffix(".cubin")
            ptx.write_bytes(_compile_ptx(sm, channels, dtype, vector).value)
            asm = subprocess.run(
                [args.cuda_bin / "ptxas", "-arch", arch, "-v", ptx, "-o", cubin],
                capture_output=True, text=True, check=True,
            ).stderr
            sass = subprocess.run(
                [args.cuda_bin / "cuobjdump", "-sass", cubin], capture_output=True,
                text=True, check=True,
            ).stdout
            loads = sass.count("LDG.E.128")
            stores = sass.count("STG.E.128")
            name = f"{arch} {dtype} C={channels} V={vector} LDG.128={loads} STG.128={stores}"
            zero_spills = "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"
            passed = asm.count(zero_spills) == 3
            return f"{name}\n{asm}", passed

    ok = True
    with args.out.open("w") as out, ThreadPoolExecutor(max_workers=8) as pool:
        for result, passed in pool.map(compile_case, cases):
            out.write(result)
            ok &= passed
    if not ok:
        raise SystemExit("at least one specialization uses stack or spills")


if __name__ == "__main__":
    main()
