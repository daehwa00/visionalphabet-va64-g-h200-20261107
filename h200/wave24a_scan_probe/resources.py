"""Compile probe scans off-device and record ptxas and SASS resource data."""

import argparse
import json
import re
import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from variants import TrialKernel, experimental, source

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, FBT003, S603, S607


def measure(job: tuple) -> dict:
    sm, dtype, name, key, cap = job
    trial = experimental(key, name)
    if cap:
        trial = TrialKernel(**vars(trial), cap=cap) if type(trial) is scans.Kernel else replace(
            trial, cap=cap,
        )
    ptx = scans._compile_ptx(trial, (sm // 10, sm % 10)).value
    with TemporaryDirectory() as folder:
        src = Path(folder) / "scan.ptx"
        dst = Path(folder) / "scan.cubin"
        src.write_bytes(ptx)
        command = ["ptxas", "-v", "-arch", f"sm_{sm}", str(src), "-o", str(dst)]
        report = subprocess.run(command, capture_output=True, text=True, check=True).stderr
        sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(dst)], text=True)
    return {
        "sm": sm, "dtype": str(dtype), "trial": name, "cap": cap,
        "key": {**vars(key), "dtype": str(dtype)}, "ptxas": report,
        "registers": int(re.search(r"Used (\d+) registers", report)[1]),
        "stack_spill": [int(v) for v in re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", report,
        ).groups()],
        "sass": {op: len(re.findall(rf"\b{op}\b", sass)) for op in ("LDS", "LDG", "LDL", "STL")},
    }


def jobs(sms: list[int]) -> Iterator[tuple]:
    for sm, dtype, coarse in product(sms, (torch.bfloat16, torch.float16), (False, True)):
        vertical = scans.Kernel(dtype, 1, False, vertical=True, coarse=coarse,
                                hidden8=True, cache_weights=True, vector_weights=True,
                                real_interleaved=True)
        for warps, line in ((1, False), (4, True)):
            key = replace(vertical, warps=warps, cache_line=line, batch_path=not line)
            for name in ("baseline", "reg1", "reg2", "reg4", "float1", "float2", "float4"):
                yield sm, dtype, name, key, 0
        for warps, chunk, line, cache_source in (
            (2, 8, False, False), (4, 8, False, False), (4, 0, True, True),
            (4, 0, False, True), (2, 14, True, False), (4, 14, False, False),
        ):
            key = replace(vertical, backward=True, warps=warps, chunk=chunk,
                          cache_line=line, cache_source=cache_source)
            for name in ("baseline", "packed", "unroll1", "unroll2", "unroll4", "vy"):
                yield sm, dtype, name, key, 0
        key = replace(vertical, backward=True, warps=2, chunk=8)
        for name, cap in product(("baseline", "unroll1", "unroll2", "unroll4"),
                                 (128, 160, 168, 192, 255)):
            yield sm, dtype, name, key, cap
        for warps in range(1, 5):
            key = scans.Kernel(dtype, warps, True, cache_line=True, compact=coarse)
            for name in ("baseline", "hu1", "hu8", "hldg") if coarse else (
                "baseline", "hu1", "hu2", "hu8", "hldg",
            ):
                yield sm, dtype, name, key, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sm", type=int, nargs="+", default=[86, 89, 90])
    args = parser.parse_args()
    with (
        patch.object(scans, "_kernel_source", source),
        args.out.open("w") as stream,
        ThreadPoolExecutor(max_workers=8) as executor,
    ):
        for row in executor.map(measure, jobs(args.sm)):
            print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    main()
