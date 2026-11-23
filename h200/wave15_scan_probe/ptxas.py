# ruff: noqa: SLF001, S603, ANN001, ANN201, T201, E501
"""Check Wave 15A CUDA resources without creating a CUDA context."""

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lnet import general_cuda_scans as scans

DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def compile_case(case, root, toolkit):
    arch, variant, key = case
    stem = f"{arch}-{variant}-{key.dtype}-{key.coarse}-{key.vertical}-{key.backward}"
    ptx = root / f"{stem}.ptx"
    ptx.write_bytes(scans._compile_ptx(key, divmod(arch, 10)).value)
    cubin = ptx.with_suffix(".cubin")
    log = subprocess.run(
        [str(toolkit / "ptxas"), "-v", f"-arch=sm_{arch}", str(ptx), "-o", str(cubin)],
        capture_output=True, text=True, check=True,
    ).stderr
    sass = subprocess.run(
        [str(toolkit / "cuobjdump"), "-sass", str(cubin)], capture_output=True, text=True, check=True,
    ).stdout
    instructions = re.findall(r"\b(?:LDG|STG)(?:\.[A-Z0-9_]+)*\b", sass)
    opcodes = [item.split(".") for item in instructions]
    return {
        "arch": arch, "variant": variant, "dtype": str(key.dtype), "warps": key.warps,
        "vertical": key.vertical, "backward": key.backward, "coarse": key.coarse,
        "registers": int(re.search(r"Used (\d+) registers", log)[1]),
        "spills": list(map(int, re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log).groups())),
        "memory_instructions": {
            op + (f".{width}" if width else ""): sum(
                parts[0] == op and (not width or width in parts[1:]) for parts in opcodes
            )
            for op in ("LDG", "STG") for width in ("", "64", "128")
        },
        "log": log,
    }


def cases():
    for arch, dtype, backward in product((86, 89, 90), DTYPES, (False, True)):
        for layout in range(3):
            for compact in range(1 + backward):
                yield arch, f"layout{layout}-{compact}", scans.Kernel(
                    dtype, 2, backward, cache_line=not backward, compact=bool(compact),
                    state_layout=layout,
                )
        for coarse in (False, True):
            key = scans.Kernel(
                dtype, 2, backward, vertical=True, coarse=coarse, hidden8=True,
                compact=backward and coarse, cache_weights=True, cache_source=backward,
                precise=not backward, vector_weights=backward,
            )
            yield arch, "planar", key
            yield arch, "local", replace(key, local_scratch=True)
            for layout in (1, 2):
                packed = replace(key, state_layout=layout)
                yield arch, f"layout{layout}", packed
                if backward:
                    single = replace(packed, local_scratch=True, single_store=True)
                    yield arch, f"single{layout}", single
                    yield arch, f"drop{layout}", replace(single, cache_source=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit", type=Path, required=True)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="va-t59-ptxas-") as directory, ThreadPoolExecutor(6) as pool:
        for row in pool.map(lambda case: compile_case(case, Path(directory), args.toolkit), cases()):
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
