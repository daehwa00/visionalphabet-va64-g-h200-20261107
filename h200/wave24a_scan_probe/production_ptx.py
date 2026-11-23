"""Record PTX identities and backend resources for the production scan policy."""

import argparse
import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from itertools import product
from pathlib import Path

import torch
from resources import measure

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, FBT003


def keys() -> Iterator[scans.Kernel]:
    for dtype, coarse, layouts, precision in product(
        (torch.bfloat16, torch.float16), (False, True), product((False, True), repeat=2),
        ((False, False), (False, True), (True, False)),
    ):
        vector, precise = precision
        yield scans.Kernel(
            dtype, 1, False, vertical=True, coarse=coarse, hidden8=True,
            cache_weights=True, vector_weights=vector, precise=precise, batch_path=True,
            real_interleaved=layouts[0], imag_interleaved=layouts[1],
        )
    for dtype, layout in product((torch.bfloat16, torch.float16), product((False, True), repeat=4)):
        yield scans.Kernel(
            dtype, 1, True, cache_line=True, real_interleaved=layout[0],
            imag_interleaved=layout[1], real_transposed=layout[2], imag_transposed=layout[3],
        )
    for dtype, warps, horizontal, backward in product(
        (torch.bfloat16, torch.float16, torch.float32), (1, 4), (False, True), (False, True),
    ):
        yield scans.Kernel(
            dtype, warps, backward, vertical=not horizontal, hidden8=not horizontal,
            cache_weights=not horizontal, cache_line=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", required=True, type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--ptx-dir", type=Path)
    args = parser.parse_args()
    if args.ptx_dir is not None:
        args.ptx_dir.mkdir(parents=True, exist_ok=True)
    with args.off.open("w") as output:
        for index, (sm, key) in enumerate(product((86, 89, 90), keys())):
            data = scans._compile_ptx(key, (sm // 10, sm % 10)).value
            print(sm, hashlib.sha256(data).hexdigest(), file=output, flush=True)
            if args.ptx_dir is not None:
                (args.ptx_dir / f"{index:03d}-sm{sm}.ptx").write_bytes(data)
    if args.resources is not None:
        with args.resources.open("w") as output:
            for sm, key in product((86, 89, 90), keys()):
                candidate = None
                if key.batch_path and not key.coarse:
                    candidate = replace(key, register_path=True, cache_weights=False)
                if not key.vertical and key.backward and key.warps == 1 and key.dtype.itemsize == 2:
                    candidate = replace(key, horizontal_unroll8=True)
                if candidate is not None:
                    for trial in (key, candidate):
                        row = measure((sm, key.dtype, "baseline", trial, 0))
                        print(json.dumps(row), file=output, flush=True)


if __name__ == "__main__":
    main()
