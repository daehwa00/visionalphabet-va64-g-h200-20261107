"""Compare baseline PTX and audit Wave 25A register and local-memory use."""

import argparse
import hashlib
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch
from resources import measure

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, FBT003

DTYPES = torch.bfloat16, torch.float16
SMS = 86, 89, 90


def keys() -> Iterator[scans.Kernel]:
    for dtype, warps, coarse, line, batch, policy, interleaved in product(
        DTYPES, range(1, 5), (False, True), (False, True), (False, True),
        ((False, False), (False, True), (True, False)), (False, True),
    ):
        vector, precise = policy
        yield scans.Kernel(
            dtype, warps, False, vertical=True, coarse=coarse, hidden8=True,
            cache_weights=True, cache_line=line, vector_weights=vector,
            precise=precise, batch_path=batch,
            real_interleaved=interleaved, imag_interleaved=interleaved,
        )
    for dtype, warps, backward, compact, line, layout in product(
        DTYPES, range(1, 5), (False, True), (False, True), (False, True),
        product((False, True), repeat=4),
    ):
        if compact and not backward:
            continue
        yield scans.Kernel(
            dtype, warps, backward, compact=compact, cache_line=line,
            real_interleaved=layout[0], imag_interleaved=layout[1],
            real_transposed=layout[2], imag_transposed=layout[3],
        )


def candidates(key: scans.Kernel) -> Iterator[scans.Kernel]:
    if key.vertical:
        candidate = replace(key, register_path=True, cache_weights=False)
        yield candidate
        if not key.batch_path:
            yield replace(candidate, batch_path=True, register_batch2=True)
    elif key.backward:
        yield replace(key, horizontal_unroll8=True)
    else:
        for unroll in (4, 8):
            yield replace(key, horizontal_forward_unroll=unroll)


def selected_keys() -> set[scans.Kernel]:
    selected = set()
    with patch.object(scans, "_resident", lambda *_: 32):
        for key, length in product(keys(), (7, 14, 28, 56)):
            threads = key.warps * 32
            if key.vertical:
                weights = 11 * 8 * 32 * (4 if key.vector_weights else key.dtype.itemsize)
                line = length // (1 + key.coarse) * 4 * key.dtype.itemsize * threads
            else:
                weights = 0
                line = length // (1 + key.compact) * (8 if key.backward else 4) * threads
            plan = scans.ScanPlan(
                key.warps, weights + (line if key.cache_line else 0),
                cache_weights=key.cache_weights, cache_line=key.cache_line,
            )
            base = scans.Launch(plan, key, 1024, 1024)
            launch = scans._with_wave25a(base, 0, (1024, length, length, 160))
            if launch != base:
                selected.add(launch.key)
    return selected


def fingerprint(job: tuple[int, scans.Kernel]) -> str:
    sm, key = job
    ptx = scans._compile_ptx(key, (sm // 10, sm % 10)).value
    return f"{sm} {hashlib.sha256(ptx).hexdigest()}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--selected", action="store_true")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        if args.off:
            with args.off.open("w") as output:
                output.writelines(pool.map(fingerprint, product(SMS, keys())))
        if args.resources:
            trials = (
                sorted(selected_keys(), key=repr) if args.selected else
                [candidate for key in keys() for candidate in candidates(key)]
            )
            with args.resources.open("w") as output:
                for row in pool.map(measure, product(SMS, trials)):
                    print(json.dumps(row), file=output, flush=True)


if __name__ == "__main__":
    main()
