"""Compile the exact NVRTC templates without exposing a CUDA device."""

import argparse
import json
import re
import subprocess
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from variants import BACKWARD, FORWARD, Trial, geometry

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, S603


type Case = tuple[int, str, bool, bool, str, scans.ScanPlan, scans.Kernel]


def build(item: Case, ptxas: str, folder: Path) -> dict:
    arch, dtype, coarse, backward, name, plan, key = item
    label = f"{arch}-{dtype}-{int(coarse)}-{int(backward)}-{plan.warps}-{name}"
    ptx = folder / f"{label}.ptx"
    ptx.write_bytes(scans._compile_ptx(key, (arch // 10, arch % 10)).value)
    result = subprocess.run(
        [ptxas, "-v", f"-arch=sm_{arch}", str(ptx), "-o", str(ptx.with_suffix(".cubin"))],
        text=True, capture_output=True, check=True,
    )
    log = result.stderr
    resources = re.search(r"Used (\d+) registers", log)
    spills = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    if not resources or not spills:
        raise RuntimeError(log)
    return {
        "arch": arch, "dtype": dtype, "coarse": coarse, "backward": backward,
        "variant": name, "plan": plan._asdict(), "registers": int(resources[1]),
        "spill_store_bytes": int(spills[1]), "spill_load_bytes": int(spills[2]), "log": log,
    }


def candidates(
    arches: Iterable[int], dtypes: Iterable[torch.dtype], warps: Iterable[int],
) -> Iterator[Case]:
    for arch in arches:
        for dtype in dtypes:
            for coarse in (False, True):
                for backward, options in ((False, FORWARD), (True, BACKWARD)):
                    for nwarps in warps:
                        plan = scans.ScanPlan(
                            nwarps, 0, 8, cache_weights=True, cache_source=backward,
                        )
                        key = scans.Kernel(
                            dtype, nwarps, backward, vertical=True, coarse=coarse, hidden8=True,
                            cache_weights=True, precise=not backward, compact=backward and coarse,
                            cache_source=backward,
                        )
                        base = scans.Launch(plan, key, 0, 0)
                        candidates = {"baseline": Trial(vector_weights=False), **options}
                        for name, trial in candidates.items():
                            plan, key = geometry(
                                base, trial, 56, 8, nwarps, dtype.itemsize,
                                backward=backward, coarse=coarse,
                            )
                            yield arch, str(dtype), coarse, backward, name, plan, key


def selected_candidates(
    arches: Iterable[int], dtypes: Iterable[torch.dtype], warps: Iterable[int],
) -> Iterator[Case]:
    for arch in arches:
        for dtype in dtypes:
            for coarse in (False, True):
                for backward, heights in ((False, (28, 56)), (True, (14, 28, 56, 128))):
                    for h in heights:
                        plans = scans._scan_plans(
                            h, 227 * 1024, vertical=True, backward=backward, coarse=coarse,
                            hidden=8, itemsize=dtype.itemsize,
                        )
                        for packed in plans:
                            if (
                                packed.warps not in warps or not packed.cache_weights
                                or packed.shared_acc or packed.global_checkpoints
                                or (coarse and not backward and not packed.cache_line)
                                or (backward and packed.chunk not in (4, 8, h))
                            ):
                                continue
                            chunk = 0 if packed.full_trajectory else packed.chunk
                            shape = f"{h}-{chunk}-{int(packed.cache_line)}"
                            shape += f"-{int(packed.cache_source)}"
                            plan = packed._replace(smem=packed.smem + 11 * 8 * 32 * 2)
                            if plan.smem > 227 * 1024:
                                continue
                            key = scans.Kernel(
                                dtype, plan.warps, backward, vertical=True, coarse=coarse,
                                hidden8=True, cache_weights=True, cache_line=plan.cache_line,
                                cache_source=plan.cache_source, compact=backward and coarse,
                                chunk=chunk, vector_weights=True,
                            )
                            yield (
                                arch, str(dtype), coarse, backward, f"vector-{shape}", plan, key,
                            )
                            yield (
                                arch, str(dtype), coarse, backward, f"packed-{shape}", packed,
                                replace(key, vector_weights=False, precise=not backward),
                            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptxas", default="/usr/local/cuda/bin/ptxas")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", type=int, nargs="+", default=(86, 89, 90))
    parser.add_argument("--warps", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--selected", action="store_true")
    args = parser.parse_args()
    dtypes = [torch.bfloat16, torch.float16] if args.fp16 else [torch.bfloat16]
    with (
        TemporaryDirectory(prefix="va-t57-ptxas-") as tmp,
        args.out.open("w") as out,
        ThreadPoolExecutor(max_workers=args.jobs) as pool,
    ):
        matrix = selected_candidates if args.selected else candidates
        for row in pool.map(
            partial(build, ptxas=args.ptxas, folder=Path(tmp)),
            matrix(args.arch, dtypes, args.warps),
        ):
            print(json.dumps(row), file=out, flush=True)


if __name__ == "__main__":
    main()
