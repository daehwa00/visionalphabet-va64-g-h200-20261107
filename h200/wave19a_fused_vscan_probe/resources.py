"""Measure static fused-kernel resources without a visible CUDA device."""

import argparse
import itertools
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from lnet import general_cuda_scans as scans

# ruff: noqa: S603, SLF001


def compile_case(case: tuple, cuda: Path, output: Path) -> dict:
    arch, dtype, coarse, backward, warps, vector, staged, height, trajectory = case
    chunk = 0 if trajectory == "full" else (
        scans._SHORT_CHECKPOINT_CHUNK if height <= scans._SHORT_SCAN else scans._CHECKPOINT_CHUNK
    )
    key = scans.Kernel(
        dtype, warps, backward, vertical=True, coarse=coarse, hidden8=True,
        chunk=chunk, cache_weights=True, cache_line=True, cache_source=trajectory != "replay",
        shared_path_rows=staged, precise=not backward and not vector, vector_weights=vector,
        compact=coarse and backward,
    )
    plans = scans._scan_plans(
        height, 227 * 1024, vertical=True, backward=backward, coarse=coarse,
        hidden=8, itemsize=dtype.itemsize,
    )
    plan = next(
        p for p in plans if p.warps == warps and p.cache_weights and p.cache_line
        and not p.shared_acc and p.cache_source == key.cache_source
        and p.full_trajectory == (trajectory == "full")
        and (p.full_trajectory or p.chunk == chunk)
    )
    threads = warps * 32
    smem = plan.smem + vector * 88 * 32 * (4 - dtype.itemsize) + staged * 8 * threads * 4
    stem = output / "-".join(map(str, case))
    ptx, cubin = Path(f"{stem}.ptx"), Path(f"{stem}.cubin")
    ptx.write_bytes(scans._compile_ptx(key, divmod(arch, 10)).value + b"\n")
    log = subprocess.run(
        [cuda / "ptxas", "-v", f"-arch=sm_{arch}", ptx, "-o", cubin],
        check=True, capture_output=True, text=True,
    ).stderr
    sass = subprocess.run(
        [cuda / "cuobjdump", "--dump-sass", cubin], check=True, capture_output=True, text=True,
    ).stdout
    Path(f"{stem}.sass").write_text(sass)
    regs = int(re.search(r"Used (\d+) registers", log)[1])
    stack, stores, loads = map(int, re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log,
    ).groups())
    # Register granularity is 256 registers/warp. SM reserves 1 KiB per block.
    sm_capacity = {86: 100, 89: 100, 90: 228}[arch] * 1024
    active = min(
        65536 // (((regs + 7) // 8) * 256 * warps),
        sm_capacity // (((smem + 255) // 256) * 256 + 1024),
        (64 if arch == 90 else 48) // warps,
        {86: 16, 89: 24, 90: 32}[arch],
    )
    return {
        "arch": arch, "dtype": str(dtype), "coarse": coarse, "backward": backward,
        "height": height, "trajectory": trajectory, "warps": warps,
        "vector": vector, "staged": staged, "smem": smem,
        "registers": regs, "active_ctas_estimate": active,
        "stack": stack, "spill_stores": stores, "spill_loads": loads, "ptxas": log,
        "sass": {op: len(re.findall(rf"\b{op}(?:\.|\s)", sass))
                 for op in ("LDG", "STG", "LDS", "STS", "FFMA", "HMMA", "MUFU")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda/bin"))
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (
        (*case, 28, "replay") for case in itertools.product(
            (86, 89, 90), (torch.bfloat16, torch.float16, torch.float32),
            (False, True), (False, True), scans._SCAN_WARPS,
            (False, True), (0, scans._STAGED_PATH_ROWS),
        )
        if (case[1] != torch.float32 or not case[5]) and (case[3] or not case[6])
    )
    if args.extended:
        trajectory_cases = (
            (arch, dtype, coarse, True, warps, vector, staged, height, trajectory)
            for arch, dtype, coarse, warps, vector, staged, height, trajectory in itertools.product(
                (86, 89, 90), (torch.bfloat16, torch.float16, torch.float32),
                (False, True), (1, 2), (False, True), (0, scans._STAGED_PATH_ROWS),
                (14, 28, 56), ("replay", "cached", "full"),
            )
            if (dtype != torch.float32 or not vector)
            and (arch == 90 or trajectory == "replay")
            and (height != 56 or trajectory != "full")
            and (height != 28 or trajectory != "replay")
        )
        cases = itertools.chain(cases, trajectory_cases)
    with (args.output / "resources.jsonl").open("w") as out, ThreadPoolExecutor(8) as pool:
        for row in pool.map(lambda case: compile_case(case, args.cuda, args.output), cases):
            print(json.dumps(row), file=out, flush=True)


if __name__ == "__main__":
    main()
