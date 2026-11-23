"""Compile the in-SMEM warp transposition and check baseline PTX invariance."""

import argparse
import itertools
import json
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch

from lnet import general_cuda_scans as scans

# ruff: noqa: S603, S607, SLF001, T201
BASE = "039440f2c0a9a9d006948fcbbf45dd5c96e36efb"


def baseline_module() -> types.ModuleType:
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:src/lnet/general_cuda_scans.py"], text=True,
    )
    baseline = types.ModuleType("lnet.wave21c_scan")
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, scans.__file__, "exec"), baseline.__dict__)  # noqa: S102
    return baseline


def compile_row(
    key: scans.Kernel, sm: tuple[int, int], directory: Path, baseline: types.ModuleType,
) -> dict:
    arch = f"sm_{sm[0]}{sm[1]}"
    ptx = directory / "scan.ptx"
    cubin = directory / "scan.cubin"
    ptx.write_bytes(scans._compile_ptx(key, sm).value)
    assemble = subprocess.run(
        ["ptxas", "-v", "-arch", arch, str(ptx), "-o", str(cubin)],
        capture_output=True, text=True, check=True,
    ).stderr
    registers = int(re.search(r"Used (\d+) registers", assemble)[1])
    stack, stores, loads = map(int, re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", assemble,
    ).groups())
    sass = subprocess.check_output(["cuobjdump", "-sass", str(cubin)], text=True)
    config = {
        "dtype": str(key.dtype), "warps": key.warps, "chunk": key.chunk,
        "cache_source": key.cache_source, "cache_line": key.cache_line, "coarse": key.coarse,
    }
    compiled = {
        "arch": arch, **config, "registers": registers, "stack": stack, "stores": stores,
        "loads": loads, **{op: len(re.findall(rf"\b{op}[.; ]", sass))
                          for op in ("HMMA", "FFMA", "LDS", "STS")},
    }
    height = 56 if key.chunk != 4 else 14
    plans = scans._scan_plans(
        height, 128 * 1024, vertical=True, backward=True, coarse=key.coarse,
        hidden=8, itemsize=2,
    )
    equivalent = [
        p for p in plans if p.warps == key.warps and p.cache_source == key.cache_source
        and (0 if p.full_trajectory else p.chunk) == key.chunk and p.cache_line == key.cache_line
        and p.cache_weights and not p.shared_acc and not p.global_checkpoints
    ]
    if equivalent:
        smem = equivalent[0].smem + 32 * 162 * 2
        threads = key.warps * 32
        reg_warp = ((registers * 32 + 255) // 256) * 256
        smem_per_sm = {"sm_86": 100, "sm_89": 100, "sm_90": 228}[arch] * 1024
        max_threads = 2048 if arch == "sm_90" else 1536
        compiled.update(height=height, smem=smem, cta_upper_bound=min(
            65536 // (reg_warp * key.warps), max_threads // threads,
            smem_per_sm // (((smem + 1024 + 255) // 256) * 256),
        ))
    base_args = dict(vars(key))
    del base_args["mma_path"]
    off = scans._compile_ptx(scans.Kernel(**base_args), sm).value
    ref = baseline._compile_ptx(baseline.Kernel(**base_args), sm).value
    compiled["off_equal"] = off == ref
    return compiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--arch", type=int, choices=(86, 89, 90), nargs="+",
                        default=(86, 89, 90))
    args = parser.parse_args()
    baseline = baseline_module()
    chunks = ((0, True), (4, False), (4, True), (8, False), (8, True), (16, True))
    layouts = ((True, False),) if args.core_only else itertools.product((True, False), repeat=2)
    matrix = itertools.product(
        (divmod(arch, 10) for arch in args.arch), (torch.bfloat16, torch.float16), (2, 4),
        chunks, layouts,
    )
    with tempfile.TemporaryDirectory(prefix="vscan-mma-") as tmp:
        for sm, dtype, warps, (chunk, cache_source), (cache_line, coarse) in matrix:
            key = scans.Kernel(
                dtype=dtype, vertical=True, hidden8=True, warps=warps, chunk=chunk,
                backward=True, cache_weights=True, cache_line=cache_line, cache_source=cache_source,
                coarse=coarse, compact=coarse, mma_path=True,
            )
            print(json.dumps(compile_row(key, sm, Path(tmp), baseline)), flush=True)


if __name__ == "__main__":
    main()
