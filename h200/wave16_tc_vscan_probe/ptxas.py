"""Compile the runtime NVRTC kernel offline and save assembler and SASS evidence."""

import argparse
import csv
import itertools
import re
import subprocess
from pathlib import Path

import torch

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, S603, S607


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extended-layouts", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = itertools.product(
        (86, 89, 90), (torch.bfloat16, torch.float16), (False, True), (False, True), (4, 8),
    )
    layouts = [(True, True, False), (False, False, False), (True, True, True)]
    if args.extended_layouts:
        layouts += [(True, False, False), (False, True, False), (True, False, True)]
    rows = [
        compile_case(
            args.out, sm, dtype, warps, chunk,
            coarse=coarse, backward=backward, cached=cached, index32=index32,
            cache_source=cache_source,
        )
        for sm, dtype, coarse, backward, warps in cases
        for chunk in ((0, 8, 16) if backward else (0,))
        for cached, index32, cache_source in layouts
    ]

    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(
        row["spill_store"] or row["spill_load"] or not row["hmma1688"]
        or (row["backward"] and not row["hmma16816"])
        for row in rows
    ):
        raise SystemExit("Stage 0 assembler gate failed")


def compile_case(
    out: Path, sm: int, dtype: torch.dtype, warps: int, chunk: int, *,
    coarse: bool, backward: bool, cached: bool, index32: bool, cache_source: bool,
) -> dict:
    variant = f"sm{sm}-{str(dtype)[6:]}-{int(coarse)}-{int(backward)}-w{warps}-c{chunk}"

    if not cached:
        variant += "-global"
    if not index32:
        variant += "-i64"
    if cache_source:
        variant += "-cached-source"
    root = out / variant
    ptx = root.with_suffix(".ptx")
    cubin = root.with_suffix(".cubin")
    key = scans.Kernel(
        dtype, warps, backward, vertical=True, coarse=coarse, compact=coarse and backward,
        hidden8=True, spatial=True, chunk=chunk, cache_line=cached, index32=index32,
        real_interleaved=not index32, imag_interleaved=not index32, cache_source=cache_source,
    )
    ptx.write_bytes(scans._compile_ptx(key, divmod(sm, 10)).value)
    compiled = subprocess.run(
        ["ptxas", "-v", f"-arch=sm_{sm}", str(ptx), "-o", str(cubin)],
        text=True, capture_output=True, check=False,
    )
    root.with_suffix(".ptxas.log").write_text(compiled.stderr)
    compiled.check_returncode()
    sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(cubin)], text=True)
    root.with_suffix(".sass").write_text(sass)
    resources = re.search(r"Used (\d+) registers", compiled.stderr)
    spills = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", compiled.stderr)
    if not resources or not spills:
        raise RuntimeError(compiled.stderr)
    h = 56
    span = chunk or h
    stored = span + (h - 1) // span if backward else 0
    per_lane = stored * 8 + h // (1 + coarse) * 4 * cached + h * 4 * cache_source
    smem = per_lane * warps * 32 + 128 * (warps + 1)
    row = {
        "backward": backward,
        "variant": variant,
        "registers": int(resources[1]),
        "spill_store": int(spills[1]),
        "spill_load": int(spills[2]),
        "dynamic_smem_h56": smem,
        "hmma1688": sass.count("HMMA.1688"),
        "hmma16816": sass.count("HMMA.16816"),
    }
    ptx.unlink()
    cubin.unlink()
    return row


if __name__ == "__main__":
    main()
