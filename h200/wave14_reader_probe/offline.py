"""Compile and disassemble the Wave 14B variants without using a GPU."""

from __future__ import annotations

# Tool names are fixed and all invocations use argument lists.
# ruff: noqa: S603, S607
import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

from sourcegen import RING, SHAPES, SPATIAL, WEIGHT, WEIGHT_STREAM, generate


def run(out: Path, arch: str, names: set[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    variants = {"tiled": generate(SHAPES[0], names, native=True)}
    variants.update((name, generate(SHAPES[0], set(), streaming=name)) for name in WEIGHT_STREAM)
    for name, source in variants.items():
        cu = out / f"{name}.cu"
        ptx, cubin = cu.with_suffix(".ptx"), cu.with_suffix(".cubin")
        cu.write_text(source, encoding="utf-8")
        nvcc = ["nvcc", "-std=c++17", "-ptx", f"-arch=compute_{arch}", str(cu), "-o", str(ptx)]
        with cu.with_suffix(".nvcc.log").open("w", encoding="utf-8") as log:
            subprocess.run(nvcc, check=True, stdout=log, stderr=subprocess.STDOUT)
        ptxas = ["ptxas", "-v", f"-arch=sm_{arch}", str(ptx), "-o", str(cubin)]
        with cu.with_suffix(".ptxas.log").open("w", encoding="utf-8") as log:
            subprocess.run(ptxas, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = subprocess.run(
            ["cuobjdump", "--dump-sass", str(cubin)], check=True, capture_output=True, text=True
        )
        loads = []
        for segment in result.stdout.split("Function : ")[1:]:
            function, _, sass = segment.partition("\n")
            ops = Counter(re.findall(r"\b(?:LDGSTS|LDG|LDS|STG)[.\w]*\b", sass))
            loads.append(f"{function.strip()}: {dict(sorted(ops.items()))}")
        cu.with_suffix(".sass-loads.log").write_text("\n".join(loads) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", choices=("86", "89", "90"), required=True)
    args = parser.parse_args()
    run(args.out, args.arch, {tile.name for tile in (*RING, *SPATIAL, *WEIGHT)})
