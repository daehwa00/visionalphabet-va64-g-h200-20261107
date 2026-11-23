"""Summarize ptxas registers, static shared memory, and CTA residency."""

# ruff: noqa: T201
import argparse
import re
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    registers = defaultdict(set)
    residency = defaultdict(set)
    shared = defaultdict(set)
    sass = defaultdict(set)
    for line in args.log.read_text().splitlines():
        if line.startswith("sm_"):
            arch, _dtype, _ci, co, _layout, _bias = line.split()
        elif "Compiling entry function" in line:
            function = line.split("'")[1]
        elif match := re.search(r"Used (\d+) registers", line):
            key = (arch, co, function)
            regs = int(match[1])
            registers[key].add(regs)
            # Each 128-thread block places one warp on each of the four partitions.
            # The register allocator rounds to 256 registers per warp.
            limit = 16 if arch == "sm_90" else 12
            residency[key].add(min(limit, 64 // ((regs + 7) // 8)))
            smem = re.search(r"(\d+) bytes smem", line)
            shared[key].add(int(smem[1]) if smem else 0)
        elif " SASS " in line:
            function, _, *widths = line.split()
            sass[(arch, co, function)].update(widths)
    print("SM CO function registers/thread static_smem_bytes resident_CTA/SM SASS")
    for key, values in sorted(registers.items()):
        cells = [
            "-".join(map(str, sorted({min(v), max(v)})))
            for v in (values, shared[key], residency[key])
        ]
        print(*key, *cells, ", ".join(sorted(sass[key])))


if __name__ == "__main__":
    main()
