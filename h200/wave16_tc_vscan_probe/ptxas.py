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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = itertools.product(
        (86, 89, 90), (torch.bfloat16, torch.float16), (False, True), (False, True), (4, 8),
    )
    rows = []
    for sm, dtype, coarse, backward, warps in cases:
        variant = f"sm{sm}-{str(dtype)[6:]}-{int(coarse)}-{int(backward)}-w{warps}"
        root = args.out / variant
        ptx = root.with_suffix(".ptx")
        cubin = root.with_suffix(".cubin")
        key = scans.Kernel(
            dtype, warps, backward, vertical=True, coarse=coarse, compact=coarse and backward,
            hidden8=True, spatial=True,
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
        smem = (h * 8 * backward + h // (1 + coarse) * 4) * warps * 32 + 128 * (warps + 1)
        rows.append({
            "variant": variant,
            "registers": int(resources[1]),
            "spill_store": int(spills[1]),
            "spill_load": int(spills[2]),
            "dynamic_smem_h56": smem,
            "hmma1688": sass.count("HMMA.1688"),
            "hmma16816": sass.count("HMMA.16816"),
        })
        ptx.unlink()
        cubin.unlink()

    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(
        row["spill_store"] or row["spill_load"] or not row["hmma1688"]
        or ("-1-w" in row["variant"] and not row["hmma16816"])
        for row in rows
    ):
        raise SystemExit("Stage 0 assembler gate failed")


if __name__ == "__main__":
    main()
