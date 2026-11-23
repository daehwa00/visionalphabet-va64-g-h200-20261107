"""Collect NVRTC/ptxas/SASS resource evidence without a CUDA device."""

# ruff: noqa: ANN001, ANN201, S603, SLF001
import argparse
import itertools
import json
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from lnet import general_cuda_scans as scans

OPS = ("LDG", "STG", "LDS", "STS", "FFMA", "MUFU")
VARIANTS = (
    (False, False, False, False), (False, False, True, False),
    (True, False, False, False), (True, False, True, False),
    (True, True, False, False), (True, True, True, False),
    (True, True, True, True),
)


def compile_one(job):
    cuda, sm, dtype, warps, variant, index32, layout = job
    backward, compact, cache, paired = variant
    real, imag = layout
    key = scans.Kernel(
        dtype, warps, backward, compact=compact, cache_line=cache, paired_hscan=paired,
        index32=index32, real_interleaved=real == "packed", imag_interleaved=imag == "packed",
        real_transposed=real == "transposed", imag_transposed=imag == "transposed",
    )
    image = scans._compile_ptx(key, tuple(map(int, sm))).value
    with tempfile.TemporaryDirectory() as directory:
        ptx, cubin = (Path(directory) / f"scan.{ext}" for ext in ("ptx", "cubin"))
        ptx.write_bytes(image)
        log = subprocess.run(
            [cuda / "ptxas", "-v", f"-arch=sm_{sm}", ptx, "-o", cubin],
            capture_output=True, text=True, check=True,
        ).stderr
        sass = subprocess.run(
            [cuda / "cuobjdump", "--dump-sass", cubin],
            capture_output=True, text=True, check=True,
        ).stdout
    registers = int(re.search(r"Used (\d+) registers", log)[1])
    stack, stores, loads = map(int, re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log,
    ).groups())
    return {
        "sm": sm, "dtype": str(dtype), "warps": warps, "backward": backward,
        "compact": compact, "cache": cache, "paired": paired, "index32": index32,
        "layout": layout, "registers": registers, "stack": stack, "spill_stores": stores,
        "spill_loads": loads, "sass": {op: len(re.findall(rf"\b{op}\b", sass)) for op in OPS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda/bin"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    layouts = tuple(itertools.product(("plain", "packed", "transposed"), repeat=2))
    jobs = itertools.product(
        (args.cuda,), ("86", "89", "90"), scans._DTYPES, scans._SCAN_WARPS, VARIANTS,
        (True, False) if args.extended else (True,),
        layouts if args.extended else layouts[:1],
    )
    with args.output.open("w") as output, ProcessPoolExecutor(args.jobs) as pool:
        spill = False
        for row in pool.map(compile_one, jobs):
            output.write(json.dumps(row) + "\n")
            spill |= bool(row["stack"] or row["spill_stores"] or row["spill_loads"])
    if spill:
        raise SystemExit("At least one offline geometry has stack or spill usage")


if __name__ == "__main__":
    main()
