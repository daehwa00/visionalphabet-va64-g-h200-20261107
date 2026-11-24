"""Compile probe scans off-device and record ptxas and SASS resource data."""

import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001, S603, S607


def measure(job: tuple) -> dict:
    sm, key = job
    ptx = scans._compile_ptx(key, (sm // 10, sm % 10)).value
    with TemporaryDirectory() as folder:
        src = Path(folder) / "scan.ptx"
        dst = Path(folder) / "scan.cubin"
        src.write_bytes(ptx)
        command = ["ptxas", "-v", "-arch", f"sm_{sm}", str(src), "-o", str(dst)]
        report = subprocess.run(command, capture_output=True, text=True, check=True).stderr
        sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(dst)], text=True)
    return {
        "sm": sm, "key": {**vars(key), "dtype": str(key.dtype)}, "ptxas": report,
        "registers": int(re.search(r"Used (\d+) registers", report)[1]),
        "stack_spill": [int(v) for v in re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", report,
        ).groups()],
        "sass": {op: len(re.findall(rf"\b{op}\b", sass)) for op in ("LDS", "LDG", "LDL", "STL")},
    }
