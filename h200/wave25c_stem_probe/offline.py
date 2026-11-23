"""Compile and disassemble all Stage 0 variants without a CUDA device."""

# ruff: noqa: S603, T201
import itertools
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lnet.fused_stem_ops import _compile_conv_stem_ptx


def main() -> None:
    variants = itertools.product(
        ((8, 6), (8, 9), (9, 0)),
        (torch.bfloat16, torch.float16),
        (3, 4),
        (32, 64),
        (False, True),
        (False, True),
    )
    spills = False
    with TemporaryDirectory(prefix="wave25c-") as root:
        ptx, cubin = Path(root) / "stem.ptx", Path(root) / "stem.cubin"
        for arch, dtype, ci, co, layout, bias in variants:
            sm = f"sm_{arch[0]}{arch[1]}"
            ptx.write_bytes(_compile_conv_stem_ptx(arch, dtype, ci, co, layout, bias).value)
            result = subprocess.run(
                ["/usr/local/cuda/bin/ptxas", "-v", f"-arch={sm}", ptx, "-o", cubin],
                text=True,
                capture_output=True,
                check=True,
            )
            sass = subprocess.check_output(
                ["/usr/local/cuda/bin/cuobjdump", "-sass", cubin],
                text=True,
            )
            print(sm, dtype, ci, co, layout, bias, flush=True)
            print(result.stderr, end="")
            for function in sass.split("Function : ")[1:]:
                name = function.splitlines()[0]
                widths = sorted(set(re.findall(r"\b(?:LDG|STG)(?:\.[\w]+)*", function)))
                print(name, "SASS", *widths)
            spills |= bool(re.search(r"[1-9]\d* bytes (?:stack frame|spill)", result.stderr))
    if spills:
        raise SystemExit("Nonzero stack or spill usage")


if __name__ == "__main__":
    main()
