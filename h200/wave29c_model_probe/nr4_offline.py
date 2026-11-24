"""Compare the shipped NR4 tile with wider candidates without launching a GPU."""

# ruff: noqa: S603, T201
import argparse
import itertools
import re
import subprocess
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lnet.terminal_imaginary_correlation import _nr4_correlation_kernel


def audit(arch: int, cuobjdump: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for dtype, transposed, (backward, normalize), cols in itertools.product(
        ("bf16", "fp16"), (False, True), ((False, False), (False, True), (True, False)), (32, 64),
    ):
        ptr = f"*{dtype}"
        signature = dict(zip(
            _nr4_correlation_kernel.arg_names[:12],
            [ptr] * 4 + ["*fp32"] * 6 + [ptr, "*fp32"],
            strict=True,
        ))
        strides = (6272, 128, 896, 1) if transposed else (6272, 896, 128, 1)
        source = ASTSource(
            _nr4_correlation_kernel, signature=signature,
            constexprs={
                "height": 7, "width": 7, "modes": 128, "strides": strides,
                "block_rows": 8, "block_cols": cols, "normalize": normalize, "backward": backward,
            },
            attrs={(i,): [["tt.divisibility", 16]] for i in range(12)},
        )
        compiled = triton.compile(
            source, target=GPUTarget("cuda", arch, 32),
            options={"num_warps": 1, "enable_fp_fusion": False},
        )
        name = f"{dtype}-t{int(transposed)}-b{int(backward)}-n{int(normalize)}-c{cols}"
        cubin = output / f"{name}.cubin"
        cubin.write_bytes(compiled.asm["cubin"])
        result = subprocess.run(
            [cuobjdump, "--dump-resource-usage", "--dump-sass", cubin],
            check=True, capture_output=True, text=True,
        ).stdout
        cubin.unlink()
        (output / f"{name}.sass").write_text(result)
        usage = re.search(r"REG:\d+[^\n]+", result)
        if usage is None or any(f"{key}:0" not in usage[0] for key in ("STACK", "LOCAL")):
            raise RuntimeError(f"sm_{arch} {name}: invalid resource usage")
        vectors = re.findall(r"\b(?:LDG|STG)\.E\.(?:128|64)\b", result)
        print(f"sm_{arch} {name} {usage[0]} vector={len(vectors)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuobjdump", type=Path, default=Path("/usr/local/cuda/bin/cuobjdump"))
    args = parser.parse_args()
    for arch in (90, 86, 89):
        audit(arch, args.cuobjdump, args.output / f"sm_{arch}")


if __name__ == "__main__":
    main()
