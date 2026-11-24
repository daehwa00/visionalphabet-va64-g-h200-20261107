"""Assemble Wave 28C Triton kernels across Ampere, Ada, and Hopper."""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import tempfile
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lnet.model_w28c import triton_w28c_packed_swiglu_kernel
from lnet.terminal_imaginary_correlation import _nr4_correlation_kernel


def audit(architecture: int, cuda: Path, output: Path) -> int:
    target = GPUTarget("cuda", architecture, 32)
    count = 0
    variants = itertools.product(("*bf16", "*fp16"), (False, True), (96, 144, 192, 384))
    for dtype, backward, hidden in variants:
        constants = {"hidden": hidden, "block": 256, "backward": backward}
        signature = {"joint": dtype, "gradient": dtype, "output": dtype, "elements": "i32"}
        source = ASTSource(
            triton_w28c_packed_swiglu_kernel, signature=signature, constexprs=constants,
        )
        name = f"swiglu-{dtype[1:]}-{hidden}-{int(backward)}"
        inspect(source, target, cuda, output / name, warps=4)
        count += 1

    names = tuple(_nr4_correlation_kernel.arg_names[:12])
    for dtype, height, mode in itertools.product(("*bf16", "*fp16"), (7, 14), ("base", "warp")):
        width = height
        modes = 128 if height == 7 else 160
        columns = width * modes
        strides = (height * columns, columns, modes, 1)
        signature = dict(zip(names, [dtype] * 4 + ["*fp32"] * 6 + [dtype, "*fp32"], strict=True))
        for backward, normalize in ((False, False), (False, True), (True, False)):
            constants = {
                "height": height, "width": width, "modes": modes, "strides": strides,
                "block_rows": triton.next_power_of_2(height), "block_cols": 32,
                "normalize": normalize, "backward": backward,
            }
            source = ASTSource(_nr4_correlation_kernel, signature=signature, constexprs=constants)
            name = f"nr4-{dtype[1:]}-{height}-{mode}-{int(backward)}-{int(normalize)}"
            inspect(source, target, cuda, output / name, warps=4 if mode == "base" else 1)
            count += 1
    return count


def inspect(
    source: ASTSource, target: GPUTarget, cuda: Path, output: Path, *, warps: int,
) -> None:
    compiled = triton.compile(
        source, target=target, options={"num_warps": warps, "enable_fp_fusion": False},
    )
    with tempfile.TemporaryDirectory() as temp:
        cubin = Path(temp) / "kernel.cubin"
        cubin.write_bytes(compiled.asm["cubin"])
        resources = subprocess.run(  # noqa: S603
            [cuda / "cuobjdump", "--dump-resource-usage", cubin],
            check=True, capture_output=True, text=True,
        ).stdout
    output.with_suffix(".txt").write_text(resources, encoding="utf-8")
    if any(not re.search(rf"\b{key}:0\b", resources) for key in ("STACK", "LOCAL")):
        raise RuntimeError(f"{output}: {resources}")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda/bin"))
    args = parser.parse_args()
    for architecture in (86, 89, 90):
        output = args.output / f"sm_{architecture}"
        output.mkdir(parents=True, exist_ok=True)
        print(f"sm_{architecture}: {audit(architecture, args.cuda, output)} cases passed")  # noqa: T201


if __name__ == "__main__":
    main()
