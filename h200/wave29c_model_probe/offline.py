"""Compile and disassemble the Wave 29C kernels without CUDA devices."""

# ruff: noqa: S603, T201
import argparse
import itertools
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import JITFunction

from lnet.model_w29c import (
    _rmsnorm_backward_kernel,
    _rmsnorm_backward_rows,
    _rmsnorm_forward_kernel,
    _swiglu_vec2d_kernel,
)


def cases() -> Iterator[tuple[str, JITFunction, list[str], dict, bool]]:
    for dtype, width, backward in itertools.product(
        ("bf16", "fp16"), (96, 144, 192, 384, 65, 520), (False, True),
    ):
        kernel = _swiglu_vec2d_kernel
        args = [f"*{dtype}"] * 3 + ["i32"]
        constants = {
            "hidden": width, "block_rows": 8, "block_cols": min(512, triton.next_power_of_2(width)),
            "backward": backward,
        }
        yield f"swiglu-{dtype}-{width}-{int(backward)}", kernel, args, constants, width % 8 == 0

    for dtype, width, pitched in itertools.product(
        ("bf16", "fp16"), (64, 96, 128, 256, 65, 1024), (False, True),
    ):
        ptr = f"*{dtype}"
        pitch = ((width + 15) // 16 + 1) * 16 if pitched else width
        constants = {"width": width, "block_cols": triton.next_power_of_2(width)}
        for save in (False, True):
            args = [ptr, ptr, ptr, "*fp32", "i32"]
            yield (
                f"rms-fwd-{dtype}-{width}-{pitch}-{int(save)}", _rmsnorm_forward_kernel, args,
                dict(constants, pitch=pitch, eps=1e-6, block_rows=8, save_inverse=save),
                width % 16 == 0,
            )
        for dx, dw in ((True, True), (True, False), (False, True)):
            args = [ptr, ptr, "*fp32", ptr, ptr, "*fp32", "i32"]
            yield (
                f"rms-bwd-{dtype}-{width}-{pitch}-{int(dx)}{int(dw)}", _rmsnorm_backward_kernel,
                args, dict(constants, x_pitch=pitch, dy_pitch=pitch,
                           block_rows=_rmsnorm_backward_rows(width),
                           need_dx=dx, need_dw=dw), width % 16 == 0,
            )


def audit(architecture: int, cuobjdump: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, kernel, args, constants, aligned in cases():
        names = kernel.arg_names[:len(args)]
        signature = dict(zip(names, args, strict=True))
        # Match runtime specialization for aligned pointers and row pitches.
        attrs = {(i,): [["tt.divisibility", 16]] for i, arg in enumerate(args) if arg[0] == "*"}
        for pitch in ("pitch", "x_pitch", "dy_pitch"):
            if pitch in constants:
                signature[pitch] = "i32"
                if constants.pop(pitch) % 16 == 0:
                    attrs[(kernel.arg_names.index(pitch),)] = [["tt.divisibility", 16]]
        source = ASTSource(kernel, signature=signature, constexprs=constants, attrs=attrs)
        compiled = triton.compile(source, target=GPUTarget("cuda", architecture, 32),
                                  options={"enable_fp_fusion": False})
        cubin = output / f"{name}.cubin"
        cubin.write_bytes(compiled.asm["cubin"])
        result = subprocess.run(
            [cuobjdump, "--dump-resource-usage", "--dump-sass", cubin],
            check=True, capture_output=True, text=True,
        ).stdout
        cubin.unlink()
        (output / f"{name}.sass").write_text(result)
        (output / f"{name}.ptx").write_text(compiled.asm["ptx"])
        loads = re.findall(r"\bLDG\.E\.(?:128|64)\b", result)
        stores = re.findall(r"\bSTG\.E\.(?:128|64)\b", result)
        scalar = re.findall(r"\b(?:LDG|STG)\.E\.[US]16\b", result)
        if any(not re.search(rf"\b{key}:0\b", result) for key in ("STACK", "LOCAL")):
            raise RuntimeError(f"{name}: stack or local memory used")
        print(f"sm_{architecture} {name}: STACK:0 LOCAL:0 LD={len(loads)} "
              f"ST={len(stores)} scalar16={len(scalar)}", flush=True)
        if architecture == 90 and aligned:
            dw_only = name.startswith("rms-bwd") and name.endswith("-01")
            if not loads or (not stores and not dw_only):
                raise RuntimeError(f"{name}: expected vector load/store instructions")
            if name.startswith("swiglu") and scalar:
                raise RuntimeError(f"{name}: scalar 16-bit memory instructions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuobjdump", type=Path, default=Path("/usr/local/cuda/bin/cuobjdump"))
    parser.add_argument("--arch", type=int, nargs="+", default=(86, 89, 90))
    args = parser.parse_args()
    for architecture in args.arch:
        audit(architecture, args.cuobjdump, args.output / f"sm_{architecture}")


if __name__ == "__main__":
    main()
