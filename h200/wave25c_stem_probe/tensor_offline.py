"""Check final Tensor Core stem resources without a CUDA device."""

# ruff: noqa: S603, T201
import itertools
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lnet.fused_stem_ops import stem_conv_tensor_reduce, stem_conv_tensor_stages


def audit(
    fn: triton.JITFunction, arch: int, pointers: list[str], constants: tuple, output: Path
) -> None:
    source = ASTSource(
        fn,
        dict(zip(fn.arg_names, [*pointers, *["constexpr"] * len(constants)], strict=True)),
        dict(zip(fn.arg_names[len(pointers) :], constants, strict=True)),
        attrs={(i,): [["tt.divisibility", 16]] for i in range(len(pointers))},
    )
    target = GPUTarget("cuda", arch, 32)
    image = triton.compile(source, target=target, options={"num_warps": 4})
    output.write_bytes(image.asm["cubin"])
    print("SMEM", image.metadata.shared, flush=True)
    resources = subprocess.check_output(
        ["/usr/local/cuda/bin/cuobjdump", "-res-usage", output], text=True
    )
    print(resources, end="")
    match = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resources)
    if match is None or int(match[2]) or int(match[4]):
        raise RuntimeError("Unexpected stack or local memory usage")
    registers, static_shared = int(match[1]), int(match[3])
    reserved = static_shared + image.metadata.shared + 1024
    shared_limit = (228 if arch == 90 else 100) * 1024 // reserved
    resident = min(16 if arch == 90 else 12, 64 // ((registers + 7) // 8), shared_limit)
    print("Estimated CTA/SM", resident)
    sass = subprocess.check_output(["/usr/local/cuda/bin/cuobjdump", "-sass", output], text=True)
    opcodes = sorted(set(re.findall(r"\b(?:LDG|STG|HMMA|HGMMA)(?:\.[\w]+)*", sass)))
    print("SASS", *opcodes)


def main() -> None:
    with TemporaryDirectory(prefix="wave25c-tensor-") as root:
        output = Path(root) / "stem.cubin"
        for arch, dtype, layout, bias, backward, hw in itertools.product(
            (86, 89, 90),
            ("bf16", "fp16"),
            (False, True),
            (False, True),
            (False, True),
            ((224, 224), (113, 127)),
        ):
            h, w = hw
            pixels = (1024 if arch == 90 else 128) * ((h + 1) // 2) * ((w + 1) // 2)
            pointers = [f"*{dtype}"] * 2 + ["*fp32"] * 3 + [f"*{dtype}"] * 2 + ["*fp32"] * 2
            constants = (
                h,
                w,
                3,
                32,
                pixels,
                1e-5,
                layout,
                bias,
                backward,
                64 if backward else 32,
                32,
                True,
                False,
            )
            print(arch, dtype, layout, bias, backward, hw, flush=True)
            audit(stem_conv_tensor_stages, arch, pointers, constants, output)
        for arch, dtype, bias in itertools.product((86, 89, 90), ("bf16", "fp16"), (False, True)):
            print(arch, dtype, bias, "reduce", flush=True)
            audit(
                stem_conv_tensor_reduce,
                arch,
                ["*fp32", f"*{dtype}", *(["*fp32"] * 3)],
                (3, 32, bias, 1024, 1024),
                output,
            )


if __name__ == "__main__":
    main()
