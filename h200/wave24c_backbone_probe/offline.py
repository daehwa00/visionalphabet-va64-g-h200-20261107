"""Compile the isolated Triton candidates without a GPU."""

# ruff: noqa: ANN001, ANN201, T201
import triton
from direct import conv3
from isolated import swiglu
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lnet.terminal_imaginary_correlation import _nr4_correlation_kernel


def compile_kernel(arch, fn, ptrs, constants, warps):
    signature = dict(zip(fn.arg_names, ptrs + ["constexpr"] * len(constants), strict=True))
    constexprs = dict(zip(fn.arg_names[len(ptrs) :], constants, strict=True))
    return triton.compile(
        ASTSource(
            fn,
            signature,
            constexprs,
            attrs={(i,): [["tt.divisibility", 16]] for i in range(len(ptrs))},
        ),
        target=GPUTarget("cuda", arch, 32),
        options={"enable_fp_fusion": False, "num_warps": warps},
    )


def main():
    for arch in (86, 89, 90):
        for dtype in ("bf16", "fp16"):
            for fused, tensor_cores in ((False, True), (True, True), (False, False)):
                image = compile_kernel(
                    arch,
                    conv3,
                    [f"*{dtype}"] * 2 + ["*fp32"] * 3 + [f"*{dtype}"],
                    [128 * 112 * 112, fused, tensor_cores],
                    4,
                )
                print(arch, dtype, "conv3", fused, tensor_cores, image.metadata)
            for rows, width in (
                (401408, 384),
                (100352, 576),
                (25088, 768),
                (6272, 1536),
                (97, 383),
            ):
                for backward in (False, True):
                    image = compile_kernel(
                        arch,
                        swiglu,
                        [f"*{dtype}"] * 3,
                        [rows * width, width, backward, 1024],
                        4,
                    )
                    ptx = image.asm["ptx"]
                    print(
                        arch,
                        dtype,
                        "swiglu",
                        width,
                        backward,
                        image.metadata.shared,
                        "v4 loads",
                        ptx.count("ld.global.v4"),
                        "v4 stores",
                        ptx.count("st.global.v4"),
                    )
            for cols, warps in ((16, 4), (32, 4), (64, 4), (32, 8), (16, 2)):
                for backward in (False, True):
                    image = compile_kernel(
                        arch,
                        _nr4_correlation_kernel,
                        [f"*{dtype}"] * 4 + ["*fp32"] * 6 + [f"*{dtype}", "*fp32"],
                        [7, 7, 128, (6272, 896, 128, 1), 8, cols, False, backward],
                        warps,
                    )
                    print(arch, dtype, "nr4", cols, warps, backward, image.metadata)


if __name__ == "__main__":
    main()
