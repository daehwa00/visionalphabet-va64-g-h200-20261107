"""Compile Wave 35C CUDA variants and measure code generation without a GPU."""

# ruff: noqa: ANN001, ANN201, PLC0415, S603, T201
import itertools
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from lnet.fused_stem_ops import stem_conv_tensor_reduce, stem_conv_tensor_stages
from lnet.model_w29c import (
    _rmsnorm_backward_kernel,
    _rmsnorm_backward_rows,
    _swiglu_vec2d_kernel,
)

CUDA = Path("/usr/local/cuda-12.8/bin")
WIDTHS = (64, 96, 128, 144, 160, 192, 256, 320, 384, 512, 1024)


def cases(dtype):
    p = f"*{dtype}"
    for width, backward, tuned in itertools.product(WIDTHS, (False, True), (False, True)):
        cols = min(128, width & -width) if tuned else min(512, triton.next_power_of_2(width))
        rows = 1024 // cols if tuned else 8
        yield f"swiglu:{width}:{backward}:{tuned}", _swiglu_vec2d_kernel, (
            p, p, p, "i32", width, rows, cols, backward, tuned,
        ), False
    for width in WIDTHS:
        yield f"rmsnorm:{width}", _rmsnorm_backward_kernel, (
            p, p, "*fp32", p, p, "*fp32", "i32", width, "i32", "i32",
            _rmsnorm_backward_rows(width), triton.next_power_of_2(width), True, True, False,
        ), False
    for width, backward, layout, tuned in itertools.product((32, 64), *(3 * [(False, True)])):
        yield f"stem:{width}:{backward}:{layout}:{tuned}", stem_conv_tensor_stages, (
            p, p, "*fp32", "*fp32", "*fp32", p, p, "*fp32", "*fp32",
            224, 224, 3, width, 1024 * 112 * 112, 1e-5, layout, True, backward,
            (16 if width == 64 else 32) if backward and tuned else (64 if backward else 32),
            32, not tuned, tuned and layout,
        ), True
    for width in (32, 64):
        yield f"stem_reduce:{width}", stem_conv_tensor_reduce, (
            "*fp32", p, "*fp32", "*fp32", "*fp32", 3, width, True, 1024, 1024,
        ), True


def audit(fn, args, arch, fp_fusion, root, warps=4, stages=3):
    signature = {}
    constants = {}
    attrs = {}
    for i, (name, arg) in enumerate(zip(fn.arg_names, args, strict=True)):
        signature[name] = arg if isinstance(arg, str) else "constexpr"
        if not isinstance(arg, str):
            constants[name] = arg
        else:
            attrs[(i,)] = [["tt.divisibility", 16]]
    image = triton.compile(
        ASTSource(fn, signature, constants, attrs),
        target=GPUTarget("cuda", arch, 32),
        options={
            "num_warps": warps, "num_stages": stages,
            "enable_fp_fusion": fp_fusion, "ptx_version": 87,
        },
    )
    ptx, cubin = root / "kernel.ptx", root / "kernel.cubin"
    ptx.write_text(image.asm["ptx"])
    target = re.search(r"\.target\s+(sm_\w+)", image.asm["ptx"])[1]
    meta = subprocess.check_output(
        [CUDA / "ptxas", "-v", f"-arch={target}", ptx, "-o", cubin],
        stderr=subprocess.STDOUT, text=True,
    )
    sass = subprocess.check_output([CUDA / "nvdisasm", "-c", cubin], text=True)
    return resource_row(meta, sass, arch, image.metadata.shared, warps)


def resource_row(meta, sass, arch, smem, warps):
    regs = int(re.search(r"Used (\d+) registers", meta)[1])
    stack, stores, loads = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", meta,
    ).groups()
    blocks = min(
        {80: 64, 86: 48, 89: 48, 90: 64, 100: 64}[arch] // warps,
        256 // (((regs + 7) // 8) * warps),
        {80: 164, 86: 100, 89: 100, 90: 228, 100: 228}[arch] * 1024 // (smem + 1024),
    )
    instructions = re.findall(r"/\*[0-9a-f]+\*/\s+[^;]+;", sass)
    vector_bits = []
    for op in ("LDG", "STG"):
        matches = re.findall(rf"\b{op}(?:\.[\w]+)*", sass)
        vector_bits.append(max(
            (max(map(int, re.findall(r"\.(16|32|64|128)(?:\.|$)", ins)), default=32)
             for ins in matches), default=0,
        ))
    return [
        regs, smem, blocks * warps, *vector_bits, len(instructions), stack, stores, loads,
        len(re.findall(r"\b(?:HMMA|HGMMA|UTCHMMA|QGMMA)\b", sass)),
    ]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs="+", type=int, default=[86, 89, 90, 100])
    parser.add_argument("--dtype", nargs="+", default=["bf16", "fp16"])
    parser.add_argument("--match", default="")
    args = parser.parse_args()
    print("sm,dtype,case,regs,smem,warps,ld_bits,st_bits,sass,stack,spill_st,spill_ld,tc")
    with TemporaryDirectory() as tmp:
        for arch, dtype in itertools.product(args.arch, args.dtype):
            for label, fn, signature, fp_fusion in cases(dtype):
                if re.search(args.match, label):
                    wide_stem = label.startswith("stem:64:True:") and label.endswith(":True")
                    warps = 8 if wide_stem else 4
                    result = audit(fn, signature, arch, fp_fusion, Path(tmp), warps)
                    print(arch, dtype, label, *result, sep=",", flush=True)


if __name__ == "__main__":
    main()
