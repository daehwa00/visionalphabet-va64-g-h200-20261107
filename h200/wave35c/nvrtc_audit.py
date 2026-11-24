"""Compile the existing NVRTC stem and a CUDA vector SwiGLU comparison."""

# ruff: noqa: S603, T201
import itertools
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from audit import CUDA, WIDTHS, resource_row

from lnet.fused_stem_ops import _compile_conv_stem_ptx, _compile_ptx, _compile_source

_SWIGLU = r"""
#include <cuda_fp16.h>
#if BF16
#include <cuda_bf16.h>
using T = __nv_bfloat16;
#else
using T = __half;
#endif

union Pack { float4 vector; T values[8]; };

template<bool backward>
__device__ __forceinline__ void calculate(
    const float4* joint, const float4* gradient, float4* output, int count, int width
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int offset = index + index / width * width;
    Pack value, gate, upstream, out, gate_out;
    value.vector = joint[offset];
    gate.vector = joint[offset + width];
    if constexpr (backward) upstream.vector = gradient[index];
    #pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
        float x = float(value.values[lane]);
        float g = float(gate.values[lane]);
        float sigmoid = 1.0f / (1.0f + __expf(-g));
        float activated = float(T(g * sigmoid));
        if constexpr (backward) {
            float grad = float(upstream.values[lane]);
            out.values[lane] = T(grad * activated);
            float inner = float(T(grad * x));
            gate_out.values[lane] = T(inner * (sigmoid * (1 + g * (1 - sigmoid))));
        } else {
            out.values[lane] = T(x * activated);
        }
    }
    if constexpr (backward) {
        output[offset] = out.vector;
        output[offset + width] = gate_out.vector;
    } else {
        output[index] = out.vector;
    }
}
extern "C" __global__ void swiglu_f(
    const float4* joint, const float4* gradient, float4* output, int count
) { calculate<false>(joint, gradient, output, count, WIDTH / 8); }
extern "C" __global__ void swiglu_b(
    const float4* joint, const float4* gradient, float4* output, int count
) { calculate<true>(joint, gradient, output, count, WIDTH / 8); }
"""


def assemble(image: bytes, arch: int, root: Path, *, disable_fma: bool = False) -> dict:
    ptx, cubin = root / "kernel.ptx", root / "kernel.cubin"
    ptx.write_bytes(image)
    args = [CUDA / "ptxas", "-v", f"-arch=sm_{arch}", ptx, "-o", cubin]
    if disable_fma:
        args.insert(1, "--fmad=false")
    resources = subprocess.check_output(args, stderr=subprocess.STDOUT, text=True)
    sass = subprocess.check_output([CUDA / "nvdisasm", "-c", cubin], text=True)
    labels_and_bodies = iter(re.split(r"(?m)^\.text\.(\w+):\s*$", sass)[1:])
    sections = dict(zip(labels_and_bodies, labels_and_bodies, strict=True))
    rows = {}
    for section in re.split("Compiling entry function ", resources)[1:]:
        name = section.split("'")[1]
        smem = re.search(r"(\d+) bytes smem", section)
        rows[name] = resource_row(section, sections[name], arch, int(smem[1]) if smem else 0, 4)
    return rows


def main() -> None:
    print("sm,dtype,case,regs,smem,warps,ld_bits,st_bits,sass,stack,spill_st,spill_ld,tc")
    with TemporaryDirectory() as tmp:
        for arch, dtype in itertools.product((86, 89, 90, 100), ("bf16", "fp16")):
            root = Path(tmp)
            cap = divmod(arch, 10)
            dtype_t = torch.bfloat16 if dtype == "bf16" else torch.float16
            scalar = "__nv_bfloat16" if dtype == "bf16" else "__half"
            for layout in (False, True):
                image = _compile_conv_stem_ptx(
                    cap, dtype_t, 3, 64, channels_last=layout, has_bias=True,
                )
                for name, result in assemble(image.value, arch, root).items():
                    print(arch, dtype, f"nvrtc:{name}:{layout}", *result, sep=",", flush=True)
            for channels, vector in itertools.product((64,), (False, True)):
                image = _compile_ptx(
                    cap, channels, (scalar, "float", "float", scalar), vector=vector,
                )
                for name, result in assemble(image.value, arch, root).items():
                    print(arch, dtype, f"stage1:{name}:{vector}", *result, sep=",", flush=True)
            for width in WIDTHS:
                source = f"#define BF16 {int(dtype == 'bf16')}\n#define WIDTH {width}\n" + _SWIGLU
                image = _compile_source(cap, source)
                for name, result in assemble(image.value, arch, root, disable_fma=True).items():
                    print(arch, dtype, f"nvrtc:{name}:{width}", *result, sep=",", flush=True)


if __name__ == "__main__":
    main()
