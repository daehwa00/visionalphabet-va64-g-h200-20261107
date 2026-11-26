"""Experimental BF16 vertical scans using the wave57 persistent worker ABI."""

from __future__ import annotations

import ctypes as ct

# The adapter deliberately reuses private planner/driver entry points.
# ruff: noqa: SLF001
import re
import subprocess
import tempfile
from dataclasses import replace
from functools import lru_cache, partial
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING

import torch

from . import general_cuda_scans as scans

if TYPE_CHECKING:
    from collections.abc import Callable

VF = "w64_vf_hfma2"
SAME = "w64_vb_hfma2_sameocc"
HIGH = "w64_vb_hfma2_highocc"
BASE = "w57_lockin"

_SCALE = r"""
__device__ float column_scale(float peak) {
    unsigned bits = __float_as_uint(fmaxf(1e-24f, peak));
    return __uint_as_float(min(0x7f000000u, (bits + 0x7fffffu) & 0x7f800000u));
}
__device__ float inverse_scale(float s) {
    return __uint_as_float(0x7f000000u - __float_as_uint(s));
}
"""
_FP8 = r"""
struct FP8Quad {
    __nv_fp8x4_e4m3 v;
    __device__ static FP8Quad pack(float4 a, float inv) {
        return {__nv_fp8x4_e4m3(make_float4(a.x*inv,a.y*inv,a.z*inv,a.w*inv))};
    }
    __device__ float4 unpack(float scale) const {
        float4 a = static_cast<float4>(v);
        return make_float4(a.x*scale,a.y*scale,a.z*scale,a.w*scale);
    }
};
"""
_SCAN = r"""
    float peak = 0;
    for (int i=0; i<c.H/(1+COARSE); ++i) {
        const I offset = c.cell(i) - c.m;
        peak = fmaxf(peak, fabsf(read(gr[offset*(1+REAL_INTERLEAVED)+c.m])));
        peak = fmaxf(peak, fabsf(read(gi[offset*(1+IMAG_INTERLEAVED)+c.m])));
    }
    const float scale = column_scale(peak), inv = inverse_scale(scale);
    __half2 acc2[45]{};
"""
_SOURCE_SCALE = r"""
    float src_peak = 0;
    for (int i=0; i<c.H; ++i) {
        const auto s = c.source(i);
        src_peak = fmaxf(src_peak, fmaxf(fmaxf(fabsf(s.x),fabsf(s.y)),
                                         fmaxf(fabsf(s.z),fabsf(s.w))));
    }
    const float src_scale = column_scale(src_peak)*0.00390625f;
    const float src_inv = inverse_scale(src_scale);
"""
_ACC = r"""
            dzs[cell] = dz;
            const float scaled = dz * inv;
            const auto dzpair = __float2half2_rn(scaled);
            #pragma unroll
            for (int k=0; k<8; k+=2) {
                const auto pair = __floats2half2_rn(p[cell][k],p[cell][k+1]);
                acc2[4*j+k/2] = __hfma2(dzpair,pair,acc2[4*j+k/2]);
            }
            const auto outs = __floats2half2_rn(d.x*inv,d.y*inv);
            acc2[36+j] = __hfma2(outs,__float2half2_rn(act),acc2[36+j]);
            const auto bias_pair = __floats2half2_rn(j%2 ? 0 : scaled,j%2 ? scaled : 0);
            acc2[32+j/2] = __hadd2(acc2[32+j/2],bias_pair);
"""
_FLUSH = r"""
    #pragma unroll
    for (int i=0; i<45; ++i) {
        asm volatile("" ::: "memory");
        const auto v = __half22float2(acc2[i]);
        const int lo = i>=36 && i<44 ? 72+i-36 : 2*i;
        const int hi = i>=36 && i<44 ? 80+i-36 : 2*i+1;
        acc[lo*acc_stride] += v.x*scale;
        acc[hi*acc_stride] += v.y*scale;
    }
"""


def plan(launch: scans.Launch, height: int, variant: str) -> scans.Launch:
    key = launch.key
    if variant == BASE:
        return launch
    if variant == VF:
        return launch._replace(
            key=replace(
                key,
                batch_path=True,
                register_batch2=True,
                fp32_path=True,
                tanh_gain=True,
                vertical_w55="v55_bf16x2_predp" if key.vertical_w55 == "off" else key.vertical_w55,
            )
        )
    if variant not in (SAME, HIGH):
        raise ValueError(variant)
    threads = key.warps * 32
    span = min(16, height) if variant == HIGH else launch.plan.chunk
    chunk = span if span < height else 0
    # Keep tail handling for both short final chunks and odd heights.
    key = replace(
        key,
        chunk=chunk,
        backward_batch2=True,
        backward_batch_tail=bool(
            (height % (2 * (1 + key.coarse))) or (span % (2 * (1 + key.coarse)))
        ),
        vertical_w55="v55_bf16x2_bf16w",
        backward_reduce_warps=0,
        path_bwd_pipe=0,
        fp32_path=True,
        tanh_gain=True,
        shared_acc=False,
        cache_weights=True,
        vector_weights=True,
        cache_source=True,
        seg16=variant == SAME and key.seg16,
    )
    extra = (height // (1 + key.coarse) * key.cache_line + launch.plan.sgs_rows) * 8
    checkpoints = 0 if key.seg16 or key.global_checkpoints else (height - 1) // span * 16
    smem = 5632 + threads * (
        extra
        + span * (8 if key.seg16 or variant == HIGH else 16)
        + height * (4 if variant == HIGH else 8)
        + checkpoints
    )
    return launch._replace(
        key=key,
        partials=launch.workers,
        plan=launch.plan._replace(
            smem=smem, chunk=span, full_trajectory=not chunk, cache_source=True, shared_acc=False
        ),
    )


def kernel_source(key: scans.Kernel, variant: str) -> str:
    src = scans._kernel_source(key)
    if variant in (BASE, VF):
        return src
    src = src.replace(
        "#include <cuda_bf16.h>",
        "#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n"
        + _SCALE
        + (_FP8 if variant == HIGH else ""),
    )
    start = src.index("void path_adjoint_pair(")
    end = src.index("\n}\n#endif", start)
    fn = src[start:end].replace("float* acc, I stride", "__half2* acc2, float inv")
    a = fn.index("        acc[88")
    b = fn.index("\n    }", a)
    fn = (
        fn[:a]
        + """        acc2[44] = __hadd2(acc2[44], __floats2half2_rn(
            dy[cell].x*inv,dy[cell].y*inv));"""
        + fn[b:]
    )
    a = fn.index("            acc[(64")
    b = fn.index("\n        }\n#if", a)
    src = src[:start] + fn[:a] + _ACC + fn[b:] + src[end:]
    src = src.replace(
        "float pre[2]{}, weights[8];", 'asm volatile("" ::: "memory"); float pre[2]{}, weights[8];'
    )
    a = src.index("void column_backward(")
    b = src.index("#undef CACHED_SOURCE_AT", a)
    col = src[a:b].replace("const auto ar =", _SCAN + "    const auto ar =")
    col = col.replace("p, dy, path, acc, acc_stride, dp", "p, dy, path, acc2, inv, dp")
    if variant == HIGH:
        col = col.replace("float4* seg, Quad<T>* seg_src", "Quad<T>* seg, FP8Quad* seg_src")
        col = col.replace("Quad<T>* source_cache", "FP8Quad* source_cache")
        col = col.replace("    float peak", _SOURCE_SCALE + "    float peak")
        col = re.sub(r"(source_cache\[[^\]]+\])\.unpack\(\)", r"\1.unpack(src_scale)", col)
        col = col.replace("= packed;", "= FP8Quad::pack(packed.unpack(),src_inv);")
        col = re.sub(
            r"(seg_src\[j \* THREADS\] = )Quad<T>::pack\((\w+)\)",
            r"\1FP8Quad::pack(\2,src_inv)",
            col,
        )
        col = re.sub(r"(seg\[[^\]]+\]) = ([^;]+);", r"\1 = Quad<T>::pack(\2);", col)
        col = re.sub(r"(seg\[[^\]]+\])(?! =)", r"\1.unpack()", col)
    src = src[:a] + col[: col.rindex("\n}")] + _FLUSH + "\n}\n" + src[b:]
    src = src.replace("BACKWARD && HIDDEN && !SHARED_ACC", "false")
    if variant == HIGH:
        src = src.replace("auto* seg = shared", "auto* seg_base = shared")
        src = src.replace(
            "auto* ckpt = seg + (BACKWARD ? span * threads : 0);",
            "auto* seg = reinterpret_cast<Quad<T>*>(seg_base);\n"
            "    auto* ckpt = reinterpret_cast<float4*>(seg + span * threads);",
        )
        src = src.replace(
            "auto* seg_src = reinterpret_cast<Quad<T>*>",
            "auto* seg_src = reinterpret_cast<FP8Quad*>",
        )
        src = src.replace("__launch_bounds__(128)", "__launch_bounds__(128, 3)")
    return src


def compile_cubin(key: scans.Kernel, variant: str, path: Path) -> Path:
    path.write_text(kernel_source(key, variant))
    cubin = path.with_suffix(".cubin")
    proc = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "nvcc",
            "-std=c++17",
            "-arch=sm_90a",
            "--fmad=" + str(key.backward or key.vector_weights).lower(),
            *(["-Xptxas=-O1"] if variant in (HIGH, SAME) else []),
            "-cubin",
            str(path),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return cubin


@lru_cache(maxsize=128)
def _function(device: int, key: scans.Kernel, variant: str, smem: int) -> tuple:
    with tempfile.TemporaryDirectory() as folder:
        image = ct.create_string_buffer(
            compile_cubin(key, variant, Path(folder) / "scan.cu").read_bytes()
        )
    driver = scans._load_cuda_driver()
    scans._ensure_context(driver, device)
    module, fn = ct.c_void_p(), ct.c_void_p()
    scans._check_cu(driver.cuModuleLoadData(ct.byref(module), image), "load wave64 cubin")
    scans._check_cu(
        driver.cuModuleGetFunction(ct.byref(fn), module, scans._kernel_name(key).encode()),
        "load wave64 function",
    )
    scans._configure_smem(driver, fn, smem)
    return driver, module, fn


def _dispatch(real: torch.Tensor, launch: scans.Launch, args: tuple, *, variant: str) -> None:
    key = launch.key
    with torch.cuda.device(real.device):
        driver, _, fn = _function(real.get_device(), key, variant, launch.plan.smem)
        argv = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
        grid = scans._ceil(real.shape[-1], 32) * scans._ceil(launch.workers, key.warps)
        stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        scans._check_cu(
            driver.cuLaunchKernel(
                fn, grid, 1, 1, 32 * key.warps, 1, 1, launch.plan.smem, stream, argv, None
            ),
            "launch wave64",
        )


@lru_cache
def _runner(variant: str) -> Callable:
    # Reuse the production allocator, argument packing and reduction with isolated hooks.
    def planner(states: torch.Tensor, shape: tuple, **kwargs: object) -> scans.Launch:
        return plan(scans._plan_launch(states, shape, **kwargs), shape[1], variant)

    fn = scans._vscan
    scope = dict(
        fn.__globals__, _plan_launch=planner, _dispatch=partial(_dispatch, variant=variant)
    )
    result = FunctionType(fn.__code__, scope, fn.__name__, fn.__defaults__, fn.__closure__)
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


def vertical(
    pole: list,
    states: torch.Tensor,
    aux: list,
    path: list,
    coarse: bool,  # noqa: FBT001
    variant: str,
    *,
    grad: tuple | None = None,
    compact: bool = False,
) -> torch.Tensor | list:
    if states.dtype != torch.bfloat16 or states.device.type != "cuda" or path[0].shape[1] != 8:
        raise ValueError("expected CUDA BF16 states and eight hidden units")
    if variant not in ((BASE, VF) if grad is None else (BASE, SAME, HIGH)):
        raise ValueError(variant)
    fn = scans._vscan if variant == BASE else _runner(variant)
    return fn(
        pole, states, aux, path, coarse, 1e-6, grad is None, grad, packed=True, compact=compact
    )
