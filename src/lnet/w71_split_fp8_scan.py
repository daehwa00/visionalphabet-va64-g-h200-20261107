# ruff: noqa: ANN001, ANN201, ANN202, ARG001, FBT002, RUF059, S603, S607, SLF001
"""Wave 71 split-plane E4M3 and BF16 zero-scratch scan kernels."""

import ctypes as ct
import re
import subprocess
import tempfile
from dataclasses import replace as with_fields
from functools import lru_cache
from pathlib import Path

import torch

from . import general_cuda_scans as scans
from . import w66_split_bf16_scan as w66

ARM = "w71_split_fp8"
BF16 = "w71_bf16_zero_scratch"
compile_flags = w66.compile_flags

FP8 = """
struct Fp8Pair {
    unsigned short value;
    __device__ static Fp8Pair pack(Complex c, float scale=1.f) {
        unsigned short word;
        // PTX packs its second source (real) low and first source (imag) high.
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(word) : "f"(c.r*scale),"f"(c.i*scale));
        return {word};
    }
    __device__ Complex unpack(float inv=1.f) const {
        unsigned bits;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(bits) : "h"(value));
        auto f=__half22float2(*reinterpret_cast<__half2*>(&bits));
        return {f.x*inv,f.y*inv};
    }
};
struct alignas(4) Fp8Quad {
    Fp8Pair a,b;
    __device__ static Fp8Quad pack(float4 v,float scale=1.f) {
        return {Fp8Pair::pack({v.x,v.y},scale),Fp8Pair::pack({v.z,v.w},scale)};
    }
    __device__ float4 unpack(float inv=1.f) const {
        auto x=a.unpack(inv), y=b.unpack(inv);
        return make_float4(x.r,x.i,y.r,y.i);
    }
};
static_assert(sizeof(Fp8Quad)==4);
"""


def profile_name(key, variant):
    return variant + "_" + ("v" if key.vertical else "h") + ("b" if key.backward else "f")


def _horizontal(key, height):
    # Use the contiguous two-mode store on both heights.
    src = w66.horizontal_source(key, 56, w66.ARM)
    src = src.replace("int M, int workers\n", "int M, int workers, const float* scales\n")
    src = src.replace("M, workers);", "M, workers, scales);")
    src = src.replace("const Pair<T>*>(adj?grad:states)", "const Fp8Pair*>(adj?grad:states)")
    src = src.replace("W/step,neg)].unpack()", "W/step,neg)].unpack(adj?scales[1]:1.f)")
    src = src.replace("const uint2*>(words+", "const Fp8Quad*>(words+")
    for lane, field in (("x", "a"), ("y", "b")):
        src = src.replace(
            f"reinterpret_cast<Pair<T>*>(&pair.{lane})->unpack()",
            f"pair.{field}.unpack(adj?scales[1]:1.f)",
        )
    src = src.replace("Pair<T>::pack(output", "Fp8Pair::pack(output")
    src = src.replace("Pair<T>*>(out)[index", "Fp8Pair*>(out)[index")
    src = re.sub(r"(out\)\[index[^;]+?=)Pair<T>::pack", r"\1Fp8Pair::pack", src)
    src = re.sub(
        r"auto words=reinterpret_cast<Pair<T>\*>\(out\);.*?\*dst=pair;",
        "auto words=reinterpret_cast<Fp8Pair*>(out);\n"
        "*reinterpret_cast<Fp8Quad*>(words+index(x,W,pass))={a,b};",
        src,
        flags=re.DOTALL,
    )
    return src.replace("Pair<T>::pack(neg).unpack()", "Fp8Pair::pack(neg).unpack()")


def _vertical(key, variant):
    # Keep the original column_adjoint path, whose positive store and negative RMW use gs.
    raw = with_fields(key, sgs=0, l2_single_write=False) if key.backward else key
    src = w66.vertical_source(raw)
    if variant == BF16:
        return w66.reuse_source_rows(src, with_fields(raw, sgs=1))
    src = src.replace("Quad<T> packed_source(", "Fp8Quad packed_source(")
    src = src.replace("const Pair<T>*>(states)", "const Fp8Pair*>(states)")
    if not key.backward:
        return src
    src = src.replace("int xa,xb,x;", "int xa,xb,x; float scale,inv;")
    src = src.replace("xa,xb,x};", "xa,xb,x,scales[0],scales[1]};")
    src = src.replace("long long grad_plane\n", "long long grad_plane, const float* scales\n")
    src = src.replace("input_plane, grad_plane);", "input_plane, grad_plane, scales);")
    src = src.replace("Pair<T>*>(gs)", "Fp8Pair*>(gs)")
    src = re.sub(r"(words\[i[ab]\])\.unpack\(\)", r"\1.unpack(c.inv)", src)
    src = re.sub(r"Pair<T>::pack\((ds_[ab])\)", r"Fp8Pair::pack(\1,c.scale)", src)
    src = src.replace("Quad<T>* seg_src", "Fp8Quad* seg_src").replace(
        "Quad<T>* source_cache", "Fp8Quad* source_cache"
    )
    src = src.replace(
        "auto* seg_src = reinterpret_cast<Quad<T>*>", "auto* seg_src = reinterpret_cast<Fp8Quad*>"
    )
    src = src.replace("Quad<T>* g2,", "Fp8Quad* g2,")
    src = re.sub(r"(seg_src\[j \* THREADS\] = )Quad<T>", r"\1Fp8Quad", src)
    src = re.sub(
        r"(g2\[[^\]]+\] =\s*)(normalize_packed<T>\(.*?\));",
        r"\1Fp8Quad::pack(\2.unpack(),c.scale);",
        src,
        flags=re.DOTALL,
    )
    src = src.replace("g2_stride].unpack()", "g2_stride].unpack(c.inv)")
    return src.replace(
        "buf, stride, chunks, da, dg",
        "seg_src + I(H)*threads + threadIdx.x, I(threads), chunks, da, dg",
    )


def kernel_source(key, variant, width, height, modes, full_rows=False):
    src = _vertical(key, variant) if key.vertical else _horizontal(key, height)
    if variant == ARM:
        pos = src.index("template<class T> struct alignas")
        src = "#include <cuda_fp8.h>\n" + src[:pos] + FP8 + src[pos:]
    if not (key.vertical and key.backward):
        pos = src.rindex(") {") + 3
        src = src[:pos] + f"\n H={height}; W={width}; M={modes};\n" + src[pos:]
    entry_key = (
        with_fields(key, sgs=0, l2_single_write=False) if key.vertical and key.backward else key
    )
    src = w66.replace(
        src, f" void {scans._kernel_name(entry_key)}(", f" void {profile_name(key, variant)}("
    )
    return src + SCALE_KERNEL if key.vertical and key.backward and variant == ARM else src


def shared_bytes(launch, height):
    key = launch.key
    if key.vertical and key.backward:
        if not (key.cache_source and not key.shared_acc and (key.seg16 or not key.chunk)):
            raise ValueError("Wave 71 requires the Wave 57 packed trajectory preset")
        return 5632 + key.warps * 32 * (
            8 * height + (8 if key.seg16 else 16) * (key.chunk or height)
        )
    return launch.plan.smem


SCALE_KERNEL = """
extern "C" __global__ void w71_dq_scale(
    const __nv_bfloat16* re, const __nv_bfloat16* im, float* scales,
    long long n, int modes, int stride_r, int stride_i) {
    __shared__ float warp_max[8];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float peak = 0;
    for (int j = threadIdx.x; j < 4096; j += 256) {
        long long k = (j * n + n / 2) / 4096;
        long long row = k / modes;
        peak = fmaxf(peak, fabsf(__bfloat162float(re[row * stride_r + k % modes])));
        peak = fmaxf(peak, fabsf(__bfloat162float(im[row * stride_i + k % modes])));
    }
    for (int d=16; d; d/=2) peak=fmaxf(peak,__shfl_down_sync(0xffffffff,peak,d));
    if (!lane) warp_max[warp]=peak;
    __syncthreads();
    if (!threadIdx.x) {
        for (int w=0; w<8; ++w) peak=fmaxf(peak,warp_max[w]);
        int e = max(-100,min(100,5-ilogbf(fmaxf(peak,0x1p-100f))));
        scales[0]=ldexpf(1.f,e);
        scales[1]=ldexpf(1.f,-e);
    }
}
"""


@lru_cache(maxsize=256)
def _function(device, key, variant, width, height, modes, full_rows, smem):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "packed.cu"
        path.write_text(kernel_source(key, variant, width, height, modes, full_rows))
        cubin = path.with_suffix(".cubin")
        subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-arch=sm_90a",
                *compile_flags(key, variant, width),
                "-cubin",
                str(path),
                "-o",
                str(cubin),
            ],
            check=True,
            capture_output=True,
        )
        image = ct.create_string_buffer(cubin.read_bytes())
    with torch.cuda.device(device):
        driver = scans._load_cuda_driver()
        scans._ensure_context(driver, device)
        module, fn = ct.c_void_p(), ct.c_void_p()
        scans._check_cu(driver.cuModuleLoadData(ct.byref(module), image), "load packed cubin")
        scans._check_cu(
            driver.cuModuleGetFunction(ct.byref(fn), module, profile_name(key, variant).encode()),
            "get packed function",
        )
        scans._configure_smem(driver, fn, smem)
        scale_fn = ct.c_void_p()
        if key.vertical and key.backward and variant == ARM:
            scans._check_cu(
                driver.cuModuleGetFunction(ct.byref(scale_fn), module, b"w71_dq_scale"),
                "get dq scale",
            )
        return driver, module, fn, scale_fn


def _dispatch(real, launch, variant, width, args, scale_args=()):
    device = real.device.index or 0
    key = launch.key
    driver, module, fn, scale_fn = _function(
        device,
        key,
        variant,
        width,
        real.shape[1],
        real.shape[-1],
        launch.workers == real.shape[0] * real.shape[1],
        shared_bytes(launch, real.shape[1]),
    )
    grid = scans._ceil(real.shape[-1], 32 * (1 + key.horizontal_vec2)) * scans._ceil(
        launch.workers, key.warps
    )
    argv = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
    with torch.cuda.device(device):
        stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        if scale_args:
            scale_argv = (ct.c_void_p * len(scale_args))(*(ct.addressof(v) for v in scale_args))
            scans._check_cu(
                driver.cuLaunchKernel(scale_fn, 1, 1, 1, 256, 1, 1, 0, stream, scale_argv, None),
                "launch dq scale",
            )
        scans._check_cu(
            driver.cuLaunchKernel(
                fn,
                grid,
                1,
                1,
                32 * key.warps,
                1,
                1,
                shared_bytes(launch, real.shape[1]),
                stream,
                argv,
                None,
            ),
            "launch packed scan",
        )


def _check_source(real, variant):
    if variant not in (ARM, BF16):
        raise ValueError("unknown packed representation")
    if real.ndim != 4 or not real.is_cuda or real.dtype != torch.bfloat16:
        raise ValueError("expected CUDA BF16 [B,H,W,M] source")
    if torch.cuda.get_device_capability(real.device) != (9, 0):
        raise ValueError("packed kernels require SM90a")


def _check_storage(real, words, width, variant):
    b, h, _, m = real.shape
    tiles = (m + 31) // 32
    dtype = torch.uint16 if variant == ARM else torch.uint32
    if (
        tuple(words.shape) != (b, tiles, 2, width, h, 32)
        or words.dtype != dtype
        or words.device != real.device
        or not words.is_contiguous()
    ):
        raise ValueError("invalid packed words")


def _words(real, width, variant):
    b, h, _, m = real.shape
    return torch.empty(
        (b, (m + 31) // 32, 2, width, h, 32),
        device=real.device,
        dtype=torch.uint16 if variant == ARM else torch.uint32,
    )


def horizontal(pole, r, i, variant=ARM, states=None, adjoints=None, compact=False, scales=None):
    if variant == BF16:
        return w66.horizontal(pole, r, i, w66.ARM, states, adjoints, compact)
    if adjoints is not None and (
        scales is None
        or scales.device != r.device
        or scales.dtype != torch.float32
        or scales.shape != (2,)
        or not scales.is_contiguous()
    ):
        raise ValueError("HB requires VB scale tensor")
    _check_source(r, variant)
    if (
        r.shape != i.shape
        or r.device != i.device
        or r.dtype != i.dtype
        or not scans._pole_ok(pole, r)
    ):
        raise ValueError("invalid horizontal scan inputs")
    shape = tuple(r.shape)
    b, h, w, m = shape
    backward = adjoints is not None
    if compact and w % 2:
        raise ValueError("wave57 horizontal backward requires even compact width")
    layout = scans._horizontal_layout(r, i)
    r, i = [
        t if a or z else t.contiguous()
        for t, a, z in zip((r, i), layout["interleaved"], layout["transposed"], strict=True)
    ]
    launch = scans._plan_launch(
        r,
        shape,
        backward=backward,
        vertical=False,
        coarse=False,
        hidden=0,
        precise=False,
        compact=compact,
        aligned=scans._vec2_aligned(r, i),
        **layout,
    )
    # Short rows may use a register-only Wave 57 launch which reloads inputs.
    # Use the identical cached template for the return sweep in that case.
    if not backward and not launch.key.cache_line:
        threads = launch.key.warps * 32
        smem = max(launch.plan.smem, w * threads * (8 if launch.key.horizontal_vec2 else 4))
        launch = launch._replace(
            key=with_fields(launch.key, cache_line=True),
            plan=launch.plan._replace(cache_line=True, smem=smem),
        )
    partial = scratch = None
    if backward:
        _check_storage(r, states, w, variant)
        _check_storage(r, adjoints, w // (1 + compact), variant)
        q = states
        dq = adjoints
        out = r.new_empty((b, h, w, 2 * m))
        partial = r.new_empty((launch.partials, 4, m), dtype=torch.float32)
        if not launch.plan.cache_line:
            scratch = r.new_empty((w // (1 + compact), launch.workers, m, 2), dtype=torch.float32)

    else:
        q = dq = None
        out = _words(r, w, variant)

    inputs = (*(t.float().contiguous() for t in pole), r, i, q, dq, out, partial, scratch)
    _dispatch(
        r,
        launch,
        variant,
        w,
        (
            *map(scans._ptr, inputs),
            *map(ct.c_int, (*shape, launch.workers)),
            scans._ptr(scales),
        ),
    )
    if backward:
        return [*scans._reduce_partials(partial, pole), out]
    return out


def vertical(
    pole,
    r,
    states,
    aux,
    path,
    coarse,
    variant=ARM,
    grad=None,
    *,
    planar=False,
    separate=False,
    epsilon=1e-6,
):
    if variant == BF16 and grad is None:
        return w66.vertical(
            pole, r, states, aux, path, coarse, planar=planar, separate=separate, epsilon=epsilon
        )
    _check_source(r, variant)
    b, h, w, m = r.shape
    _check_storage(r, states, w, variant)
    logical = r.unsqueeze(0).expand(4, -1, -1, -1, -1)
    if (
        not scans._pole_ok(pole, r)
        or not scans._vertical_inputs_ok(logical, aux, path, coarse=coarse)
        or path[0].shape[1] != 8
    ):
        raise ValueError("invalid vertical scan inputs")
    q = states
    backward = grad is not None
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    gr, gi = scans._vertical_adjoints(grad, logical, (b, oh, ow, m)) if backward else (None, None)
    interleaved = (
        (not gr.is_contiguous(), not gi.is_contiguous())
        if backward
        else (not (planar or separate),) * 2
    )
    # Plan with the logical four-plane numel and dtype without allocating legacy states.
    launch = scans._plan_launch(
        logical,
        (b, h, w, m),
        backward=backward,
        vertical=True,
        coarse=coarse,
        hidden=8,
        precise=not backward,
        compact=backward and coarse,
        interleaved=interleaved,
    )
    if backward and variant == ARM and h == 28:
        launch = scans._plan_w57(r.device.index or 0, launch, (b, h, w, m), "v57_bwd_seg16_2chunk")
    outr = outi = dq = partial = scratch = checkpoints = None
    if backward:
        dq = _words(r, ow, variant)
        partial = r.new_empty((launch.partials, 94, m), dtype=torch.float32)
        ck = scans._checkpoint_rows(h, launch.plan.chunk)
        if launch.plan.global_checkpoints and ck:
            checkpoints = r.new_empty((ck, launch.workers, m, 4), dtype=torch.float32)
    elif separate:
        outr, outi = (r.new_empty((b, oh, ow, m)) for _ in range(2))
        out = (outr, outi)
    else:
        out = r.new_empty((2, b, oh, ow, m) if planar else (b, oh, ow, 2 * m))
        outr, outi = out.unbind(0) if planar else out.split(m, -1)
    if not backward and not launch.plan.cache_line:
        scratch = r.new_empty((oh, launch.workers, m, 4))
    inputs = (
        *(t.float().contiguous() for t in pole),
        q,
        *(t.float().contiguous() for t in aux),
        *(t.contiguous() for t in path),
        gr,
        gi,
        outr,
        outi,
        dq,
        partial,
        scratch,
        checkpoints,
    )
    args = (
        *map(scans._ptr, inputs),
        *map(ct.c_int, (b, h, w, m, launch.workers, 8, scans._scalar_types(path))),
        ct.c_float(epsilon),
        ct.c_longlong(0),
        ct.c_longlong(0),
    )
    scales = r.new_empty((2,), dtype=torch.float32) if backward and variant == ARM else None
    scale_args = ()
    if scales is not None:
        args += (scans._ptr(scales),)
        scale_args = (
            *map(scans._ptr, (gr, gi, scales)),
            ct.c_longlong(gr.numel()),
            *map(ct.c_int, (m, gr.stride(-2), gi.stride(-2))),
        )
    _dispatch(r, launch, variant, w, args, scale_args)
    if backward:
        params = scans._reduce_partials(partial, [*pole, *path], 8)
        return [*params[:4], dq, *params[4:], scales]
    return out


def pipeline(pole, r, i, aux, path, coarse, grad, variant=ARM):
    states = horizontal(pole, r, i, variant)
    out = vertical(pole, r, states, aux, path, coarse, variant)
    v = vertical(pole, r, states, aux, path, coarse, variant, grad)
    h = horizontal(pole, r, i, variant, states, v[4], coarse, scales=v[-1])
    return [out, *v[:4], *v[5:-1], *h]
