# ruff: noqa: ANN001, ANN201, ANN202, ARG001, FBT001, FBT002, RUF059, S603, S607, SLF001
"""Shape-selected SM90 split-plane scans with E4M3 states and E5M2 adjoints."""

import ctypes as ct
import os
import re
import subprocess
import tempfile
from dataclasses import replace as with_fields
from functools import lru_cache
from pathlib import Path

import torch

from . import general_cuda_scans as scans
from . import w66_split_bf16_scan as w66


def compile_flags(key, width):
    return w66.compile_flags(key, w66.ARM, width)


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
# Both formats have the same packed layout and differ only in their PTX conversions.
FP8 += FP8.replace("Fp8", "E5m2").replace("e4m3x2", "e5m2x2")


def profile_name(key):
    axis = "vscan" if key.vertical else "hscan"
    direction = "bwd" if key.backward else "fwd"
    return f"va_profile_{axis}_{direction}__w72_split_fp8"


def _horizontal(key):
    # Use the contiguous two-mode store on both heights.
    src = w66.horizontal_source(key, 56, w66.ARM)
    src = src.replace("int M, int workers\n", "int M, int workers, const float* scales\n")
    src = src.replace("M, workers);", "M, workers, scales);")
    # w66 passes literal booleans to both read helpers. Specialize their packed types.
    src = re.sub(r"((?:get|fetch)\([^,]+,)(true|false),", r"\1std::bool_constant<\2>{},", src)
    src = src.replace("bool adj,", "auto adj_tag,")
    src = src.replace(
        "int step=1+int(adj", "constexpr bool adj=decltype(adj_tag)::value; int step=1+int(adj"
    )
    src = src.replace(
        "const Pair<T>*>(adj?grad:states)",
        "const std::conditional_t<adj,E5m2Pair,Fp8Pair>*>(adj?grad:states)",
    )
    src = src.replace("W/step,neg)].unpack()", "W/step,neg)].unpack(adj?scales[1]:1.f)")
    src = src.replace(
        "const uint2*>(words+", "const std::conditional_t<adj,E5m2Quad,Fp8Quad>*>(words+"
    )
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


def _vertical(key):
    # Keep the original column_adjoint path, whose positive store and negative RMW use gs.
    raw = with_fields(key, sgs=0, l2_single_write=False) if key.backward else key
    src = w66.vertical_source(raw)
    src = src.replace("Quad<T> packed_source(", "Fp8Quad packed_source(")
    src = src.replace("const Pair<T>*>(states)", "const Fp8Pair*>(states)")
    if not key.backward:
        return src
    src = src.replace("int xa,xb,x;", "int xa,xb,x; float scale,inv;")
    src = src.replace("xa,xb,x};", "xa,xb,x,scales[0],scales[1]};")
    src = src.replace("long long grad_plane\n", "long long grad_plane, const float* scales\n")
    src = src.replace("input_plane, grad_plane);", "input_plane, grad_plane, scales);")
    src = src.replace("Pair<T>*>(gs)", "E5m2Pair*>(gs)")
    src = re.sub(r"(words\[i[ab]\])\.unpack\(\)", r"\1.unpack(c.inv)", src)
    src = re.sub(r"Pair<T>::pack\((ds_[ab])\)", r"E5m2Pair::pack(\1,c.scale)", src)
    src = src.replace("Quad<T>* seg_src", "Fp8Quad* seg_src").replace(
        "Quad<T>* source_cache", "Fp8Quad* source_cache"
    )
    src = src.replace(
        "auto* seg_src = reinterpret_cast<Quad<T>*>", "auto* seg_src = reinterpret_cast<Fp8Quad*>"
    )
    src = src.replace("Quad<T>* g2,", "E5m2Quad* g2,")
    src = re.sub(r"(seg_src\[j \* THREADS\] = )Quad<T>", r"\1Fp8Quad", src)
    src = re.sub(
        r"(g2\[[^\]]+\] =\s*)(normalize_packed<T>\(.*?\));",
        r"\1E5m2Quad::pack(\2.unpack(),c.scale);",
        src,
        flags=re.DOTALL,
    )
    src = src.replace("g2_stride].unpack()", "g2_stride].unpack(c.inv)")
    return src.replace(
        "buf, stride, chunks, da, dg",
        "reinterpret_cast<E5m2Quad*>(seg_src + I(H)*threads) + threadIdx.x, "
        "I(threads), chunks, da, dg",
    )


def configure(launch, height):
    key = launch.key
    if key.backward:
        if height != 56:
            return launch
        key = with_fields(
            key,
            chunk=0,
            warps=4,
            cache_weights=True,
            cache_source=True,
            global_checkpoints=False,
            seg16=True,
            shared_acc=False,
            backward_reduce_warps=0,
            cache_line=False,
            fp32_path=True,
            tanh_gain=True,
            path_bwd_pipe=0,
            backward_batch2=True,
            backward_batch_tail=False,
            sgs=0,
            l2_single_write=False,
            vector_weights=True,
            vertical_w55="v55_bf16x2_bf16w",
            register_path=False,
        )
        smem = 5632 + 1536 * height
        return launch._replace(
            key=key,
            partials=(launch.workers + 3) // 4,
            plan=launch.plan._replace(chunk=0, smem=smem, global_checkpoints=False),
        )
    if height == 28 and key.coarse:
        return launch
    key = with_fields(key, cache_line=True)
    weights = 11264 if key.vector_weights else 5632
    smem = weights + key.warps * 32 * (height // (1 + key.coarse)) * 4
    return launch._replace(key=key, plan=launch.plan._replace(smem=smem, cache_line=True))


def _upgrade_vertical(src, key, width, height, modes):
    if not key.backward:
        if height == 56 or not key.coarse:
            src = src.replace(
                "auto* buf = CACHE_LINE ? line_buf + threadIdx.x : scratch + I(worker) * M + m;",
                "auto* buf = reinterpret_cast<Fp8Quad*>(line_buf) + threadIdx.x;",
            )
            src = src.replace(
                "normalize_packed<T>(state, c.gain(y, false));",
                "Fp8Quad::pack(normalize_packed<T>(state, c.gain(y, false)).unpack());",
            )
        return src
    if height != 56:
        return src
    src = src.replace("Quad<T>* seg,", "Fp8Quad* seg,").replace(
        "auto* seg = reinterpret_cast<Quad<T>*>", "auto* seg = reinterpret_cast<Fp8Quad*>"
    )
    src = re.sub(r"(seg\[[^\]]+\] = )Quad<T>::pack", r"\1Fp8Quad::pack", src)
    pos = (
        src.index(
            ") {",
            src.index(
                'extern "C" __global__ __launch_bounds__', src.index("#undef CACHED_SOURCE_AT")
            ),
        )
        + 3
    )
    return src[:pos] + f"\n H={height}; W={width}; M={modes};\n" + src[pos:]


def kernel_source(key, width, height, modes):
    src = _vertical(key) if key.vertical else _horizontal(key)
    pos = src.index("template<class T> struct alignas")
    src = "#include <cuda_fp8.h>\n#include <type_traits>\n" + src[:pos] + FP8 + src[pos:]
    if not (key.vertical and key.backward):
        pos = src.rindex(") {") + 3
        src = src[:pos] + f"\n H={height}; W={width}; M={modes};\n" + src[pos:]
    entry_key = (
        with_fields(key, sgs=0, l2_single_write=False) if key.vertical and key.backward else key
    )
    src = w66.replace(src, f" void {scans._kernel_name(entry_key)}(", f" void {profile_name(key)}(")
    if key.vertical:
        src = _upgrade_vertical(src, key, width, height, modes)
    return src + SCALE_KERNEL if key.vertical and key.backward else src


def shared_bytes(launch, height):
    key = launch.key
    if key.vertical and key.backward and height != 56:
        if not (key.cache_source and not key.shared_acc and (key.seg16 or not key.chunk)):
            raise ValueError("Wave 71 requires the Wave 57 packed trajectory preset")
        return 5632 + key.warps * 32 * (
            8 * height + (8 if key.seg16 else 16) * (key.chunk or height)
        )
    return launch.plan.smem


SCALE_KERNEL = """
extern "C" __global__ void va_profile_scale__w72_split_fp8(
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
def _function(device, key, width, height, modes, smem):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "packed.cu"
        path.write_text(kernel_source(key, width, height, modes))
        cubin = path.with_suffix(".cubin")
        subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-arch=sm_90a",
                *compile_flags(key, width),
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
            driver.cuModuleGetFunction(ct.byref(fn), module, profile_name(key).encode()),
            "get packed function",
        )
        scans._configure_smem(driver, fn, smem)
        scale_fn = ct.c_void_p()
        if key.vertical and key.backward:
            scans._check_cu(
                driver.cuModuleGetFunction(
                    ct.byref(scale_fn), module, b"va_profile_scale__w72_split_fp8"
                ),
                "get dq scale",
            )
        return driver, module, fn, scale_fn


def _dispatch(real, launch, width, args, scale_args=()):
    device = real.device.index or 0
    key = launch.key
    driver, module, fn, scale_fn = _function(
        device,
        key,
        width,
        real.shape[1],
        real.shape[-1],
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


def _check_source(real):
    if real.ndim != 4 or not real.is_cuda or real.dtype != torch.bfloat16:
        raise ValueError("expected CUDA BF16 [B,H,W,M] source")
    if torch.cuda.get_device_capability(real.device) != (9, 0):
        raise ValueError("packed kernels require SM90a")


def _check_storage(real, words, width):
    b, h, _, m = real.shape
    tiles = (m + 31) // 32
    dtype = torch.int16
    if (
        tuple(words.shape) != (b, tiles, 2, width, h, 32)
        or words.dtype != dtype
        or words.device != real.device
        or not words.is_contiguous()
    ):
        raise ValueError("invalid packed words")


def _words(real, width):
    b, h, _, m = real.shape
    return torch.empty(
        (b, (m + 31) // 32, 2, width, h, 32),
        device=real.device,
        dtype=torch.int16,
    )


def horizontal(pole, r, i, states=None, adjoints=None, compact=False, scales=None):
    if adjoints is not None and (
        scales is None
        or scales.device != r.device
        or scales.dtype != torch.float32
        or scales.shape != (2,)
        or not scales.is_contiguous()
    ):
        raise ValueError("HB requires VB scale tensor")
    _check_source(r)
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
        _check_storage(r, states, w)
        _check_storage(r, adjoints, w // (1 + compact))
        q = states
        dq = adjoints
        out = r.new_empty((b, h, w, 2 * m))
        partial = r.new_empty((launch.partials, 4, m), dtype=torch.float32)
        if not launch.plan.cache_line:
            scratch = r.new_empty((w // (1 + compact), launch.workers, m, 2), dtype=torch.float32)

    else:
        q = dq = None
        out = _words(r, w)

    inputs = (*(t.float().contiguous() for t in pole), r, i, q, dq, out, partial, scratch)
    _dispatch(
        r,
        launch,
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
    grad=None,
    *,
    planar=False,
    separate=False,
    epsilon=1e-6,
):
    _check_source(r)
    b, h, w, m = r.shape
    _check_storage(r, states, w)
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
    if backward and h == 28:
        launch = scans._plan_w57(r.device.index or 0, launch, (b, h, w, m), "v57_bwd_seg16_2chunk")
    launch = configure(launch, h)
    outr = outi = dq = partial = scratch = checkpoints = None
    if backward:
        dq = _words(r, ow)
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
    scales = r.new_empty((2,), dtype=torch.float32) if backward else None
    scale_args = ()
    if scales is not None:
        args += (scans._ptr(scales),)
        scale_args = (
            *map(scans._ptr, (gr, gi, scales)),
            ct.c_longlong(gr.numel()),
            *map(ct.c_int, (m, gr.stride(-2), gi.stride(-2))),
        )
    _dispatch(r, launch, w, args, scale_args)
    if backward:
        params = scans._reduce_partials(partial, [*pole, *path], 8)
        return [*params[:4], dq, *params[4:], scales]
    return out


def _is_hopper(real):
    return real.is_cuda and torch.cuda.get_device_capability(real.device) == (9, 0)


def enabled(real, path, epilogue, kind, swiglu):
    return (
        os.environ.get("VA_W72_SPLIT_FP8") == "1"
        and real.dtype == torch.bfloat16
        and real.ndim == 4
        and real.shape[1:] in ((56, 56, 160), (28, 28, 160))
        and epilogue in ("full16", "coarse_path")
        and kind == 0
        and not swiglu
        and len(path) == 4
        and path[0].shape == (160, 8, 8)
        and scans.general_scans_enabled(forward=True)
        and _is_hopper(real)
    )


@torch.library.custom_op("lnet::w72_fp8_horizontal", mutates_args=())
def hforward(pole: list[torch.Tensor], r: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    return horizontal(pole, r, i)


@hforward.register_fake
def _fake_hf(pole, r, i):
    return _words(r, r.shape[2])


@torch.library.custom_op("lnet::w72_fp8_vertical", mutates_args=())
def vforward(
    pole: list[torch.Tensor],
    r: torch.Tensor,
    states: torch.Tensor,
    aux: list[torch.Tensor],
    path: list[torch.Tensor],
    coarse: bool,
    planar: bool,
    epsilon: float,
) -> torch.Tensor:
    return vertical(pole, r, states, aux, path, coarse, planar=planar, epsilon=epsilon)


@vforward.register_fake
def _fake_vf(pole, r, states, aux, path, coarse, planar, epsilon):
    b, h, w, m = r.shape
    spatial = (b, h // (1 + coarse), w // (1 + coarse))
    return r.new_empty((2, *spatial, m) if planar else (*spatial, 2 * m))


@torch.library.custom_op("lnet::w72_fp8_vertical_planes", mutates_args=())
def vforward_planes(
    pole: list[torch.Tensor],
    r: torch.Tensor,
    states: torch.Tensor,
    aux: list[torch.Tensor],
    path: list[torch.Tensor],
    coarse: bool,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return vertical(pole, r, states, aux, path, coarse, separate=True, epsilon=epsilon)


@vforward_planes.register_fake
def _fake_vf_planes(pole, r, states, aux, path, coarse, epsilon):
    b, h, w, m = r.shape
    shape = (b, h // (1 + coarse), w // (1 + coarse), m)
    return r.new_empty(shape), r.new_empty(shape)


@torch.library.custom_op("lnet::w72_fp8_scan_backward", mutates_args=())
def scan_backward(
    px: list[torch.Tensor],
    py: list[torch.Tensor],
    r: torch.Tensor,
    i: torch.Tensor,
    states: torch.Tensor,
    aux: list[torch.Tensor],
    path: list[torch.Tensor],
    gr: torch.Tensor,
    gi: torch.Tensor,
    coarse: bool,
    epsilon: float,
) -> list[torch.Tensor]:
    v = vertical(py, r, states, aux, path, coarse, grad=(gr, gi), epsilon=epsilon)
    h = horizontal(px, r, i, states=states, adjoints=v[4], compact=coarse, scales=v[-1])
    return [*h[:4], *v[:4], h[4], *v[5:-1]]


@scan_backward.register_fake
def _fake_backward(px, py, r, i, states, aux, path, gr, gi, coarse, epsilon):
    packed = r.new_empty((*r.shape[:-1], 2 * r.shape[-1]))
    return [
        *(p.new_empty(p.shape) for p in (*px, *py)),
        packed,
        *(p.new_empty(p.shape) for p in path),
    ]
