# ruff: noqa: ANN001, ANN201, ANN202, ARG001, FBT002, RUF059, S603, S607, SLF001
"""Small-resolution W57 SGS scans with E4M3 states and E5M2 adjoints."""

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
from . import w72_split_fp8_scan as w72

compile_flags = w72.compile_flags
_words = w72._words
_check_storage = w72._check_storage


def profile_name(key):
    return w72.profile_name(key).replace("w72_split_fp8", "w73b_small_fp8")


def configure(launch, height):
    if not launch.key.backward:
        return launch
    key = with_fields(
        launch.key,
        sgs=1,
        chunk=0,
        seg16=True,
        cache_line=True,
        cache_source=True,
        shared_acc=False,
        global_checkpoints=False,
        l2_single_write=False,
    )
    return launch._replace(
        key=key, plan=launch.plan._replace(chunk=0, cache_line=True, global_checkpoints=False)
    )


def _vertical(key):
    src = w66.vertical_source(key)
    src = src.replace("Quad<T> packed_source(", "Fp8Quad packed_source(")
    src = src.replace("const Pair<T>*>(states)", "const Fp8Pair*>(states)")
    if not key.backward:
        return src
    if not (key.sgs and key.cache_line and key.seg16 and key.cache_source):
        raise ValueError("expected cached SGS plan")
    src = src.replace("int xa,xb,x;", "int xa,xb,x; float scale,inv;")
    src = src.replace("xa,xb,x};", "xa,xb,x,scales[0],scales[1]};")
    src = src.replace("long long grad_plane\n", "long long grad_plane, const float* scales\n")
    src = src.replace("input_plane, grad_plane);", "input_plane, grad_plane, scales);")
    src = src.replace("Pair<T>*>(gs)", "E5m2Pair*>(gs)")
    for slot, field in (("ia", "a"), ("ib", "b")):
        src = w66.replace(
            src,
            f"(ACCUMULATE ? words[{slot}].unpack() : positive.{field}.unpack())",
            f"positive.{field}.unpack()",
        )
        src = w66.replace(
            src,
            f"words[{slot}]=Pair<T>::pack(ds_{field});",
            f"words[{slot}]=E5m2Pair::pack(ds_{field},c.scale);",
        )
    src = src.replace("Quad<T>* seg", "Fp8Quad* seg")
    src = src.replace(
        "reinterpret_cast<Quad<T>*>(seg_base)", "reinterpret_cast<Fp8Quad*>(seg_base)"
    )
    src = src.replace(
        "auto* seg_src = reinterpret_cast<Quad<T>*>", "auto* seg_src = reinterpret_cast<Fp8Quad*>"
    )
    src = src.replace("Quad<T>* source_cache", "Fp8Quad* source_cache")
    src = re.sub(r"(seg(?:_src)?\[[^\]]+\] = )Quad<T>::pack", r"\1Fp8Quad::pack", src)
    src = src.replace("Quad<T>* g2,", "E5m2Quad* g2,")
    src = re.sub(
        r"(g2\[[^\]]+\] =\s*)(normalize_packed<T>\(.*?\));",
        r"\1E5m2Quad::pack(\2.unpack(),c.scale);",
        src,
        flags=re.DOTALL,
    )
    src = src.replace("g2_stride].unpack()", "g2_stride].unpack(c.inv)")
    return w66.replace(
        src,
        "buf, stride, chunks, da, dg, line_buf + (CACHE_LINE ? I(oh) * threads : 0) + threadIdx.x",
        "reinterpret_cast<E5m2Quad*>(line_buf)+threadIdx.x, I(threads), chunks, da, dg, "
        "reinterpret_cast<Quad<T>*>(reinterpret_cast<E5m2Quad*>(line_buf)+I(oh)*threads)+threadIdx.x",
    )


def kernel_source(key, width, height, modes):
    src = _vertical(key) if key.vertical else w72._horizontal(key)
    pos = src.index("template<class T> struct alignas")
    src = "#include <cuda_fp8.h>\n#include <type_traits>\n" + src[:pos] + w72.FP8 + src[pos:]
    if not key.vertical or key.backward:
        pos = src.rindex(") {") + 3
        src = src[:pos] + f"\n H={height}; W={width}; M={modes};\n" + src[pos:]
    src = w66.replace(src, f" void {scans._kernel_name(key)}(", f" void {profile_name(key)}(")
    return src + w72.SCALE_KERNEL if key.vertical and key.backward else src


def shared_bytes(launch, height):
    k = launch.key
    if not (k.vertical and k.backward):
        return launch.plan.smem
    weights = (
        11 * 8 * 32 * (4 if k.vector_weights and k.vertical_w55 != "v55_bf16x2_bf16w" else 2)
        if k.cache_weights
        else 0
    )
    return weights + k.warps * 32 * (16 * height + 4 * (height // (1 + k.coarse)))


def _check_source(real):
    w72._check_source(real)
    if tuple(real.shape[1:]) not in ((14, 14, 160), (7, 7, 128)):
        raise ValueError("unsupported small scan shape")


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


def cached_horizontal(launch, w):
    if not launch.key.cache_line:
        threads = launch.key.warps * 32
        smem = max(launch.plan.smem, w * threads * (8 if launch.key.horizontal_vec2 else 4))
        launch = launch._replace(
            key=with_fields(launch.key, cache_line=True),
            plan=launch.plan._replace(cache_line=True, smem=smem),
        )
    return launch


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
    if not backward:
        launch = cached_horizontal(launch, w)
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
    if coarse and h == 7:
        raise ValueError("unsupported coarse 7x7 shape")
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
