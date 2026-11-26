# ruff: noqa: ANN001, ANN201, ANN202, FBT002, S603, S607, SLF001
"""Experimental Wave 73A kernels. Each caller selects one operator variant explicitly."""

import ctypes as ct
import re
import subprocess
import tempfile
from dataclasses import replace as with_fields
from functools import lru_cache
from pathlib import Path

import torch

from . import w72_split_fp8_scan as base
from .w72_split_fp8_scan import (
    _check_source,
    _check_storage,
    _words,
    compile_flags,
    configure,
    profile_name,
    scans,
)

VB = ("vb_single_dq", "vb_single_dq_56c_tile", "vb_fp16x2_dwi", "vb_combo")
VF = ("vf_bf16x2_hidden", "vf_fp16x2_hidden")
HB = ("hb_vec2_mode",)


def _single(src, tile):
    # Both the dense and compact layouts store a positive value at every input row.
    slot = (
        "g2 + (c.H/2 + y)*THREADS" if tile else "reinterpret_cast<E5m2Quad*>(seg_src) + y*THREADS"
    )
    src = src.replace("Quad<T> positive = {}", "E5m2Quad positive = {}")
    for field in ("a", "b"):
        src = src.replace(f"positive.{field}.unpack()", f"positive.{field}.unpack(c.inv)")
    begin = src.index("void column_adjoint(")
    end = (
        src.index("template<class T> struct NativeWeights", begin)
        if "template<class T> struct NativeWeights" in src[begin:]
        else src.index("\n}", begin) + 2
    )
    helper = (
        src[begin:end]
        .replace("T* gs\n", "T* gs, E5m2Quad* slot\n")
        .replace(
            "write_column_gradient<REVERSE, COMPACT, REVERSE>(c, y, ds, gs);",
            """if constexpr (REVERSE) write_column_gradient<true, COMPACT>(c,y,ds,gs,*slot);
    else *slot=E5m2Quad::pack(ds,c.scale);""",
        )
    )
    src = src[:begin] + helper + src[end:]
    pattern = r"(column_adjoint<(?:true|false), COMPACT>\(\s*c, (y(?: - 1)?),[^;]+?), gs\);"
    src, count = re.subn(
        pattern, lambda m: m[1] + ", gs, " + slot.replace("y", "(" + m[2] + ")") + ");", src
    )
    if count != 5:
        raise ValueError("unexpected column adjoint schedule")
    if not tile:
        src = src.replace("CACHED_SOURCE_AT(j), in, ar", "c.source(y), in, ar")
    return src


def _dwi(src):
    src = src.replace("float (&dp)[2][8]\n", "float (&dp)[2][8], __half2 (&dwi)[32], float scale\n")
    src = src.replace(
        "path, acc, acc_stride, dp);\n            #pragma",
        "path, acc, acc_stride, dp,dwi,c.scale*0.25f);\n            #pragma",
    )
    src = src.replace(
        "float4 pos{}, neg{}, boundary{};",
        "__half2 dwi[32]{};\n    float4 pos{}, neg{}, boundary{};",
        1,
    )
    src = src.replace(
        "acc[(8 * j + k) * stride] += dz * p[cell][k];",
        """
                if (!(k&1)) {
                    auto x=__floats2half2_rn(p[cell][k],p[cell][k+1]);
                    auto& sum=dwi[4*j+k/2];
                    sum=__hfma2(__float2half2_rn(dz*scale),x,sum);
                }""",
    )
    return src.replace(
        "    adj_a = adj_b = {};",
        """
    #pragma unroll
    for (int k=0;k<32;++k) {
        auto x=__half22float2(dwi[k]);
        acc[2*k*acc_stride]+=x.x*(4.f*c.inv);
        acc[(2*k+1)*acc_stride]+=x.y*(4.f*c.inv);
    }
    adj_a = adj_b = {};""",
    )


def _hidden(src, arm):
    pack, unpack = (
        ("__floats2bfloat162_rn", "__bfloat1622float2")
        if arm == VF[0]
        else ("__floats2half2_rn", "__half22float2")
    )
    start = src.index(
        "        #pragma unroll\n        for (int t = 0; t < K; ++t) {\n            float pre = 0;"
    )
    anchor = src.index("const float wr = path.output(j), wj =", src.index("void path_mlp_batch"))
    packed = src.find("#if __CUDA_ARCH__", anchor, start)
    if packed >= 0:
        start = packed
    end = src.index("            pre = ((pre) + b);", start)
    src = (
        src[:start]
        + f"""
        static_assert(K%2==0);
        #pragma unroll
        for (int pair=0;pair<K;pair+=2) {{
            auto v={pack}(0.f,0.f);
            #pragma unroll
            for (int k=0;k<8;++k)
                v=__hfma2({pack}(weights[k],weights[k]),{pack}(p[pair][k],p[pair+1][k]),v);
            auto z={unpack}(v);
            #pragma unroll
            for (int lane=0;lane<2;++lane) {{
            int t=pair+lane;
            float pre=lane?z.y:z.x;
"""
        + src[end:]
    )
    return src.replace("out_i[t].add(wj * act);", "out_i[t].add(wj * act);\n            }", 1)


def _hb(src):
    # Adjacent lanes share one aligned four-BF16 transaction per real/imag plane.
    start = src.index("    auto read2=[]")
    end = src.index("    if (m<M", start)
    return (
        src[:start]
        + """
    auto read2=[&](const T* ptr) {
        unsigned mask=__activemask();
        uint2 bits{};
        if (!(lane&1)) bits=__ldg(reinterpret_cast<const uint2*>(ptr));
        unsigned hi=__shfl_sync(mask,bits.y,lane&~1);
        unsigned word=(lane&1)?hi:bits.x;
        return reinterpret_cast<Pair<T>*>(&word)->unpack();
    };
    auto store2=[&](T* ptr,float x,float y) {
        auto packed=Pair<T>::pack({x,y});
        unsigned lo=*reinterpret_cast<unsigned*>(&packed);
        unsigned hi=__shfl_sync(__activemask(),lo,lane|1);
        if (!(lane&1)) *reinterpret_cast<uint2*>(ptr)=make_uint2(lo,hi);
    };
"""
        + src[end:]
    )


def candidate_plan(launch, height, arm):
    if arm in HB:
        k = with_fields(
            launch.key,
            horizontal_vec2=True,
            warps=4,
            cache_line=True,
            horizontal_unroll=4,
            horizontal_bf16_buf=True,
        )
        size = 128 * 8 * (height // (1 + k.compact))
        return launch._replace(
            key=k,
            plan=launch.plan._replace(smem=size, warps=4, cache_line=True),
            partials=(launch.workers + 3) // 4,
        )
    return launch


def shared_bytes(launch, height, arm):
    size = base.shared_bytes(launch, height)
    return size + (14336 if arm == VB[1] else 0)


def kernel_source(key, width, height, modes, arm):
    src = base.kernel_source(key, width, height, modes)
    if arm == "w72":
        return src
    if (height, width, modes) not in ((56, 56, 160), (28, 28, 160)):
        raise ValueError("Wave 73A requires a core scan shape")
    if arm in VB and key.vertical and key.backward and not key.chunk:
        if arm == VB[1] and (height != 56 or not key.coarse):
            raise ValueError("dedicated positive tile requires 56c")
        if arm in (VB[0], VB[1], VB[3]):
            src = _single(src, arm == VB[1])
        if arm in VB[2:]:
            src = _dwi(src)
    elif arm in VF and key.vertical and not key.backward and not (height == 28 and key.coarse):
        src = _hidden(src, arm)
    elif arm not in HB or key.vertical or not key.backward or not key.horizontal_vec2:
        raise ValueError(f"unsupported kernel variant {arm}")
    else:
        src = _hb(src)
        if not key.compact:
            src = src.replace("#pragma unroll 4", "#pragma unroll 2", 1)
    return src


@lru_cache(maxsize=256)
def _function(device, key, width, height, modes, smem, arm):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "packed.cu"
        path.write_text(kernel_source(key, width, height, modes, arm))
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


def _dispatch(real, launch, width, args, scale_args=(), *, arm):
    device = real.device.index or 0
    key = launch.key
    driver, _module, fn, scale_fn = _function(
        device,
        key,
        width,
        real.shape[1],
        real.shape[-1],
        shared_bytes(launch, real.shape[1], arm),
        arm,
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
                shared_bytes(launch, real.shape[1], arm),
                stream,
                argv,
                None,
            ),
            "launch packed scan",
        )


def _check_vector_storage(r, i, states, adjoints):
    if states is None or adjoints is None or any(t.data_ptr() % 8 for t in (r, i)):
        raise ValueError("vector HB requires aligned BF16 drive planes and adjoints")
    if any(t.data_ptr() % 4 for t in (states, adjoints)):
        raise ValueError("vector HB requires aligned packed planes")


def horizontal(pole, r, i, states=None, adjoints=None, compact=False, scales=None, *, arm="w72"):
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
    if arm in HB:
        _check_vector_storage(r, i, states, adjoints)
    launch = candidate_plan(launch, h, arm)
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
        arm=arm,
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
    arm="w72",
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
    _dispatch(r, launch, w, args, scale_args, arm=arm)
    if backward:
        params = scans._reduce_partials(partial, [*pole, *path], 8)
        return [*params[:4], dq, *params[4:], scales]
    return out
