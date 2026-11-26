# ruff: noqa: ANN001, ANN201, ANN202, ARG001, FBT002, RUF059, S603, S607, SLF001
"""Wave 72 shared-memory vertical scan kernels."""

import ctypes as ct
import re
import subprocess
import tempfile
from dataclasses import replace as with_fields
from functools import lru_cache
from pathlib import Path

import torch

from . import general_cuda_scans as scans
from . import w71_split_fp8_scan as w71

VF = "w72_fp8_vf_smem"
BF = "w72_bf16_vf_smem"
FULL = "w72_fp8_vb_full"
AW = "w72_fp8_vb_aw12_shared8k"
compile_flags = w71.compile_flags
profile_name = w71.profile_name


def configure(launch, height, variant):
    key = launch.key
    if height not in (28, 56) or variant not in ((FULL, AW) if key.backward else (VF, BF)):
        raise ValueError("unsupported Wave 72 vertical scan")
    if key.backward:
        chunk = 8 if variant == AW and height == 56 else 0
        key = with_fields(
            key,
            chunk=chunk,
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
        smem = 5632 + 512 * (2 * height + (chunk or height)) + 8192 * (variant == AW)
        return launch._replace(
            key=key,
            partials=(launch.workers + 3) // 4,
            plan=launch.plan._replace(chunk=chunk, smem=smem, global_checkpoints=False),
        )
    key = with_fields(key, cache_line=True)
    weights = 11264 if key.vector_weights else 5632
    smem = weights + key.warps * 32 * (height // (1 + key.coarse)) * (4 if variant == VF else 8)
    return launch._replace(key=key, plan=launch.plan._replace(smem=smem, cache_line=True))


def shared_bytes(launch, height):
    return launch.plan.smem


def kernel_source(key, variant, width, height, modes, full_rows=False):
    src = w71.kernel_source(key, w71.ARM, width, height, modes, full_rows)
    src = src.replace(profile_name(key, w71.ARM), profile_name(key, variant))
    if not key.backward:
        if variant == VF:
            src = src.replace(
                "auto* buf = CACHE_LINE ? line_buf + threadIdx.x : scratch + I(worker) * M + m;",
                "auto* buf = reinterpret_cast<Fp8Quad*>(line_buf) + threadIdx.x;",
            )
            src = src.replace(
                "normalize_packed<T>(state, c.gain(y, false));",
                "Fp8Quad::pack(normalize_packed<T>(state, c.gain(y, false)).unpack());",
            )
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
    src = src[:pos] + f"\n H={height}; W={width}; M={modes};\n" + src[pos:]
    if variant == FULL:
        return src
    src = src.replace("block ? c.H - span : 0", "block * span").replace(
        "full || block ? span : c.H - span", "min(span, c.H - y0)"
    )
    src = src.replace(
        "float4 pos{}, neg{}, boundary{};",
        """
    float4 pos{}, neg{};
    Fp8Quad checkpoint[6]{};
    auto save=[&](int slot,float4 v){
        #pragma unroll
        for(int n=0;n<6;++n) if(n==slot) checkpoint[n]=Fp8Quad::pack(v);
    };
    auto load=[&](int slot){
        float4 v{};
        #pragma unroll
        for(int n=0;n<6;++n) if(n==slot) v=checkpoint[n].unpack();
        return v;
    };""",
    )
    src = src.replace("boundary = pos;", "save(block,pos);").replace(
        "boundary = neg;", "save(block,neg);"
    )
    src = src.replace("block ? boundary :", "block ? load(block-1) :").replace(
        "chunks ? boundary :", "chunks ? load(block) :"
    )
    return accumulator_source(src)


def load_weight_rows(s):
    """Load each packed weight row at its point of use to bound register lifetimes."""
    pos = s.index("        float pre[2]", s.index("void path_adjoint_pair"))
    end = s.index("        float dzs", pos)
    s = (
        s[:pos]
        + """        float pre[2];
        unsigned ptr=__cvta_generic_to_shared(path.cache)+(j*32+path.lane)*16;
        uint4 row;
        asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
            :"=r"(row.x),"=r"(row.y),"=r"(row.z),"=r"(row.w):"r"(ptr));
        auto weights=reinterpret_cast<T*>(&row);
#if __CUDA_ARCH__ >= 800
"""
        + s[end:]
    )
    return s.replace(
        "packed_weight<T>(path,j,k,weights[k])", "Pair<T>::from(weights[k],weights[k]).value"
    )


def accumulator_source(s):
    ty = "__half2"
    pack = "__floats2half2_rn"
    s = s.replace("__launch_bounds__(128)", "__launch_bounds__(128,3)")
    s = s.replace(
        "float* acc, I stride, float (&dp)",
        f"{ty}* dwi, float scale, float* acc, I stride, float (&dp)",
    )
    s = s.replace(
        "T* gs, float* acc, I acc_stride,",
        "T* gs, __half2* dwi, float scale, float* acc, I acc_stride,",
    )
    s = s.replace(
        "acc, acc_stride, (seg + threadIdx.x)", "dwi, scale, acc, acc_stride, (seg + threadIdx.x)"
    )
    s = s.replace(
        "float local_acc[register_acc ? 11 * HIDDEN + 2 : 1]{};", "float local_acc[26]{};"
    )
    s = s.replace(
        "write_vscan_partial<WARPS, 11 * HIDDEN + 6, I>",
        "if(warp==0 && m<M) {\n#pragma unroll\n"
        "for(int j=0;j<64;++j) row[(4+j)*M]=sums[j*32+lane];}\n"
        "write_vscan_partial<WARPS, 30, I>",
    )
    s = s.replace("row[I(begin + j) * M]", "row[I(begin+j+(begin+j>=4 ? 64 : 0))*M]")
    for original, tail in (
        ("88 * stride", "24"),
        ("89 * stride", "25"),
        ("(64 + j) * stride", "j"),
        ("(72 + j) * stride", "8+j"),
        ("(80 + j) * stride", "16+j"),
    ):
        s = s.replace("acc[" + original + "]", "acc[" + tail + "]")
    s = s.replace("acc[(8 * j + k) * stride] += dz * p[cell][k];", "")
    s = s.replace(
        "auto dz2=P::pack",
        f"""
        #pragma unroll
        for(int k=0;k<8;k+=2){{
            auto v=dwi[4*j+k/2];
            #pragma unroll
            for(int i=0;i<2;++i)
                v=__hfma2({pack}(dzs[i]*scale,dzs[i]*scale),{pack}(p[i][k],p[i][k+1]),v);
            dwi[4*j+k/2]=v;
        }}
        auto dz2=P::pack""",
    )
    s = load_weight_rows(s)
    s = s.replace("path, acc, acc_stride, dp);", "path, dwi, scale, acc, acc_stride, dp);")
    offset = s.index("auto* acc_base", s.index("auto* seg = reinterpret_cast<Fp8Quad*>"))
    left, v = s[:offset], s[offset:]
    v = v.replace(
        "Complex da{}, dg{};\n    if (active) {",
        """
    float* sums=reinterpret_cast<float*>(seg_src+2*I(H)*threads);
    for(int j=threadIdx.x;j<2048;j+=threads) sums[j]=0;
    __syncthreads();
    Complex da{}, dg{};
    for(I base_line=0;base_line<I(B)*ow;base_line+=workers){
        I line=base_line+worker;
        bool active_col=active && line<I(B)*ow;
        __half2 dwi[32]{};
        float scale=1.f;
        if(active_col){""",
        1,
    )
    v = v.replace("for (I line = worker; line < I(B) * ow; line += workers) {", "{", 1)
    v = v.replace(
        "auto* boundaries = GLOBAL_CKPT",
        """
                float peak=0.f;
                #pragma unroll 1
                for(int y=0;y<oh;++y){
                    I cell=c.cell(y)-m;
                    peak=fmaxf(peak,fabsf(read(gr[cell*(1+REAL_INTERLEAVED)+m])));
                    peak=fmaxf(peak,fabsf(read(gi[cell*(1+IMAG_INTERLEAVED)+m])));
                }
                int e=((__float_as_uint(peak)>>23)&255)-127;
                scale=peak>0 ? ldexpf(1.f,max(-8,min(20,4-e))) : 1.f;
                auto* boundaries = GLOBAL_CKPT""",
        1,
    )
    end = v.index("    if constexpr (BACKWARD) {\n        if constexpr (PURE && reduce)")
    v = (
        v[:end]
        + """
        #pragma unroll
        for(int w=0;w<WARPS;++w){
            if(warp==w && active_col){
                #pragma unroll
                for(int k=0;k<32;++k){
                    auto pair=__half22float2(dwi[k]);
                    sums[2*k*32+lane]+=pair.x/scale;
                    sums[(2*k+1)*32+lane]+=pair.y/scale;
                }
            }
            __syncthreads();
        }
    }
"""
        + v[end:]
    )
    return left + v


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
        if key.vertical and key.backward:
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


def vertical(
    pole,
    r,
    states,
    aux,
    path,
    coarse,
    variant=VF,
    grad=None,
    *,
    planar=False,
    separate=False,
    epsilon=1e-6,
):
    w71._check_source(r, w71.ARM)
    b, h, w, m = r.shape
    w71._check_storage(r, states, w, w71.ARM)
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
    launch = configure(launch, h, variant)
    outr = outi = dq = partial = scratch = checkpoints = None
    if backward:
        dq = w71._words(r, ow, w71.ARM)
        partial = r.new_empty((launch.partials, 94, m), dtype=torch.float32)
    elif separate:
        outr, outi = (r.new_empty((b, oh, ow, m)) for _ in range(2))
        out = (outr, outi)
    else:
        out = r.new_empty((2, b, oh, ow, m) if planar else (b, oh, ow, 2 * m))
        outr, outi = out.unbind(0) if planar else out.split(m, -1)
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
    _dispatch(r, launch, variant, w, args, scale_args)
    if backward:
        params = scans._reduce_partials(partial, [*pole, *path], 8)
        return [*params[:4], dq, *params[4:], scales]
    return out
