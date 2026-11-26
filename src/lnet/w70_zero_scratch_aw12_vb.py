# ruff: noqa: ANN001, ANN201, ANN202, S603, S607, SLF001
"""Fused split-plane Hopper vertical backward candidates."""

import ctypes as ct
import re
import subprocess
import tempfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import torch

from . import general_cuda_scans as scans
from . import w66_split_bf16_scan as fp

BF = "w70_cachesrc_bf16dwi_aw12"
HALF = "w70_cachesrc_fp16dwi_dyn_aw12"
FULL = "w70_full_bf16dwi_aw12"
ARMS = (BF, HALF, FULL)


def configure(base, h, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    warps = 4
    chunk = 8 if h == 56 and arm != FULL else 0
    key = replace(
        base.key,
        warps=warps,
        chunk=chunk,
        cache_weights=True,
        cache_source=arm != FULL,
        cache_line=False,
        shared_acc=False,
        global_checkpoints=False,
        sgs=0,
        l2_single_write=False,
        backward_reduce_warps=0,
        seg16=True,
        fp32_path=True,
        tanh_gain=True,
        path_bwd_pipe=0,
        backward_batch2=True,
        backward_batch_tail=False,
        vector_weights=True,
        vertical_w55="v55_bf16x2_bf16w",
        register_path=False,
    )
    threads = warps * 32
    smem = 5632 + threads * 8 * ((chunk or h) + (h if arm != FULL else 0))
    workers = max(1584, base.workers)
    return base._replace(
        key=key,
        workers=workers,
        partials=workers,
        plan=base.plan._replace(
            smem=smem,
            warps=warps,
            chunk=chunk,
            cache_line=False,
            cache_source=arm != FULL,
            global_checkpoints=False,
            full_trajectory=not chunk,
        ),
    )


def kernel_source(key, h, w, m, arm):
    s = fp.vertical_source(key)
    s = s.replace("static_assert(!full || CACHE_SOURCE);", "")
    s = s.replace("block ? c.H - span : 0", "block * span").replace(
        "full || block ? span : c.H - span", "min(span, c.H - y0)"
    )
    s = s.replace(
        "float4 pos{}, neg{}, boundary{};",
        """
    float4 pos{}, neg{};
    Quad<T> checkpoint[6]{};
    auto save=[&](int slot,float4 v){
        #pragma unroll
        for(int n=0;n<6;++n) if(n==slot) checkpoint[n]=Quad<T>::pack(v);
    };
    auto load=[&](int slot){
        float4 v{};
        #pragma unroll
        for(int n=0;n<6;++n) if(n==slot) v=checkpoint[n].unpack();
        return v;
    };""",
    )
    s = s.replace("boundary = pos;", "save(block,pos);").replace(
        "boundary = neg;", "save(block,neg);"
    )
    s = s.replace("block ? boundary :", "block ? load(block-1) :").replace(
        "chunks ? boundary :", "chunks ? load(block) :"
    )
    begin = s.index("void column_backward(")
    split = s.index("    adj_a = adj_b = {};", begin)
    end = s.index("#undef CACHED_SOURCE_AT", split)
    descending, ascending = s[begin:split], s[split:end]
    if arm != FULL:
        descending = re.sub(
            r"g2\[I\((?:COARSE \? y / 2 : y|yo)\) \* g2_stride\] =", "auto injection =", descending
        )
        descending = re.sub(
            r"(?m)^( +)if constexpr \(full\) \{",
            r"\1seg_src[I(y-int(COARSE))*THREADS]=injection;\n\1if constexpr (full) {",
            descending,
        )
        ascending = ascending.replace(
            "g2[I(COARSE ? y / 2 : y) * g2_stride]", "seg_src[I(y)*THREADS]"
        )
    else:
        descending = descending.replace("source_cache[(j) * THREADS].unpack()", "c.source(y0+(j))")
    descending = descending.replace(
        "if constexpr (COARSE && !CACHE_SOURCE) seg_src[j * THREADS] = Quad<T>::pack(source);", ""
    )
    ascending = ascending.replace("source_cache[j * THREADS].unpack()", "c.source(y0+j)").replace(
        "CACHED_SOURCE_AT(j)", "c.source(y)"
    )
    ascending = ascending.replace(
        "if constexpr (!CACHE_SOURCE) seg_src[j * THREADS] = Quad<T>::pack(s);", ""
    )
    s = s[:begin] + descending + ascending + s[end:]
    if arm != FULL:
        s = fp.replace(s, "scratch + I(worker) * M + m;", "nullptr;")
    s = accumulator_source(s, arm)
    if h == 28 and arm != HALF:
        s = s.replace("2*step)%8==0", f"2*step)%{12 if key.coarse else 14}==0")
    pos = s.rindex(") {") + 3
    s = s[:pos] + f"\n H={h}; W={w}; M={m};\n" + s[pos:]
    return s.replace(" void " + scans._kernel_name(key) + "(", " void " + arm + "(")


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
    s = s.replace(
        "packed_weight<T>(path,j,k,weights[k])", "Pair<T>::from(weights[k],weights[k]).value"
    )
    return s.replace("float* sums,", "volatile float* sums,")


def accumulator_source(s, arm):
    half = arm == HALF
    ty = "__half2" if half else "__nv_bfloat162"
    pack = "__floats2half2_rn" if half else "__floats2bfloat162_rn"
    unpack = "__half22float2" if half else "__bfloat1622float2"
    s = s.replace("__launch_bounds__(128)", "__launch_bounds__(128,3)")
    s = s.replace(
        "float* acc, I stride, float (&dp)",
        f"{ty}* dwi, float scale, float* acc, I stride, float (&dp)",
    )
    s = s.replace(
        "T* gs, float* acc, I acc_stride,", "T* gs, float* sums, float* acc, I acc_stride,"
    )
    s = s.replace(
        "acc, acc_stride, (seg + threadIdx.x)", "row+4*M, acc, acc_stride, (seg + threadIdx.x)"
    )
    s = s.replace(
        "float local_acc[register_acc ? 11 * HIDDEN + 2 : 1]{};", "float local_acc[26]{};"
    )
    s = s.replace("I(reduce ? group : worker)", "I(worker)")
    s = s.replace(
        "acc = local_acc;", "acc=local_acc; if(active) for(int k=0;k<64;++k) row[(4+k)*M]=0;"
    )
    old = """write_vscan_partial<WARPS, 11 * HIDDEN + 6, I>(
                reinterpret_cast<float*>(shared), row, local_acc, da, dg, m, M);"""
    new = """if(active){
        row[0]=da.r; row[M]=da.i; row[2*M]=dg.r; row[3*M]=dg.i;
        #pragma unroll
        for(int k=0;k<26;++k) row[(68+k)*M]=local_acc[k];
    }"""
    s = fp.replace(s, old, new)
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
    marker = "const auto ar = a.conj(), gg = g.conj();\n    // CHUNK"
    s = fp.replace(s, marker, f"{ty} dwi[32]{{}}; float scale=1.f;\n    " + marker)
    s = load_weight_rows(s)
    call = "path_adjoint_pair<T, I>(p, dy, path, acc, acc_stride, dp);"
    # Scan a full chunk before selecting a scale, including zero-leading chunks.
    scaling = (
        """
            if (((length-1-end)%8)==0){
                float mag=0;
                for(int q=end;q>=max(0,end-7);q-=step){
                    I cell=c.cell(COARSE ? (y0+q)/2 : y0+q);
                    mag=fmaxf(mag,fabsf(read(gr[(cell-c.m)*(1+REAL_INTERLEAVED)+c.m])));
                    mag=fmaxf(mag,fabsf(read(gi[(cell-c.m)*(1+IMAG_INTERLEAVED)+c.m])));
                }
                int exp=((__float_as_uint(mag)>>23)&255)-127;
                scale=mag>0.f ? __int_as_float((127+max(-8,min(20,4-exp)))<<23) : 1.f;
            }
"""
        if half
        else ""
    )
    return fp.replace(
        s,
        call,
        scaling
        + call.replace("path, acc", "path, dwi, scale, acc")
        + f"""
            if((length-1-end+2*step)%8==0 || end<2*step){{
                #pragma unroll
                for(int k=0;k<32;++k){{
                    auto v={unpack}(dwi[k]);
                    sums[2*k*c.M]+=v.x/scale; sums[(2*k+1)*c.M]+=v.y/scale;
                    dwi[k]={pack}(0.f,0.f);
                }}
            }}""",
    )


def compile_flags(arm):
    """Keep BF16 checkpoint and accumulator live ranges within the three-CTA register budget."""
    return [] if arm == HALF else ["-Xptxas=-O1"]


@lru_cache(maxsize=128)
def _function(device, key, h, w, m, arm, smem):
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "scan.cu"
        cu.write_text(kernel_source(key, h, w, m, arm))
        cubin = cu.with_suffix(".cubin")
        subprocess.run(
            [
                "nvcc",
                "-std=c++17",
                "-arch=sm_90a",
                *compile_flags(arm),
                "-cubin",
                str(cu),
                "-o",
                str(cubin),
            ],
            check=True,
        )
        image = ct.create_string_buffer(cubin.read_bytes())
    driver = scans._load_cuda_driver()
    scans._ensure_context(driver, device)
    mod, fn = ct.c_void_p(), ct.c_void_p()
    scans._check_cu(driver.cuModuleLoadData(ct.byref(mod), image), "load w70")
    scans._check_cu(driver.cuModuleGetFunction(ct.byref(fn), mod, arm.encode()), "resolve w70")
    scans._configure_smem(driver, fn, smem)
    return driver, mod, fn


def backward(pole, r, states, aux, path, coarse, grad, arm):
    fp._check_source(r, fp.ARM)
    b, h, w, m = r.shape
    if (h, w, m) not in ((56, 56, 160), (28, 28, 160)):
        raise ValueError("unsupported Wave 70 shape")
    fp._check_storage(r, states, w)
    logical = r.unsqueeze(0).expand(4, -1, -1, -1, -1)
    if (
        not scans._pole_ok(pole, r)
        or not scans._vertical_inputs_ok(logical, aux, path, coarse=coarse)
        or path[0].shape[1] != 8
    ):
        raise ValueError("invalid vertical scan inputs")
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    gr, gi = scans._vertical_adjoints(grad, logical, (b, oh, ow, m))
    base = scans._plan_launch(
        logical,
        tuple(r.shape),
        backward=True,
        vertical=True,
        coarse=coarse,
        hidden=8,
        precise=False,
        compact=coarse,
        interleaved=(not gr.is_contiguous(), not gi.is_contiguous()),
    )
    launch = configure(base, h, arm)
    dq = fp._words(r, ow)
    partial = r.new_empty((launch.partials, 94, m), dtype=torch.float32)
    scratch = r.new_empty((oh, launch.workers, m, 4)) if arm == FULL else None
    tensors = (
        *(v.float().contiguous() for v in pole),
        states,
        *(v.float().contiguous() for v in aux),
        *(v.contiguous() for v in path),
        gr,
        gi,
        None,
        None,
        dq,
        partial,
        scratch,
        None,
    )
    args = (
        *map(scans._ptr, tensors),
        *map(ct.c_int, (b, h, w, m, launch.workers, 8, scans._scalar_types(path))),
        ct.c_float(1e-6),
        ct.c_longlong(0),
        ct.c_longlong(0),
    )
    argv = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
    key = launch.key
    with torch.cuda.device(r.device):
        driver, _, fn = _function(r.device.index, key, h, w, m, arm, launch.plan.smem)
        grid = scans._ceil(m, 32) * scans._ceil(launch.workers, key.warps)
        scans._check_cu(
            driver.cuLaunchKernel(
                fn,
                grid,
                1,
                1,
                32 * key.warps,
                1,
                1,
                launch.plan.smem,
                ct.c_void_p(torch.cuda.current_stream().cuda_stream),
                argv,
                None,
            ),
            arm,
        )
    params = scans._reduce_partials(partial, [*pole, *path], 8)
    return [*params[:4], dq, *params[4:]]
