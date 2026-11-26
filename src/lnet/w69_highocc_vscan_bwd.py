# ruff: noqa: ANN001, ANN201, ANN202, E501, S603, S607, SLF001
"""Fused split-plane Hopper vertical backward candidates."""

import ctypes as ct
import subprocess
import tempfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import torch

from . import general_cuda_scans as scans
from . import w66_split_bf16_scan as fp

FULL = "w69_full_nocachesrc"
AW12 = "w69_smemacc_aw12"
AW16 = "w69_smemacc_aw16"
PACK = "w69_dwi_bf16x2_or_fp16x2_aw12"
ARMS = (FULL, AW12, AW16, PACK)


def configure(base, h, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    warps = 2 if arm in (AW16, PACK) else 4
    chunk = 0 if arm == FULL else 8
    key = replace(
        base.key,
        warps=warps,
        chunk=chunk,
        cache_weights=True,
        cache_source=False,
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
    rows = (chunk or h) + (h - 1) // (chunk or h)
    smem = 5632 + threads * (rows * 8 + (256 if arm != FULL else 0))
    if arm == PACK:
        smem += threads * 26 * 4
    # Baseline packs both transient lines in BF16 workspace allocations.
    oh = h // (1 + key.coarse)
    base_rows = h if h != 56 or base.key.l2_single_write or not base.key.sgs else 0
    if h != 56 and base.key.sgs and not base.key.l2_single_write:
        base_rows = 0
    if h == 56 and not base.key.cache_line:
        base_rows += oh
    # Retain the original scratch + partials byte budget.
    budget = base.workers * base_rows * 8 + base.partials * 94 * 4
    workers = min(base.workers, max(1, (budget // (warps * (h + oh) * 8 + 376)) * warps))
    return base._replace(
        key=key,
        workers=workers,
        partials=scans._ceil(workers, warps),
        plan=base.plan._replace(
            smem=smem,
            warps=warps,
            chunk=chunk,
            cache_line=False,
            cache_source=False,
            global_checkpoints=False,
            full_trajectory=not chunk,
        ),
    )


def kernel_source(key, h, w, m, arm):
    s = scans._sgs_source(fp.vertical_source(key), persistent=True)
    s = s.replace("static_assert(!full || CACHE_SOURCE);", "")
    s = s.replace(
        "#define CACHED_SOURCE_AT(j) source_cache[(j) * THREADS].unpack()",
        "#define CACHED_SOURCE_AT(j) c.source(y0 + (j))",
    )
    for line in (
        "if constexpr (COARSE && !CACHE_SOURCE) seg_src[j * THREADS] = Quad<T>::pack(source);",
        "if constexpr (!CACHE_SOURCE) seg_src[j * THREADS] = Quad<T>::pack(s);",
    ):
        s = s.replace(line, "")
    s = s.replace("float4 pos{}, neg{}, boundary{};", "float4 pos{}, neg{};")
    s = s.replace("block ? c.H - span : 0", "block * span").replace(
        "full || block ? span : c.H - span", "min(span, c.H - y0)"
    )
    s = s.replace("float4* ckpt,", "Quad<T>* ckpt,").replace(
        "reinterpret_cast<float4*>(seg + span * threads)", "seg + span * threads"
    )
    s = s.replace("boundary = pos;", "ckpt[I(block)*ckpt_stride] = Quad<T>::pack(pos);")
    s = s.replace("boundary = neg;", "ckpt[I(block)*ckpt_stride] = Quad<T>::pack(neg);")
    s = s.replace("block ? boundary :", "block ? ckpt[I(block-1)*ckpt_stride].unpack() :")
    s = s.replace("chunks ? boundary :", "chunks ? ckpt[I(block)*ckpt_stride].unpack() :")
    s = s.replace(
        "global_checkpoints + I(worker) * M + m",
        "reinterpret_cast<Quad<T>*>(global_checkpoints) + I(worker)*M+m",
    )
    s = s.replace(
        "seg_src + (BACKWARD ? I(CACHE_SOURCE ? H : span) * threads : 0)",
        "ckpt + I(chunks-1)*threads",
    )
    if arm != FULL:
        s = accumulator_source(s, key, arm)
    pos = s.rindex(") {") + 3
    s = s[:pos] + f"\n H={h}; W={w}; M={m};\n" + s[pos:]
    return s.replace(" void " + scans._kernel_name(key) + "(", " void " + arm + "(")


def accumulator_source(s, key, arm):
    threads = key.warps * 32
    blocks = 7 if arm == AW16 else (6 if arm == PACK else 3)
    s = s.replace(f"__launch_bounds__({threads})", f"__launch_bounds__({threads},{blocks})")
    packed = arm == PACK
    ty = "__half2" if packed else "float"
    s = s.replace(
        "float* acc, I stride, float (&dp)", f"{ty}* dwi, float* acc, I stride, float (&dp)"
    )
    s = s.replace("path, acc, acc_stride, dp);", "path, dwi, acc, acc_stride, dp);")
    s = s.replace(
        "T* gs, float* acc, I acc_stride,", "T* gs, float* sums, float* acc, I acc_stride,"
    )
    s = s.replace(
        "acc, acc_stride, (seg + threadIdx.x)", "sums, acc, acc_stride, (seg + threadIdx.x)"
    )
    s = s.replace(
        "float local_acc[register_acc ? 11 * HIDDEN + 2 : 1]{};",
        """
    float local_acc[26]{};
    float* sums = acc_base + threadIdx.x;
    #pragma unroll
    for (int k=0;k<64;++k) sums[k*threads]=0;
""",
    )
    s = s.replace(
        "const auto ar = a.conj(), gg = g.conj();\n    // CHUNK",
        (
            "__half2 dwi[32];\n    #pragma unroll\n    for(int k=0;k<32;++k) dwi[k]=__float2half2_rn(0);"
            if packed
            else "float* dwi=sums;"
        )
        + "\n    const auto ar = a.conj(), gg = g.conj();\n    // CHUNK",
    )
    for old, new in (
        ("88 * stride", "24 * stride"),
        ("89 * stride", "25 * stride"),
        ("(64 + j) * stride", "j * stride"),
        ("(72 + j) * stride", "(8+j)*stride"),
        ("(80 + j) * stride", "(16+j)*stride"),
    ):
        s = s.replace("acc[" + old + "]", "acc[" + new + "]")
    update = """
        #pragma unroll
        for(int k=0;k<8;++k)
            dwi[(8*j+k)*THREAD_COUNT] += __fmaf_rn(dzs[0],p[0][k],dzs[1]*p[1][k]);
"""
    if packed:
        update = """
        #pragma unroll
        for(int k=0;k<8;k+=2){
            auto v=dwi[4*j+k/2];
            #pragma unroll
            for(int i=0;i<2;++i)
                v=__hfma2(__float2half2_rn(dzs[i]*256.f),__floats2half2_rn(p[i][k],p[i][k+1]),v);
            dwi[4*j+k/2]=v;
        }
"""
        pos = s.index(
            "\n}",
            s.index("if constexpr (COARSE) clear_unused_columns", s.index("void column_backward(")),
        )
        s = (
            s[:pos]
            + """
    #pragma unroll
    for(int k=0;k<32;++k){
        float2 v=__half22float2(dwi[k]);
        sums[2*k*THREADS]+=v.x*(1.f/256.f);
        sums[(2*k+1)*THREADS]+=v.y*(1.f/256.f);
    }
"""
            + s[pos:]
        )
    if packed:
        # Stage the FP32 tail beside dwi to free registers for the packed column.
        s = s.replace("__half2* dwi, float* acc", "__half2* dwi, float* bias_sum, float* acc")
        s = s.replace("path, dwi, acc", "path, dwi, sums+64*THREADS, acc")
        s = s.replace("acc[j * stride] += dz", f"bias_sum[j*{threads}] += dz")
        s = s.replace("k<64;++k) sums", "k<90;++k) sums")
        for field in ("(8+j)", "(16+j)", "24 ", "25 "):
            s = s.replace(f"acc[{field}*stride]", f"bias_sum[{field}*{threads}]")
        for field in (24, 25):
            s = s.replace(f"acc[{field} * stride]", f"bias_sum[{field}*{threads}]")
    s = s.replace("acc[(8 * j + k) * stride] += dz * p[cell][k];", "")
    s = s.replace(
        "auto dz2=P::pack", update.replace("THREAD_COUNT", str(threads)) + "auto dz2=P::pack"
    )
    # Reduce dwi directly before reusing the allocation as CTA staging.
    old = """write_vscan_partial<WARPS, 11 * HIDDEN + 6, I>(
                reinterpret_cast<float*>(shared), row, local_acc, da, dg, m, M);"""
    reduction = """
            __syncthreads();
            if(threadIdx.x<32 && m<M){
                for(int j=0;j<64;++j){
                    float sum=sums[j*threads];
                    #pragma unroll
                    for(int k=1;k<WARPS;++k) sum+=sums[j*threads+k*32];
                    row[I(4+j)*M]=sum;
                }
            }
            __syncthreads();
            write_vscan_partial<WARPS,30,I>(reinterpret_cast<float*>(shared),
                row,local_acc,da,dg,m,M);
"""
    if packed:
        reduction = reduction.replace("j<64", "j<90").replace("WARPS,30", "WARPS,4")
    s = fp.replace(s, old, reduction)
    return s.replace(
        "row[I(begin + j) * M]", "row[I(begin+j + (begin+j>=4 ? (FIELDS==30 ? 64 : 0) : 0))*M]"
    )


@lru_cache(maxsize=128)
def _function(device, key, h, w, m, arm, smem):
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "scan.cu"
        cu.write_text(kernel_source(key, h, w, m, arm))
        cubin = cu.with_suffix(".cubin")
        subprocess.run(
            ["nvcc", "-std=c++17", "-arch=sm_90a", "-cubin", str(cu), "-o", str(cubin)], check=True
        )
        image = ct.create_string_buffer(cubin.read_bytes())
    driver = scans._load_cuda_driver()
    scans._ensure_context(driver, device)
    mod, fn = ct.c_void_p(), ct.c_void_p()
    scans._check_cu(driver.cuModuleLoadData(ct.byref(mod), image), "load w69")
    scans._check_cu(driver.cuModuleGetFunction(ct.byref(fn), mod, arm.encode()), "resolve w69")
    scans._configure_smem(driver, fn, smem)
    return driver, mod, fn


def backward(pole, r, states, aux, path, coarse, grad, arm):
    fp._check_source(r, fp.ARM)
    b, h, w, m = r.shape
    if (h, w, m) not in ((56, 56, 160), (28, 28, 160)):
        raise ValueError("unsupported Wave 69 shape")
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
    scratch = r.new_empty((h + oh, launch.workers, m, 4))
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
        grid = scans._ceil(m, 32) * launch.partials
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
