"""Opt-in SM90a vertical forward screening kernels. No production dispatch changes."""

import ctypes as ct
import math
import subprocess
import tempfile
from functools import cache
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor

from . import general_cuda_scans as scans

MODES = ("v58_fwd_smem_src", "v58_fwd_smem_src_2w", "v58_fwd_smem_tc", "v58_fwd_smem_b8")


class Config(NamedTuple):
    threads: int
    grid: int
    smem: int


def config(mode, batch, h, w, m, coarse):
    if mode not in MODES or min(batch, h, w, m) < 1 or (coarse and min(h, w) < 2):
        raise ValueError("Invalid scan configuration")
    tc = mode == MODES[2]
    tm = 16 if tc else 32
    threads = 256 if tc else 64 if mode == MODES[1] else 128
    step = 1 + coarse
    grid = scans._ceil(batch * (w // step), threads // tm) * scans._ceil(m, tm)
    smem = threads * (h + h // step) * 8
    if smem > 227 * 1024:
        raise ValueError("Tile exceeds Hopper block shared memory")
    return Config(threads, grid, smem)


# Each MMA warp handles two modes. Reuse consumed source and positive rows
# for the K=8 input halves and then for the two FP32 output channels.
_TC = r"""
    const int lane = tid % 32, group = lane / 4, pair = 2 * (lane % 4);
    unsigned weights[2];
    float bias[2][2], wr[2][2], wj[2][2];
    #pragma unroll
    for (int s=0; s<2; ++s) {
        int mode = mode0 + 2*(tid/32) + s;
        PathWeights<T,int,false> pw{wi,bi,wo,types,min(mode,M-1),8};
        weights[s] = bits(pw.input_pair(4*group + pair/2));
        #pragma unroll
        for (int i=0;i<2;++i) {
            bias[s][i]=pw.bias(pair+i);
            wr[s][i]=pw.output(pair+i);
            wj[s][i]=pw.output(8+pair+i);
        }
    }
    for (int y=H-1; y>=0; --y) {
        state=advance(state,src[y*N+tid].unpack(),a.conj(),g.conj());
        if (STEP==2 && (y%2 || y/2>=OH)) continue;
        int yo=y/STEP;
        auto p=pos[yo*N+tid].unpack(), n=norm(state,c.gain(y,true)).unpack();
        auto* re=pos+yo*N;
        auto* im=src+y*N;
        re[tid]=Quad<T>::pack(make_float4(p.x,p.z,n.x,n.z));
        im[tid]=Quad<T>::pack(make_float4(p.y,p.w,n.y,n.w));
        __syncthreads();
        float result[2][2][2];
        #pragma unroll
        for(int s=0;s<2;++s) {
            int col=2*(tid/32)+s;
            unsigned aa[2];
            #pragma unroll
            for(int half=0;half<2;++half) {
                const auto q=(pair<4?re:im)[(group+8*half)*TM+col];
                aa[half]=bits((pair%4==0?q.a:q.b).unpack());
            }
            float d[4];
            asm("mma.sync.aligned.m16n8k16.row.col.f32.FMT.FMT.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%6},{%7,%6},{%8,%8,%8,%8};"
                :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
                :"r"(aa[0]),"r"(aa[1]),"r"(0u),"r"(weights[s]),"f"(0.f));
            #pragma unroll
            for(int half=0;half<2;++half) {
                float r=0,i=0;
                #pragma unroll
                for(int k=0;k<2;++k) {
                    float z=d[2*half+k]+bias[s][k];
                    float act=z*__fdividef(1.f,1.f+__expf(-z));
                    r+=act*wr[s][k]; i+=act*wj[s][k];
                }
                r+=__shfl_xor_sync(0xffffffff,r,1);
                i+=__shfl_xor_sync(0xffffffff,i,1);
                result[s][half][0]=r+__shfl_xor_sync(0xffffffff,r,2);
                result[s][half][1]=i+__shfl_xor_sync(0xffffffff,i,2);
            }
        }
        __syncthreads();
        auto* out=reinterpret_cast<float2*>(re);
        if(pair==0) {
            #pragma unroll
            for(int s=0;s<2;++s)
                for(int half=0;half<2;++half)
                    out[(group+8*half)*TM+2*(tid/32)+s]=
                        make_float2(result[s][half][0],result[s][half][1]);
        }
        __syncthreads();
        float2 v=out[tid];
        if(valid) store(yo,v);
    }
"""

_SCALAR = r"""
    if(!valid) return;
    PathWeights<T,int,false> path{wi,bi,wo,types,m,8};
    for(int top=OH-1;top>=0;top-=BW) {
        float p[BW][8];
        #pragma unroll
        for(int t=0;t<BW;++t) {
            int yo=top-t;
            float4 n{},v{};
            if(yo>=0) {
                int y=yo*STEP;
                if(STEP==2) state=advance(state,src[(y+1)*N+tid].unpack(),a.conj(),g.conj());
                state=advance(state,src[y*N+tid].unpack(),a.conj(),g.conj());
                n=norm(state,c.gain(y,true)).unpack();
                v=pos[yo*N+tid].unpack();
            }
            p[t][0]=v.x;p[t][1]=v.z;p[t][2]=n.x;p[t][3]=n.z;
            p[t][4]=v.y;p[t][5]=v.w;p[t][6]=n.y;p[t][7]=n.w;
        }
        float2 out[BW]{};
        #pragma unroll
        for(int j=0;j<8;++j) {
            auto bj=path.bias(j),wr=path.output(j),wj=path.output(8+j);
            #pragma unroll
            for(int t=0;t<BW;t+=2) {
                auto dot=Pair<T>::pack({0,0}).value;
                #pragma unroll
                for(int k=0;k<8;++k) {
                    float w=path.input(8*j+k);
                    dot=__hfma2(Pair<T>::pack({w,w}).value,
                                Pair<T>::pack({p[t][k],p[t+1][k]}).value,dot);
                }
                Complex z=Pair<T>{dot}.unpack();
                #pragma unroll
                for(int q=0;q<2;++q) {
                    float v=(q?z.i:z.r)+bj;
                    float act=v*__fdividef(1.f,1.f+__expf(-v));
                    out[t+q].x+=wr*act; out[t+q].y+=wj*act;
                }
            }
        }
        #pragma unroll
        for(int t=0;t<BW;++t) if(top>=t) store(top-t,out[t]);
    }
"""

_BODY = r"""
__device__ Quad<T> norm(float4 v,float2 gain) {
    return Quad<T>::pack(make_float4(v.x*gain.x,v.y*gain.x,v.z*gain.y,v.w*gain.y));
}
__device__ unsigned bits(Complex z) {
    auto p=Pair<T>::pack(z).value;
    return *reinterpret_cast<const unsigned*>(&p);
}
extern "C" __global__ void scan(const float* dr,const float* di,const float* gr,const float* gi,
    const T* states,const float* vx,const float* vy,const void* wi,const void* bi,
    const void* wo,const void* bo,T* outr,T* outi,int B,int types,int inter,float eps) {
    extern __shared__ float4 shared[];
    auto* src=reinterpret_cast<Quad<T>*>(shared);
    auto* pos=src+H*N;
    int tid=threadIdx.x,tiles=(M+TM-1)/TM;
    int mode0=blockIdx.x%tiles*TM, raw=mode0+tid%TM;
    int line=blockIdx.x/tiles*(N/TM)+tid/TM;
    bool valid=raw<M && line<B*OW;
    int m=min(raw,M-1);
    line=min(line,B*OW-1);
    int x=line%OW,b=line/OW,xa=x*STEP+STEP-1,xb=x*STEP;
    long long plane=1LL*B*H*W*M,base=1LL*b*H*W*M+m;
    Column<T,long long> c{states,vy,base+xa*M,base+xb*M,plane,plane,
        (1LL*b*OH*OW+x)*M+m,1LL*B*OH*OW*M,H,W,M,OW,m,vx[xa*M+m],vx[(W+xb)*M+m],eps};
    Complex a{dr[m],di[m]},g{gr[m],gi[m]};
    float br=rounded<T>(load_scalar(bo,2*m,types>>6));
    float bj=rounded<T>(load_scalar(bo,2*m+1,types>>6));
    auto store=[&](int yo,float2 v) {
        auto cell=(c.cell(yo)-m)*(1+inter)+m;
        outr[cell]=write<T>(v.x+br); outi[cell]=write<T>(v.y+bj);
    };
    float4 state{};
    for(int y=0;y<H;++y) {
        auto q=c.packed_source(y);
        src[y*N+tid]=q;
        state=advance(state,q.unpack(),a,g);
        if(y%STEP==STEP-1) pos[(y/STEP)*N+tid]=norm(state,c.gain(y,false));
    }
    state={};
"""


@cache
def source(mode, h, w, m, coarse, dtype):
    cfg = config(mode, 1, h, w, m, coarse)
    typ, fmt = {torch.float16: ("__half", "f16"), torch.bfloat16: ("__nv_bfloat16", "bf16")}[dtype]
    tc = mode == MODES[2]
    prefix = scans._CUDA_SOURCE.split("template<class T>\n__device__ Quad<T> normalize_packed")[0]
    step = 1 + coarse
    constants = f"""
using T={typ};
constexpr int H={h},W={w},M={m},STEP={step},OH=H/STEP,OW=W/STEP;
constexpr int N={cfg.threads},TM={16 if tc else 32},BW={8 if mode == MODES[3] else 4};
"""
    body = _TC.replace("FMT", fmt) if tc else _SCALAR
    if not tc and coarse and h % 2:
        body = "state=advance(state,src[(H-1)*N+tid].unpack(),a.conj(),g.conj());\n" + body
    return prefix + constants + _BODY + body + "}\n"


def compile_cubin(code, directory):
    cu = Path(directory) / "scan.cu"
    cubin = cu.with_suffix(".cubin")
    cu.write_text(code)
    proc = subprocess.run(
        [
            "nvcc",
            "-arch=sm_90a",
            "-std=c++17",
            "--cubin",
            "--ptxas-options=-v",
            str(cu),
            "-o",
            str(cubin),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stdout)
    return cubin, proc.stdout


@cache
def function(device, mode, h, w, m, coarse, dtype):
    driver = scans._load_cuda_driver()
    module, fn = ct.c_void_p(), ct.c_void_p()
    with torch.cuda.device(device), tempfile.TemporaryDirectory() as directory:
        scans._ensure_context(driver, device)
        cubin, _ = compile_cubin(source(mode, h, w, m, coarse, dtype), directory)
        scans._check_cu(driver.cuModuleLoadData(ct.byref(module), cubin.read_bytes()), "load cubin")
        scans._check_cu(driver.cuModuleGetFunction(ct.byref(fn), module, b"scan"), "load scan")
        scans._configure_smem(driver, fn, config(mode, 1, h, w, m, coarse).smem)
    return driver, fn


@torch.library.custom_op("lnet::wave58b_packed", mutates_args=())
def _packed(
    pole: list[Tensor],
    states: Tensor,
    aux: list[Tensor],
    path: list[Tensor],
    coarse: bool,
    epsilon: float,
    mode: str,
    planar: bool,
) -> Tensor:
    if (
        not states.is_cuda
        or not scans._pole_ok(pole, states)
        or not scans._vertical_inputs_ok(states, aux, path, coarse=coarse)
        or path[0].shape[1] != 8
        or not 0 < epsilon < math.inf
    ):
        raise ValueError("Invalid Wave 58B forward inputs")
    b, h, w, m = states.shape[1:]
    cfg = config(mode, b, h, w, m, coarse)
    output = _fake(pole, states, aux, path, coarse, epsilon, mode, planar)
    real, imag = output.unbind(0) if planar else output.split(m, dim=-1)
    inputs = [
        *(p.float().contiguous() for p in pole),
        states.contiguous(),
        *(v.float().contiguous() for v in aux),
        *(p.contiguous() for p in path),
    ]
    with torch.cuda.device(states.device):
        driver, fn = function(states.device.index, mode, h, w, m, coarse, states.dtype)
        args = [
            *(scans._ptr(p) for p in [*inputs, real, imag]),
            ct.c_int(b),
            ct.c_int(scans._scalar_types(path)),
            ct.c_int(not planar),
            ct.c_float(epsilon),
        ]
        argv = (ct.c_void_p * len(args))(*(ct.addressof(a) for a in args))
        scans._check_cu(
            driver.cuLaunchKernel(
                fn,
                cfg.grid,
                1,
                1,
                cfg.threads,
                1,
                1,
                cfg.smem,
                ct.c_void_p(torch.cuda.current_stream().cuda_stream),
                argv,
                None,
            ),
            "scan",
        )
    return output


@_packed.register_fake
def _fake(pole, states, aux, path, coarse, epsilon, mode, planar):
    b, h, w, m = states.shape[1:]
    shape = (b, h // (1 + coarse), w // (1 + coarse))
    return states.new_empty((2, *shape, m) if planar else (*shape, 2 * m))


def vscan_fwd(
    pole,
    states,
    auxiliary,
    path,
    coarse,
    epsilon=1e-6,
    precise=False,
    *,
    mode=MODES[0],
    packed=False,
    planar=False,
):
    if precise:
        raise ValueError("Wave 58B supports precise=False")
    out = _packed(pole, states, auxiliary, path, coarse, epsilon, mode, planar or not packed)
    return out if packed else out.unbind(0)
