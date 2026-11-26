# ruff: noqa: ANN001, ANN201, ANN202, E501, S603, S607, SLF001
"""Standalone SM90a dense vertical scan. Saved BF16 features approximate prior FP32 states."""

import ctypes as ct
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import torch

from . import general_cuda_scans as scans

ARM = "w67_decoupled_bf16"
NAMES = ("VF_pure", "PathMLP_fwd", "PathMLP_bwd", "VB_pure")
HEADER = r"""
#include <cuda_bf16.h>
using bf = __nv_bfloat16;
using I = long long;
__device__ float rnd(float x){return __bfloat162float(__float2bfloat16(x));}
__device__ unsigned pack(float a,float b){
    return __bfloat16_as_ushort(bf(a)) | (unsigned(__bfloat16_as_ushort(bf(b)))<<16);
}
__device__ float2 unpack(unsigned u){
    return {float(__ushort_as_bfloat16(u)),float(__ushort_as_bfloat16(u>>16))};
}
__device__ float2 mul(float2 a,float2 b){return {a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x};}
__device__ float2 add(float2 a,float2 b){return {a.x+b.x,a.y+b.y};}
__device__ float2 conjv(float2 a){return {a.x,-a.y};}
"""
SCAN = r"""
extern "C" __global__ void NAME(const unsigned *s,const float *ar,const float *ai,
    const float *gr,const float *gi,const float *vx,const float *vy,bf *p,
    const bf *dp,unsigned *ds,float *pg,int B,int W,int M){
    const int lane=threadIdx.x%32, mt=blockIdx.y, m=32*mt+lane;
    const int col=blockIdx.x*4+threadIdx.x/32;
    if(col>=B*W)return;
    const int b=col/W,x=col%W;
    float2 da={},dg={};
    for(int rev=0;rev<2;++rev){
        const float2 a={ar[m],rev?-ai[m]:ai[m]}, g={gr[m],rev?-gi[m]:gi[m]};
        #pragma unroll
        for(int plane=0;plane<2;++plane){
            const float xx=vx[(plane*W+x)*M+m];
            float2 state={};
            const int k=2*rev+plane;
            for(int i=0;i<HEIGHT;++i){
                const int y=(rev^BACK)?HEIGHT-1-i:i;
                I si=((((I(b)*(M/32)+mt)*2+plane)*W+x)*HEIGHT+y)*32+lane;
                I pi=(((I(mt)*B*W+col)*HEIGHT+y)*8+k)*32+lane;
                float2 src=unpack(s[si]);
                const float gain=rsqrtf(fmaxf(xx*vy[(rev*HEIGHT+y)*M+m],1.e-6f));
#if BACK
                float2 inj={rnd(float(dp[pi])*gain),rnd(float(dp[pi+128])*gain)};
                state=add(mul(conjv(a),state),inj);
                const int prev=y+(rev?1:-1);
                if(prev>=0 && prev<HEIGHT){
                    const I q=pi+(rev?256:-256);
                    float prev_gain=rsqrtf(fmaxf(xx*vy[(rev*HEIGHT+prev)*M+m],1.e-6f));
                    float2 term=mul(state,{float(p[q])/prev_gain,-float(p[q+128])/prev_gain});
                    da=add(da,rev?conjv(term):term);
                }
                float2 term=mul(state,conjv(src));
                dg=add(dg,rev?conjv(term):term);
                float2 grad=mul(conjv(g),state);
                if(rev) grad=add(unpack(ds[si]),{rnd(grad.x),rnd(grad.y)});
                ds[si]=pack(grad.x,grad.y);
#else
                state=add(mul(a,state),mul(g,src));
                p[pi]=bf(state.x*gain);p[pi+128]=bf(state.y*gain);
#endif
            }
        }
    }
#if BACK
    float *part=pg+(I(col)*4)*M+m;
    part[0]=da.x;part[M]=da.y;part[2*M]=dg.x;part[3*M]=dg.y;
#endif
}
"""
MLP = r"""
__device__ void mma8(float (&d)[4],unsigned a0,unsigned a1,unsigned b){
    asm("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(b));
}
__device__ I outidx(I n,int m,int W,int M){
    I col=n/HEIGHT;return ((col/W*HEIGHT+n%HEIGHT)*W+col%W)*M+m;
}
extern "C" __global__ __launch_bounds__(1024) void NAME(
    const bf *p,const bf *wi,const bf *bi,const bf *wo,const bf *bo,
    const bf *dy,bf *out,bf *dp,float *part,int B,int W,int M){
    const int tid=threadIdx.x,mode=tid/32,lane=tid%32,g=lane/4,t=lane%4;
    const int mt=blockIdx.y,m=32*mt+mode;
    const I N=I(B)*W*HEIGHT;
    // A 129-word pitch makes each coalesced input warp scatter to distinct banks.
    __shared__ float tile[32][129];
#if BACK
    __shared__ float deriv[32][129];
#endif
    unsigned wt=pack(wi[m*64+8*g+2*t],wi[m*64+8*g+2*t+1]);
    unsigned wback=pack(wi[m*64+16*t+g],wi[m*64+16*t+8+g]);
    float bias[2],wr[2],wj[2];
    for(int j=0;j<2;j++){
        int k=2*t+j;bias[j]=bi[m*8+k];wr[j]=wo[m*16+k];wj[j]=wo[m*16+8+k];
    }
    float dw[4]={},db[2]={},dr[2]={},di[2]={},dbr=0,dbi=0;
    const I end=min(N,I(blockIdx.x+1)*STRIPE);
    for(I base=I(blockIdx.x)*STRIPE;base<end;base+=16){
        for(int e=tid;e<4096;e+=1024){
            I n=base+e/256;
            tile[e%32][e/32]=n<N?float(p[(I(mt)*N+base)*256+e]):0;
        }
        __syncthreads();
        float *row=tile[mode];
        float z[4]={};
        mma8(z,pack(row[8*g+2*t],row[8*g+2*t+1]),
               pack(row[8*(g+8)+2*t],row[8*(g+8)+2*t+1]),wt);
        for(int r=0;r<2;r++){
            I n=base+g+8*r;
            I ix=outidx(n,m,W,M);
            float yr=0,yi=0;
#if BACK
            if(n<N){yr=dy[ix];yi=dy[N*M+ix];}
            dbr+=yr;dbi+=yi;
#endif
            float sr=0,si=0;
            for(int j=0;j<2;j++){
                float pre=z[2*r+j]+bias[j], sig=1.f/(1.f+__expf(-pre)), act=pre*sig;
#if BACK
                float dz=rnd((yr*wr[j]+yi*wj[j])*sig*(1.f+pre*(1.f-sig)));
                deriv[mode][(g+8*r)*8+2*t+j]=dz;
                db[j]+=dz;dr[j]+=yr*act;di[j]+=yi*act;
#else
                sr+=act*wr[j];si+=act*wj[j];
#endif
            }
#if !BACK
            for(int d=2;d;d/=2){
                sr+=__shfl_xor_sync(~0u,sr,d);si+=__shfl_xor_sync(~0u,si,d);
            }
            if(t==0 && n<N){out[ix]=bf(sr+float(bo[2*m]));out[N*M+ix]=bf(si+float(bo[2*m+1]));}
#endif
        }
#if BACK
        __syncwarp();
        float *dz=deriv[mode];
        unsigned a0=pack(dz[16*t+g],dz[16*t+8+g]);
        unsigned a2=pack(dz[64+16*t+g],dz[72+16*t+g]);
        unsigned b0=pack(row[16*t+g],row[16*t+8+g]);
        unsigned b1=pack(row[64+16*t+g],row[72+16*t+g]);
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%5},{%7,%8},{%0,%1,%2,%3};"
            :"+f"(dw[0]),"+f"(dw[1]),"+f"(dw[2]),"+f"(dw[3])
            :"r"(a0),"r"(0u),"r"(a2),"r"(b0),"r"(b1));
        float grad[4]={};
        mma8(grad,pack(dz[8*g+2*t],dz[8*g+2*t+1]),pack(dz[64+8*g+2*t],dz[65+8*g+2*t]),wback);
        __syncwarp();
        for(int j=0;j<4;j++)row[(g+8*(j/2))*8+2*t+j%2]=grad[j];
        __syncthreads();
        for(int e=tid;e<4096;e+=1024)
            if(base+e/256<N)dp[(I(mt)*N+base)*256+e]=bf(tile[e%32][e/32]);
#endif
        __syncthreads();
    }
#if BACK
    // One partial stripe per CTA. Only the upper 8x8 accumulator is live.
    float *dst=part+(I(blockIdx.x)*M+m)*90;
    for(int j=0;j<2;j++){
        dst[g*8+2*t+j]=dw[j];
        for(int d=16;d>=4;d/=2){
            db[j]+=__shfl_xor_sync(~0u,db[j],d);
            dr[j]+=__shfl_xor_sync(~0u,dr[j],d);
            di[j]+=__shfl_xor_sync(~0u,di[j],d);
        }
        if(!g){dst[64+2*t+j]=db[j];dst[72+2*t+j]=dr[j];dst[80+2*t+j]=di[j];}
    }
    for(int d=16;d>=4;d/=2){dbr+=__shfl_xor_sync(~0u,dbr,d);dbi+=__shfl_xor_sync(~0u,dbi,d);}
    if(!lane){dst[88]=dbr;dst[89]=dbi;}
#endif
}
"""


def source(name, height):
    backward = name in ("VB_pure", "PathMLP_bwd")
    stripe = 256 if backward else 16
    return (f"#define HEIGHT {height}\n#define BACK {int(backward)}\n#define STRIPE {stripe}\n"
            + HEADER + (SCAN if name.endswith("pure") else MLP).replace("NAME", name))


def compile_kernel(name, height, folder):
    path = folder / (name + ".cu")
    path.write_text(source(name, height))
    cubin = path.with_suffix(".cubin")
    subprocess.run(["nvcc", "-arch=sm_90a", "-std=c++17", "-cubin", str(path), "-o", str(cubin)], check=True)
    return cubin


@lru_cache(maxsize=64)
def _function(name, height, device):
    with tempfile.TemporaryDirectory() as tmp:
        image = ct.create_string_buffer(compile_kernel(name, height, Path(tmp)).read_bytes())
    driver = scans._load_cuda_driver()
    scans._ensure_context(driver, device)
    module, fn = ct.c_void_p(), ct.c_void_p()
    scans._check_cu(driver.cuModuleLoadData(ct.byref(module), image), "load Wave 67")
    scans._check_cu(driver.cuModuleGetFunction(ct.byref(fn), module, name.encode()), name)
    return driver, module, fn


def launch(name, shape, grid, tensors):
    b, h, w, m = shape
    device = tensors[0].device
    with torch.cuda.device(device):
        driver, _, fn = _function(name, h, device.index)
        args = [*(scans._ptr(x) for x in tensors), *map(ct.c_int, (b, w, m))]
        argv = (ct.c_void_p * len(args))(*(ct.addressof(x) for x in args))
        threads = 128 if name.endswith("pure") else 1024
        stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        scans._check_cu(driver.cuLaunchKernel(fn, grid, m//32, 1, threads, 1, 1, 0, stream, argv, None), name)


class Vertical:
    def __init__(self, states, poles, aux, path) -> None:
        b, mt, planes, w, h, lanes = states.shape
        if (planes, lanes) != (2, 32) or states.dtype != torch.uint32 or not states.is_cuda:
            raise ValueError("expected split-plane CUDA uint32 states")
        m = mt*32
        self.shape = b, h, w, m
        self.states = states.contiguous()
        self.scan = [self.states, *(v.float().contiguous() for v in (*poles, *aux))]
        self.path = [v.bfloat16().contiguous() for v in path]
        shapes = [(m, 8, 8), (m, 8), (m, 2, 8), (m, 2)]
        expected = [(m,)] * 4 + [(2, w, m), (2, h, m), *shapes]
        args = [*self.scan[1:], *self.path]
        if len(args) != len(expected) or any(t.shape != s or t.device != states.device for t, s in zip(args, expected, strict=True)):
            raise ValueError("invalid dense hidden-8 parameter shapes or devices")
        self.tiles = mt, b, w, h, 8, 32
        self.cols = (b*w+3)//4
        self.tokens = b*w*h

    def vf(self):
        p = self.path[0].new_empty(self.tiles)
        launch("VF_pure", self.shape, self.cols, [*self.scan, p, None, None, None])
        return p

    def forward(self, p):
        out = p.new_empty((2, *self.shape))
        launch("PathMLP_fwd", self.shape, (self.tokens+15)//16, [p, *self.path, None, out, None, None])
        return out

    def backward(self, p, dy):
        if dy.shape != (2, *self.shape) or dy.device != p.device or dy.dtype != p.dtype:
            raise ValueError("expected planar BF16 adjoints")
        m = self.shape[-1]
        stripes = (self.tokens+255)//256
        part = p.new_empty((stripes, m, 90), dtype=torch.float32)
        dp = torch.empty_like(p)
        launch("PathMLP_bwd", self.shape, stripes, [p, *self.path, dy.contiguous(), None, dp, part])
        params = [v.reshape(w.shape).to(w.dtype) for v, w in zip(part.sum(0).split((64, 8, 16, 2), -1), self.path, strict=True)]
        return dp, params

    def vb(self, p, dp):
        b, _, w, m = self.shape
        part = p.new_empty((b*w, 4, m), dtype=torch.float32)
        ds = torch.empty_like(self.states)
        launch("VB_pure", self.shape, self.cols, [*self.scan, p, dp, ds, part])
        return ds, part.sum(0)
