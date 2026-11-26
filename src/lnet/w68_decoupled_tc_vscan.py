# ruff: noqa: ANN001, ANN201, ANN202, E501, S603, S607, SLF001
"""Standalone SM90a dense vertical scan. Saved BF16 features approximate prior FP32 states."""

import ctypes as ct
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import torch

from . import general_cuda_scans as scans

ARM = "w68_decoupled_bf16"
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
    const float *gr,const float *gi,const float *vx,const float *vy,unsigned *p,
    const unsigned *dp,unsigned *ds,float *pg,int B,int W,int M){
    const int tid=threadIdx.x,lane=tid%32, mt=blockIdx.y, m=32*mt+lane;
    extern __shared__ unsigned buf[];
    float2 da={},dg={};
    for(int col=blockIdx.x*4+tid/32;col<B*W;col+=gridDim.x*4){
        const int b=col/WIDTH,x=col%WIDTH;
        for(int plane=0;plane<2;++plane){
            const I si0=(((I(b)*(M/32)+mt)*2+plane)*W+x)*HEIGHT*32+lane;
            const unsigned first=s[si0];
            for(int y=1;y<HEIGHT;++y) buf[(y-1)*128+tid]=s[si0+y*32];
            const float xx=vx[(plane*W+x)*M+m];
            for(int rev=0;rev<2;++rev){
                const float2 a={ar[m],rev?-ai[m]:ai[m]}, g={gr[m],rev?-gi[m]:gi[m]};
                float2 state={};
                for(int i=0;i<HEIGHT;++i){
                    const int y=(rev^BACK)?HEIGHT-1-i:i;
                    const I pi=(((I(mt)*B*W+col)*HEIGHT+y)*4+2*rev+plane)*32+lane;
                    float2 src=unpack(y?buf[(y-1)*128+tid]:first);
                    const float gain=rsqrtf(fmaxf(xx*vy[(rev*HEIGHT+y)*M+m],1.e-6f));
#if BACK
                    float2 d=unpack(dp[pi]);
                    state=add(mul(conjv(a),state),{rnd(d.x*gain),rnd(d.y*gain)});
                    const int prev=y+(rev?1:-1);
                    if(prev>=0 && prev<HEIGHT){
                        float v=fmaxf(xx*vy[(rev*HEIGHT+prev)*M+m],1.e-6f);
                        float inv=v*rsqrtf(v);
                        float2 saved=unpack(p[pi+(rev?128:-128)]);
                        float2 term=mul(state,{saved.x*inv,-saved.y*inv});
                        da=add(da,rev?conjv(term):term);
                    }
                    float2 term=mul(state,conjv(src));
                    dg=add(dg,rev?conjv(term):term);
                    float2 grad=mul(conjv(g),state);
                    unsigned &pos=buf[(HEIGHT-1+y)*128+tid];
                    if(rev){
                        grad=add(unpack(pos),{rnd(grad.x),rnd(grad.y)});
                        ds[si0+y*32]=pack(grad.x,grad.y);
                    } else pos=pack(grad.x,grad.y);
#else
                    state=add(mul(a,state),mul(g,src));
                    p[pi]=pack(state.x*gain,state.y*gain);
#endif
                }
            }
        }
    }
#if BACK
    float *part=pg+(I(blockIdx.x*4+tid/32)*4)*M+m;
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
    I col=n/HEIGHT;return ((col/WIDTH*HEIGHT+n%HEIGHT)*WIDTH+col%WIDTH)*M+m;
}
extern "C" __global__ __launch_bounds__(1024) void NAME(
    const unsigned *p,const bf *wi,const bf *bi,const bf *wo,const bf *bo,
    const bf *dy,bf *out,unsigned *dp,float *part,int B,int W,int M){
    const int tid=threadIdx.x,mode=tid/32,lane=tid%32,g=lane/4,t=lane%4;
    const int mt=blockIdx.y,m=32*mt+mode;
    const I N=I(B)*W*HEIGHT;
    __shared__ unsigned tile[32][65];
    __shared__ bf io_r[16][33],io_i[16][33];
#if BACK
    __shared__ unsigned deriv[32][65];
#endif
    unsigned wt=pack(wi[m*64+8*g+2*t],wi[m*64+8*g+2*t+1]);
    unsigned wback=pack(wi[m*64+16*t+g],wi[m*64+16*t+8+g]);
    float bias[2],wr[2],wj[2];
    for(int j=0;j<2;j++){
        int k=2*t+j;bias[j]=bi[m*8+k];wr[j]=wo[m*16+k];wj[j]=wo[m*16+8+k];
    }
    float dw[4]={},db[2]={},dr[2]={},di[2]={},dbr=0,dbi=0;
    for(I base=I(blockIdx.x)*16;base<N;base+=I(gridDim.x)*16){
        for(int e=tid;e<2048;e+=1024){
            tile[lane][e/32]=base+e/128<N?p[(I(mt)*N+base)*128+e]:0;
        }
#if BACK
        if(mode<16){
            I ix=outidx(base+mode,32*mt+lane,W,M);
            io_r[mode][lane]=base+mode<N?dy[ix]:bf(0.f);
            io_i[mode][lane]=base+mode<N?dy[N*M+ix]:bf(0.f);
        }
#endif
        __syncthreads();
        unsigned *row=tile[mode];
        auto channel=[&](int r,int k){return (row[4*r+(k%4)]>>(16*(k/4)))&65535u;};
        auto pair=[&](int r,int k){return channel(r,k)|(channel(r,k+1)<<16);};
        float z[4]={};
        mma8(z,pair(g,2*t),pair(g+8,2*t),wt);
        for(int r=0;r<2;r++){
            unsigned dzpair=0;
            float yr=0,yi=0;
#if BACK
            yr=io_r[g+8*r][mode];yi=io_i[g+8*r][mode];
            dbr+=yr;dbi+=yi;
#endif
            float sr=0,si=0;
            for(int j=0;j<2;j++){
                float pre=z[2*r+j]+bias[j], sig=1.f/(1.f+__expf(-pre)), act=pre*sig;
#if BACK
                float dz=rnd((yr*wr[j]+yi*wj[j])*sig*(1.f+pre*(1.f-sig)));
                dzpair|=unsigned(__bfloat16_as_ushort(bf(dz)))<<(16*j);
                db[j]+=dz;dr[j]+=yr*act;di[j]+=yi*act;
#else
                sr+=act*wr[j];si+=act*wj[j];
#endif
            }
#if !BACK
            for(int d=2;d;d/=2){
                sr+=__shfl_xor_sync(~0u,sr,d);si+=__shfl_xor_sync(~0u,si,d);
            }
            if(t==0){io_r[g+8*r][mode]=bf(sr+float(bo[2*m]));io_i[g+8*r][mode]=bf(si+float(bo[2*m+1]));}
#else
            deriv[mode][4*(g+8*r)+t]=dzpair;
#endif
        }
#if BACK
        __syncwarp();
        unsigned *dz=deriv[mode];
        auto dchannel=[&](int r){return (dz[4*r+g/2]>>(16*(g%2)))&65535u;};
        unsigned a0=dchannel(2*t)|(dchannel(2*t+1)<<16);
        unsigned a2=dchannel(8+2*t)|(dchannel(9+2*t)<<16);
        unsigned b0=channel(2*t,g)|(channel(2*t+1,g)<<16);
        unsigned b1=channel(8+2*t,g)|(channel(9+2*t,g)<<16);
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%5},{%7,%8},{%0,%1,%2,%3};"
            :"+f"(dw[0]),"+f"(dw[1]),"+f"(dw[2]),"+f"(dw[3])
            :"r"(a0),"r"(0u),"r"(a2),"r"(b0),"r"(b1));
        float grad[4]={};
        mma8(grad,dz[4*g+t],dz[4*(g+8)+t],wback);
        __syncwarp();
        for(int j=0;j<4;j++){
            float imag=__shfl_xor_sync(~0u,grad[j],2);
            if(t<2) row[(g+8*(j/2))*4+2*t+j%2]=pack(grad[j],imag);
        }
        __syncthreads();
        for(int e=tid;e<2048;e+=1024)
            if(base+e/128<N)dp[(I(mt)*N+base)*128+e]=tile[lane][e/32];
#else
        __syncthreads();
        if(mode<16 && base+mode<N){
            I ix=outidx(base+mode,32*mt+lane,W,M);
            out[ix]=io_r[mode][lane];out[N*M+ix]=io_i[mode][lane];
        }
#endif
        __syncthreads();
    }
#if BACK
    // Five CTAs share each partial group at the full grid size.
    float *dst=part+(I(blockIdx.x%32)*M+m)*90;
    for(int j=0;j<2;j++){
        atomicAdd(dst+(g*8+2*t+j),dw[j]);
        for(int d=16;d>=4;d/=2){
            db[j]+=__shfl_xor_sync(~0u,db[j],d);
            dr[j]+=__shfl_xor_sync(~0u,dr[j],d);
            di[j]+=__shfl_xor_sync(~0u,di[j],d);
        }
        if(!g){atomicAdd(dst+(64+2*t+j),db[j]);atomicAdd(dst+(72+2*t+j),dr[j]);atomicAdd(dst+(80+2*t+j),di[j]);}
    }
    for(int d=16;d>=4;d/=2){dbr+=__shfl_xor_sync(~0u,dbr,d);dbi+=__shfl_xor_sync(~0u,dbi,d);}
    if(!lane){atomicAdd(dst+(88),dbr);atomicAdd(dst+(89),dbi);}
#endif
}
"""


def source(name, height, width=None):
    backward = name in ("VB_pure", "PathMLP_bwd")
    return (f"#define WIDTH {width or height}\n#define HEIGHT {height}\n#define BACK {int(backward)}\n"
            + HEADER + (SCAN if name.endswith("pure") else MLP).replace("NAME", name))


def compile_kernel(name, height, folder, width=None):
    path = folder / (name + ".cu")
    path.write_text(source(name, height, width))
    cubin = path.with_suffix(".cubin")
    subprocess.run(["nvcc", "-arch=sm_90a", "-std=c++17", "-cubin", str(path), "-o", str(cubin)], check=True)
    return cubin


@lru_cache(maxsize=64)
def _function(name, height, width, device):
    with tempfile.TemporaryDirectory() as tmp:
        image = ct.create_string_buffer(compile_kernel(name, height, Path(tmp), width).read_bytes())
    driver = scans._load_cuda_driver()
    scans._ensure_context(driver, device)
    module, fn = ct.c_void_p(), ct.c_void_p()
    scans._check_cu(driver.cuModuleLoadData(ct.byref(module), image), "load Wave 68")
    scans._check_cu(driver.cuModuleGetFunction(ct.byref(fn), module, name.encode()), name)
    if name == "VB_pure" and (2*height-1)*512 > 49152:
        scans._check_cu(driver.cuFuncSetAttribute(fn, 8, (2*height-1)*512), "dynamic shared memory")
    return driver, module, fn


def launch(name, shape, grid, tensors):
    b, h, w, m = shape
    device = tensors[0].device
    with torch.cuda.device(device):
        driver, _, fn = _function(name, h, w, device.index)
        args = [*(scans._ptr(x) for x in tensors), *map(ct.c_int, (b, w, m))]
        argv = (ct.c_void_p * len(args))(*(ct.addressof(x) for x in args))
        threads = 128 if name.endswith("pure") else 1024
        shared = (h * (2 if name == "VB_pure" else 1) - 1) * 512 if name.endswith("pure") else 0
        stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        scans._check_cu(driver.cuLaunchKernel(fn, grid, m//32, 1, threads, 1, 1, shared, stream, argv, None), name)


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
        self.tiles = mt, b, w, h, 4, 32
        self.cols = min(132, (b*w+3)//4)
        self.stripes = min(160, (b*w*h+15)//16)

    def vf(self):
        p = self.path[0].new_empty(self.tiles, dtype=torch.uint32)
        launch("VF_pure", self.shape, (self.shape[0]*self.shape[2]+3)//4, [*self.scan, p, None, None, None])
        return p

    def forward(self, p):
        out = self.path[0].new_empty((2, *self.shape))
        launch("PathMLP_fwd", self.shape, self.stripes, [p, *self.path, None, out, None, None])
        return out

    def backward(self, p, dy):
        if dy.shape != (2, *self.shape) or dy.device != p.device or dy.dtype != torch.bfloat16:
            raise ValueError("expected planar BF16 adjoints")
        m = self.shape[-1]
        stripes = self.stripes
        part = p.new_zeros((min(stripes, 32), m, 90), dtype=torch.float32)
        dp = torch.empty_like(p)
        launch("PathMLP_bwd", self.shape, stripes, [p, *self.path, dy.contiguous(), None, dp, part])
        params = [v.reshape(w.shape).to(w.dtype) for v, w in zip(part.sum(0).split((64, 8, 16, 2), -1), self.path, strict=True)]
        return dp, params

    def vb(self, p, dp):
        m = self.shape[-1]
        part = p.new_empty((self.cols*4, 4, m), dtype=torch.float32)
        ds = torch.empty_like(self.states)
        launch("VB_pure", self.shape, self.cols, [*self.scan, p, dp, ds, part])
        return ds, part.sum(0)
