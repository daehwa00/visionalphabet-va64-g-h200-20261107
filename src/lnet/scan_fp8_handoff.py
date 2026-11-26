"""Standalone Hopper scan handoffs. State planes are positive real/imag then negative real/imag."""

from functools import cache
import ctypes as ct

import torch

from . import cuda_nvrtc as nv
from .general_cuda_scans import _CUDA_SOURCE


CUDA = r"""
using T = __nv_bfloat16;
using I = long long;
#if __CUDA_ARCH__ >= 890
template<bool GRAD> __device__ unsigned short encode(float x, float y) {
    unsigned short r;
    if constexpr (GRAD) asm("cvt.rn.satfinite.e5m2x2.f32 %0,%2,%1;" : "=h"(r) : "f"(x),"f"(y));
    else asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;" : "=h"(r) : "f"(x),"f"(y));
    return r;
}
template<bool GRAD> __device__ Complex decode(unsigned short v, float scale) {
    unsigned int r;
    if constexpr (GRAD) asm("cvt.rn.f16x2.e5m2x2 %0,%1;" : "=r"(r) : "h"(v));
    else asm("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(r) : "h"(v));
    return {__half2float(__ushort_as_half(r & 65535))*scale,
            __half2float(__ushort_as_half(r >> 16))*scale};
}
__device__ float warpmax(float v) {
    #pragma unroll
    for(int d=16;d;d/=2) v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,d));
    return v;
}
__device__ float mag(Complex v) { return fmaxf(fabsf(v.r),fabsf(v.i)); }
__device__ void put(unsigned char* q,I i,I p, unsigned short v) {
    q[i]=v; q[i+p]=v>>8;
}
__device__ unsigned short get(const unsigned char* q,I i,I p) {
    return q[i] | (static_cast<unsigned short>(q[i+p]) << 8);
}
extern "C" __global__ void hf(const float* ar,const float* ai,const float* gr,const float* gi,
 const T* xr,const T* xi,unsigned char* q,float* scales,int B,int M,I bs,I rs,I cs,I ms,I bj,I ri,I ci,I mi) {
    __shared__ float4 row[W][32];
    int lane=threadIdx.x, tile=blockIdx.x, m=32*tile+lane;
    I line=blockIdx.y+I(gridDim.y)*blockIdx.z;
    if(line>=I(B)*H) return;
    Complex a{},g{},pos{},neg{};
    if(m<M) {a={ar[m],ai[m]}; g={gr[m],gi[m]};}
    float mx=0;
    for(int t=0;t<W;t++) {
        if(m<M) {
            I r=(line/H)*bs+(line%H)*rs+m*ms,i=(line/H)*bj+(line%H)*ri+m*mi;
            Complex v{read(xr[r+t*cs]),read(xi[i+t*ci])};
            pos=a*pos+g*v;
            row[t][lane]=make_float4(pos.r,pos.i,v.r,v.i);
            mx=fmaxf(mx,mag(pos));
        }
    }
    for(int u=W-1;u>=0;u--) {
        if(m<M) {
            auto v=row[u][lane];
            neg=a.conj()*neg+g.conj()*Complex{v.z,v.w};
            row[u][lane].z=neg.r; row[u][lane].w=neg.i;
            mx=fmaxf(mx,mag(neg));
        }
    }
    float scale=fmaxf(warpmax(mx),1e-12f)/448.f;
    if(!lane) scales[line*gridDim.x+tile]=scale;
    if(m>=M) return;
    I p=I(B)*H*W*M;
    for(int x=0;x<W;x++) {
        auto z=row[x][lane]; I i=(line*W+x)*M+m;
        put(q,i,p,encode<false>(z.x/scale,z.y/scale));
        put(q,i+2*p,p,encode<false>(z.z/scale,z.w/scale));
    }
}
extern "C" __global__ void vf(const float* ar,const float* ai,const float* gr,const float* gi,
 const void* src,const float* scales,const float* vx,const float* vy,const T* wi,const T* bi,
 const T* wo,const T* bo,T* out,int B,int M,float eps,int planar) {
    constexpr int step=1+COARSE, oh=H/step, ow=W/step;
    __shared__ unsigned int cache[H][32];
    __shared__ float scale[H];
    __shared__ Quad<T> positive[oh][32];
    int lane=threadIdx.x%32, warp=threadIdx.x/32, tile=blockIdx.x,m=tile*32+lane;
    I line=blockIdx.y+I(gridDim.y)*blockIdx.z;
    if(line>=I(B)*ow) return;
    int b=line/ow,x=line%ow,xa=x*step+COARSE,xb=x*step;
    I p=I(B)*H*W*M;
    for(int y=warp;y<H;y+=4) {
        I r=I(b)*H+y, i=(r*W+xa)*M+m,j=(r*W+xb)*M+m+2*p;
        unsigned short a=0,c=0;
        if constexpr (PACKED) {
            auto q=(const unsigned char*)src;
            if(m<M) {a=get(q,i,p);c=get(q,j,p);}
            if(!lane) scale[y]=scales[r*gridDim.x+tile];
        } else {
            auto q=(const T*)src; Complex l{},v{};
            if(m<M) {l=load_complex(q,i,p);v=load_complex(q,j,p);}
            float s=fmaxf(warpmax(fmaxf(mag(l),mag(v))),1e-12f)/448.f;
            if(!lane) scale[y]=s;
            a=encode<false>(l.r/s,l.i/s); c=encode<false>(v.r/s,v.i/s);
        }
        cache[y][lane]=unsigned(a)|(unsigned(c)<<16);
    }
    __syncthreads();
    // Each warp owns eight modes during the two dependent sweeps.
    int k=warp*8+lane;
    if(lane>=8 || tile*32+k>=M) return;
    m=tile*32+k;
    Complex a{ar[m],ai[m]},g{gr[m],gi[m]};
    float4 f{};
    auto source=[&](int y) {
        auto raw=cache[y][k]; auto l=decode<false>(raw,scale[y]),r=decode<false>(raw>>16,scale[y]);
        return make_float4(l.r,l.i,r.r,r.i);
    };
    auto norm=[&](float4 z,int y,bool rev) {
        float v=vy[(rev*H+y)*M+m];
        float l=rsqrtf(fmaxf(v*vx[xa*M+m],eps)),r=rsqrtf(fmaxf(v*vx[(W+xb)*M+m],eps));
        return make_float4(z.x*l,z.y*l,z.z*r,z.w*r);
    };
    for(int y=0;y<H;y++) {
        f=advance(f,source(y),a,g);
        if(y%step==step-1) positive[y/step][k]=Quad<T>::pack(norm(f,y,false));
    }
    f={};
    for(int y=H-1;y>=0;y--) {
        f=advance(f,source(y),a.conj(),g.conj());
        if(y%step || y/step>=oh) continue;
        auto u=positive[y/step][k].unpack(),v=norm(f,y,true);
        float features[]={u.x,u.z,v.x,v.z,u.y,u.w,v.y,v.w};
        float re=0,im=0;
        #pragma unroll
        for(int h=0;h<8;h++) {
            float z=0;
            #pragma unroll
            for(int j=0;j<8;j++) z=fmaf(read(wi[(m*8+h)*8+j]),features[j],z);
            z+=read(bi[m*8+h]);
            float s;
            if constexpr (FAST) {
                asm("tanh.approx.f32 %0,%1;" : "=f"(s) : "f"(z*.5f)); s=.5f*(1+s);
            } else s=1.f/(1.f+__expf(-z));
            re=fmaf(z*s,read(wo[m*16+h]),re);
            im=fmaf(z*s,read(wo[m*16+8+h]),im);
        }
        I i=((I(b)*oh+y/step)*ow+x)*(planar?M:2*M)+m;
        out[i]=write<T>(re+read(bo[2*m]));out[i+(planar?I(B)*oh*ow*M:M)]=write<T>(im+read(bo[2*m+1]));
    }
}
extern "C" __global__ void hb(const float* ar,const float* ai,const float* gr,const float* gi,
 const T* xr,const T* xi,const unsigned char* q,const float* ss,const unsigned char* dq,
 const float* ds,T* dx,float* partial,int B,int M,I bs,I rs,I cs,I ms,I bj,I ri,I ci,I mi) {
    __shared__ Complex buf[W][32];
    int lane=threadIdx.x,tile=blockIdx.x,m=tile*32+lane;
    I line=blockIdx.y+I(gridDim.y)*blockIdx.z;
    if(m>=M || line>=I(B)*H) return;
    constexpr int step=1+COARSE, dw=W/step;
    I p=I(B)*H*W*M,dp=I(B)*H*dw*M;
    float s=ss[line*gridDim.x+tile],d=*ds;
    Complex a{ar[m],ai[m]},g{gr[m],gi[m]},right{},left{},da{},dg{},neg{};
    for(int x=0;x<W;x++) {
        Complex inc{};
        if(x%step==0 && x/step<dw) inc=decode<true>(get(dq,2*dp+(line*dw+x/step)*M+m,dp),d);
        right=a*right+inc;buf[x][lane]=right;
    }
    for(int x=W-1;x>=0;x--) {
        Complex inc{};
        if(x%step==step-1) inc=decode<true>(get(dq,(line*dw+x/step)*M+m,dp),d);
        left=a.conj()*left+inc;right=buf[x][lane];
        I i=(line*W+x)*M+m;
        Complex prev{},next{};
        if(x) prev=decode<false>(get(q,i-M,p),s);
        next=decode<false>(encode<false>(neg.r/s,neg.i/s),s);
        Complex z{read(xr[(line/H)*bs+(line%H)*rs+x*cs+m*ms]),read(xi[(line/H)*bj+(line%H)*ri+x*ci+m*mi])};
        neg=a.conj()*neg+g.conj()*z;
        da.r+=left.r*prev.r+left.i*prev.i+right.r*next.r+right.i*next.i;
        da.i+=left.i*prev.r-left.r*prev.i+right.r*next.i-right.i*next.r;
        Complex sum=left+right,diff{left.r-right.r,left.i-right.i};
        dg.r+=sum.r*z.r+sum.i*z.i; dg.i+=diff.i*z.r-diff.r*z.i;
        store_complex(dx,2*(i-m)+m,I(M),Complex{sum.r*g.r+diff.i*g.i,sum.i*g.r-diff.r*g.i});
    }
    I i=line*4*M+m;
    partial[i]=da.r;partial[i+M]=da.i;partial[i+2*M]=dg.r;partial[i+3*M]=dg.i;
}
extern "C" __global__ void quant(const T* x,unsigned char* q,const float* s,I n) {
    I i=2*(I(blockIdx.x)*blockDim.x+threadIdx.x);
    if(i>=n) return;
    auto bits=encode<true>(read(x[i])/(*s),i+1<n?read(x[i+1])/(*s):0.f);
    q[i]=bits;if(i+1<n) q[i+1]=bits>>8;
}
#endif
"""


@cache
def image(h, w, coarse=False, packed=False, fast=False):
    # Share the production scalar, complex and packed BF16 primitives.
    base = _CUDA_SOURCE[: _CUDA_SOURCE.index("template<class T, class I, bool CACHED>")]
    quad = _CUDA_SOURCE.split("template<class T> struct Quad {", 1)[1].split(
        "template<class T, class I>", 1
    )[0]
    source = base + "template<class T> struct Quad {" + quad + CUDA
    flags = [
        "--gpu-architecture=sm_90a",
        "--std=c++17",
        f"-DH={h}",
        f"-DW={w}",
        f"-DCOARSE={int(coarse)}",
        f"-DPACKED={int(packed)}",
        f"-DFAST={int(fast)}",
        *(f"-I{p}" for p in nv._include_dirs()),
    ]
    lib = nv._nvrtc()
    prog = ct.c_void_p()
    nv._check_nvrtc(
        lib, lib.nvrtcCreateProgram(ct.byref(prog), source.encode(), b"scan.cu", 0, None, None)
    )
    try:
        options = (ct.c_char_p * len(flags))(*(v.encode() for v in flags))
        nv._check_nvrtc(lib, lib.nvrtcCompileProgram(prog, len(flags), options), prog)
        size = ct.c_size_t()
        nv._check_nvrtc(lib, lib.nvrtcGetCUBINSize(prog, ct.byref(size)), prog)
        buf = ct.create_string_buffer(size.value)
        nv._check_nvrtc(lib, lib.nvrtcGetCUBIN(prog, buf), prog)
        return buf.raw
    finally:
        lib.nvrtcDestroyProgram(ct.byref(prog))


@cache
def _module(device, config):
    mod = nv._CudaModule(ct.create_string_buffer(image(*config)), device)
    for name in ("vf", "hf", "hb", "quant"):
        mod.load_function(name, 0)
    return mod


def _launch(name, config, device, grid, threads, tensors, args):
    with torch.cuda.device(device):
        mod = _module(device.index, config)
        mod.launch(
            name,
            grid,
            (threads, 1, 1),
            0,
            torch.cuda.current_stream().cuda_stream,
            [
                *(ct.c_void_p(t.data_ptr()) if t is not None else ct.c_void_p() for t in tensors),
                *args,
            ],
        )


def _grid(lines, m):
    rows = min(lines, 65535)
    return ((m + 31) // 32, rows, (lines + rows - 1) // rows)


def _validate_pole(pole, m, device):
    if len(pole) != 4 or any(p.shape != (m,) or p.device != device for p in pole):
        raise ValueError("expected four pole vectors on the input device")


def _packed(tensor, shape, dtype, device):
    return (
        tensor is not None
        and tensor.shape == shape
        and tensor.dtype == dtype
        and tensor.is_contiguous()
        and tensor.device == device
    )


def _horizontal(pole, real, imag, states=None, scale=None, grad=None, grad_scale=None):
    b, h, w, m = real.shape
    if real.dtype != torch.bfloat16 or imag.shape != real.shape or imag.dtype != real.dtype:
        raise ValueError("expected BF16 real/imag BHWM")
    if not real.is_cuda or imag.device != real.device or min(b, h, w, m) < 1:
        raise ValueError("expected nonempty CUDA tensors on the same device")
    _validate_pole(pole, m, real.device)
    args = [ct.c_int(b), ct.c_int(m), *(ct.c_longlong(s) for x in (real, imag) for s in x.stride())]
    pointers = [*(p.float().contiguous() for p in pole), real, imag]
    grid = _grid(b * h, m)
    if grad is None:
        out = real.new_empty((4, b, h, w, m), dtype=torch.uint8)
        scale = real.new_empty((b, h, grid[0]), dtype=torch.float32)
        _launch("hf", (h, w), real.device, grid, 32, [*pointers, out, scale], args)
        return out, scale
    compact = grad.shape[3] != w
    if not (
        _packed(grad, (4, b, h, w // (1 + compact), m), torch.uint8, real.device)
        and _packed(states, (4, b, h, w, m), torch.uint8, real.device)
        and _packed(scale, (b, h, grid[0]), torch.float32, real.device)
        and _packed(grad_scale, (), torch.float32, real.device)
    ):
        raise ValueError("invalid adjoint shape")
    out = real.new_empty((b, h, w, 2 * m))
    partial = real.new_empty((b, h, 4, m), dtype=torch.float32)
    _launch(
        "hb",
        (h, w, compact),
        real.device,
        grid,
        32,
        [*pointers, states, scale, grad, grad_scale, out, partial],
        args,
    )
    reduced = partial.sum((0, 1))
    return [*reduced.bfloat16().unbind(), out]


def w61_hscan_fwd_fp8_out(pole, real, imag):
    return _horizontal(pole, real, imag)


def w61_hscan_bwd_fp8_in(pole, real, imag, states, scale, grad, grad_scale):
    return _horizontal(pole, real, imag, states, scale, grad, grad_scale)


def quantize_d_states(grad):
    if (
        not grad.is_cuda
        or not grad.numel()
        or grad.dtype != torch.bfloat16
        or not grad.is_contiguous()
    ):
        raise ValueError("expected contiguous BF16 adjoints")
    low, high = torch.aminmax(grad)
    scale = torch.maximum(low.abs(), high.abs()).float().clamp_min(1e-12) / 57344
    out = torch.empty_like(grad, dtype=torch.uint8)
    _launch(
        "quant",
        (1, 1),
        grad.device,
        ((grad.numel() + 511) // 512, 1, 1),
        256,
        [grad, out, scale],
        [ct.c_longlong(grad.numel())],
    )
    return out, scale


def _vertical(pole, states, scale, auxiliary, path, coarse, epsilon, fast, planar=False):
    _, b, h, w, m = states.shape
    device = states.device
    _validate_pole(pole, m, device)
    if (
        not states.is_cuda
        or min(b, h, w, m) < 1
        or not _packed(
            states, (4, b, h, w, m), torch.bfloat16 if scale is None else torch.uint8, device
        )
        or (scale is not None and not _packed(scale, (b, h, (m + 31) // 32), torch.float32, device))
    ):
        raise ValueError("invalid state planes or row scales")
    values = [*auxiliary, *path]
    shapes = [(2, w, m), (2, h, m), (m, 8, 8), (m, 8), (m, 2, 8), (m, 2)]
    if (
        len(auxiliary) != 2
        or len(path) != 4
        or any(t.shape != s or t.device != device for t, s in zip(values, shapes))
    ):
        raise ValueError("invalid gain or path tensor")
    step = 1 + coarse
    if coarse and (h < 2 or w < 2):
        raise ValueError("empty coarse output")
    shape = (2, b, h // step, w // step, m) if planar else (b, h // step, w // step, 2 * m)
    out = states.new_empty(shape, dtype=torch.bfloat16)
    _launch(
        "vf",
        (h, w, coarse, scale is not None, fast),
        states.device,
        _grid(b * (w // step), m),
        128,
        [
            *(p.float().contiguous() for p in pole),
            states,
            scale,
            *(a.float().contiguous() for a in auxiliary),
            *(p.bfloat16().contiguous() for p in path),
            out,
        ],
        [ct.c_int(b), ct.c_int(m), ct.c_float(epsilon), ct.c_int(planar)],
    )
    return out


def w61_vscan_fwd_fp8_smem(
    pole, states, auxiliary, path, coarse=False, epsilon=1e-6, planar=False, *, fast_silu=False
):
    return _vertical(pole, states, None, auxiliary, path, coarse, epsilon, fast_silu, planar)


def w61_vscan_fwd_fp8_smem_fastsilu(pole, states, auxiliary, path, coarse=False, epsilon=1e-6):
    return w61_vscan_fwd_fp8_smem(pole, states, auxiliary, path, coarse, epsilon, fast_silu=True)


def w61_vscan_fwd_fp8_in(pole, states, scale, auxiliary, path, coarse=False, epsilon=1e-6):
    return _vertical(pole, states, scale, auxiliary, path, coarse, epsilon, False)
