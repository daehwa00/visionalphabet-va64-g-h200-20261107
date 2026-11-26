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
extern "C" __global__ void w62_hscan_fwd_packed(
    const float* ar, const float* ai, const float* gr, const float* gi,
    const T* xr, const T* xi, unsigned* states, float* scales,
    int M, I bs, I rs, I cs, I ms, I bj, I ri, I ci, I mi) {
    extern __shared__ unsigned sm[];
    auto forward=reinterpret_cast<__nv_bfloat162 (*)[W][32]>(sm);
    auto row=forward+4;
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    int tile=blockIdx.x, m=32*tile+lane, y=blockIdx.y*4+warp, b=blockIdx.z;
    if(y>=H) return;
    I group=I(b)*gridDim.x+tile;
    Complex a{},g{},pos{},neg{};
    if(m<M) {a={ar[m],ai[m]}; g={gr[m],gi[m]};}
    auto source=[&](int x) -> Complex {
        if(m>=M) return {};
        return {read(xr[b*bs+y*rs+x*cs+m*ms]),read(xi[b*bj+y*ri+x*ci+m*mi])};
    };
    float mx=0;
    // Read the BF16 input once; overwrite it with the reverse state after its last use.
    for(int x=0;x<W;x++) {
        auto v=source(x);
        pos=a*pos+g*v;
        row[warp][x][lane]=__floats2bfloat162_rn(v.r,v.i);
        forward[warp][x][lane]=__floats2bfloat162_rn(pos.r,pos.i);
        mx=fmaxf(mx,mag(pos));
    }
    for(int x=W-1;x>=0;x--) {
        auto v=__bfloat1622float2(row[warp][x][lane]);
        neg=a.conj()*neg+g.conj()*Complex{v.x,v.y};
        row[warp][x][lane]=__floats2bfloat162_rn(neg.r,neg.i);
        mx=fmaxf(mx,mag(neg));
    }
    float scale=exp2f(ceilf(log2f(fmaxf(warpmax(mx),1e-12f)/448.f)));
    if(!lane) scales[group*H+y]=scale;
    for(int x=0;x<W;x++) {
        auto p=__bfloat1622float2(forward[warp][x][lane]), n=__bfloat1622float2(row[warp][x][lane]);
        unsigned lo=encode<false>(p.x/scale,p.y/scale);
        unsigned hi=encode<false>(n.x/scale,n.y/scale);
        states[((group*W+x)*H+y)*32+lane]=lo|(hi<<16);
    }
}

extern "C" __global__ void VERTICAL_NAME(const float* ar,const float* ai,const float* gr,const float* gi,
 const unsigned* src,const float* scales,const float* vx,const float* vy,const T* wi,const T* bi,
 const T* wo,const T* bo,T* out,int B,int M,float eps,int planar) {
    constexpr int step=1+COARSE, oh=H/step, ow=W/step;
    extern __shared__ unsigned sm[];
    auto cache=reinterpret_cast<unsigned (*)[H][32]>(sm);
    constexpr int span=(oh+1)/2;
    auto positive=reinterpret_cast<unsigned (*)[span][32]>(sm+4*H*32);
    auto weights=reinterpret_cast<T (*)[32]>(sm+4*(H+span)*32);
    float* scale=reinterpret_cast<float*>(weights+88);
    int lane=threadIdx.x%32, warp=threadIdx.x/32, tile=blockIdx.x,m=tile*32+lane;
    int b=blockIdx.z,x=blockIdx.y*4+warp,xa=x*step+COARSE,xb=x*step;
    I group=I(b)*gridDim.x+tile;
    if constexpr(!PURE) for(int k=warp;k<88;k+=4){
        T v{};
        if(m<M) v=k<64?wi[m*64+k]:k<72?bi[m*8+k-64]:wo[m*16+k-72];
        weights[k][lane]=v;
    }
    for(int y=threadIdx.x;y<H;y+=blockDim.x) scale[y]=scales[group*H+y];
    __syncthreads();
    if(x>=ow) return;
    bool active=m<M;
    m=min(m,M-1);
    for(int y=0;y<H;y++){
        unsigned l=src[((group*W+xa)*H+y)*32+lane];
        if constexpr(COARSE) l=(l&65535)|(src[((group*W+xb)*H+y)*32+lane]&0xffff0000u);
        cache[warp][y][lane]=l;
    }
    Complex a{ar[m],ai[m]},g{gr[m],gi[m]};
    float br=PURE?0:read(bo[2*m]),bi_out=PURE?0:read(bo[2*m+1]);
    float4 f{};
    auto source=[&](int y) {
        auto raw=cache[warp][y][lane]; auto l=decode<false>(raw,scale[y]),r=decode<false>(raw>>16,scale[y]);
        return make_float4(l.r,l.i,r.r,r.i);
    };
    auto norm=[&](float4 z,int y,bool rev) {
        float v=vy[(rev*H+y)*M+m];
        float l=rsqrtf(fmaxf(v*vx[xa*M+m],eps)),r=rsqrtf(fmaxf(v*vx[(W+xb)*M+m],eps));
        return make_float4(z.x*l,z.y*l,z.z*r,z.w*r);
    };
    auto save=[&](int y,float4 f){if(y%step==step-1 && y/step<oh) {auto z=norm(f,y,false); positive[warp][(y/step)%span][lane]=unsigned(encode<false>(z.x,z.y))|(unsigned(encode<false>(z.z,z.w))<<16);}};
    for(int y=0;y<H;y++) {
        f=advance(f,source(y),a,g);
        save(y,f);
    }
    f={};
    for(int y=H-1;y>=0;y--) {
        if(y==span*step-1 && oh>span) {
            float4 replay{};
            for(int k=0;k<span*step;k++) {replay=advance(replay,source(k),a,g);save(k,replay);}
        }
        f=advance(f,source(y),a.conj(),g.conj());
        if(y%step || y/step>=oh) continue;
        unsigned word=positive[warp][(y/step)%span][lane]; auto l=decode<false>(word,1),r=decode<false>(word>>16,1); float4 u={l.r,l.i,r.r,r.i},v=norm(f,y,true);
        float features[]={u.x,u.z,v.x,v.z,u.y,u.w,v.y,v.w};
        if constexpr(PURE) {
            I cell=((I(b)*oh+y/step)*ow+x)*M+m,plane=I(B)*oh*ow*M;
            if(active) for(int j=0;j<8;j++) out[j*plane+cell]=write<T>(features[j]);
            continue;
        }
        float re=0,im=0;
        #pragma unroll
        for(int h=0;h<8;h++) {
            float z=0;
            #pragma unroll
            for(int j=0;j<8;j++) z=fmaf(read(weights[h*8+j][lane]),features[j],z);
            z+=read(weights[64+h][lane]);
            float s=1.f/(1.f+__expf(-z));
            re=fmaf(z*s,read(weights[72+h][lane]),re);
            im=fmaf(z*s,read(weights[80+h][lane]),im);
        }
        I i=((I(b)*oh+y/step)*ow+x)*(planar?M:2*M)+m;
        if(active) {out[i]=write<T>(re+br);out[i+(planar?I(B)*oh*ow*M:M)]=write<T>(im+bi_out);}
    }
}
extern "C" __global__ void w62_hscan_bwd_packed(const float* ar,const float* ai,const float* gr,const float* gi,
 const T* xr,const T* xi,const unsigned* q,const float* ss,const unsigned* dq,
 const float* ds,T* dx,float* partial,int B,int M,I bs,I rs,I cs,I ms,I bj,I ri,I ci,I mi) {
    __shared__ __nv_bfloat162 buf[4][W][32];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,tile=blockIdx.x,m=tile*32+lane;
    int y=blockIdx.y*4+warp,b=blockIdx.z; I line=I(b)*H+y,group=I(b)*gridDim.x+tile;
    if(y>=H) return;
    bool active=m<M;
    m=min(m,M-1);
    constexpr int step=1+COARSE, dw=W/step;
    float s=ss[group*H+y];
    Complex a{ar[m],ai[m]},g{gr[m],gi[m]},right{},left{},da{},dg{};
    for(int x=0;x<W;x++) {
        Complex inc{};
        if(x%step==0 && x/step<dw) inc=decode<true>(dq[((group*dw+x/step)*H+y)*32+lane]>>16,ds[group*dw+x/step]);
        right=a*right+inc;buf[warp][x][lane]=__floats2bfloat162_rn(right.r,right.i);
    }
    for(int x=W-1;x>=0;x--) {
        Complex inc{};
        if(x%step==step-1) inc=decode<true>(dq[((group*dw+x/step)*H+y)*32+lane],ds[group*dw+x/step]);
        left=a.conj()*left+inc;auto saved=__bfloat1622float2(buf[warp][x][lane]); right={saved.x,saved.y};
        I i=(line*W+x)*M+m;
        Complex prev{},next{};
        if(x) prev=decode<false>(q[((group*W+x-1)*H+y)*32+lane],s);
        if(x+1<W) next=decode<false>(q[((group*W+x+1)*H+y)*32+lane]>>16,s);
        Complex z{read(xr[(line/H)*bs+(line%H)*rs+x*cs+m*ms]),read(xi[(line/H)*bj+(line%H)*ri+x*ci+m*mi])};

        da.r+=left.r*prev.r+left.i*prev.i+right.r*next.r+right.i*next.i;
        da.i+=left.i*prev.r-left.r*prev.i+right.r*next.i-right.i*next.r;
        Complex sum=left+right,diff{left.r-right.r,left.i-right.i};
        dg.r+=sum.r*z.r+sum.i*z.i; dg.i+=diff.i*z.r-diff.r*z.i;
        if(active) store_complex(dx,2*(i-m)+m,I(M),Complex{sum.r*g.r+diff.i*g.i,sum.i*g.r-diff.r*g.i});
    }
    if(!active) return;
    I i=line*4*M+m;
    partial[i]=da.r;partial[i+M]=da.i;partial[i+2*M]=dg.r;partial[i+3*M]=dg.i;
}
#endif
"""


def image(h, w, coarse=False):
    # Share the production scalar, complex and packed BF16 primitives.
    base = _CUDA_SOURCE.split("struct ProjectWeights")[0]
    start = CUDA.index('extern "C" __global__ void VERTICAL_NAME')
    end = CUDA.index('extern "C" __global__ void w62_hscan_bwd')
    forward = CUDA[start:end]
    cuda = CUDA[:start] + forward.replace("VERTICAL_NAME", "w62_vscan_fwd_packed").replace(
        "PURE", "false"
    )
    cuda += (
        forward.replace("VERTICAL_NAME", "w62_pure_vscan_packed").replace("PURE", "true")
        + CUDA[end:]
    )
    source = base + f"\n#define H {h}\n#define W {w}\n" + cuda + _BACKWARD
    flags = [
        "--gpu-architecture=sm_90a",
        "--std=c++17",
        f"-DCOARSE={int(coarse)}",
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


_BACKWARD = r"""
template<bool D> __device__ unsigned compress(float4 v,float s) {
    return unsigned(encode<D>(v.x/s,v.y/s)) | (unsigned(encode<D>(v.z/s,v.w/s))<<16);
}
template<bool D> __device__ float4 expand(unsigned v,float s) {
    auto l=decode<D>(v,s),r=decode<D>(v>>16,s);
    return {l.r,l.i,r.r,r.i};
}
template<bool ACC,int N> __device__ void path_grad(float4 p,float4 n,float2 gp,float2 gn,
    const T (*wt)[32],int lane,float yr,float yi,__half2 (&acc)[N],volatile unsigned* spill,float as,float4 &ip,float4 &in) {
    p=normalize<T>(p,gp); n=normalize<T>(n,gn);
    float feat[8]={p.x,p.z,n.x,n.z,p.y,p.w,n.y,n.w},dp[8]{};
    auto add=[&](int k,__half2 v){
        if(k<N) acc[k]=__hadd2(acc[k],v);
        else {unsigned word=spill[(k-N)*128];__half2 old=*reinterpret_cast<__half2*>(&word);
            old=__hadd2(old,v); spill[(k-N)*128]=*reinterpret_cast<unsigned*>(&old);}
    };
    #pragma unroll
    for(int u=0;u<8;u++) {
        float z=0;
        #pragma unroll
        for(int k=0;k<8;k++) z=fmaf(feat[k],read(wt[8*u+k][lane]),z);
        z+=read(wt[64+u][lane]);
        float sig=1.f/(1.f+__expf(-z)),act=z*sig;
        float dz=(yr*read(wt[72+u][lane])+yi*read(wt[80+u][lane]))*sig*(1.f+z*(1.f-sig));
        #pragma unroll
        for(int k=0;k<8;k+=2) {
            dp[k]+=dz*read(wt[8*u+k][lane]); dp[k+1]+=dz*read(wt[8*u+k+1][lane]);
            if constexpr(ACC) add(4*u+k/2,__floats2half2_rn(dz*feat[k]/as,dz*feat[k+1]/as));
        }
        if constexpr(ACC) {
            // Pack adjacent mode-local parameter indices in their output order.
            float vals[3]={dz,yr*act,yi*act};
            #pragma unroll
            for(int j=0;j<3;j++) add(32+4*j+u/2,
                __floats2half2_rn((u%2?0:vals[j])/as,(u%2?vals[j]:0)/as));
        }
    }
    if constexpr(ACC) add(44,__floats2half2_rn(yr/as,yi/as));
    ip=normalize<T>({dp[0],dp[4],dp[1],dp[5]},gp);
    in=normalize<T>({dp[2],dp[6],dp[3],dp[7]},gn);
}
#define VBARGS const float* ar,const float* ai,const float* gr,const float* gi,const void* input,const float* scales,const float* vx,const float* vy,const T* wi,const T* bi,const T* wo,const T* dyr,const T* dyi,void* output,float* ds,float* partial,int B,int M,I rs,I is,float eps
#define VBVAL ar,ai,gr,gi,input,scales,vx,vy,wi,bi,wo,dyr,dyi,output,ds,partial,B,M,rs,is,eps
template<bool FULL,bool LEGACY> __device__ __forceinline__ void backward(VBARGS) {
    constexpr int step=COARSE+1, oh=H/step, ow=W/step, span=FULL?H:(H<=7?7:14);
    constexpr int chunks=(H+span-1)/span;
    extern __shared__ unsigned sm[];
    int tid=threadIdx.x,lane=tid%32,warp=tid/32,tile=blockIdx.x,m=32*tile+lane;
    int b=blockIdx.z,x=blockIdx.y*4+warp,xb=x*step,xa=xb+step-1;
    I group=I(b)*gridDim.x+tile, plane=I(B)*H*W*M, gplane=I(B)*H*ow*M;
    auto wt=reinterpret_cast<T (*)[32]>(sm);
    unsigned* src=sm+1408+tid;
    unsigned* pos=src+H*128;
    unsigned* inj=pos+span*128;
    constexpr int extra=FULL?0:(H<=14?18:4);
    unsigned* pc=inj+(span+extra)*128;
    unsigned* nc=pc+(chunks-1)*128;
    unsigned* ac=nc+(chunks-1)*128;
    auto ss=reinterpret_cast<float*>(ac-tid+(chunks-1)*128);
    for(int k=warp;k<88;k+=4) {
        T v{};
        if(m<M) v=k<64?wi[m*64+k]:k<72?bi[m*8+k-64]:wo[m*16+k-72];
        wt[k][lane]=v;
    }
#if H<=14
    #pragma unroll 1
#endif
    for(int y=tid;y<H;y+=128) ss[y]=scales[group*H+y];
    __syncthreads();
    if(x>=ow) return;
    Complex a{},g{};
    if(m<M) {a={ar[m],ai[m]};g={gr[m],gi[m]};}
    float mx=0;
    auto source=[&](int y){return expand<false>(src[128*y],ss[y]);};
    auto index=[&](int y){return ((group*ow+x)*H+y)*32+lane;};
    auto cell=[&](int y){return ((I(b)*oh+y/step)*ow+x)*M;};
    auto gain=[&](int y,bool rev)->float2 {
        float v=vy[(int(rev)*H+y)*M+m];
        return {rsqrtf(fmaxf(v*vx[xa*M+m],eps)),rsqrtf(fmaxf(v*vx[(W+xb)*M+m],eps))};
    };
    float4 p{},neg{};
#if H<=14
    #pragma unroll 1
#endif
    for(int y=0;y<H;y++) {
        if constexpr(LEGACY) {
            auto q=(const T*)input;
            I i=((I(b)*H+y)*W+xa)*M+m,j=i-(xa-xb)*M+2*plane;
            src[y*128]=m<M?compress<false>({read(q[i]),read(q[i+plane]),read(q[j]),read(q[j+plane])},ss[y]):0;
        } else {
            auto q=(const unsigned*)input;
            unsigned v=q[((group*W+xa)*H+y)*32+lane];
            if constexpr(COARSE) v=(v&65535)|(q[((group*W+xb)*H+y)*32+lane]&0xffff0000u);
            src[y*128]=v;
        }
        p=advance(p,source(y),a,g);
        // E5M2 checkpoints allow the column scale to be selected after the sweep.
        if constexpr(FULL) pos[y*128]=compress<true>(p,1);
        else if((y+1)%span==0 && y+1<H) pc[y/span*128]=compress<true>(p,1);
        if(m<M && y%step==0 && y/step<oh) mx=fmaxf(mx,fmaxf(fabsf(read(dyr[cell(y)*rs+m])),fabsf(read(dyi[cell(y)*is+m]))));
    }
    float scale=exp2f(ceilf(log2f(fmaxf(warpmax(mx),1e-12f)/64.f)));
    if(!lane && !LEGACY) ds[group*ow+x]=scale;
    bool active=m<M;
    m=min(m,M-1);
    Complex aa{},ab{},da{},dg{};
    // The segment injection buffer is available for FP16x2 Path partials
    // until the downward sweep has flushed all 94 per-column values.
    constexpr int na=FULL?45:45-span-extra;
    __half2 acc[na]{};
    if constexpr(!FULL) for(int j=0;j<span+extra;j++) inj[j*128]=0;
    auto restore=[&](int block,int lo,int hi) {
        float4 f=block?expand<true>(pc[(block-1)*128],1):float4{};
#if H<=14
        #pragma unroll 1
#endif
        for(int y=lo;y<hi;y++) {f=advance(f,source(y),a,g);pos[(y-lo)*128]=compress<true>(f,1);}
    };
    auto storegrad=[&](int y,float4 v) {
        if constexpr(LEGACY) {
            I i=((I(b)*H+y)*ow+x)*M+m;
            auto q=(T*)output;
            float vals[]={v.x,v.y,v.z,v.w};
            #pragma unroll
            for(int j=0;j<4;j++) if(active) q[i+j*gplane]=write<T>(vals[j]);
        } else {
            auto q=(unsigned*)output+index(y);
            *q=active?compress<true>(v,scale):0;
        }
    };
    // Forward-mode derivatives of the negative recurrence let the ascending
    // adjoint consume injections without retaining a third segment buffer.
    float4 tan_a{},tan_g{};
    // Positive adjoints run from the bottom. Save reverse checkpoints for replay.
#if H<=14
    #pragma unroll 1
#endif
    for(int block=chunks-1;block>=0;block--) {
        int lo=block*span,hi=min(H,lo+span);
        if constexpr(!FULL) {restore(block,lo,hi); if(block+1<chunks) {nc[block*128]=compress<true>(neg,1);ac[block*128]=compress<true>({aa.r,aa.i,ab.r,ab.i},scale);}}
#if H<=14
        #pragma unroll 1
#endif
        for(int y=hi-1;y>=lo;y--) {
            float4 next=neg;
            tan_a=advance(tan_a,next,a.conj(),Complex{1,0});
            tan_g=advance(tan_g,source(y),a.conj(),Complex{1,0});
            neg=advance(neg,source(y),a.conj(),g.conj());
            if(step==2 && y%2) continue;
            float4 ip{},in{};
            bool valid=y/step<oh;
            if(valid) {
                path_grad<true>(expand<true>(pos[(y-lo+step-1)*128],1),neg,gain(y+step-1,false),gain(y,true),wt,lane,
                    read(dyr[cell(y)*rs+m]),read(dyi[cell(y)*is+m]),acc,inj,scale*64,ip,in);
                pole_grad({in.x,in.y},{tan_a.x,tan_a.y},{tan_g.x,tan_g.y},da,dg,true);
                pole_grad({in.z,in.w},{tan_a.z,tan_a.w},{tan_g.z,tan_g.w},da,dg,true);
                if constexpr(FULL) inj[(y-lo)*128]=compress<true>(in,scale);
            }
            for(int r=valid?step-1:0;r>=0;r--) {
                int j=y+r-lo;
                float4 prev=j?expand<true>(pos[(j-1)*128],1):(block?expand<true>(pc[(block-1)*128],1):float4{});
                auto v=column_adjoint_step<false>(prev,source(y+r),r==step-1?ip:float4{},a,g,aa,ab,da,dg);
                if constexpr(FULL) pos[j*128]=compress<true>(v,scale);
            }
        }
    }
    float* out=partial+(I(b)*ow+x)*94*M+m;
    if(active) {
    out[0]=da.r;out[M]=da.i;out[2*M]=dg.r;out[3*M]=dg.i;
    #pragma unroll
    for(int k=0;k<45;k++) {unsigned bits=k<na?0:inj[(k-na)*128]; float2 v=__half22float2(k<na?acc[k]:*reinterpret_cast<__half2*>(&bits));out[(4+2*k)*M]=v.x*scale*64;out[(5+2*k)*M]=v.y*scale*64;}
    }
    aa={}; ab={};
#if H<=14
    #pragma unroll 1
#endif
    for(int block=0;block<chunks;block++) {
        int lo=block*span,hi=min(H,lo+span);
        if constexpr(!FULL) {
            restore(block,lo,hi);
            neg=block+1<chunks?expand<true>(nc[block*128],1):float4{};
            float4 boundary=block+1<chunks?expand<true>(ac[block*128],scale):float4{};
            Complex pa{boundary.x,boundary.y},pb{boundary.z,boundary.w};
            float4 saved{};
#if H<=14
            #pragma unroll 1
#endif
            for(int y=hi-1;y>=lo;y--) {
                neg=advance(neg,source(y),a.conj(),g.conj());
                if(step==2 && y%2) {saved=expand<true>(pos[(y-lo)*128],1);continue;}
                float4 ip{},in{};
                bool valid=y/step<oh;
                if(valid) {
                    path_grad<false>(step==2?saved:expand<true>(pos[(y-lo)*128],1),neg,gain(y+step-1,false),gain(y,true),wt,lane,
                        read(dyr[cell(y)*rs+m]),read(dyi[cell(y)*is+m]),acc,inj,scale*64,ip,in);
                    inj[(y-lo)*128]=compress<true>(in,scale);
                }
                for(int r=valid?step-1:0;r>=0;r--) {
                    float4 d=r==step-1?ip:float4{};
                    pa=a.conj()*pa+Complex{d.x,d.y};pb=a.conj()*pb+Complex{d.z,d.w};
                    auto l=g.conj()*pa,rr=g.conj()*pb;
                    pos[(y+r-lo)*128]=compress<true>({l.r,l.i,rr.r,rr.i},scale);
                }
            }
        }
#if H<=14
        #pragma unroll 1
#endif
        for(int y=lo;y<hi;y++) {
            float4 d=y%step==0 && y/step<oh?expand<true>(inj[(y-lo)*128],scale):float4{};
            aa=a*aa+Complex{d.x,d.y};ab=a*ab+Complex{d.z,d.w};
            auto l=g*aa,r=g*ab;
            float4 v=expand<true>(pos[(y-lo)*128],scale);
            storegrad(y,{v.x+l.r,v.y+l.i,v.z+r.r,v.w+r.i});
        }
    }

}
extern "C" __global__ __maxnreg__(128) void w62_vscan_bwd_packed_seg(VBARGS){backward<false,false>(VBVAL);}
extern "C" __global__ void w62_vscan_bwd_packed_full(VBARGS){backward<true,false>(VBVAL);}
extern "C" __global__ __maxnreg__(128) void w62_vscan_bwd_bf16io(VBARGS){backward<false,true>(VBVAL);}
"""

HF = "w62_hscan_fwd_packed"
HB = "w62_hscan_bwd_packed"
VF = "w62_vscan_fwd_packed"
PURE = "w62_pure_vscan_packed"
SEG = "w62_vscan_bwd_packed_seg"
FULL = "w62_vscan_bwd_packed_full"
BF16 = "w62_vscan_bwd_bf16io"
NAMES = (HF, VF, SEG, FULL, BF16, PURE, HB)


def shared_bytes(name, h, w, coarse):
    if name == HF:
        return 1024 * w
    if name == HB:
        return 0
    if name in (VF, PURE):
        return 5632 + 4 * h + 512 * (h + (h // (1 + coarse) + 1) // 2)
    span = h if name == FULL else (7 if h <= 7 else 14)
    return (
        5632
        + 4 * h
        + 512
        * (h + 2 * span + 3 * ((h - 1) // span) + (0 if name == FULL else 18 if h <= 14 else 4))
    )


@cache
def _module(device, h, w, coarse):
    mod = nv._CudaModule(ct.create_string_buffer(image(h, w, coarse)), device)
    for name in NAMES:
        mod.load_function(name, shared_bytes(name, h, w, coarse))
    return mod


def _launch(name, shape, coarse, tensors, scalars):
    b, h, w, m = shape
    device = tensors[0].device
    lines = h if name in (HF, HB) else w // (1 + coarse)
    with torch.cuda.device(device):
        mod = _module(device.index, h, w, coarse)
        mod.launch(
            name,
            ((m + 31) // 32, (lines + 3) // 4, b),
            (128, 1, 1),
            shared_bytes(name, h, w, coarse),
            torch.cuda.current_stream().cuda_stream,
            [
                *(ct.c_void_p(t.data_ptr()) if t is not None else ct.c_void_p() for t in tensors),
                *scalars,
            ],
        )


def _tensor(t, shape, dtype, device, *, contiguous=True):
    if (
        not isinstance(t, torch.Tensor)
        or tuple(t.shape) != tuple(shape)
        or t.dtype != dtype
        or t.device != device
        or device.type != "cuda"
        or (contiguous and not t.is_contiguous())
    ):
        raise ValueError(f"expected {shape} {dtype} on {device}")


def _poles(pole, m, device):
    if len(pole) != 4 or any(
        p.shape != (m,) or p.device != device or not p.is_floating_point() for p in pole
    ):
        raise ValueError("four floating pole vectors must match the mode count and device")


def _horizontal_inputs(pole, real, imag):
    if real.ndim != 4 or min(real.shape) < 1:
        raise ValueError("expected nonempty BHWM source")
    for t in (real, imag):
        _tensor(t, real.shape, torch.bfloat16, real.device, contiguous=False)
    _poles(pole, real.shape[-1], real.device)


def _vertical_inputs(pole, q, s, aux, path, coarse, legacy):
    if q.ndim != 5 or min(q.shape) < 1:
        raise ValueError("expected nonempty five-dimensional states")
    m = pole[0].numel()
    if legacy:
        _, b, h, w, _ = q.shape
        _tensor(q, (4, b, h, w, m), torch.bfloat16, q.device)
    else:
        b, _, w, h, _ = q.shape
        _tensor(q, (b, (m + 31) // 32, w, h, 32), torch.uint32, q.device)
    _tensor(s, (b, (m + 31) // 32, h), torch.float32, q.device)
    _poles(pole, m, q.device)
    if coarse and min(h, w) < 2:
        raise ValueError("coarse projection needs two rows and columns")
    dims = [(2, w, m), (2, h, m), (m, 8, 8), (m, 8), (m, 2, 8), (m, 2)]
    values = [*aux, *path]
    if (
        len(aux) != 2
        or len(path) != 4
        or any(
            t.shape != k or t.device != q.device or not t.is_floating_point()
            for t, k in zip(values, dims)
        )
    ):
        raise ValueError("invalid auxiliary gain or path weight shape/device")
    return b, h, w, m


def w62_hscan_fwd_packed(pole, real, imag):
    _horizontal_inputs(pole, real, imag)
    b, h, w, m = real.shape
    t = (m + 31) // 32
    q = real.new_empty((b, t, w, h, 32), dtype=torch.uint32)
    s = real.new_empty((b, t, h), dtype=torch.float32)
    strides = [ct.c_longlong(v) for x in (real, imag) for v in x.stride()]
    _launch(
        HF,
        real.shape,
        False,
        [*(v.float().contiguous() for v in pole), real, imag, q, s],
        [ct.c_int(m), *strides],
    )
    return q, s


def w62_hscan_bwd_packed(pole, real, imag, q, s, dq, ds):
    b, h, w, m = real.shape
    _horizontal_inputs(pole, real, imag)
    tiles = (m + 31) // 32
    coarse = dq.shape[2] != w
    dw = w // (1 + coarse)
    _tensor(q, (b, tiles, w, h, 32), torch.uint32, real.device)
    _tensor(s, (b, tiles, h), torch.float32, real.device)
    _tensor(dq, (b, tiles, dw, h, 32), torch.uint32, real.device)
    _tensor(ds, (b, tiles, dw), torch.float32, real.device)
    dx = real.new_empty((b, h, w, 2 * m))
    partial = real.new_empty((b, h, 4, m), dtype=torch.float32)
    strides = [ct.c_longlong(v) for x in (real, imag) for v in x.stride()]
    _launch(
        HB,
        real.shape,
        coarse,
        [*(v.float().contiguous() for v in pole), real, imag, q, s, dq, ds, dx, partial],
        [ct.c_int(b), ct.c_int(m), *strides],
    )
    return [*partial.sum((0, 1)).unbind(), dx]


def w62_vscan_fwd_packed(pole, q, s, aux, path, coarse=False, *, pure=False, epsilon=1e-6):
    b, h, w, m = _vertical_inputs(pole, q, s, aux, path, coarse, False)
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    out = q.new_empty((8, b, oh, ow, m) if pure else (b, oh, ow, 2 * m), dtype=torch.bfloat16)
    _launch(
        PURE if pure else VF,
        (b, h, w, m),
        coarse,
        [
            *(p.float().contiguous() for p in pole),
            q,
            s,
            *(p.float().contiguous() for p in aux),
            *(p.bfloat16().contiguous() for p in path),
            out,
        ],
        [ct.c_int(b), ct.c_int(m), ct.c_float(epsilon), ct.c_int(0)],
    )
    return out


def w62_vscan_bwd_packed(pole, q, s, aux, path, gr, gi, coarse=False, *, mode=SEG, epsilon=1e-6):
    if mode not in (SEG, FULL, BF16):
        raise ValueError("unknown vertical backward mode")
    b, h, w, m = _vertical_inputs(pole, q, s, aux, path, coarse, mode == BF16)
    for grad in (gr, gi):
        _tensor(
            grad,
            (b, h // (1 + coarse), w // (1 + coarse), m),
            torch.bfloat16,
            q.device,
            contiguous=False,
        )
    t = (m + 31) // 32
    ow = w // (1 + coarse)
    dims = (4, b, h, ow, m) if mode == BF16 else (b, t, ow, h, 32)
    dq = q.new_empty(dims, dtype=torch.bfloat16 if mode == BF16 else torch.uint32)
    ds = None if mode == BF16 else q.new_empty((b, t, ow), dtype=torch.float32)
    partial = q.new_empty((b, ow, 94, m), dtype=torch.float32)

    def interleaved(t):
        return t.stride() == (t.shape[1] * t.shape[2] * 2 * m, t.shape[2] * 2 * m, 2 * m, 1)

    gr = gr if interleaved(gr) else gr.contiguous()
    gi = gi if interleaved(gi) else gi.contiguous()
    _launch(
        mode,
        (b, h, w, m),
        coarse,
        [
            *(p.float().contiguous() for p in pole),
            q,
            s,
            *(p.float().contiguous() for p in aux),
            *(p.bfloat16().contiguous() for p in path[:3]),
            gr,
            gi,
            dq,
            ds,
            partial,
        ],
        [
            ct.c_int(b),
            ct.c_int(m),
            ct.c_longlong(gr.stride(2) // m),
            ct.c_longlong(gi.stride(2) // m),
            ct.c_float(epsilon),
        ],
    )
    params = partial.sum((0, 1)).T
    weights = [v.reshape_as(p) for v, p in zip(params[:, 4:].split((64, 8, 16, 2), 1), path)]
    return [*params[:, :4].T.unbind(), dq, ds, *weights]
