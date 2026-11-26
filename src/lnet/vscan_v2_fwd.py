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

MODES = (
    "v59_fwd_smem_regw",
    "v59_fwd_fused_tc_v2",
    "v59_fwd_decoupled_tc",
    "v59_fwd_decoupled_tc_l2chunk",
)


def chunk_batch(b, h, w, m, coarse):
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    # A tile contains four lines and a multiple of four output rows.
    tile_cells = 16 * scans._ceil(oh, 4)
    max_tiles = (32 << 20) // (16 * m * tile_cells)
    return min(b, max(scans._ceil(b, 64), 4 * max_tiles // ow))


def pitch(batch, oh, ow):
    return scans._ceil(batch * ow, 4) * 16 * scans._ceil(oh, 4)


class Config(NamedTuple):
    threads: int
    grid: int
    smem: int


def cache_source(mode, h, oh):
    return h <= 28 and (mode == MODES[0] or (mode == MODES[1] and (h != 28 or oh != 28)))


def register_rows(mode, h, oh):
    # Keep the first few source rows in registers to fit four 28d CTAs.
    return max(0, h + oh + 6 - 56) if mode == MODES[0] and h <= 28 else 0


def config(mode, batch, h, w, m, coarse, stage="scan"):
    if mode not in MODES or min(batch, m, h // (1 + coarse), w // (1 + coarse)) < 1:
        raise ValueError("Invalid scan configuration")
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    tiles = scans._ceil(m, 32)
    if stage == "mlp":
        return Config(128, tiles * min(256, pitch(batch, oh, ow) // 16), 4096)
    fused = mode == MODES[1]
    cache = cache_source(mode, h, oh)
    size = 1024 * (oh + cache * h - register_rows(mode, h, oh)) + (
        8192 + max(0, 4 - oh) * 1024 if fused else 8192 if mode in MODES[2:] else 0
    )
    if mode == MODES[0]:
        size += 32 * 88 * 2
    if size > 227 * 1024:
        raise ValueError("Shared memory exceeds Hopper capacity")
    return Config(
        128,
        tiles
        * (
            scans._ceil(batch * ow, 4)
            if mode in MODES[2:]
            else min(256, scans._ceil(batch * ow, 4))
        ),
        size,
    )


_SCALAR = r"""
    for(int top=OH-1;top>=0;top-=BW) {
        float p[BW][8];
        #pragma unroll
        for(int t=0;t<BW;++t) {
            int yo=top-t;
            float4 n{},v{};
            if(yo>=0) {
                int y=yo*STEP;
                if(STEP==2) state=advance(state,get(y+1).unpack(),a.conj(),g.conj());
                state=advance(state,get(y).unpack(),a.conj(),g.conj());
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
                    float w=((k&1)?path.input_pair(4*j+k/2).i:path.input_pair(4*j+k/2).r);
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
        for(int t=0;t<BW;++t) if(valid && top>=t) store(top-t,out[t]);
    }
"""

_HEADER = r"""
// Stage eight consecutive halfwords per lane, then transpose to field-major
// shared memory. Contiguous tensor views can still have unaligned offsets.
template<int WIDTH,int BASE>
__device__ void stage_weights(T* cache,const void* ptr,int mode0){
    auto src=static_cast<const T*>(ptr)+1LL*mode0*WIDTH;
    int groups=WIDTH/8;
    for(int i=threadIdx.x;i<min(32,M-mode0)*groups;i+=N){
        uint4 v;
        if((reinterpret_cast<unsigned long long>(src)&15)==0)
            v=reinterpret_cast<const uint4*>(src)[i];
        else {
            #pragma unroll
            for(int j=0;j<8;j++) reinterpret_cast<T*>(&v)[j]=src[8*i+j];
        }
        int m=i/groups,field=i%groups*8;
        if constexpr(BASE==0){
            #pragma unroll
            for(int j=0;j<4;j++)
                reinterpret_cast<unsigned*>(cache)[(field/2+j)*32+m]=reinterpret_cast<unsigned*>(&v)[j];
        } else {
            #pragma unroll
            for(int j=0;j<8;j++) cache[(BASE+field+j)*32+m]=reinterpret_cast<T*>(&v)[j];
        }
    }
}

__device__ int slot(int row,int col){return row*32+(col^row);}
__device__ Quad<T> norm(float4 v,float2 gain) {
    return Quad<T>::pack(make_float4(v.x*gain.x,v.y*gain.x,v.z*gain.y,v.w*gain.y));
}
__device__ unsigned bits(Complex z) {
    auto p=Pair<T>::pack(z).value;
    return *reinterpret_cast<const unsigned*>(&p);
}
"""
_KERNEL = r"""
extern "C" __global__ __maxnreg__(REGISTER_LIMIT) void ENTRY(const float* dr,const float* di,const float* gr,const float* gi,
    const T* states,const float* vx,const float* vy,const void* wi,const void* bi,
    const void* wo,const void* bo,T* outr,T* outi,int B,int inter,float eps,uint4* P,int start,int count,int pitch) {
    extern __shared__ float4 shared[];
"""
_BODY = r"""
    auto* src=reinterpret_cast<Quad<T>*>(shared);
    auto* pos=src+(CACHE?(H-REG_ROWS)*N:0);
    int tid=threadIdx.x,tiles=(M+TM-1)/TM;
    int mode0=blockIdx.x%tiles*TM, raw=mode0+tid%TM;
    WEIGHTS
    TILE_LOOP {
    int line=tile*4+tid/TM;
    bool valid=raw<M && line<count*OW;
    int m=min(raw,M-1);
    line=min(line,count*OW-1);
    int x=line%OW,b=start+line/OW,xa=x*STEP+STEP-1,xb=x*STEP;
    long long plane=1LL*B*H*W*M,base=1LL*b*H*W*M+m;
    Column<T,long long> c{states,vy,base+xa*M,base+xb*M,plane,plane,
        (1LL*b*OH*OW+x)*M+m,1LL*B*OH*OW*M,H,W,M,OW,m,vx[xa*M+m],vx[(W+xb)*M+m],eps};
    Complex a{dr[m],di[m]},g{gr[m],gi[m]};
    float br=rounded<T>(load_scalar(bo,2*m,PATH_TYPES>>6));
    float bj=rounded<T>(load_scalar(bo,2*m+1,PATH_TYPES>>6));
    auto store=[&](int yo,float2 v) {
        auto cell=(c.cell(yo)-m)*(1+inter)+m;
        outr[cell]=write<T>(v.x+br); outi[cell]=write<T>(v.y+bj);
    };
    SAVED_ROWS
    float4 state{};
    for(int y=0;y<H;++y) {
        auto q=c.packed_source(y);
        SAVE_SOURCE
        state=advance(state,q.unpack(),a,g);
        if(y%STEP==STEP-1) pos[(y/STEP)*N+tid]=norm(state,c.gain(y,false));
    }
    auto get=[&](int y){READ_SOURCE};
    state={};
"""


_MMA = r"""

__device__ void project(unsigned aa,unsigned ab,unsigned weight,float2 bias,float2 wr,float2 wi,float2 (&out)[2]){
    float d[4];
    asm("mma.sync.aligned.m16n8k16.row.col.f32.FMT.FMT.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%6},{%7,%6},{%8,%8,%8,%8};"
        :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        :"r"(aa),"r"(ab),"r"(0u),"r"(weight),"f"(0.f));
    #pragma unroll
    for(int h=0;h<2;h++){
        float a=d[2*h]+bias.x,b=d[2*h+1]+bias.y;
        a*=__fdividef(1.f,1.f+__expf(-a)); b*=__fdividef(1.f,1.f+__expf(-b));
        float r=a*wr.x+b*wr.y,i=a*wi.x+b*wi.y;
        r+=__shfl_xor_sync(~0u,r,1); i+=__shfl_xor_sync(~0u,i,1);
        out[h]={r+__shfl_xor_sync(~0u,r,2),i+__shfl_xor_sync(~0u,i,2)};
    }
}
"""
_WEIGHTS = r"""
    int group=(tid%32)/4,k=tid%4,warp=tid/32;
    unsigned weight[8];
    PathWeights<T,int,false> owner{wi,bi,wo,PATH_TYPES,min(mode0+warp*8+group,M-1),8};
    float2 bias={owner.bias(2*k),owner.bias(2*k+1)};
    float2 wr={owner.output(2*k),owner.output(2*k+1)},wj={owner.output(8+2*k),owner.output(9+2*k)};
    auto gather=[&](float2 v,int s){return make_float2(__shfl_sync(~0u,v.x,4*s+k),__shfl_sync(~0u,v.y,4*s+k));};
    #pragma unroll
    for(int s=0;s<8;s++){
        PathWeights<T,int,false> pw{wi,bi,wo,PATH_TYPES,min(mode0+warp*8+s,M-1),8};
        weight[s]=bits(pw.input_pair(4*group+k));
    }
"""
_BATCH = r"""
    auto* transpose=reinterpret_cast<uint4*>(pos+((FUSED && OH<4)?4:OH)*N);
    // The four highest positive rows have been consumed before the first MMA store.
    auto* out=reinterpret_cast<float2*>(pos+(OH>4?OH-4:0)*N);
    for(int top=OH-1;top>=0;top-=4){
        #pragma unroll 1
        for(int t=0;t<4;t++){
            int yo=top-t;
            uint4 value{};
            if(yo>=0){
                int y=yo*STEP;
                if(STEP==2) state=advance(state,get(y+1).unpack(),a.conj(),g.conj());
                state=advance(state,get(y).unpack(),a.conj(),g.conj());
                auto n=norm(state,c.gain(y,true)).unpack(),v=pos[yo*N+tid].unpack();
                value={bits({v.x,v.z}),bits({n.x,n.z}),bits({v.y,v.w}),bits({n.y,n.w})};
            }
            transpose[slot(4*t+tid/32,tid%32)]=value;
        }
        __syncthreads();
        PATH
        __syncthreads();
        STORE
    }
"""
_FUSED = r"""
    #pragma unroll 2
    for(int s=0;s<8;s++){
        int col=warp*8+s;
        // Statically indexed reads keep all eight fragments in registers.
        unsigned w=weight[0];
        #pragma unroll
        for(int j=1;j<8;j++) if(s==j) w=weight[j];
        auto words=reinterpret_cast<unsigned*>(transpose);
        float2 v[2];
        project(words[(slot(group,col))*4+k],words[(slot(group+8,col))*4+k],
                w,gather(bias,s),gather(wr,s),gather(wj,s),v);
        if(k==0){out[slot(group,col)]=v[0];out[slot(group+8,col)]=v[1];}
    }
"""
_PSTORE = r"""
    // Physical cell order packs four column lines in reverse scan order.
    for(int idx=tid;idx<512;idx+=128){
        int col=idx/16,row=idx%16;
        if(mode0+col<M){
            int cell=tile*TILE_CELLS+(OH-1-top)*4+row;
            P[(long long)(mode0+col)*pitch+cell]=transpose[slot(row,col)];
        }
    }
"""
_MLP = r"""
    auto* out=reinterpret_cast<float2*>(shared);
    int tid=threadIdx.x,tiles=(M+31)/32,mode0=blockIdx.x%tiles*32;
    int group=tid%32/4,k=tid%4,warp=tid/32;
    int cells=((count*OW+3)/4)*TILE_CELLS;
    for(int n=(blockIdx.x/tiles)*16;n<cells;n+=gridDim.x/tiles*16){
        #pragma unroll 1
        for(int s=0;s<8;s++){
            int col=min(mode0+warp*8+s,M-1);
            PathWeights<T,int,false> pw{wi,bi,wo,PATH_TYPES,col,8};
            auto words=reinterpret_cast<unsigned*>(P+(long long)col*pitch+n);
            float2 v[2];
            project(words[group*4+k],words[(group+8)*4+k],bits(pw.input_pair(4*group+k)),
                    {pw.bias(2*k),pw.bias(2*k+1)},{pw.output(2*k),pw.output(2*k+1)},
                    {pw.output(8+2*k),pw.output(9+2*k)},v);
            if(k==0){out[slot(group,warp*8+s)]=v[0];out[slot(group+8,warp*8+s)]=v[1];}
        }
        __syncthreads();
        for(int idx=tid;idx<512;idx+=128){
            int cell=n+idx/32,m=mode0+idx%32;
            if(cell<cells && m<M){
                int tile=cell/TILE_CELLS,sub=cell%TILE_CELLS;
                int line=tile*4+sub%4,yo=OH-1-sub/4;
                if(line>=count*OW || yo<0) continue;
                auto off=(((long long)(start+line/OW)*OH+yo)*OW+line%OW)*M*(1+inter)+m;
                outr[off]=write<T>(out[slot(idx/32,idx%32)].x+rounded<T>(load_scalar(bo,2*m,PATH_TYPES>>6)));
                outi[off]=write<T>(out[slot(idx/32,idx%32)].y+rounded<T>(load_scalar(bo,2*m+1,PATH_TYPES>>6)));
            }
        }
        __syncthreads();
    }
}
"""


@cache
def source(mode, h, w, m, coarse, dtype, stage="scan"):
    typ, fmt = {torch.float16: ("__half", "f16"), torch.bfloat16: ("__nv_bfloat16", "bf16")}[dtype]
    original = scans._CUDA_SOURCE
    prefix = original.split("template<class T>\n__device__ Quad<T> normalize_packed")[0]
    cfg_oh = h // (1 + coarse)
    nr = register_rows(mode, h, cfg_oh)
    # The launcher converts all four path tensors to the state's scalar type.
    scalar_code = scans._SCALAR_TYPES[dtype] * 0x55
    constants = f"""
using T={typ};
constexpr int H={h},W={w},M={m},STEP={1 + coarse},OH=H/STEP,OW=W/STEP;
constexpr int TILE_CELLS=16*((OH+3)/4);
constexpr int N=128,TM=32,BW=4,PATH_TYPES={scalar_code},REG_ROWS={nr};
constexpr bool CACHE={int(cache_source(mode, h, cfg_oh))},FUSED={int(mode == MODES[1])};
"""
    decoupled = mode in MODES[2:]
    entry = "tc_path_mlp_fwd" if stage == "mlp" else "pure_vscan_fwd_p" if decoupled else "scan"
    cap = 48 if decoupled else 128 if nr else 80
    head = (
        prefix
        + constants
        + _HEADER
        + _MMA.replace("FMT", fmt)
        + _KERNEL.replace("REGISTER_LIMIT", str(cap)).replace("ENTRY", entry)
    )
    if stage == "mlp":
        return head + _MLP
    weights = _WEIGHTS if mode == MODES[1] else ""
    if mode == MODES[0]:
        weights = """
        auto* cache=reinterpret_cast<T*>(pos+OH*N);
        stage_weights<64,0>(cache,wi,mode0);
        stage_weights<8,64>(cache,bi,mode0);
        stage_weights<16,72>(cache,wo,mode0);
        __syncthreads();
        auto path=path_weights<T,int,true>(cache,wi,bi,wo,PATH_TYPES,raw,8,tid%32);
        """
    declarations = " ".join(f"Quad<T> saved{i}{{}};" for i in range(nr))
    save = "".join(f"if(y=={i}) saved{i}=q; else " for i in range(nr))
    read = "".join(f"if(y=={i}) return saved{i}; " for i in range(nr))
    tile_loop = (
        "{int tile=blockIdx.x/tiles;"
        if decoupled
        else "for(int tile=blockIdx.x/tiles;tile*4<count*OW;tile+=gridDim.x/tiles)"
    )
    body = _BODY.replace("WEIGHTS", weights).replace(
        "TILE_LOOP {", tile_loop if decoupled else tile_loop + " {"
    )
    body = body.replace("SAVED_ROWS", declarations).replace(
        "SAVE_SOURCE", save + "if constexpr(CACHE) src[(y-REG_ROWS)*N+tid]=q;"
    )
    body = body.replace(
        "READ_SOURCE", read + "return CACHE?src[(y-REG_ROWS)*N+tid]:c.packed_source(y);"
    )
    if coarse and h % 2:
        body += "state=advance(state,get(H-1).unpack(),a.conj(),g.conj());\n"
    fused = mode == MODES[1]
    suffix = (
        _SCALAR
        if mode == MODES[0]
        else _BATCH.replace("PATH", _FUSED if fused else _PSTORE).replace(
            "STORE",
            "for(int t=0;t<4;t++) if(valid && top>=t) store(top-t,out[slot(t*4+tid/32,tid%32)]);"
            if fused
            else "",
        )
    )
    return head + body + suffix + ("__syncwarp();\n" if fused else "") + "}}\n"


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
def function(device, mode, h, w, m, coarse, dtype, stage="scan"):
    driver = scans._load_cuda_driver()
    module, fn = ct.c_void_p(), ct.c_void_p()
    with torch.cuda.device(device), tempfile.TemporaryDirectory() as directory:
        scans._ensure_context(driver, device)
        cubin, _ = compile_cubin(source(mode, h, w, m, coarse, dtype, stage), directory)
        scans._check_cu(driver.cuModuleLoadData(ct.byref(module), cubin.read_bytes()), "load cubin")
        scans._check_cu(
            driver.cuModuleGetFunction(
                ct.byref(fn),
                module,
                (
                    b"tc_path_mlp_fwd"
                    if stage == "mlp"
                    else b"pure_vscan_fwd_p"
                    if mode in MODES[2:]
                    else b"scan"
                ),
            ),
            "load scan",
        )
        scans._configure_smem(driver, fn, config(mode, 1, h, w, m, coarse, stage).smem)
    return driver, fn


@torch.library.custom_op("lnet::vscan_v2_fwd_packed", mutates_args=())
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
        states.dtype not in (torch.float16, torch.bfloat16)
        or not states.is_cuda
        or not scans._pole_ok(pole, states)
        or not scans._vertical_inputs_ok(states, aux, path, coarse=coarse)
        or path[0].shape[1] != 8
        or not 0 < epsilon < math.inf
    ):
        raise ValueError("Invalid Wave 59B forward inputs")
    b, h, w, m = states.shape[1:]
    config(mode, b, h, w, m, coarse)
    output = _fake(pole, states, aux, path, coarse, epsilon, mode, planar)
    real, imag = output.unbind(0) if planar else output.split(m, dim=-1)
    inputs = [
        *(p.float().contiguous() for p in pole),
        states.contiguous(),
        *(v.float().contiguous() for v in aux),
        *(p.to(states.dtype).contiguous() for p in path),
    ]
    with torch.cuda.device(states.device):
        chunk = chunk_batch(b, h, w, m, coarse) if mode == MODES[3] else b
        stride = pitch(chunk, h // (1 + coarse), w // (1 + coarse))
        work = states.new_empty((m, stride, 8)) if mode in MODES[2:] else None
        for start in range(0, b, chunk):
            count = min(b - start, chunk)
            for stage in ("scan", "mlp") if work is not None else ("scan",):
                cfg = config(mode, count, h, w, m, coarse, stage)
                driver, fn = function(
                    states.device.index, mode, h, w, m, coarse, states.dtype, stage
                )
                args = [
                    *(scans._ptr(p) for p in [*inputs, real, imag]),
                    ct.c_int(b),
                    ct.c_int(not planar),
                    ct.c_float(epsilon),
                    ct.c_void_p(work.data_ptr() if work is not None else 0),
                    ct.c_int(start),
                    ct.c_int(count),
                    ct.c_int(stride),
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
    if mode not in MODES:
        raise ValueError("Unknown Wave 59B mode")
    if precise:
        raise ValueError("Wave 59B supports precise=False")
    out = _packed(pole, states, auxiliary, path, coarse, epsilon, mode, planar or not packed)
    return out if packed else out.unbind(0)
