"""Packed state handoffs for fused Wave 57 BF16 scans on Hopper.

Set the production w57_lockin preset before planning. Horizontal forward
returns (words, row_scales). Vertical backward returns four pole gradients,
words, column_scales and the Path gradients. BF16 words use a uint64 tensor
for the CUDA uint2 storage and both scale slots are None.
"""

from functools import lru_cache
import ctypes as ct
from pathlib import Path
import tempfile
import subprocess

import torch


from . import general_cuda_scans as scans


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"unexpected wave57 source around {old!r}")
    return source.replace(old, new)


_HELPERS = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#error Packed Wave 57 scan requires Hopper
#endif
__device__ unsigned packed_shuffle(unsigned mask,unsigned v,int lane) {
    return __shfl_sync(mask,v,lane);
}
__device__ uint2 packed_shuffle(unsigned mask,uint2 v,int lane) {
    return make_uint2(__shfl_sync(mask,v.x,lane),__shfl_sync(mask,v.y,lane));
}
__device__ __forceinline__ unsigned fp8_pair(Complex v, float inv) {
#if __CUDA_ARCH__ >= 900
    unsigned short bits;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;" : "=h"(bits) : "f"(v.i*inv),"f"(v.r*inv));
    return bits;
#else
    asm("trap;");
    return 0;
#endif
}
__device__ __forceinline__ unsigned bf_pair(Complex v) {
    auto packed=__floats2bfloat162_rn(v.r,v.i);
    return *reinterpret_cast<unsigned*>(&packed);
}
__device__ __forceinline__ Complex unpack_pos(unsigned bits) {
    float2 z=__bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&bits));
    return {z.x,z.y};
}
__device__ __forceinline__ float component_max(Complex z) {
    return fmaxf(fabsf(z.r),fabsf(z.i));
}
__device__ __forceinline__ float row_inverse(float amax, float* scales, long long slot, int width) {
    unsigned mask=__activemask();
    for(int step=width/2; step; step/=2) {
        float peer=__shfl_xor_sync(mask,amax,step,width);
        if(mask & (1U<<((threadIdx.x&31)^step))) amax=fmaxf(amax,peer);
    }
    float ratio=fmaxf(amax,1.e-12f)*(1.f/112.f);
    unsigned bits=(__float_as_uint(ratio)+0x7fffffU)&0x7f800000U;
    if ((threadIdx.x&(width-1))==0) scales[slot]=__uint_as_float(bits);
    return __uint_as_float(0x7f000000U-bits);
}
"""


def horizontal_forward_source(key, variant, width):
    """Add a trailing float* row-scale argument; out is packed storage."""
    if key.backward or key.vertical or variant not in ("packed_fp8", "packed_bf16"):
        raise ValueError("expected packed horizontal forward")
    source = scans._kernel_source(key)
    start = source.rindex("template<", 0, source.index("void horizontal("))
    end = source.index("template<bool PRECISE>", start)
    body = source[start:end]
    vector = key.horizontal_vec2
    if not vector and not key.cache_line:
        raise ValueError("scalar forward requires wave57 row staging")
    if vector and not key.cache_line and width != 7:
        raise ValueError("register staging requires W=7")
    body = _replace(body, "int M, int workers", "int M, int workers, float* scales")
    mode = "m+q" if vector else "m"
    slot = f"((line/H)*((M+31)/32)+({mode})/32)"
    offset = f"((({slot}*W+x)*H+line%H)*32+(({mode})&31))"
    fp8 = variant == "packed_fp8"
    store = (
        f"reinterpret_cast<unsigned*>(out)[{offset}]=fp8_pair(posval,inv)|(fp8_pair(negval,inv)<<16);"
        if fp8
        else f"reinterpret_cast<uint2*>(out)[{offset}]=make_uint2(bf_pair(posval),bf_pair(negval));"
    )
    reduce = (
        f"inv=row_inverse(amax,scales,{slot.replace('m+q', 'm')}*H+line%H,{16 if vector else 32});"
        if fp8
        else ""
    )
    if vector:
        if key.cache_line:
            first = body.index("auto* slot=")
            last = body.index("if constexpr (BACKWARD)", first)
            body = body[:first] + "inputs(x,r,i);\n                    " + body[last:]
            staged = "reinterpret_cast<uint2*>(shared)[I(x)*threads+threadIdx.x]"
        else:
            body = _replace(
                body,
                "Complex right[2]{},left[2]{};",
                "uint2 regs[7]; Complex right[2]{},left[2]{};",
            )
            body = _replace(
                body,
                "#pragma unroll 4\n                for (int t=0;t<W;++t)",
                "#pragma unroll\n                for (int t=0;t<7;++t)",
            )
            body = _replace(body, "int x=pass?W-1-t:t;", "int x=pass?6-t:t;")
            staged = "regs[x]"
        body = _replace(
            body,
            "Complex right[2]{},left[2]{};",
            "float amax=0,inv=1; Complex right[2]{},left[2]{};",
        )
        marker = "for (int pass=0;pass<2;++pass) {"
        body = _replace(body, marker, marker + " if(pass){" + reduce + "}")
        first = body.index("if (!BACKWARD || pass) {")
        last = body.index("\n                    }", first) + len("\n                    }")
        body = (
            body[:first]
            + f"""if(!pass) {{
                        {staged}=make_uint2(bf_pair(output[0]),bf_pair(output[1]));
                        amax=fmaxf(amax,fmaxf(component_max(output[0]),component_max(output[1])));
                    }} else {{
                        auto p={staged};
                        #pragma unroll
                        for(int q=0;q<2;++q) {{
                            Complex posval=unpack_pos(q?p.y:p.x), negval=output[q];
                            {store}
                        }}
                    }}"""
            + body[last:]
        )
    else:
        body = _replace(body, "Complex pos{}, neg{};", "float amax=0,inv=1; Complex pos{}, neg{};")
        body = _replace(body, "if constexpr (CACHE) cache[I(x) * threads] = input;", "")
        body = _replace(
            body,
            "store_complex(out, base + I(x) * M, plane, pos);",
            "cache[I(x)*threads]=Pair<T>::pack(pos); amax=fmaxf(amax,component_max(pos));",
        )
        body = _replace(
            body,
            "for (int x = W - 1; x >= 0; --x) {\n                    const auto s = CACHE ? cache[I(x) * threads].unpack() : source(x);",
            reduce + "\n                for (int x=W-1;x>=0;--x) { const auto s=source(x);",
        )
        body = _replace(
            body,
            "store_complex(out, 2 * plane + base + I(x) * M, plane, neg);",
            "Complex posval=cache[I(x)*threads].unpack(),negval=neg; " + store,
        )
    tail = source[end:]
    tail = _replace(
        tail,
        "int B, int H, int W, int M, int workers\n) {\n    horizontal",
        "int B, int H, int W, int M, int workers, float* scales\n) {\n    horizontal",
    )
    tail = _replace(tail, "B, H, W, M, workers);\n}", "B, H, W, M, workers, scales);\n}")
    if vector:
        first = body.index(
            "                        #pragma unroll\n                        for(int q=0;q<2;++q)"
        )
        last = body.index("\n                        }", first) + len("\n                        }")
        kind = "unsigned" if fp8 else "uint2"
        encode = (
            "fp8_pair(unpack_pos(POS),inv)|(fp8_pair(output[Q],inv)<<16)"
            if fp8
            else "make_uint2(POS,bf_pair(output[Q]))"
        )
        a = encode.replace("POS", "p.x").replace("Q", "0")
        b = encode.replace("POS", "p.y").replace("Q", "1")
        body = (
            body[:first]
            + f"""
                        {kind} first={a},second={b};
                        const unsigned mask=__activemask();
                        const int active=min(32,(M-(m-2*lane))/2);
                        #pragma unroll
                        for(int phase=0;phase<2;++phase) {{
                            const int logical=phase*active+lane;
                            const int mode=m-2*lane+logical;
                            const auto lo=packed_shuffle(mask,first,logical/2);
                            const auto hi=packed_shuffle(mask,second,logical/2);
                            const I tile=(line/H)*((M+31)/32)+mode/32;
                            reinterpret_cast<{kind}*>(out)[((tile*W+x)*H+line%H)*32+(mode&31)]
                                =(logical&1)?hi:lo;
                        }}"""
            + body[last:]
        )
    if vector and not key.cache_line:
        body = body.replace(
            "uint2 regs[7];", "uint2 " + ",".join(f"p{i}{{}}" for i in range(7)) + ";"
        )
        # Keep the seven rows in scalar registers. The reverse sweep pops the
        # forward values in LIFO order and never indexes a local array.
        body = body.replace(
            "regs[x]=make_uint2(bf_pair(output[0]),bf_pair(output[1]));",
            "uint2 tmp=make_uint2(bf_pair(output[0]),bf_pair(output[1]));"
            + "".join(f"p{i}=p{i - 1};" for i in range(6, 0, -1))
            + "p0=tmp;",
        )
        body = body.replace(
            "auto p=regs[x];", "uint2 p=p0;" + "".join(f"p{i}=p{i + 1};" for i in range(6))
        )
        marker = "const I base=line*W*M+m, gb=line*(W/(1+COMPACT))*M+m;"
        body = _replace(
            body,
            marker,
            marker + " auto rowbase=((line/H)*((M+31)/32)+(m/32))*W*H*32+(line%H)*32+(m&31);",
        )
        body = body.replace(
            "((((line/H)*((M+31)/32)+(m+q)/32)*W+x)*H+line%H)*32+((m+q)&31)", "rowbase+x*H*32+q"
        )
        body = body.replace(
            "#pragma unroll\n                for (int t=0;t<7;",
            "#pragma unroll 1\n                for (int t=0;t<7;",
        )
    return source[:start] + _HELPERS + body + tail


def _packed_helpers(variant):
    if variant == "packed_bf16":
        return (
            _HELPERS
            + r"""
using Word=uint2;
__device__ __forceinline__ Word load_word(const Word* ptr) {
    Word v;
    asm volatile("ld.volatile.global.v2.u32 {%0,%1},[%2];":"=r"(v.x),"=r"(v.y):"l"(ptr));
    return v;
}
__device__ Complex decode(Word word,bool hi,bool grad,float scale) {
    return unpack_pos(hi?word.y:word.x);
}
__device__ Word gradient_word(Complex a,Complex b,float inv) {
    return make_uint2(bf_pair(a),bf_pair(b));
}
"""
        )
    if variant != "packed_fp8":
        raise ValueError(variant)
    return (
        _HELPERS
        + r"""
using Word=unsigned;
__device__ __forceinline__ Word load_word(const Word* ptr) {
    Word v;
    asm("ld.global.u32 %0,[%1];":"=r"(v):"l"(ptr));
    return v;
}
__device__ Complex decode(Word word,bool hi,bool grad,float scale) {
    unsigned short bits=hi?word>>16:word;
    unsigned halves;
    if(grad) asm("cvt.rn.f16x2.e5m2x2 %0,%1;" : "=r"(halves) : "h"(bits));
    else asm("cvt.rn.f16x2.e4m3x2 %0,%1;" : "=r"(halves) : "h"(bits));
    float2 z=__half22float2(*reinterpret_cast<__half2*>(&halves));
    return {z.x*scale,z.y*scale};
}
__device__ unsigned grad_pair(Complex z,float inv) {
    unsigned short bits;
    asm("cvt.rn.satfinite.e5m2x2.f32 %0,%1,%2;" : "=h"(bits) : "f"(z.i*inv),"f"(z.r*inv));
    return bits;
}
__device__ Word gradient_word(Complex a,Complex b,float inv) {
    return grad_pair(a,inv)|(grad_pair(b,inv)<<16);
}
"""
    )


def vertical_source(key, variant):
    """Append states and adjoint scales to the wave57 vertical ABI."""
    if not key.vertical or key.pure_vertical:
        raise ValueError("expected fused vertical scan")
    s = scans._kernel_source(key)
    first = s.index("template<class T, class I>\nstruct Column")
    s = s[:first] + _packed_helpers(variant) + s[first:]
    first = s.index("    __device__ Quad<T> packed_source(int y)")
    last = s.index("    __device__ float2 gain(", first)
    s = (
        s[:first]
        + r"""
    I tile;
    int xa,xb,x;
    const float* scales;
    float* ds_scales;
    float ds_scale,inv_ds;
    __device__ I idx(int x,int y,int width) const {
        return ((tile*width+x)*H+y)*32+(m&31);
    }
    __device__ float4 source(int y) const {
        const Word* words=reinterpret_cast<const Word*>(states);
        const float scale=sizeof(Word)==4?__ldg(scales+tile*H+y):1.f;
        auto lo=load_word(words+idx(xa,y,W));
        Complex a=decode(lo,false,false,scale);
        Complex b=decode(COARSE_IO?load_word(words+idx(xb,y,W)):lo,true,false,scale);
        return make_float4(a.r,a.i,b.r,b.i);
    }
    __device__ Quad<T> packed_source(int y) const {return Quad<T>::pack(source(y));}
"""
        + s[last:]
    )
    s = s.replace("COARSE_IO", str(key.coarse).lower())
    s = s.replace(
        "long long input_plane, long long grad_plane\n",
        "long long input_plane, long long grad_plane, const float* scales, float* ds_scales\n",
    )
    s = _replace(s, "input_plane, grad_plane);", "input_plane, grad_plane, scales, ds_scales);")
    first = s.index("Column<T, I> c{states, vy,")
    last = s.index("};", first)
    s = s[:last] + ", I(b)*((M+31)/32)+m/32,xa,xb,x,scales,ds_scales,1.f,1.f" + s[last:]
    if not key.backward:
        return s
    if key.coarse and not key.compact:
        raise ValueError("coarse gradient requires compact storage")
    if variant == "packed_fp8":
        first = s.index("void column_backward(")
        last = s.index("#undef CACHED_SOURCE_AT", first)
        adj = s[first:last]
        gather = r"""
            auto observe=[&](float2 value) {
                mx=fmaxf(mx,fmaxf(fabsf(value.x),fabsf(value.y)));
                return value;
            };
"""
        pos = adj.index("    const auto ar")
        adj = adj[:pos] + "    float mx=0;" + gather + adj[pos:]
        finish = r"""
    c.inv_ds=row_inverse(mx*(1.f/128.f),c.ds_scales,c.tile*c.OW+c.x,32);
    c.ds_scale=__uint_as_float(0x7f000000U-__float_as_uint(c.inv_ds));
"""
        import re

        dy = r"make_float2\(read\(gr\[\(cell - c.m\) \* \(1 \+ REAL_INTERLEAVED\) \+ c.m\]\),\s*read\(gi\[\(cell - c.m\) \* \(1 \+ IMAG_INTERLEAVED\) \+ c.m\]\)\)"
        if "column_adjoint_positive<T" in adj:
            adj, n = re.subn(dy, lambda m: "observe(" + m[0] + ")", adj)
            assert n == 2
            adj = _replace(adj, "    adj_a = adj_b = {};", finish + "    adj_a = adj_b = {};")
        else:
            # g2 is dead until each row's Path derivative replaces its dy pair.
            cached = r"""auto cached_dy=[&](int yo) {
                Complex d=g2[I(yo)*g2_stride].a.unpack();
                return make_float2(d.r,d.i);
            };
"""
            adj, n = re.subn(dy, "cached_dy(COARSE ? y/2 : y)", adj)
            assert n == 2
            adj = _replace(
                adj,
                "    Complex adj_a{}, adj_b{};",
                finish + cached + "    Complex adj_a{}, adj_b{};",
            )
            adj = _replace(
                adj,
                "            float4 source;",
                r"""
            int y=y0+j;
            if (!COARSE || (y&1)) {
                int yo=y/(1+COARSE);
                I cell=c.cell(yo);
                float2 d=observe(make_float2(read(gr[(cell-c.m)*(1+REAL_INTERLEAVED)+c.m]),
                                             read(gi[(cell-c.m)*(1+IMAG_INTERLEAVED)+c.m])));
                g2[I(yo)*g2_stride].a=Pair<T>::pack({d.x,d.y});
            }
            float4 source;""",
            )
        s = s[:first] + adj + s[last:]
    start = s.index("    const I row", s.index("void write_column_gradient("))
    end = s.index("    if constexpr (REVERSE)", start)
    s = (
        s[:start]
        + r"""    Word* words=reinterpret_cast<Word*>(gs);
    const auto slot=c.idx(c.x,y,c.OW);
"""
        + s[end:]
    )
    s = s.replace(
        "load_complex(gs, ia, plane)", "decode(words[slot],false,true,c.ds_scale)"
    ).replace("load_complex(gs, ib, plane)", "decode(words[slot],true,true,c.ds_scale)")
    s = _replace(
        s,
        "store_complex(gs, ia, plane, ds_a);\n    store_complex(gs, ib, plane, ds_b);",
        "words[slot]=gradient_word(ds_a,ds_b,c.inv_ds);",
    )
    return s


def horizontal_backward_source(key, variant, width):
    import re

    if not key.backward or key.vertical:
        raise ValueError("expected horizontal backward")
    s = scans._kernel_source(key)
    first = s.rindex("template<", 0, s.index("void horizontal("))
    last = s.index("template<bool PRECISE>", first)
    body = s[first:last]
    params = "int M, int workers, const float* scales, const float* ds_scales"
    body = _replace(body, "int M, int workers", params)
    insert = body.index("{", body.index("for (I line")) + 1
    body = body[:insert] + f"\n            W={width};\n" + body[insert:]
    body = (
        body[:insert]
        + r"""
            auto fetch=[&](int x,int q,bool gradword,bool high)->Complex {
                const int dw=gradword?W/(1+COMPACT):W;
                const int col=gradword?x/(1+COMPACT):x;
                if(COMPACT && W%2 && col>=dw) return {};
                const I tile=(line/H)*((M+31)/32)+(m+q)/32;
                auto words=reinterpret_cast<const Word*>(gradword?grad:states);
                const int y=line%H;
                float scale=1;
                if constexpr(sizeof(Word)==4)
                    scale=gradword?ds_scales[tile*dw+col]:scales[tile*H+y];
                return decode(load_word(words+((tile*dw+col)*H+y)*32+((m+q)&31)),high,gradword,scale);
            };
"""
        + body[insert:]
    )
    if key.horizontal_vec2:
        start = body.index(
            "                    if constexpr (BACKWARD) {\n                        if (!COMPACT"
        )
        end = body.index("                    Complex output[2]", start)
        body = (
            body[:start]
            + r"""
                    auto get2=[&](int column,bool gradient,bool hi,V& re,V& im) {
                        const unsigned mask=__activemask();
                        const int active=min(32,(M-(m-2*lane))/2);
                        int dw=gradient?W/(1+COMPACT):W;
                        int col=gradient?column/(1+COMPACT):column;
                        Word raw[2];

                        #pragma unroll
                        for(int j=0;j<2;++j) {
                            int mode=m-2*lane+lane+j*active;
                            I tile=(line/H)*((M+31)/32)+mode/32;
                            const Word* base=reinterpret_cast<const Word*>(gradient?grad:states);
                            raw[j]=load_word(base+((tile*dw+col)*H+line%H)*32+(mode&31));

                        }
                        #pragma unroll
                        for(int j=0;j<2;++j) {
                            int logical=2*lane+j;
                            int upper=logical>=active;
                            int peer=logical-upper*active;
                            Word a=packed_shuffle(mask,raw[0],peer),b=packed_shuffle(mask,raw[1],peer);
                            float factor=1;
                            if constexpr(sizeof(Word)==4) {
                                I own=(line/H)*((M+31)/32)+m/32;
                                factor=gradient?ds_scales[own*dw+col]:scales[own*H+line%H];
                            }
                            auto val=decode(upper?b:a,hi,gradient,factor);
                            if(j){re.i=val.r;im.i=val.i;}else{re.r=val.r;im.r=val.i;}
                        }
                    };
                    if (!COMPACT || ((x&1)==pass)) get2(x,true,!pass,gr,gim);
                    if (pass && x) get2(x-1,false,false,pr,pi);
                    if (pass && x+1<W) get2(x+1,false,true,nr,ni);
"""
            + body[end:]
        )
    else:

        def replacement(match):
            pointer, expr = match.groups()
            high = "true" if "2" in expr else "false"
            if pointer == "grad":
                return f"fetch(x,0,true,{high})"
            return "fetch(x+1,0,false,true)" if high == "true" else "fetch(x-1,0,false,false)"

        body = re.sub(
            r"load_complex(?:_readonly)?\((states|grad),\s*([^,]+),\s*(?:grad_)?plane\)",
            replacement,
            body,
        )
    tail = s[last:].replace(
        "int M, int workers\n) {\n    horizontal", params + "\n) {\n    horizontal"
    )
    tail = _replace(tail, "B, H, W, M, workers);\n}", "B, H, W, M, workers, scales, ds_scales);\n}")
    return s[:first] + _packed_helpers(variant) + body + tail


def compile_flags(key, variant, width):
    flags = ["--fmad=" + str(not key.vertical or key.backward or key.vector_weights).lower()]
    if (
        key.backward
        and not key.vertical
        and key.compact
        and width == 28
        and variant == "packed_fp8"
    ):
        flags.append("--maxrregcount=48")
    if not key.vertical and variant == "packed_fp8":
        if key.backward and key.compact and width == 56:
            flags.append("-Xptxas=--register-usage-level=0")
        if not key.backward and width == 7:
            flags.append("--maxrregcount=40")
    return flags


def kernel_source(key, variant, width, height=None, modes=None, full_rows=False):
    if key.dtype != torch.bfloat16 or key.reduction or key.projection:
        raise ValueError("expected a BF16 fused scan launch")
    if key.vertical:
        source = vertical_source(key, variant)
    elif key.backward:
        source = horizontal_backward_source(key, variant, width)
    else:
        source = horizontal_forward_source(key, variant, width)
    if key.vertical and not key.backward and variant == "packed_fp8" and height is not None:
        marker = "    const int ow = COARSE ? W / 2 : W, oh = COARSE ? H / 2 : H;"
        source = _replace(
            source,
            marker,
            marker
            + f"""
    __shared__ float scale_cache[{height}];
    const int first_batch=(group*WARPS)/ow;
    for(int y=threadIdx.x;y<H;y+=threads)
        scale_cache[y]=scales[(I(first_batch)*tiles+mode0/32)*H+y];
    __syncthreads();""",
        )
        source = _replace(
            source,
            "x,scales,ds_scales,1.f,1.f",
            "x,scales,ds_scales,1.f,1.f,scale_cache,b==first_batch",
        )
        source = _replace(
            source,
            "float ds_scale,inv_ds;",
            "float ds_scale,inv_ds; const float* shared_row; bool cached;",
        )
        source = _replace(
            source,
            "sizeof(Word)==4?__ldg(scales+tile*H+y):1.f",
            "cached?shared_row[y]:__ldg(scales+tile*H+y)",
        )
    if (
        height is not None
        and modes is not None
        and (
            not key.vertical
            or (variant == "packed_fp8" and height == 14 and key.coarse and not key.backward)
        )
    ):
        if modes % 64 == 0:
            source = source.replace("min(32,(M-(m-2*lane))/2)", "32")
        elif modes % 64 == 32:
            source = source.replace("min(32,(M-(m-2*lane))/2)", f"((m-2*lane)<{modes - 32}?32:16)")
        if key.horizontal_vec2 and modes % 32 == 0:
            # Each half-warp supplies one component of each adjacent pair.
            # One cross-half exchange makes both stores (or loads) contiguous.
            if not key.backward:
                first = source.index(
                    "                        #pragma unroll\n                        for(int phase"
                )
                last = source.index("                        }", first) + len(
                    "                        }"
                )
                kind = "unsigned" if variant == "packed_fp8" else "uint2"
                source = (
                    source[:first]
                    + f"""
                        const int half=active/2;
                        const bool lower=lane<half;
                        auto exchanged=packed_shuffle(mask,lower?second:first,lane^half);
                        const int mode0=m-2*lane+2*(lane&(half-1))+int(!lower);
                        #pragma unroll
                        for(int phase=0;phase<2;++phase) {{
                            const int mode=mode0+phase*active;
                            const I tile=(line/H)*((M+31)/32)+mode/32;
                            reinterpret_cast<{kind}*>(out)[((tile*W+x)*H+line%H)*32+(mode&31)]
                                =phase?(lower?exchanged:second):(lower?first:exchanged);
                        }}"""
                    + source[last:]
                )
            else:
                source = _replace(
                    source,
                    "int mode=m-2*lane+lane+j*active;",
                    "int mode=m-2*lane+2*(lane&(active/2-1))+int(lane>=active/2)+j*active;",
                )
                first = source.index(
                    "                        #pragma unroll\n                        for(int j=0;j<2;++j) {\n                            int logical="
                )
                last = source.index(
                    "\n                        }", source.index("if(j){re.i=val.r", first)
                ) + len("\n                        }")
                source = (
                    source[:first]
                    + """
                        bool lower=lane<active/2;
                        Word exch=packed_shuffle(mask,lower?raw[1]:raw[0],lane^(active/2));
                        float factor=1.f;
                        if constexpr(sizeof(Word)==4){
                            I tile=(line/H)*((M+31)/32)+m/32;
                            factor=gradient?ds_scales[tile*dw+col]:scales[tile*H+line%H];
                        }
                        auto a=decode(lower?raw[0]:exch,hi,gradient,factor);
                        auto b=decode(lower?exch:raw[1],hi,gradient,factor);
                        re={a.r,b.r}; im={a.i,b.i};"""
                    + source[last:]
                )
        if (
            not key.vertical
            and not key.backward
            and key.horizontal_vec2
            and width == 7
            and modes % 64 == 0
        ):
            first = source.index("                        const int mode0=m-2*lane")
            last = source.index("                        }", first) + len(
                "                        }"
            )
            kind = "unsigned" if variant == "packed_fp8" else "uint2"
            source = (
                source[:first]
                + f"auto* dst=reinterpret_cast<{kind}*>(out)+rowbase+x*H*32; dst[0]=lower?first:exchanged; dst[W*H*32]=lower?exchanged:second;"
                + source[last:]
            )
            first = source.index("auto rowbase=")
            last = source.index(";", first)
            source = (
                source[:first]
                + "auto rowbase=(((line/H)*((M+31)/32)+2*(blockIdx.x%tiles))*W*H+line%H)*32+2*(lane&15)+(lane>>4)"
                + source[last:]
            )
            if variant == "packed_bf16" and full_rows:
                # workers == B*H; the enclosing active predicate selects exactly one row.
                source = _replace(
                    source, "for (I line=worker; line<I(B)*H; line+=workers) {", "{I line=worker;"
                )
            if variant == "packed_fp8":
                source = source.replace(
                    "const unsigned mask=__activemask();", "const unsigned mask=0xffffffffU;"
                )
        index = source.rindex(") {") + 3
        constants = f"H={height}; M={modes};"
        if (
            variant == "packed_bf16"
            and full_rows
            and width == 7
            and not key.vertical
            and not key.backward
        ):
            constants += " workers=B*H; W=7;"
        source = source[:index] + "\n " + constants + "\n" + source[index:]
    if variant == "packed_fp8" and (modes is None or modes % 32 not in (0, 1)):
        # In an incomplete warp only its leader is guaranteed to see every
        # active lane during the XOR reduction. Broadcast before encoding.
        source = _replace(
            source,
            "    float ratio=fmaxf(amax,1.e-12f)",
            "    amax=__shfl_sync(mask,amax,0,width);\n    float ratio=fmaxf(amax,1.e-12f)",
        )
    return source


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
            driver.cuModuleGetFunction(ct.byref(fn), module, scans._kernel_name(key).encode()),
            "get packed function",
        )
        scans._configure_smem(driver, fn, smem)
        return driver, module, fn


def _dispatch(real, launch, variant, width, args):
    device = real.device.index or 0
    key = launch.key
    driver, module, fn = _function(
        device,
        key,
        variant,
        width,
        real.shape[1],
        real.shape[-1],
        launch.workers == real.shape[0] * real.shape[1],
        launch.plan.smem,
    )
    grid = scans._ceil(real.shape[-1], 32 * (1 + key.horizontal_vec2)) * scans._ceil(
        launch.workers, key.warps
    )
    argv = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
    with torch.cuda.device(device):
        stream = ct.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        scans._check_cu(
            driver.cuLaunchKernel(
                fn, grid, 1, 1, 32 * key.warps, 1, 1, launch.plan.smem, stream, argv, None
            ),
            "launch packed scan",
        )


def _check_source(real, variant):
    if variant not in ("packed_fp8", "packed_bf16"):
        raise ValueError("unknown packed representation")
    if real.ndim != 4 or not real.is_cuda or real.dtype != torch.bfloat16:
        raise ValueError("expected CUDA BF16 [B,H,W,M] source")
    if torch.cuda.get_device_capability(real.device) != (9, 0):
        raise ValueError("packed kernels require SM90a")


def _check_storage(real, storage, variant, width, states):
    words, scale = storage
    b, h, _, m = real.shape
    tiles = (m + 31) // 32
    dtype = torch.uint32 if variant == "packed_fp8" else torch.uint64
    if (
        tuple(words.shape) != (b, tiles, width, h, 32)
        or words.dtype != dtype
        or words.device != real.device
        or not words.is_contiguous()
    ):
        raise ValueError("invalid packed words")
    if variant == "packed_fp8" and (
        scale is None
        or tuple(scale.shape) != (b, tiles, h if states else width)
        or scale.dtype != torch.float32
        or not scale.is_contiguous()
        or scale.device != real.device
    ):
        raise ValueError("invalid FP8 scales")


def _words(real, variant, width):
    b, h, _, m = real.shape
    return torch.empty(
        (b, (m + 31) // 32, width, h, 32),
        device=real.device,
        dtype=torch.uint32 if variant == "packed_fp8" else torch.uint64,
    )


def horizontal(pole, r, i, variant, states=None, adjoints=None, compact=False):
    _check_source(r, variant)
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
        for t, a, z in zip((r, i), layout["interleaved"], layout["transposed"])
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
    partial = scratch = None
    if backward:
        _check_storage(r, states, variant, w, True)
        _check_storage(r, adjoints, variant, w // (1 + compact), False)
        q, s = states
        dq, ds = adjoints
        out = r.new_empty((b, h, w, 2 * m))
        partial = r.new_empty((launch.partials, 4, m), dtype=torch.float32)
        if not launch.plan.cache_line:
            scratch = r.new_empty((w // (1 + compact), launch.workers, m, 2), dtype=torch.float32)
        extra = (s, ds)
    else:
        q = dq = None
        out = _words(r, variant, w)
        s = (
            r.new_empty((b, (m + 31) // 32, h), dtype=torch.float32)
            if variant == "packed_fp8"
            else None
        )
        extra = (s,)
    inputs = (*(t.float().contiguous() for t in pole), r, i, q, dq, out, partial, scratch)
    _dispatch(
        r,
        launch,
        variant,
        w,
        (
            *map(scans._ptr, inputs),
            *map(ct.c_int, (*shape, launch.workers)),
            *map(scans._ptr, extra),
        ),
    )
    if backward:
        return [*scans._reduce_partials(partial, pole), out]
    return out, s


def vertical(pole, r, states, aux, path, coarse, variant, grad=None):
    _check_source(r, variant)
    b, h, w, m = r.shape
    _check_storage(r, states, variant, w, True)
    logical = r.unsqueeze(0).expand(4, -1, -1, -1, -1)
    if (
        not scans._pole_ok(pole, r)
        or not scans._vertical_inputs_ok(logical, aux, path, coarse=coarse)
        or path[0].shape[1] != 8
    ):
        raise ValueError("invalid vertical scan inputs")
    q, s = states
    backward = grad is not None
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    gr, gi = scans._vertical_adjoints(grad, logical, (b, oh, ow, m)) if backward else (None, None)
    interleaved = (not gr.is_contiguous(), not gi.is_contiguous()) if backward else (True, True)
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
    outr = outi = dq = ds = partial = scratch = checkpoints = None
    if backward:
        dq = _words(r, variant, ow)
        ds = (
            r.new_empty((b, (m + 31) // 32, ow), dtype=torch.float32)
            if variant == "packed_fp8"
            else None
        )
        partial = r.new_empty((launch.partials, 94, m), dtype=torch.float32)
        ck = scans._checkpoint_rows(h, launch.plan.chunk)
        if launch.plan.global_checkpoints and ck:
            checkpoints = r.new_empty((ck, launch.workers, m, 4), dtype=torch.float32)
    else:
        out = r.new_empty((b, oh, ow, 2 * m))
        outr, outi = out.split(m, -1)
    if not launch.plan.cache_line:
        scratch = r.new_empty((oh + h * launch.key.l2_single_write, launch.workers, m, 4))
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
        ct.c_float(1e-6),
        ct.c_longlong(0),
        ct.c_longlong(0),
        scans._ptr(s),
        scans._ptr(ds),
    )
    _dispatch(r, launch, variant, w, args)
    if backward:
        params = scans._reduce_partials(partial, [*pole, *path], 8)
        return [*params[:4], dq, ds, *params[4:]]
    return out
