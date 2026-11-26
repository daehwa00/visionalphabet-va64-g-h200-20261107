"""Opt-in Hopper vertical scan backward kernels. All launches use native SM90a cubins."""

from __future__ import annotations

import ctypes as ct
import math
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch
from torch import Tensor

from . import general_cuda_scans as scans
from .cuda_nvrtc import _CudaModule

MODES = (
    "w60_fp8_smem_fp16acc",
    "w60_fp8_fullcache_1read",
    "w60_fp8_smem_fastsilu",
    "w60_decoupled_fp8_tc",
)


def padded(n):
    return (n + 15) // 16 * 16


def chunk_size(mode, b, cells, m):
    if mode != MODES[3]:
        return b
    # A 16-cell block uses 320 data bytes and eight scale bytes per mode.
    blocks = (32 << 20) // (328 * m)
    return min(b, max((b + 31) // 32, blocks * 16 // max(1, cells)))


@dataclass(frozen=True)
class Kernel:
    name: str
    h: int
    w: int
    m: int
    coarse: bool
    dtype: torch.dtype
    wide: bool
    full: bool
    fast: bool
    compact: bool
    threads = 128

    @property
    def smem(self):
        if self.name == "tc_path_mlp_bwd_fp8":
            return 3072
        if self.name == "pure_vscan_bwd_gs_fp8":
            return 4 * 4096 + 80 * 128
        if self.name == "pure_vscan_fwd_p_fp8":
            return 8192 + 4096 + 2048 + (self.h // (1 + self.coarse)) * 128 * 4
        span = self.h if self.full else 16
        return 5632 + 128 * (
            8 * span + 4 * (self.h // (1 + self.coarse)) + 4 * ((self.h - 1) // span) + 40
        )


def kernels(mode, shape, dtype, coarse, compact=False):
    if mode not in MODES or (compact and not coarse):
        raise ValueError("Invalid vertical scan policy")
    b, h, w, m = shape
    stages = ("pure_vscan_fwd_p_fp8", "tc_path_mlp_bwd_fp8", "pure_vscan_bwd_gs_fp8")
    return tuple(
        Kernel(
            name,
            h,
            w,
            m,
            coarse,
            dtype,
            4 * b * h * w * m >= 2**31,
            mode == MODES[1] or h <= 28,
            mode == MODES[2],
            compact,
        )
        for name in (stages if mode == MODES[3] else ("column_fp8",))
    )


_TC = r"""
__device__ unsigned load2(const T *p) { return *reinterpret_cast<const unsigned *>(p); }
__device__ unsigned packed(T x, T y) {
    auto v = Pair<T>::from(x, y);
    return *reinterpret_cast<unsigned *>(&v);
}
__device__ void mma(float (&v)[4], unsigned a, unsigned b, unsigned c, unsigned d, unsigned e,
                    unsigned f) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.FMT.FMT.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(v[0]), "+f"(v[1]), "+f"(v[2]), "+f"(v[3])
        : "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f));
}
// Transpose an 8x8 halfword MMA fragment without scalar shared accesses.
__device__ unsigned transpose8(unsigned value) {
    int lane = threadIdx.x % 32, group = lane / 4;
    int owner = (lane % 4) * 8 + group / 2;
    auto lo = __shfl_sync(~0u, value, owner);
    auto hi = __shfl_sync(~0u, value, owner + 4);
    return __byte_perm(lo, hi, group % 2 ? 0x7632 : 0x5410);
}
__device__ void read_pair(T (&out)[2], const T *ptr) {
    unsigned u = load2(ptr);
    auto *v = reinterpret_cast<T *>(&u);
    out[0] = v[0];
    out[1] = v[1];
}
struct Weights {
    unsigned wt, w;
    T bias[2], o0[2], o1[2];
    __device__ void init(int m, const void *wi, const void *bi, const void *wo) {
        int l = threadIdx.x % 32, g = l / 4, k = 2 * (l % 4);
        const T *weights = (const T *)wi + m * 64;
        wt = load2(weights + g * 8 + k);
        w = packed(weights[k * 8 + g], weights[(k + 1) * 8 + g]);
        for (int i = 0; i < 2; i++) {
            bias[i] = ((const T *)bi)[m * 8 + k + i];
            o0[i] = ((const T *)wo)[m * 16 + k + i];
            o1[i] = ((const T *)wo)[m * 16 + 8 + k + i];
        }
    }
};
struct Acc {

    volatile float *aux;
    __device__ Acc(float *base) : aux(base + threadIdx.x * 6) {
        for (int i = 0; i < 6; i++) {
            aux[i] = 0.f;
        }
    }
    __device__ void tile(const Weights &w, unsigned a,unsigned b,const T *dy,float (&grad)[4]) {
        int l = threadIdx.x % 32, g = l / 4, k = 2 * (l % 4);
        float pre[4]{};
        mma(pre, a, b, 0, 0, w.wt, 0);
        unsigned b0 = transpose8(a), b1 = transpose8(b);
        aux[3] += read(dy[(l % 16) * 2 + l / 16]);
        unsigned dz[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            int row = g + 8 * r;
            T pair[2];
            read_pair(pair, dy + 2 * row);
            float y0 = read(pair[0]), y1 = read(pair[1]);
            T v[2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                float z = rounded<T>(rounded<T>(pre[2 * r + i]) + read(w.bias[i]));
                float sig = __fdividef(1.f, 1.f + __expf(-z)), act = rounded<T>(z * sig);
                float d = rounded<T>(rounded<T>(y0 * read(w.o0[i]) + y1 * read(w.o1[i])) *
                                     (sig * (1.f + z * (1.f - sig))));
                v[i] = write<T>(d);
                float x = y0 * act, y = y1 * act;
                float dd = __shfl_xor_sync(~0u, d, 4), xx = __shfl_xor_sync(~0u, x, 4),
                      yy = __shfl_xor_sync(~0u, y, 4);
                if ((g % 2) == i) {
                    aux[0] += d + dd;
                    aux[1] += x + xx;
                    aux[2] += y + yy;
                }
            }
            dz[r] = packed(v[0], v[1]);
        }
        mma(grad, dz[0], dz[1], 0, 0, w.w, 0);
        unsigned at = transpose8(dz[0]), bt = transpose8(dz[1]);
        float dw[4] = {aux[4], aux[5], 0, 0};
        mma(dw, at, 0, bt, 0, b0, b1);
        aux[4] = dw[0];
        aux[5] = dw[1];
    }
    // Each lane owns two W1 values and at most one bias/W2 value.
    template <int Pitch = M> __device__ void finish(float *out, bool add) {
        int l = threadIdx.x % 32, g = l / 4, k = 2 * (l % 4);
        for (int i = 0; i < 2; i++) {
            auto *v = out + (4 + g * 8 + k + i) * Pitch;
            *v = aux[4 + i] + (add ? *v : 0);
        }
        for (int f = 0; f < 3; f++) {
            float t = aux[f];
            t += __shfl_down_sync(~0u, t, 16);
            t += __shfl_down_sync(~0u, t, 8);
            if (g < 2) {
                auto *v = out + (68 + 8 * f + k + g) * Pitch;
                *v = t + (add ? *v : 0);
            }
        }
        float t = aux[3];
        for (int d = 8; d; d /= 2) {
            t += __shfl_down_sync(~0u, t, d, 16);
        }
        if (!(l % 16)) {
            auto *v = out + (92 + l / 16) * Pitch;
            *v = t + (add ? *v : 0);
        }
    }
};
"""

_ARGS = """
 const float *dr,const float *di,const float *gr0,const float *gi0,const T *states,
 const float *vx,const float *vy,const void *wi,const void *bi,const void *wo,
 const T *gr,const T *gi,T *gs,float *partial,unsigned char *P,T *DY,unsigned char *DP,float *ps,float *ds,
 int B,int start,int count,int stride,int workers,int rs,int is,bool compact_arg,float eps
"""
_COMMON = r"""
__device__ float ceil_pow2(float positive) {
    return __uint_as_float((__float_as_uint(positive) + 0x7fffffu) & 0x7f800000u);
}
__device__ float magnitude(float4 v) {
    return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
}
__device__ float4 mul4(float4 v, float s) { return {v.x * s, v.y * s, v.z * s, v.w * s}; }
template <class F> __device__ unsigned pack4(float4 x, float scale) {
    return F(mul4(x, 1.f / scale)).__x;
}
template <class F> __device__ float4 unpack4(unsigned bits, float scale) {
    F v;
    v.__x = bits;
    return mul4(static_cast<float4>(v), scale);
}
__device__ __half2 bits_to_pair(float v) { return *reinterpret_cast<__half2 *>(&v); }
__device__ float pair_to_bits(__half2 v) { return *reinterpret_cast<float *>(&v); }
using P8 = __nv_fp8x4_e4m3;
using G8 = __nv_fp8x4_e5m2;
__device__ Column<T, I> column(const T *states, const float *vx, const float *vy, int b, int x,
                               int m, int B, float eps) {
    int xb = x * STEP, xa = xb + STEP - 1;
    I plane = I(B) * H * W * M, off = I(b) * H * W * M + m;
    return {states,
            vy,
            off + xa * M,
            off + xb * M,
            plane,
            plane,
            (I(b) * OH * OW + x) * M + m,
            I(B) * OH * OW * M,
            H,
            W,
            M,
            OW,
            m,
            vx[xa * M + m],
            vx[(W + xb) * M + m],
            eps,
            (I(b) * H * OW + x) * M + m};
}
"""
_DIRECT = r"""
    extern __shared__ float4 shared[];
    int tid = threadIdx.x, lane = tid % 32, m = blockIdx.y * 32 + lane;
    T *weights = (T *)shared;
    // Load mode-major vectors once; all four column warps reuse this tile.
    for (int k = tid; k < 32 * 88 / 8; k += NT) {
        int mat = k < 256 ? 0 : k < 288 ? 1 : 2, offset = mat == 0 ? 0 : mat == 1 ? 256 : 288;
        const T *ptr = mat == 0 ? (const T *)wi : mat == 1 ? (const T *)bi : (const T *)wo;
        int width = mat == 0 ? 64 : mat == 1 ? 8 : 16;
        ((uint4 *)weights)[k] = ((const uint4 *)(ptr + blockIdx.y * 32 * width))[k - offset];
    }
    __syncthreads();
    constexpr int span = FULL ? H : 16;
    unsigned *pos = (unsigned *)(weights + 32 * 88) + tid, *src = pos + span * NT,
             *inj = src + span * NT;
    unsigned *ck = (unsigned *)(inj - tid + OH * NT) + tid;
    volatile float *auxacc = (float *)(ck - tid + ((H - 1) / span) * NT) + tid;
    Complex a{dr[m], di[m]}, g{gr0[m], gi0[m]};
    float weight_bound = 1.f;
    for (int k = 0; k < 64; k++) {
        weight_bound = fmaxf(weight_bound, fabsf(read(weights[lane * 64 + k])));
    }
    for (int k = 0; k < 8; k++) {
        weight_bound = fmaxf(weight_bound, fabsf(read(weights[2048 + lane * 8 + k])));
    }
    for (int k = 0; k < 16; k++) {
        weight_bound = fmaxf(weight_bound, fabsf(read(weights[2304 + lane * 16 + k])));
    }
    int worker = blockIdx.x * 4 + tid / 32;
    float *out = partial + (worker * 94) * M + m;
    for (int k = 0; k < 94; k++) {
        out[k * M] = 0.f;
    }
    for (int line = worker; line < count * OW; line += gridDim.x * 4) {
        auto c = column(states, vx, vy, start + line / OW, line % OW, m, B, eps);
        float4 p{}, neg{};
        float scale = 1.e-6f, col_scale = 16.f;
        Complex da{}, dg{}, aa{}, ab{};
        __half2 acc[35]{};
        for (int i = 0; i < 10; i++) {
            auxacc[i * NT] = 0;
        }
        for (int y = 0; y < H; y++) {
            auto x = c.source(y);
            p = advance(p, x, a, g);
            float need = fmaxf(magnitude(x), magnitude(p)) / 448.f;
            if (need > scale) {
                float next = ceil_pow2(need);
                if constexpr (FULL) {
                    for (int i = 0; i < y; i++) {
                        pos[i * NT] = pack4<P8>(unpack4<P8>(pos[i * NT], scale), next);
                        src[i * NT] = pack4<P8>(unpack4<P8>(src[i * NT], scale), next);
                    }
                }
                if constexpr (!FULL) {
                    for (int i = 0; i < y / span; i++) {
                        ck[i * NT] = pack4<P8>(unpack4<P8>(ck[i * NT], scale), next);
                    }
                }
                scale = next;
            }
            if constexpr (FULL) {
                pos[y * NT] = pack4<P8>(p, scale);
                src[y * NT] = pack4<P8>(x, scale);
            } else if ((y + 1) % span == 0 && y + 1 < H) {
                ck[(y / span) * NT] = pack4<P8>(p, scale);
            }
        }
        for (int block = (H - 1) / span; block >= 0; block--) {
            int lo = block * span, hi = min(H, lo + span);
            float4 before{};
            if constexpr (!FULL) {
                if (block) {
                    before = unpack4<P8>(ck[(block - 1) * NT], scale);
                }
                if (hi < H) {
                    ck[block * NT] = pack4<P8>(neg, scale);
                }
                p = before;
                for (int y = lo; y < hi; y++) {
                    auto x = c.source(y);
                    p = advance(p, x, a, g);
                    src[(y - lo) * NT] = pack4<P8>(x, scale);
                    pos[(y - lo) * NT] = pack4<P8>(p, scale);
                }
            }
            float4 oddnext{};
            for (int y = hi - 1; y >= lo; y--) {
                int j = y - lo;
                auto x = unpack4<P8>(src[j * NT], scale);
                auto next = neg;
                neg = advance(neg, x, a.conj(), g.conj());
                // Coarse projections join the positive odd and negative even rows.
                float4 ip{}, in{};
                if (y % STEP == 0 && y / STEP < OH) {
                    int yo = y / STEP;
                    I cell = c.cell(yo);
                    auto pp = normalize<T>(unpack4<P8>(pos[(j + STEP - 1) * NT], scale),
                                           c.gain(y + STEP - 1, false));
                    auto pn = normalize<T>(neg, c.gain(y, true));
                    float feat[8] = {pp.x, pp.z, pn.x, pn.z, pp.y, pp.w, pn.y, pn.w}, dp[8]{};
                    float yr = read(gr[(cell - m) * rs + m]), yi = read(gi[(cell - m) * is + m]);
                    auto pair = [](float u, float v) { return __floats2half2_rn(u, v); };
                    float max_p = fmaxf(1.f, fmaxf(magnitude(pp), magnitude(pn)));
                    // W1 has eight inputs and |SiLU'| < 1.1. Bound the
                    // largest bias, W1 or W2 sum before moving it into FP16.
                    float need =
                        (float(H) / 2048.f) * fmaxf(fabsf(yr), fabsf(yi)) * max_p * weight_bound;
                    if (need > col_scale) {
                        float next = ceil_pow2(need);
                        auto ratio = pair(col_scale / next, col_scale / next);
#pragma unroll
                        for (int n = 0; n < 35; n++) {
                            acc[n] = __hmul2(acc[n], ratio);
                        }
#pragma unroll
                        for (int n = 0; n < 10; n++) {
                            auxacc[n * NT] =
                                pair_to_bits(__hmul2(bits_to_pair(auxacc[n * NT]), ratio));
                        }
                        col_scale = next;
                    }
                    float inv = __frcp_rn(col_scale), xs = fmaxf(1.f, max_p / 16384.f);
                    auto add = [&](int k, float u, float v) {
                        auto a = k < 35 ? acc[k] : bits_to_pair(auxacc[(k - 35) * NT]);
                        auto sum = __hadd2(a, pair(u * inv, v * inv));
                        if (k < 35) {
                            acc[k] = sum;
                        } else {
                            auxacc[(k - 35) * NT] = pair_to_bits(sum);
                        }
                    };
                    add(44, yr, yi);
#pragma unroll
                    for (int u = 0; u < 8; u++) {
                        float z = 0;
#pragma unroll
                        for (int k = 0; k < 8; k++) {
                            z = fmaf(feat[k], read(weights[lane * 64 + u * 8 + k]), z);
                        }
                        z = rounded<T>(rounded<T>(z) + read(weights[2048 + lane * 8 + u]));
                        float sig, der;
                        if constexpr (FAST) {
                            float t = tanhf_fast(z * .5f);
                            sig = .5f + .5f * t;
                            der = sig + .25f * z * (1 - t * t);
                        } else {
                            sig = 1.f / (1.f + __expf(-z));
                            der = sig * (1.f + z * (1.f - sig));
                        }
                        float act = rounded<T>(z * sig);
                        float dz =
                            rounded<T>(rounded<T>(yr * read(weights[2304 + lane * 16 + u]) +
                                                  yi * read(weights[2304 + lane * 16 + 8 + u])) *
                                       der);
                        if (u % 2 == 0) {
                            add(32 + u / 2, dz, 0);
                            add(36 + u / 2, yr * act, 0);
                            add(40 + u / 2, yi * act, 0);
                        } else {
                            add(32 + u / 2, 0, dz);
                            add(36 + u / 2, 0, yr * act);
                            add(40 + u / 2, 0, yi * act);
                        }
#pragma unroll
                        for (int k = 0; k < 8; k += 2) {
                            acc[u * 4 + k / 2] =
                                __hfma2(pair(dz * inv * xs, dz * inv * xs),
                                        pair(feat[k] / xs, feat[k + 1] / xs), acc[u * 4 + k / 2]);
                            dp[k] += dz * read(weights[lane * 64 + u * 8 + k]);
                            dp[k + 1] += dz * read(weights[lane * 64 + u * 8 + k + 1]);
                        }
                    }
                    in = normalize<T>({dp[2], dp[6], dp[3], dp[7]}, c.gain(y, true));
                    ip = normalize<T>({dp[0], dp[4], dp[1], dp[5]}, c.gain(y + STEP - 1, false));
                    inj[yo * NT] = pack4<G8>(in, 1.f);
                }
                if (STEP == 2 && y % 2) {
                    oddnext = next;
                    continue;
                }
                for (int r = (y / STEP < OH ? STEP : 1) - 1; r >= 0; r--) {
                    int q = j + r;
                    auto prev = q ? unpack4<P8>(pos[(q - 1) * NT], scale) : before;
                    auto ds0 = column_adjoint_step<false>(prev, unpack4<P8>(src[q * NT], scale),
                                                          r == STEP - 1 ? ip : float4{}, a, g, aa,
                                                          ab, da, dg);
                    if (compact) {
                        write_column_gradient<false, true>(c, y + r, ds0, gs);
                    } else {
                        write_column_gradient<false, false>(c, y + r, ds0, gs);
                    }
                    if constexpr (FULL) {
                        pos[q * NT] = pack4<P8>(r ? oddnext : next, scale);
                    }
                }
            }
        }
        aa = {};
        ab = {};
        for (int block = 0; block < ((H + span - 1) / span); block++) {
            int lo = block * span, hi = min(H, lo + span);
            if constexpr (!FULL) {
                neg = hi < H ? unpack4<P8>(ck[block * NT], scale) : float4{};
                for (int y = hi - 1; y >= lo; y--) {
                    auto x = c.source(y);
                    src[(y - lo) * NT] = pack4<P8>(x, scale);
                    pos[(y - lo) * NT] = pack4<P8>(neg, scale);
                    neg = advance(neg, x, a.conj(), g.conj());
                }
            }
            for (int y = lo; y < hi; y++) {
                float4 d{};
                if (y % STEP == 0 && y / STEP < OH) {
                    d = unpack4<G8>(inj[(y / STEP) * NT], 1.f);
                }
                auto v = column_adjoint_step<true>(unpack4<P8>(pos[(y - lo) * NT], scale),
                                                   unpack4<P8>(src[(y - lo) * NT], scale), d,
                                                   a.conj(), g.conj(), aa, ab, da, dg);
                if (compact) {
                    write_column_gradient<true, true, true>(c, y, v, gs);
                } else {
                    write_column_gradient<true, false, true>(c, y, v, gs);
                }
            }
        }
        out[0] += da.r;
        out[M] += da.i;
        out[2 * M] += dg.r;
        out[3 * M] += dg.i;
#pragma unroll
        for (int k = 0; k < 45; k++) {
            float2 v = __half22float2(k < 35 ? acc[k] : bits_to_pair(auxacc[(k - 35) * NT]));
            out[(4 + 2 * k) * M] += col_scale * v.x;
            out[(5 + 2 * k) * M] += col_scale * v.y;
        }
    }
}
"""
_FWD = r"""
    extern __shared__ float4 shared[];
    T *features = (T *)shared;
    unsigned char *bytes = (unsigned char *)(features + 4096);
    T *grad = (T *)(bytes + 4096);
    int tid = threadIdx.x, l = tid % 32, m = blockIdx.y * 32 + l, blk = blockIdx.x;
    long long tile = (long long)blockIdx.y * (stride / 16) + blk;
    auto flush = [&](long long tile) {
        __syncthreads();
        for (int mode = tid / 32; mode < 32; mode += 4) {
            float4 v;
            float *vals = (float *)&v;
            for (int k = 0; k < 4; k++) {
                vals[k] =
                    read(features[(mode * 16 + ((l / 2) ^ (mode % 8))) * 8 + (l % 2) * 4 + k]);
            }
            float s = magnitude(v);
            for (int d = 16; d; d /= 2) {
                s = fmaxf(s, __shfl_xor_sync(~0u, s, d));
            }
            s = fmaxf(1.e-6f, s) / 448.f;
            if (!l) {
                ps[tile * 32 + mode] = s;
            }
            ((unsigned *)bytes)[mode * 32 + l] = pack4<P8>(v, s);
        }
        __syncthreads();
        for (int i = tid; i < 256; i += NT) {
            ((uint4 *)P)[tile * 256 + i] = ((uint4 *)bytes)[i];
        }
        ((uint4 *)DY)[tile * 128 + tid] = ((uint4 *)grad)[tid];
        __syncthreads();
    };
    // Four neighboring columns contribute a 4x4 cell panel. Pack
    // leftover rows and columns into consecutive tiles after the full panels.
    int groups = OH >= 4 ? count * OW / 4 : 0, mainrows = OH / 4 * 4;
    if (blockIdx.x < groups) {
        int warp = tid / 32, line = 4 * blockIdx.x + warp;
        auto c = column(states, vx, vy, start + line / OW, line % OW, m, B, eps);
        Complex a{dr[m], di[m]}, g{gr0[m], gi0[m]};
        unsigned *history = (unsigned *)(grad + 1024) + tid;
        float4 p{}, q{};
        float sc = 1.e-6f;
        for (int y = 0; y < H; y++) {
            p = advance(p, c.source(y), a, g);
            if (y % STEP == STEP - 1 && y / STEP < OH) {
                float4 v = normalize<T>(p, c.gain(y, false));
                float need = magnitude(v) / 448.f;
                if (need > sc) {
                    float next = ceil_pow2(need);
                    for (int i = 0; i < y / STEP; i++) {
                        history[i * NT] = pack4<P8>(unpack4<P8>(history[i * NT], sc), next);
                    }
                    sc = next;
                }
                history[(y / STEP) * NT] = pack4<P8>(v, sc);
            }
        }
        for (int y = H - 1; y >= 0; y--) {
            q = advance(q, c.source(y), a.conj(), g.conj());
            if (y % STEP || y / STEP >= mainrows) {
                continue;
            }
            int oy = y / STEP, r = oy % 4 * 4 + warp;
            auto u = unpack4<P8>(history[oy * NT], sc), v = normalize<T>(q, c.gain(y, true));
            float feat[8] = {u.x, u.z, v.x, v.z, u.y, u.w, v.y, v.w};
            for (int k = 0; k < 8; k++) {
                features[(l * 16 + (r ^ (l % 8))) * 8 + k] = write<T>(feat[k]);
            }
            I cell = c.cell(oy);
            grad[(l * 16 + r) * 2] = gr[(cell - m) * rs + m];
            grad[(l * 16 + r) * 2 + 1] = gi[(cell - m) * is + m];
            if (oy % 4 == 0) {
                flush((long long)blockIdx.y * (stride / 16) + blockIdx.x * (OH / 4) + oy / 4);
            }
        }
        return;
    }
    for (int j = 0; j < 4; j++) {
        int r = (tid / 32) * 4 + j, n = (blk - groups) * 16 + r;
        int tails = groups * 4 * (OH - mainrows);
        int line, oy;
        if (OH != mainrows && n < tails) {
            line = n / max(1, OH - mainrows);
            oy = mainrows + n % max(1, OH - mainrows);
        } else {
            line = groups * 4 + (n - tails) / OH;
            oy = (n - tails) % OH;
        }
        tile = (long long)blockIdx.y * (stride / 16) + groups * (OH / 4) + blk - groups;
        float feat[8]{}, yr = 0, yi = 0;
        if (line < count * OW) {
            auto c = column(states, vx, vy, start + line / OW, line % OW, m, B, eps);
            Complex a{dr[m], di[m]}, g{gr0[m], gi0[m]};
            float4 p{}, q{};
            for (int y = 0; y <= oy * STEP + STEP - 1; y++) {
                p = advance(p, c.source(y), a, g);
            }
            for (int y = H - 1; y >= oy * STEP; y--) {
                q = advance(q, c.source(y), a.conj(), g.conj());
            }
            p = normalize<T>(p, c.gain(oy * STEP + STEP - 1, false));
            q = normalize<T>(q, c.gain(oy * STEP, true));
            float v[8] = {p.x, p.z, q.x, q.z, p.y, p.w, q.y, q.w};
            for (int k = 0; k < 8; k++) {
                feat[k] = v[k];
            }
            I cell = c.cell(oy);
            yr = read(gr[(cell - m) * rs + m]);
            yi = read(gi[(cell - m) * is + m]);
        }
        for (int k = 0; k < 8; k++) {
            features[(l * 16 + (r ^ (l % 8))) * 8 + k] = write<T>(feat[k]);
        }
        grad[(l * 16 + r) * 2] = write<T>(yr);
        grad[(l * 16 + r) * 2 + 1] = write<T>(yi);
    }
    flush(tile);
}
"""
_MLP = r"""
    extern __shared__ float4 shared[];
    int warp = threadIdx.x / 32, l = threadIdx.x % 32, m = blockIdx.y * 4 + warp,
        worker = blockIdx.x;
    Acc acc((float *)shared);
    for (int blk = worker; blk < stride / 16; blk += gridDim.x) {
        int chunk = (m / 32) * (stride / 16) * 32 + blk * 32 + m % 32;
        const unsigned short *pairs=(const unsigned short*)P+chunk*64;
        auto operand=[&](int offset){
            __nv_fp8x2_e4m3 q;q.__x=pairs[offset];
            auto f=__half22float2(static_cast<__half2>(q));
            return packed(write<T>(f.x*ps[chunk]),write<T>(f.y*ps[chunk]));
        };
        Weights w; w.init(m,wi,bi,wo);
        float grad[4]{};
        acc.tile(w,operand(l),operand(32+l),DY+chunk*32,grad);
        float s=magnitude({grad[0],grad[1],grad[2],grad[3]});
        for(int d=16;d;d/=2)s=fmaxf(s,__shfl_xor_sync(~0u,s,d));
        s=fmaxf(s,1.e-6f)/57344.f;
        if(!l)ds[chunk]=s;
        auto output=(unsigned short*)DP+chunk*64+l;
        for(int i=0;i<2;i++) output[i*32]=__nv_fp8x2_e5m2(
            make_float2(grad[2*i]/s,grad[2*i+1]/s)).__x;
    }
    acc.finish(partial + worker * 94 * M + m, start != 0);
}
"""
_BWD = r"""
    extern __shared__ float4 shared[];
    int tid = threadIdx.x, l = tid % 32, warp = tid / 32, m = blockIdx.y * 32 + l;
    unsigned char *tile = (unsigned char *)shared + warp * 4096;
    Complex a{dr[m], di[m]}, g{gr0[m], gi0[m]};
    volatile float *poles = (float *)(shared + 1024) + 16 * NT + tid;
    for (int i = 0; i < 4; i++) {
        poles[i * NT] = 0;
    }
    for (int line = blockIdx.x * 4 + warp; line < count * OW; line += gridDim.x * 4) {
        auto c = column(states, vx, vy, start + line / OW, line % OW, m, B, eps);

#pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            volatile float *tangents = (float *)(shared + 1024) + tid;
            for (int k = 0; k < 16; k++) {
                tangents[k * NT] = 0;
            }
            int prev = -1;
            float scale = 0;
            for (int t = 0; t < H; t++) {
                int y = pass ? H - 1 - t : t, oy = y / STEP;
                float4 pos{}, neg{};
                if (oy < OH) {
                    int bulk = OH >= 4 ? count * OW / 4 * 4 : 0, mainrows = OH / 4 * 4;
                    int n = line >= bulk
                                ? bulk * OH + (line - bulk) * OH + oy
                                : (oy < mainrows
                                       ? (line / 4 * mainrows + oy) * 4 + line % 4
                                       : bulk * mainrows + line * (OH - mainrows) + oy - mainrows);
                    int blk = n / 16;
                    bool together = line < bulk && oy < mainrows;
                    unsigned char *buf = together ? (unsigned char *)shared : tile;
                    if (line < bulk && mainrows < OH &&
                        y == (pass ? mainrows * STEP - 1 : mainrows * STEP)) {
                        __syncthreads();
                    }
                    if (blk != prev) {
                        long long base = (long long)blockIdx.y * (stride / 16) + blk;
                        if (together) {
                            __syncthreads();
                            for (int i = tid; i < 256; i += NT) {
                                ((uint4 *)buf)[i] = ((const uint4 *)DP)[base * 256 + i];
                            }
                            __syncthreads();
                        } else {
                            __syncwarp();
                            for (int i = l; i < 256; i += 32) {
                                ((uint4 *)buf)[i] = ((const uint4 *)DP)[base * 256 + i];
                            }
                            __syncwarp();
                        }
                        prev = blk;
                        scale = ds[base * 32 + l];
                    }
                    unsigned *v = (unsigned *)buf + (l * 16 + n % 16) * 2;
                    float4 re = unpack4<G8>(v[0], scale), im = unpack4<G8>(v[1], scale);
                    if (y % STEP == STEP - 1) {
                        pos = normalize<T>({re.x, im.x, re.y, im.y}, c.gain(y, false));
                    }
                    if (y % STEP == 0) {
                        neg = normalize<T>({re.z, im.z, re.w, im.w}, c.gain(y, true));
                    }
                }
                Complex ta = pass ? a.conj() : a, tg = pass ? g.conj() : g;
                auto x = c.source(y);
                float4 state{tangents[8 * NT], tangents[9 * NT], tangents[10 * NT],
                             tangents[11 * NT]};
                float4 prev = Quad<T>::pack(state).unpack();
                state = advance(state, x, ta, tg);
                tangents[8 * NT] = state.x;
                tangents[9 * NT] = state.y;
                tangents[10 * NT] = state.z;
                tangents[11 * NT] = state.w;
                float4 d0 = pass ? neg : pos;
#pragma unroll
                for (int z = 0; z < 2; z++) {
                    int off = z * 2;
                    Complex sp = ta * Complex{tangents[off * NT], tangents[(off + 1) * NT]} +
                                 Complex{((float *)&prev)[off], ((float *)&prev)[off + 1]};
                    Complex sx = ta * Complex{tangents[(off + 4) * NT], tangents[(off + 5) * NT]} +
                                 Complex{((float *)&x)[off], ((float *)&x)[off + 1]};
                    tangents[off * NT] = sp.r;
                    tangents[(off + 1) * NT] = sp.i;
                    tangents[(off + 4) * NT] = sx.r;
                    tangents[(off + 5) * NT] = sx.i;
                    Complex da{poles[0], poles[NT]}, dg{poles[2 * NT], poles[3 * NT]};
                    pole_grad({((float *)&d0)[off], ((float *)&d0)[off + 1]}, sp, sx, da, dg, pass);
                    poles[0] = da.r;
                    poles[NT] = da.i;
                    poles[2 * NT] = dg.r;
                    poles[3 * NT] = dg.i;
                }
                float4 d = pass ? pos : neg;
                Complex aa = ta * Complex{tangents[12 * NT], tangents[13 * NT]} + Complex{d.x, d.y};
                Complex ab = ta * Complex{tangents[14 * NT], tangents[15 * NT]} + Complex{d.z, d.w};
                tangents[12 * NT] = aa.r;
                tangents[13 * NT] = aa.i;
                tangents[14 * NT] = ab.r;
                tangents[15 * NT] = ab.i;
                auto ra = tg * aa, rb = tg * ab;
                float4 v{ra.r, ra.i, rb.r, rb.i};
                if (pass) {
                    if (compact) {
                        write_column_gradient<true, true, true>(c, y, v, gs);
                    } else {
                        write_column_gradient<true, false, true>(c, y, v, gs);
                    }
                } else {
                    if (compact) {
                        write_column_gradient<false, true>(c, y, v, gs);
                    } else {
                        write_column_gradient<false, false>(c, y, v, gs);
                    }
                }
            }
            if (line < count * OW / 4 * 4) {
                __syncthreads();
            }
        }
    }
    __syncthreads();
    if (!warp) {
        auto *out = partial + blockIdx.x * 4 * M + m;
        for (int i=0;i<4;i++) {
            float sum=0;
            for (int w=0;w<4;w++) sum+=poles[i*NT+w*32];
            out[i*M]=sum+(start?out[i*M]:0.f);
        }
    }
}
"""


def source(k):
    prefix = scans._CUDA_SOURCE.split("struct ProjectWeights")[0]
    prefix = (
        prefix.replace("float vx_a, vx_b, eps;", "float vx_a, vx_b, eps; I compact_base;")
        .replace("COMPACT ? c.plane / 2", "COMPACT ? c.plane / c.W * c.OW")
        .replace("(row + c.offset_a - c.m - c.M) / 2 + c.m", "c.compact_base + I(y) * c.OW * c.M")
    )
    constants = f"""
using T={"__half" if k.dtype == torch.float16 else "__nv_bfloat16"};
using I={"long long" if k.wide else "int"};
constexpr int H={k.h},W={k.w},M={k.m},NT=128,STEP={1 + int(k.coarse)},OH=H/STEP,OW=W/STEP;
constexpr bool FULL={int(k.full)},FAST={int(k.fast)},compact={int(k.compact)};
__device__ float tanhf_fast(float x) {{float y; asm("tanh.approx.f32 %0,%1;":"=f"(y):"f"(x));return y;}}
"""
    tc = _TC.replace("FMT", "f16" if k.dtype == torch.float16 else "bf16")
    body, helper, regs = {
        "column_fp8": (_DIRECT, "", 128),
        "pure_vscan_fwd_p_fp8": (_FWD, "", 96),
        "pure_vscan_bwd_gs_fp8": (_BWD, "", 64),
        "tc_path_mlp_bwd_fp8": (_MLP, tc, 40),
    }[k.name]
    return (
        "#include <cuda_fp8.h>\n"
        + prefix
        + constants
        + _COMMON
        + helper
        + f'extern "C" __global__ __maxnreg__({regs}) void {k.name}('
        + _ARGS
        + "){"
        + body
    )


def compile_cubin(k, directory):
    cu = Path(directory) / "scan.cu"
    cubin = cu.with_suffix(".cubin")
    cu.write_text(source(k))
    proc = subprocess.run(
        [
            "nvcc",
            "-arch=sm_90a",
            "-std=c++17",
            "--cubin",
            (
                "--ptxas-options=-v"
                if k.name == "column_fp8"
                and not (k.fast and k.h == 56 and k.dtype == torch.float16)
                else "--ptxas-options=-v,-O2"
                if k.name == "tc_path_mlp_bwd_fp8" and k.dtype == torch.float16
                else "--ptxas-options=-v,-O1"
            ),
            str(cu),
            "-o",
            str(cubin),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return cubin, proc.stderr


@cache
def module(device, k):
    with tempfile.TemporaryDirectory() as directory:
        cubin, _ = compile_cubin(k, directory)
        mod = _CudaModule(ct.create_string_buffer(cubin.read_bytes()), device)
    mod.load_function(k.name, k.smem)
    return mod


def _validate(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode):
    if (
        mode not in MODES
        or (compact and not coarse)
        or not scans._shape_supported(tuple(states.shape[1:]))
        or states.dtype not in (torch.float16, torch.bfloat16)
        or not scans._pole_ok(pole, states)
        or not scans._vertical_inputs_ok(states, auxiliary, path, coarse=coarse)
        or path[0].shape[1] != 8
        or not 0 < eps < math.inf
    ):
        raise ValueError("Invalid vertical scan inputs")
    b, h, w, m = states.shape[1:]
    shape = (b, h // (1 + coarse), w // (1 + coarse), m)
    if any(
        t.shape != shape or t.dtype != states.dtype or t.device != states.device for t in (gr, gi)
    ):
        raise ValueError("Invalid vertical adjoints")


@torch.library.custom_op("lnet::vscan_fp8_bwd", mutates_args=())
def vscan_fp8_bwd(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    gr: Tensor,
    gi: Tensor,
    coarse: bool,
    compact: bool,
    eps: float,
    mode: str,
) -> list[Tensor]:
    _validate(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode)
    if not states.is_cuda:
        raise ValueError("Wave 60A requires CUDA tensors")
    m = states.shape[-1]
    # Heights beyond the compiled policy retain the original backward contract.
    if states.shape[2] > 56:
        return scans._vscan(
            pole, states, auxiliary, path, coarse, eps, False, (gr, gi), compact=compact
        )
    if m % 32:
        extra = 32 - m % 32

        def pad_last(t):
            return torch.nn.functional.pad(t, (0, extra))

        def pad_modes(t):
            return torch.cat((t, t.new_zeros((extra, *t.shape[1:]))))

        result = _run(
            [pad_last(t.reshape(-1)) for t in pole],
            pad_last(states),
            [pad_last(t) for t in auxiliary],
            [pad_modes(t) for t in path],
            pad_last(gr),
            pad_last(gi),
            coarse,
            compact,
            eps,
            mode,
        )
        return [
            *(t[:m].reshape_as(v) for t, v in zip(result[:4], pole, strict=True)),
            result[4][..., :m].contiguous(),
            *(t[:m] for t in result[5:]),
        ]
    return _run(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode)


def _run(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode, timings=None):
    b, h, w, m = states.shape[1:]
    ks = kernels(mode, (b, h, w, m), states.dtype, coarse, compact)
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    gs = states.new_empty((4, b, h, ow if compact else w, m))
    if (coarse and not compact) or not oh * ow:
        gs.zero_()
    if not oh * ow:
        return [*(torch.zeros_like(t) for t in pole), gs, *(torch.zeros_like(t) for t in path)]
    gr, gi = scans._vertical_adjoints((gr, gi), states, (b, oh, ow, m))
    dec = mode == MODES[3]
    chunk = chunk_size(mode, b, oh * ow, m)
    sms = torch.cuda.get_device_properties(states.device).multi_processor_count
    groups = min((chunk * ow + 3) // 4, scans._ceil(sms * (8 if dec else 4), m // 32))
    blocks = padded(chunk * oh * ow) // 16
    ng = min(blocks, scans._ceil(sms * 12, m // 4))
    rows = ng if dec else groups * 4
    # Both stage-specific views belong to the same persistent partial allocation.
    storage = states.new_empty((rows * 94 + (4 * groups if dec else 0), m), dtype=torch.float32)
    part = storage[: rows * 94].view(rows, 94, m)
    if dec:
        pole_part = storage[rows * 94 :].view(groups, 4, m)
    shape = (m // 32, blocks, 32)
    buf = (
        [
            states.new_empty((*shape, 16, 8), dtype=torch.uint8),
            states.new_empty((*shape, 16, 2)),
            states.new_empty((*shape, 16, 8), dtype=torch.uint8),
            *(states.new_empty(shape, dtype=torch.float32) for _ in range(2)),
        ]
        if dec
        else [None] * 5
    )
    args = [
        *(v.float().contiguous() for v in pole),
        states.contiguous(),
        *(v.float().contiguous() for v in auxiliary),
        *(v.to(states.dtype).contiguous() for v in path[:3]),
        gr,
        gi,
        gs,
        part,
        *buf,
    ]

    def event():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    with torch.cuda.device(states.device):
        stream = torch.cuda.current_stream().cuda_stream
        for start in range(0, b, chunk):
            count = min(chunk, b - start)
            stride = padded(count * oh * ow)
            for k in ks:
                ptrs = list(map(scans._ptr, args))
                grid = (groups, m // 32, 1)
                if k.name == "pure_vscan_fwd_p_fp8":
                    major = count * ow // 4 if oh >= 4 else 0
                    grid = (major + stride // 16 - major * (oh // 4), m // 32, 1)
                elif k.name == "tc_path_mlp_bwd_fp8":
                    grid = (ng, m // 4, 1)
                elif k.name == "pure_vscan_bwd_gs_fp8":
                    ptrs[13] = scans._ptr(pole_part)
                mod = module(states.device.index, k)
                if timings is not None:
                    begin = event()
                mod.launch(
                    k.name,
                    grid,
                    (128, 1, 1),
                    k.smem,
                    stream,
                    [
                        *ptrs,
                        *map(
                            ct.c_int,
                            (
                                b,
                                start,
                                count,
                                stride,
                                groups,
                                1 + int(not gr.is_contiguous()),
                                1 + int(not gi.is_contiguous()),
                            ),
                        ),
                        ct.c_bool(compact),
                        ct.c_float(eps),
                    ],
                )
                if timings is not None:
                    timings.append((k.name, begin, event()))
        if timings is not None:
            begin = event()
        if dec:
            part[:, :4].zero_()
            torch.sum(pole_part, dim=0, out=part[0, :4])
        grad = scans._reduce_partials(part, [*pole, *path], 8)
        if timings is not None:
            timings.append(("reduce", begin, event()))
    return [*grad[:4], gs, *grad[4:]]


@vscan_fp8_bwd.register_fake
def _fake(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode):
    _validate(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode)
    _, b, h, w, m = states.shape
    return [
        *(torch.empty_like(t) for t in pole),
        states.new_empty((4, b, h, w // 2 if compact else w, m)),
        *(torch.empty_like(t) for t in path),
    ]
