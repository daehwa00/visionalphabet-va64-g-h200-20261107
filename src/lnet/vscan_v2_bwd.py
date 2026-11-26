"""Opt-in Hopper vertical scan backward kernels. All launches use native SM90a cubins."""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from functools import cache
import math
from pathlib import Path
import subprocess
import tempfile

import torch
from torch import Tensor

from . import general_cuda_scans as scans
from .cuda_nvrtc import _CudaModule

MODES = ("v59_decoupled_tc", "v59_decoupled_tc_l2chunk", "v59_fused_tc_v2", "v59_fused_tc_v2_4row")


def padded(n):
    return (n + 15) // 16 * 16


def chunk_size(mode, b, cells, m):
    if mode in MODES[2:] or cells == 0:
        return b
    budget = (36 << 20) if mode == MODES[1] else 20_000_000_000
    if mode == MODES[0] and padded(b * cells) * m * 36 <= budget:
        return b
    chunk = 1
    while chunk * 2 <= b and padded(chunk * 2 * cells) * m * 36 <= budget:
        chunk *= 2
    if mode == MODES[1]:
        while chunk * 64 < b:
            chunk *= 2
    return chunk


@dataclass(frozen=True)
class Kernel:
    name: str
    h: int
    w: int
    m: int
    coarse: bool
    dtype: torch.dtype
    threads: int
    wide: bool

    @property
    def cluster(self):
        if self.name == "fused":
            if self.h > 28:
                return 8
            if self.threads == 128:
                return 2 if self.h <= 7 else 4
        return 1

    @property
    def stripe(self):
        step = 1 + self.coarse
        return ((self.h // step + self.cluster - 1) // self.cluster) * step + self.h % step

    @property
    def smem(self):
        n, h = self.threads, self.h
        if self.name == "tc_path_mlp_bwd":
            return n // 32 * 16 * 10 * 2
        if self.name == "pure_vscan_fwd_p":
            return 8 * n * h * (1 + (h <= 28)) + n // 32 * 32 * 17 * 20
        if self.name == "pure_vscan_bwd_gs":
            return n * (80 + (h * 8 if h <= 14 else 0))
        # Once a column tile has finished its projections, the transpose
        # buffer holds the reverse trajectory of a single sixteen-row segment.
        tile = 32 * 296 * 2
        if self.cluster > 1:
            return (
                (5 if h <= 14 else 3) * self.stripe * n * 8
                + max(32 * 296 * 2, self.stripe * n * 8)
                + 32 * 88 * 2
                + 2 * 32 * 32 * 4
            )
        if h <= 14:
            return 3 * h * n * 8 + tile + 32 * 88 * 2 + 2048
        return h * n * 8 + 16 * n * ((h - 1) // 16) + max(tile, 16 * n * 8) + 32 * 88 * 2 + 2048


def kernels(mode, shape, dtype, coarse, compact=False):
    b, h, w, m = shape
    if (
        mode not in MODES
        or min(shape) < 1
        or h > 56
        or m % 32
        or (compact and not coarse)
        or dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError("Unsupported Wave 59A vertical scan")

    def make(name, threads):
        return Kernel(name, h, w, m, coarse, dtype, threads, 4 * b * h * w * m >= 2**31)

    if mode in MODES[:2]:
        return (
            make("pure_vscan_fwd_p", 64 if h > 28 else 128),
            make("tc_path_mlp_bwd", 128),
            make("pure_vscan_bwd_gs", 128),
        )
    return (make("fused", 256 if mode == MODES[2] else 128),)


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
    float dw[4]{}, aux[4]{};
    template <int ScratchPitch = 10>
    __device__ void tile(const Weights &w, const T *p, const T *dy, T *dp, T *s) {
        int l = threadIdx.x % 32, g = l / 4, k = 2 * (l % 4);
        unsigned a = load2(p + g * 8 + k), b = load2(p + (g + 8) * 8 + k);
        float pre[4]{}, grad[4]{};
        mma(pre, a, b, 0, 0, w.wt, 0);
        unsigned b0, b1;
        if constexpr (FUSED) {
            b0 = transpose8(a);
            b1 = transpose8(b);
        } else {
            // Transpose the two source fragments in the warp-local scratch.
            *((unsigned *)(s + g * ScratchPitch + k)) = a;
            *((unsigned *)(s + (g + 8) * ScratchPitch + k)) = b;
            __syncwarp();
            b0 = packed(s[k * ScratchPitch + g], s[(k + 1) * ScratchPitch + g]);
            b1 = packed(s[(k + 8) * ScratchPitch + g], s[(k + 9) * ScratchPitch + g]);
            __syncwarp();
        }
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
            if constexpr (!FUSED)
                *((unsigned *)(s + row * ScratchPitch + k)) = dz[r];
        }
        mma(grad, dz[0], dz[1], 0, 0, w.w, 0);
        unsigned at, bt;
        if constexpr (FUSED) {
            at = transpose8(dz[0]);
            bt = transpose8(dz[1]);
        } else {
            __syncwarp();
            at = packed(s[k * ScratchPitch + g], s[(k + 1) * ScratchPitch + g]);
            bt = packed(s[(k + 8) * ScratchPitch + g], s[(k + 9) * ScratchPitch + g]);
        }
        mma(dw, at, 0, bt, 0, b0, b1);
        __syncwarp();
#pragma unroll
        for (int r = 0; r < 2; r++)
            *((unsigned *)(dp + (g + 8 * r) * 8 + k)) =
                packed(write<T>(grad[2 * r]), write<T>(grad[2 * r + 1]));
    }
    // Each lane owns two W1 values and at most one bias/W2 value.
    __device__ float auxiliary_sum() {
        int l = threadIdx.x % 32, unit = l % 8, owner = unit / 2 + (unit % 2) * 4;
        float result = 0;
#pragma unroll
        for (int f = 0; f < 3; f++) {
            float v = aux[f];
            v += __shfl_down_sync(~0u, v, 16);
            v += __shfl_down_sync(~0u, v, 8);
            v = __shfl_sync(~0u, v, owner);
            if (l / 8 == f)
                result = v;
        }
        float bias = aux[3];
        for (int d = 8; d; d /= 2)
            bias += __shfl_down_sync(~0u, bias, d, 16);
        bias = __shfl_sync(~0u, bias, 16 * (l % 2));
        return l >= 24 ? bias : result;
    }
    template <int Pitch = M> __device__ void finish(float *out, bool add) {
        int l = threadIdx.x % 32, g = l / 4, k = 2 * (l % 4);
        for (int i = 0; i < 2; i++) {
            auto *v = out + (4 + g * 8 + k + i) * Pitch;
            *v = dw[i] + (add ? *v : 0);
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
        for (int d = 8; d; d /= 2)
            t += __shfl_down_sync(~0u, t, d, 16);
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
 const T *gr,const T *gi,T *gs,float *partial,T *P,T *DY,T *DP,
 int B,int start,int count,int stride,int workers,int rs,int is,bool compact,float eps
"""
_MLP = r"""
    extern __shared__ T scratch[];
    int warp = threadIdx.x / 32, m = blockIdx.y * (NT / 32) + warp, worker = blockIdx.x;
    Weights w;
    w.init(m, wi, bi, wo);
    Acc a;
    for (int n = worker * 16; n < stride; n += workers * 16) {
        long long idx = (long long)m * stride + n;
        a.tile(w, P + 8 * idx, DY + 2 * idx, DP + 8 * idx, scratch + warp * 160);
    }
    a.finish(partial + (worker * 94) * M + m, start != 0);
}
"""
_PURE = r"""
// In each sweep the source gradient is a reverse-mode recurrence. The opposite
// direction's two pole derivatives are forward sensitivities, so no adjoint
// injection history or replayed state trajectory is needed.
struct Tangent {
    float4 state{}, ca{}, cg{};
    __device__ void step(float4 src, Complex a, Complex g) {
        auto previous = Quad<T>::pack(state).unpack();
        ca = advance(ca, previous, a, {1, 0});
        cg = advance(cg, src, a, {1, 0});
        state = advance(state, src, a, g);
    }
    __device__ void collect(float4 d, Complex &da, Complex &dg, bool rev) {
        pole_grad({d.x, d.y}, {ca.x, ca.y}, {cg.x, cg.y}, da, dg, rev);
        pole_grad({d.z, d.w}, {ca.z, ca.w}, {cg.z, cg.w}, da, dg, rev);
    }
};
template <bool Desc>
__device__ void sweep(Column<T, I> c, Complex a, Complex g, bool valid, bool compact, int line,
                      int stride, const T *DP, T *gs, T *tile, Quad<T> *cache, Complex &da,
                      Complex &dg) {
    int tid = threadIdx.x, lane = tid % 32;
    Tangent tan;
    Complex aa{}, ab{};
    Complex ta = Desc ? a.conj() : a, tg = Desc ? g.conj() : g;
    auto row = [&](int y, float4 inject, float4 other) {
        auto src = (!Desc || !CACHE) ? c.packed_source(y) : cache[y * NT];
        if constexpr (!Desc && CACHE)
            cache[y * NT] = src;
        tan.step(src.unpack(), ta, tg);
        tan.collect(inject, da, dg, Desc);
        aa = ta * aa + Complex{other.x, other.y};
        ab = ta * ab + Complex{other.z, other.w};
        auto ra = tg * aa, rb = tg * ab;
        float4 ds = make_float4(ra.r, ra.i, rb.r, rb.i);
        if (compact)
            write_column_gradient<Desc, true, Desc>(c, y, ds, gs);
        else {
            write_column_gradient<Desc, false, Desc>(c, y, ds, gs);
            if constexpr (Desc && STEP == 2) {
                clear_unused_columns<false>(c, y, gs);
                if (W % 2 && line % OW == OW - 1) {
                    I off = c.offset_b + I(y) * W * M + 2 * M;
                    store_complex(gs, off, c.plane, {});
                    store_complex(gs, off + 2 * c.plane, c.plane, {});
                }
            }
        }
    };
    if constexpr (Desc && STEP == 2 && H % 2)
        if (valid)
            row(H - 1, {}, {});
    for (int n = 0; n < OH; n += 4) {
#pragma unroll 1
        for (int i = 0; i < 32; i += 8) {
            int mm = i + lane / 4, r = lane % 4, y = Desc ? OH - 1 - n - r : n + r;
            if (valid && y >= 0 && y < OH) {
                long long index =
                    ((long long)(blockIdx.y * 32 + mm) * stride + (long long)line * OH + y) * 8;
                *reinterpret_cast<uint4 *>(tile + (mm * 5 + r) * 8) =
                    *reinterpret_cast<const uint4 *>(DP + index);
            }
        }
        __syncwarp();
#pragma unroll 1
        for (int r = 0; r < 4 && n + r < OH; r++) {
            int oy = Desc ? OH - 1 - n - r : n + r;
            if (valid) {
                auto v = *reinterpret_cast<uint4 *>(tile + (lane * 5 + r) * 8);
                auto *s = reinterpret_cast<T *>(&v);
                float4 p = normalize<T>(make_float4(read(s[0]), read(s[4]), read(s[1]), read(s[5])),
                                        c.gain(oy * STEP + STEP - 1, false));
                float4 q = normalize<T>(make_float4(read(s[2]), read(s[6]), read(s[3]), read(s[7])),
                                        c.gain(oy * STEP, true));
                for (int j = 0; j < STEP; j++) {
                    int k = Desc ? STEP - 1 - j : j;
                    auto ip = k == STEP - 1 ? p : float4{}, in = k == 0 ? q : float4{};
                    row(oy * STEP + k, Desc ? in : ip, Desc ? ip : in);
                }
            }
        }
        __syncwarp();
    }
    if constexpr (!Desc && STEP == 2 && H % 2)
        if (valid)
            row(H - 1, {}, {});
}
"""
_PURE_BODY = r"""
    extern __shared__ float4 shared[];
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, m = blockIdx.y * 32 + lane;
    auto *cache = reinterpret_cast<Quad<T> *>(shared);
    T *tile = reinterpret_cast<T *>(cache + (CACHE ? H * NT : 0)) + warp * 32 * 5 * 8;
    Complex a{dr[m], di[m]}, g{gr0[m], gi0[m]}, da{}, dg{};
    for (int base = blockIdx.x * TW; base < count * OW; base += gridDim.x * TW) {
        int line = base + warp;
        bool valid = line < count * OW;
        int b = start + (valid ? line / OW : 0), x = valid ? line % OW : 0,
            xa = x * STEP + STEP - 1, xb = x * STEP;
        I plane = I(B) * H * W * M, offset = I(b) * H * W * M + m;
        Column<T, I> c{states,
                       vy,
                       offset + xa * M,
                       offset + xb * M,
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
                       eps};
        sweep<false>(c, a, g, valid, compact, line, stride, DP, gs, tile, cache + tid, da, dg);
        sweep<true>(c, a, g, valid, compact, line, stride, DP, gs, tile, cache + tid, da, dg);
    }
    __syncthreads();
    auto *s = reinterpret_cast<float4 *>(shared);
    s[tid] = make_float4(da.r, da.i, dg.r, dg.i);
    __syncthreads();
    if (warp == 0) {
        float v[4]{};
        for (int t = lane; t < NT; t += 32) {
            float4 q = s[t];
            v[0] += q.x;
            v[1] += q.y;
            v[2] += q.z;
            v[3] += q.w;
        }
        for (int f = 0; f < 4; f++) {
            auto *out = partial + (blockIdx.x * 4 + f) * M + m;
            *out = v[f] + (start ? *out : 0);
        }
    }
}
"""

_CLUSTER_HELPERS = r"""
__device__ int first_row(int rank) { return min(OH, rank * CELLS) * STEP; }
__device__ int last_row(int rank) { return rank == Z - 1 ? H : first_row(rank + 1); }
// DSM holds one distinct row range per CTA. Visit peer segments in scan
// order to preserve the scalar FP32 recurrence, including zero injections.
template <bool Desc> __device__ float4 boundary(Quad<T> *data, int rank, Complex a, Complex g) {
    auto cluster = cooperative_groups::this_cluster();
    float4 result{};
    for (int i = 0; i < (Desc ? Z - rank - 1 : rank); i++) {
        int peer = Desc ? Z - 1 - i : i;
        auto *ptr = cluster.map_shared_rank(data, peer) + threadIdx.x;
        int rows = last_row(peer) - first_row(peer);
        for (int j = 0; j < rows; j++)
            result = advance(result, ptr[(Desc ? rows - 1 - j : j) * NT].unpack(), a, g);
    }
    return result;
}
"""
_CLUSTER = r"""
    extern __shared__ float4 shared[];
    auto cluster = cooperative_groups::this_cluster();
    int rank = blockIdx.z, tid = threadIdx.x, lane = tid % 32, warp = tid / 32,
        m = blockIdx.y * 32 + lane;
    int lo = first_row(rank), hi = last_row(rank), nr = hi - lo,
        oc = min(CELLS, max(0, OH - rank * CELLS));
    auto *src = reinterpret_cast<Quad<T> *>(shared);
    auto *pos = src + SPAN * NT, *in = pos + SPAN * NT, *ip = FULL ? in + SPAN * NT : pos;
    auto *negprev = ip + SPAN * NT;
    auto *tiles = reinterpret_cast<T *>(src + (FULL ? 5 : 3) * SPAN * NT);
    constexpr int pitch = 296;
    auto *weights = tiles + max(32 * pitch, SPAN * NT * 4);
    for (int n = tid; n < 32 * 88; n += NT) {
        int mm = blockIdx.y * 32 + n / 88, f = n % 88;
        weights[n] = f < 64   ? ((const T *)wi)[mm * 64 + f]
                     : f < 72 ? ((const T *)bi)[mm * 8 + f - 64]
                              : ((const T *)wo)[mm * 16 + f - 72];
    }
    Complex a{dr[m], di[m]}, g{gr0[m], gi0[m]}, ar = a.conj(), gg = g.conj(), da{}, dg{};
    float sums[32 / TW]{};
    float *auxsum = reinterpret_cast<float *>(weights + 32 * 88);
    for (int n = tid; n < 2 * 32 * 32; n += NT)
        auxsum[n] = 0;
    for (int tile = blockIdx.x; tile * TW < count * OW; tile += gridDim.x) {
        int line = tile * TW + warp;
        bool valid = line < count * OW;
        int b = start + (valid ? line / OW : 0), x = valid ? line % OW : 0,
            xa = x * STEP + STEP - 1, xb = x * STEP;
        I plane = I(B) * H * W * M, base = I(b) * H * W * M + m;
        Column<T, I> c{states,
                       vy,
                       base + xa * M,
                       base + xb * M,
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
                       eps};
        for (int j = 0; j < nr; j++)
            src[j * NT + tid] = c.packed_source(lo + j);
        cluster.sync();
        auto pseed = boundary<false>(src, rank, a, g);
        auto nseed = boundary<true>(src, rank, ar, gg);
        auto pf = pseed, nf = nseed;
        for (int j = 0; j < nr; j++) {
            pf = advance(pf, src[j * NT + tid].unpack(), a, g);
            pos[j * NT + tid] = Quad<T>::pack(pf);
            in[j * NT + tid] = {};
            if constexpr (FULL)
                ip[j * NT + tid] = {};
        }
        if (nr > STEP * oc) {
            if constexpr (FULL)
                negprev[(nr - 1) * NT + tid] = Quad<T>::pack(nf);
            nf = advance(nf, src[(nr - 1) * NT + tid].unpack(), ar, gg);
            if constexpr (!FULL)
                ip[(nr - 1) * NT + tid] = {};
        }
        for (int end = oc - 1; end >= 0; end -= ROWS) {
#pragma unroll 1
            for (int r = 0; r < ROWS; r++) {
                int cell = end - r, j = cell * STEP + STEP - 1, row = r * TW + warp;
                float4 pp{}, pn{};
                unsigned dy = 0;
                if (cell >= 0) {
                    for (int k = 0; k < STEP; k++) {
                        if constexpr (FULL)
                            negprev[(j - k) * NT + tid] = Quad<T>::pack(nf);
                        nf = advance(nf, src[(j - k) * NT + tid].unpack(), ar, gg);
                    }
                    if (valid) {
                        pp = normalize<T>(pos[j * NT + tid].unpack(), c.gain(lo + j, false));
                        pn = normalize<T>(nf, c.gain(lo + j + 1 - STEP, true));
                        I offset = c.cell(lo / STEP + cell) - m;
                        dy = packed(gr[offset * rs + m], gi[offset * is + m]);
                    }
                }
                T *t = tiles + lane * pitch;
                *reinterpret_cast<uint4 *>(t + row * 8) = make_uint4(
                    packed(write<T>(pp.x), write<T>(pp.z)), packed(write<T>(pn.x), write<T>(pn.z)),
                    packed(write<T>(pp.y), write<T>(pp.w)), packed(write<T>(pn.y), write<T>(pn.w)));
                *reinterpret_cast<unsigned *>(t + 128 + row * 2) = dy;
            }
            __syncthreads();
#pragma unroll
            for (int u = 0; u < 32 / TW; u++) {
                T *t = tiles + (warp + TW * u) * pitch;
                const T *pw = weights + (warp + TW * u) * 88;
                Weights w;
                int group = lane / 4, k = (lane % 4) * 2;
                w.wt = load2(pw + group * 8 + k);
                w.w = transpose8(w.wt);
                read_pair(w.bias, pw + 64 + k);
                read_pair(w.o0, pw + 72 + k);
                read_pair(w.o1, pw + 80 + k);
                Acc local;
                local.tile<8>(w, t, t + 128, t + 160, t);
                sums[u] += local.dw[0];
                int idx = (warp + TW * u) * 32 + lane;
                auxsum[idx] += local.dw[1];
                auxsum[1024 + idx] += local.auxiliary_sum();
            }
            __syncthreads();
            for (int r = 0; r < ROWS && r <= end; r++) {
                int j = (end - r) * STEP + STEP - 1;
                auto vec =
                    *reinterpret_cast<uint4 *>(tiles + lane * pitch + 160 + (r * TW + warp) * 8);
                auto *s = reinterpret_cast<T *>(&vec);
                ip[j * NT + tid] =
                    normalize_packed<T>(make_float4(read(s[0]), read(s[4]), read(s[1]), read(s[5])),
                                        c.gain(lo + j, false));
                if constexpr (!FULL && STEP == 2)
                    ip[(j - 1) * NT + tid] = {};
                in[(j + 1 - STEP) * NT + tid] =
                    normalize_packed<T>(make_float4(read(s[2]), read(s[6]), read(s[3]), read(s[7])),
                                        c.gain(lo + j + 1 - STEP, true));
            }
        }
        cluster.sync();
        auto adjp = boundary<true>(ip, rank, ar, {1, 0});
        auto adjn = boundary<false>(in, rank, a, {1, 0});
        // The transpose tile is dead. Long stripes replay only their local
        // on-chip sources; short stripes retain both complete trajectories.
        auto *prev = reinterpret_cast<Quad<T> *>(tiles) + tid;
        if constexpr (!FULL) {
            pf = pseed;
            for (int j = 0; j < nr; j++) {
                prev[j * NT] = Quad<T>::pack(pf);
                pf = advance(pf, src[j * NT + tid].unpack(), a, g);
            }
        }
        Complex aa{adjp.x, adjp.y}, ab{adjp.z, adjp.w};
        if (valid)
            for (int j = nr - 1; j >= 0; j--) {
                auto old =
                    FULL ? (j ? pos[(j - 1) * NT + tid] : Quad<T>::pack(pseed)) : prev[j * NT];
                auto ds =
                    column_adjoint_step<false>(old.unpack(), src[j * NT + tid].unpack(),
                                               ip[j * NT + tid].unpack(), a, g, aa, ab, da, dg);
                if (compact)
                    write_column_gradient<false, true>(c, lo + j, ds, gs);
                else
                    write_column_gradient<false, false>(c, lo + j, ds, gs);
            }
        if constexpr (!FULL) {
            nf = nseed;
            for (int j = nr - 1; j >= 0; j--) {
                prev[j * NT] = Quad<T>::pack(nf);
                nf = advance(nf, src[j * NT + tid].unpack(), ar, gg);
            }
        }
        aa = {adjn.x, adjn.y};
        ab = {adjn.z, adjn.w};
        if (valid)
            for (int j = 0; j < nr; j++) {
                auto old = FULL ? negprev[j * NT + tid] : prev[j * NT];
                auto ds =
                    column_adjoint_step<true>(old.unpack(), src[j * NT + tid].unpack(),
                                              in[j * NT + tid].unpack(), ar, gg, aa, ab, da, dg);
                int y = lo + j;
                if (compact)
                    write_column_gradient<true, true, true>(c, y, ds, gs);
                else {
                    write_column_gradient<true, false, true>(c, y, ds, gs);
                    if constexpr (STEP == 2)
                        clear_unused_columns<false>(c, y, gs);
                    if constexpr (STEP == 2 && W % 2)
                        if (x == OW - 1) {
                            I off = c.offset_b + I(y) * W * M + 2 * M;
                            store_complex(gs, off, plane, {});
                            store_complex(gs, off + 2 * plane, plane, {});
                        }
                }
            }
        cluster.sync();
    }
    float *out = partial + (blockIdx.x * Z + rank) * 94 * M + blockIdx.y * 32;
#pragma unroll
    for (int u = 0; u < 32 / TW; u++) {
        int mm = warp + TW * u;
        out[(4 + lane / 4 * 8 + lane % 4 * 2) * M + mm] = sums[u];
        out[(5 + lane / 4 * 8 + lane % 4 * 2) * M + mm] = auxsum[mm * 32 + lane];
        if (lane < 26)
            out[(68 + lane) * M + mm] = auxsum[1024 + mm * 32 + lane];
    }
    shared[tid] = make_float4(da.r, da.i, dg.r, dg.i);
    __syncthreads();
    if (warp == 0) {
        float4 v{};
        for (int j = lane; j < NT; j += 32) {
            auto d = shared[j];
            v.x += d.x;
            v.y += d.y;
            v.z += d.z;
            v.w += d.w;
        }
        out[lane] = v.x;
        out[M + lane] = v.y;
        out[2 * M + lane] = v.z;
        out[3 * M + lane] = v.w;
    }
}
"""

_SCAN = r"""
    extern __shared__ float4 storage[];
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, m = blockIdx.y * 32 + lane;
    auto *hist = reinterpret_cast<Quad<T> *>(storage) + tid;
    auto *srcbuf = hist + H * NT;
    auto *checkpoint = reinterpret_cast<float4 *>(srcbuf - tid + (CACHE ? H * NT : 0)) + tid;
    auto *prev = H <= 14
                     ? reinterpret_cast<Quad<T> *>(checkpoint - tid) + tid
                     : reinterpret_cast<Quad<T> *>(checkpoint - tid + ((H - 1) / 16) * NT) + tid;
    auto *tiles = reinterpret_cast<T *>(prev - tid + (H <= 14 ? H * NT : 0));
    auto *exchange = reinterpret_cast<T *>(srcbuf - tid + (CACHE ? H * NT : 0));
    constexpr int pitch = 296, tilebytes = 32 * pitch * 2;
    T *weights = tiles + (H > 14 && 16 * NT * 8 > tilebytes ? 16 * NT * 4 : tilebytes / 2);
    float path_sum[FUSED ? 32 / TW : 1][3]{};
    float *auxcache = reinterpret_cast<float *>(weights + 32 * 88);
    if constexpr (FUSED)
        for (int n = tid; n < 512; n += NT)
            auxcache[n] = 0;
    if constexpr (FWD) {
        if (blockIdx.x == 0 && warp == 0)
            for (int j = count * OH * OW; j < stride; j++) {
                long long offset = (long long)m * stride + j;
                *reinterpret_cast<uint4 *>(P + offset * 8) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<unsigned *>(DY + offset * 2) = 0;
            }
    }
    if constexpr (FUSED) {
        for (int n = tid; n < 32 * 88; n += NT) {
            int mm = blockIdx.y * 32 + n / 88, f = n % 88;
            weights[n] = f < 64   ? ((const T *)wi)[mm * 64 + f]
                         : f < 72 ? ((const T *)bi)[mm * 8 + f - 64]
                                  : ((const T *)wo)[mm * 16 + f - 72];
        }
        __syncthreads();
    }
    Complex a{dr[m], di[m]}, g{gr0[m], gi0[m]}, ar = a.conj(), gg = g.conj(), da{}, dg{};
    for (int tile = blockIdx.x; tile * TW < count * OW; tile += gridDim.x) {
        int line = tile * TW + warp;
        bool valid = line < count * OW;
        int x = valid ? line % OW : 0, b = start + (valid ? line / OW : 0),
            xa = x * STEP + STEP - 1, xb = x * STEP;
        I plane = I(B) * H * W * M, base = I(b) * H * W * M + m;
        Column<T, I> c{states,
                       vy,
                       base + xa * M,
                       base + xb * M,
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
                       eps};
        float4 pos{}, neg{};
        for (int y = 0; y < H; y++) {
            auto s = c.packed_source(y);
            if constexpr (CACHE)
                srcbuf[y * NT] = s;
            pos = advance(pos, s.unpack(), a, g);
            hist[y * NT] = Quad<T>::pack(pos);
        }
        auto src = [&](int y) { return CACHE ? srcbuf[y * NT].unpack() : c.source(y); };
        Complex aa{}, ab{};
        for (int end = H - 1; end >= 0;) {
            if constexpr (STEP == 2 && H % 2) {
                if (end == H - 1) {
                    if constexpr (!FWD && CACHE)
                        prev[end * NT] = Quad<T>::pack(neg);
                    neg = advance(neg, src(end), ar, gg);
                    if constexpr (!FWD)
                        if (valid) {
                            if (compact)
                                write_column_gradient<false, true>(c, end, {}, gs);
                            else
                                write_column_gradient<false, false>(c, end, {}, gs);
                        }
                    --end;
                    continue;
                }
            }
#pragma unroll 1
            for (int r = 0; r < ROWS; r++) {
                int y = end - r * STEP;
                float4 pp{}, pn{};
                T *p = nullptr, *dy = nullptr;
                if (y >= 0) {
                    for (int k = 0; k < STEP; k++) {
                        int yy = y - k;
                        if constexpr (!FWD) {
                            if constexpr (CACHE)
                                prev[yy * NT] = Quad<T>::pack(neg);
                            else if (yy % 16 == 15 && yy < H - 1)
                                checkpoint[(yy / 16) * NT] = neg;
                        }
                        neg = advance(neg, src(yy), ar, gg);
                    }
                    if (valid) {
                        pp = normalize<T>(hist[y * NT].unpack(), c.gain(y, false));
                        pn = normalize<T>(neg, c.gain(y + 1 - STEP, true));
                    }
                }
                {
                    if constexpr (FUSED) {
                        T *t = tiles + lane * pitch;
                        int row = r * TW + warp;
                        p = t + row * 8;
                        dy = t + 128 + row * 2;
                    } else {
                        T *t = exchange + warp * 32 * 17 * 10;
                        p = t + (lane * 17 + r) * 8;
                        dy = t + 32 * 17 * 8 + (lane * 17 + r) * 2;
                    }
                    if (p) {
                        uint4 vec = make_uint4(packed(write<T>(pp.x), write<T>(pp.z)),
                                               packed(write<T>(pn.x), write<T>(pn.z)),
                                               packed(write<T>(pp.y), write<T>(pp.w)),
                                               packed(write<T>(pn.y), write<T>(pn.w)));
                        *reinterpret_cast<uint4 *>(p) = vec;
                        I off = y >= 0 ? c.cell(y / STEP) - m : 0;
                        *reinterpret_cast<unsigned *>(dy) =
                            valid && y >= 0 ? packed(gr[off * rs + m], gi[off * is + m]) : 0;
                    }
                }
            }
            if constexpr (FWD) {
                constexpr int fields = 10;
                T *t = exchange + warp * 32 * 17 * fields;
                __syncwarp();
                // Two independent modes per warp, sixteen contiguous cells per mode.

#pragma unroll 1
                for (int base_mode = 0; base_mode < 32; base_mode += 2) {
                    int mm = base_mode + lane / 16, r = lane % 16;
                    int row = end / STEP - r;
                    if (valid && row >= 0) {
                        long long idx =
                            (long long)(blockIdx.y * 32 + mm) * stride + (long long)line * OH + row;
                        auto *v = reinterpret_cast<uint4 *>(t + (mm * 17 + r) * 8);
                        if constexpr (FWD) {
                            *reinterpret_cast<uint4 *>(P + idx * 8) = *v;
                            *reinterpret_cast<unsigned *>(DY + idx * 2) =
                                *reinterpret_cast<unsigned *>(t + 32 * 17 * 8 + (mm * 17 + r) * 2);
                        }
                    }
                }
                __syncwarp();
            }
            if constexpr (FUSED) {
                __syncthreads();
#pragma unroll
                for (int u = 0; u < 32 / TW; u++) {
                    T *t = tiles + (warp + TW * u) * pitch;
                    const T *pw = weights + (warp + TW * u) * 88;
                    Weights w;
                    int group = lane / 4, k = (lane % 4) * 2;
                    w.wt = load2(pw + group * 8 + k);
                    w.w = transpose8(w.wt);
                    read_pair(w.bias, pw + 64 + k);
                    read_pair(w.o0, pw + 72 + k);
                    read_pair(w.o1, pw + 80 + k);
                    Acc local;
                    local.tile(w, t, t + 128, t + 160, t + 160);
                    path_sum[u][0] += local.dw[0];
                    path_sum[u][1] += local.dw[1];
                    if (u < 16 / TW)
                        auxcache[(warp + TW * u) * 32 + lane] += local.auxiliary_sum();
                    else
                        path_sum[u][2] += local.auxiliary_sum();
                }
                __syncthreads();
            }
            if constexpr (!FWD) {
                for (int r = 0; r < ROWS; r++) {
                    int y = end - r * STEP;
                    if (y < 0)
                        continue;
                    const T *p;
                    if constexpr (FUSED)
                        p = tiles + lane * pitch + 160 + (r * TW + warp) * 8;
                    else
                        p = exchange + (warp * 32 * 17 + lane * 17 + r) * 8;
                    if (valid) {
                        auto v = *reinterpret_cast<const uint4 *>(p);
                        auto *s = reinterpret_cast<const T *>(&v);
                        auto ip = normalize<T>(
                            make_float4(read(s[0]), read(s[4]), read(s[1]), read(s[5])),
                            c.gain(y, false));
                        auto in = normalize_packed<T>(
                            make_float4(read(s[2]), read(s[6]), read(s[3]), read(s[7])),
                            c.gain(y + 1 - STEP, true));
                        for (int k = 0; k < STEP; k++) {
                            int yy = y - k;
                            auto old = yy ? hist[(yy - 1) * NT].unpack() : float4{};
                            auto source = src(yy);
                            auto ds = column_adjoint_step<false>(old, source, k ? float4{} : ip, a,
                                                                 g, aa, ab, da, dg);
                            if (compact)
                                write_column_gradient<false, true>(c, yy, ds, gs);
                            else
                                write_column_gradient<false, false>(c, yy, ds, gs);
                        }
                        hist[(y + 1 - STEP) * NT] = in;
                    }
                }
            }
            if constexpr (!FUSED)
                __syncwarp();
            end -= ROWS * STEP;
        }
        if constexpr (!FWD) {
            if constexpr (FUSED && H > 14)
                __syncthreads();
            aa = ab = {};
            if (valid)
                for (int begin = 0; begin < H; begin += 16) {
                    int stop = min(H, begin + 16);
                    if constexpr (!CACHE) {
                        neg = stop == H ? float4{} : checkpoint[(begin / 16) * NT];
                        for (int y = stop - 1; y >= begin; y--) {
                            prev[(y - begin) * NT] = Quad<T>::pack(neg);
                            neg = advance(neg, src(y), ar, gg);
                        }
                    }
                    for (int y = begin; y < stop; y++) {
                        auto in = (STEP == 1 || (y % 2 == 0 && y + 1 < H)) ? hist[y * NT].unpack()
                                                                           : float4{};
                        auto ds =
                            column_adjoint_step<true>(prev[(CACHE ? y : y - begin) * NT].unpack(),
                                                      src(y), in, ar, gg, aa, ab, da, dg);
                        if (compact)
                            write_column_gradient<true, true, true>(c, y, ds, gs);
                        else {
                            write_column_gradient<true, false, true>(c, y, ds, gs);
                            if constexpr (STEP == 2)
                                clear_unused_columns<false>(c, y, gs);
                            if constexpr (STEP == 2 && W % 2)
                                if (x == OW - 1) {
                                    I off = I(y) * W * M + c.offset_b + 2 * M;
                                    store_complex(gs, off, plane, {});
                                    store_complex(gs, off + 2 * plane, plane, {});
                                }
                        }
                    }
                }
            if constexpr (FUSED && H > 14)
                __syncthreads();
        }
    }
    if constexpr (FUSED) {
#pragma unroll
        for (int u = 0; u < 32 / TW; u++) {
            float *out = partial + blockIdx.x * 94 * M + blockIdx.y * 32 + warp + TW * u;
            out[(4 + lane / 4 * 8 + lane % 4 * 2) * M] = path_sum[u][0];
            out[(5 + lane / 4 * 8 + lane % 4 * 2) * M] = path_sum[u][1];
            if (lane < 26)
                out[(68 + lane) * M] =
                    u < 16 / TW ? auxcache[(warp + TW * u) * 32 + lane] : path_sum[u][2];
        }
    }
    if constexpr (!FWD) {
        float vals[4] = {da.r, da.i, dg.r, dg.i};
        auto *s = reinterpret_cast<float *>(storage);
        __syncthreads();
        for (int f = 0; f < 4; f++) {
            s[tid] = vals[f];
            __syncthreads();
            if (warp == 0) {
                float v = 0;
                for (int j = 0; j < TW; j++)
                    v += s[j * 32 + lane];
                auto *out = partial + (blockIdx.x * (FUSED ? 94 : 4) + f) * M + m;
                *out = v + (start ? *out : 0);
            }
            __syncthreads();
        }
    }
}
"""


@cache
def source(k):
    prefix = scans._odd_compact_source(scans._CUDA_SOURCE.split("struct ProjectWeights")[0])
    fwd = k.name == "pure_vscan_fwd_p"
    bwd = k.name == "pure_vscan_bwd_gs"
    fused = k.name == "fused"
    half = k.dtype == torch.float16
    constants = f"""
using T={"__half" if half else "__nv_bfloat16"}; using I={"long long" if k.wide else "int"};
constexpr int H={k.h},W={k.w},M={k.m},NT={k.threads},TW=NT/32,STEP={1 + int(k.coarse)};
constexpr int OH=H/STEP,OW=W/STEP,SEG=H<16?H:16,ROWS={16 // (k.threads // 32) if fused else 16};
constexpr int Z={k.cluster},CELLS=(OH+Z-1)/Z,SPAN={k.stripe};
constexpr bool FULL=H<=14,FUSED={int(fused)},FWD={int(fwd)},CACHE=H<={28 if fwd else 14};
"""
    bounds = 2 if fused and k.threads == 256 else 4 if fused else 8 if bwd else 1
    launch = (
        "__maxnreg__(48)"
        if fwd
        else "__maxnreg__(40)"
        if k.name == "tc_path_mlp_bwd"
        else f"__launch_bounds__(NT,{bounds})"
    )
    clustered = k.cluster > 1
    if clustered:
        prefix = "#include <cooperative_groups.h>\n" + prefix
    attribute = f"__cluster_dims__(1,1,{k.cluster}) " if clustered else ""
    return (
        prefix
        + constants
        + (_CLUSTER_HELPERS if clustered else "")
        + (_PURE if bwd else "")
        + _TC.replace("FMT", "f16" if half else "bf16")
        + f'extern "C" __global__ {launch} void {attribute}{k.name}('
        + _ARGS
        + "){"
        + (
            _CLUSTER
            if clustered
            else _MLP
            if k.name == "tc_path_mlp_bwd"
            else _PURE_BODY
            if bwd
            else _SCAN
        )
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
            "--ptxas-options=-v,-O1" if k.name != "fused" else "--ptxas-options=-v",
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


def workers(m):
    return max(1, 528 // m)


def _validate(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode):
    if (
        mode not in MODES
        or compact
        and not coarse
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


@torch.library.custom_op("lnet::vscan_v2_bwd", mutates_args=())
def vscan_v2_bwd(
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
        raise ValueError("Wave 59A requires CUDA tensors")
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


def _run(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode):
    ks = kernels(mode, states.shape[1:], states.dtype, coarse, compact)
    b, h, w, m = states.shape[1:]
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    gr, gi = scans._vertical_adjoints((gr, gi), states, (b, oh, ow, m))
    gs = states.new_empty((4, b, h, ow if compact else w, m))
    if not oh * ow:
        return [
            *(torch.zeros_like(t) for t in pole),
            gs.zero_(),
            *(torch.zeros_like(t) for t in path),
        ]
    dec = mode in MODES[:2]
    chunk = chunk_size(mode, b, oh * ow, m)
    n = padded(chunk * oh * ow)
    nw = (
        workers(m)
        if dec
        else min(
            (b * ow + ks[0].threads // 32 - 1) // (ks[0].threads // 32),
            528 // (m // 32) or 1,
            max(1, (4 << 20) // (94 * m * 4)),
        )
    )
    z = ks[0].cluster
    nw = max(1, nw // z)
    scan_groups = max(1, 264 // (nw * (m // 32)))
    rows = nw * z
    scan_rows = nw * scan_groups if dec else 0
    # Path workers and pure scan workers share one persistent FP32 allocation.
    storage = states.new_empty((rows * 94 + scan_rows * 4, m), dtype=torch.float32)
    partial = storage[: rows * 94].view(rows, 94, m)
    pole_part = storage[rows * 94 :].view(scan_rows, 4, m)
    buf = [states.new_empty((m, n, f)) for f in (8, 2, 8)] if dec else []
    args = [
        *(v.float().contiguous() for v in pole),
        states.contiguous(),
        *(v.float().contiguous() for v in auxiliary),
        *(v.to(states.dtype).contiguous() for v in path[:3]),
        gr,
        gi,
        gs,
        partial,
        *buf,
    ]
    ptrs = [*map(scans._ptr, args), *([] if dec else [ct.c_void_p()] * 3)]
    with torch.cuda.device(states.device):
        stream = torch.cuda.current_stream().cuda_stream
        for start in range(0, b, chunk):
            count = min(chunk, b - start)
            # Last chunk may be shorter. Pack all modes at the active padded stride.
            stride = padded(count * oh * ow)
            for k in ks:
                grid = (nw, m // 32, k.cluster)
                if k.name == "pure_vscan_fwd_p":
                    grid = ((count * ow + k.threads // 32 - 1) // (k.threads // 32), m // 32, 1)
                elif k.name == "tc_path_mlp_bwd":
                    grid = (nw, m // (k.threads // 32), 1)
                stage_ptrs = ptrs
                if k.name == "pure_vscan_bwd_gs":
                    grid = (nw * scan_groups, m // 32, 1)
                    stage_ptrs = ptrs.copy()
                    stage_ptrs[13] = scans._ptr(pole_part)
                module(states.device.index, k).launch(
                    k.name,
                    grid,
                    (k.threads, 1, 1),
                    k.smem,
                    stream,
                    [
                        *stage_ptrs,
                        *map(
                            ct.c_int,
                            (
                                b,
                                start,
                                count,
                                stride,
                                nw,
                                1 + int(not gr.is_contiguous()),
                                1 + int(not gi.is_contiguous()),
                            ),
                        ),
                        ct.c_bool(compact),
                        ct.c_float(eps),
                    ],
                )
    if dec:
        torch.sum(pole_part.view(nw, scan_groups, 4, m), dim=1, out=partial[:, :4, :])
    grad = scans._reduce_partials(partial, [*pole, *path], 8)
    return [*grad[:4], gs, *grad[4:]]


@vscan_v2_bwd.register_fake
def _fake(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode):
    _validate(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode)
    _, b, h, w, m = states.shape
    return [
        *(torch.empty_like(t) for t in pole),
        states.new_empty((4, b, h, w // 2 if compact else w, m)),
        *(torch.empty_like(t) for t in path),
    ]
