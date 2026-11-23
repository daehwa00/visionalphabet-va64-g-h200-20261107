"""Runtime-shaped CUDA scans with packed states and first-order autograd.

Horizontal planes are ordered (positive real, positive imag, negative real,
negative imag). Vertical outputs are planar and their path inputs are ordered
(PP, NP, PN, NN) for each Cartesian component. Vertical gain tables are detached.
The forward mode "reference" uses PyTorch forward with the same CUDA backward.
"""

from __future__ import annotations

# Custom op schemas and their fake implementations use positional flags.
# ruff: noqa: FBT001, FBT002
import ctypes as ct
import os
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch
from torch import Tensor
from torch._inductor import config as inductor_config

from .cuda_nvrtc import (
    _check_cu,
    _check_nvrtc,
    _ensure_context,
    _include_dirs,
    _load_cuda_driver,
    _nvrtc,
)
from .product_scan_reference import (
    _causal_weights,
    bidirectional_product_scan_reference,
    d4_grouped_path_collapse_packed_reference,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Constants and types

_WARP_SIZE = 32
_SCAN_WARPS = (4, 3, 2, 1)
_REGISTER_HIDDEN = 8
_STAGED_PATH_ROWS = 5
_MAX_CACHED_HIDDEN = 16
_MAX_SAVED_DP_CACHE_CHUNKS = 8

# Preserve the temporary storage budgets on devices with up to 24 GiB.
_MAX_PARTIAL_BYTES = 64 * 1024**2
_MAX_SCRATCH_BYTES = 64 * 1024**2
_BASE_BUDGET_VRAM = 24 * 1024**3
_DTYPES = {
    torch.bfloat16: "__nv_bfloat16",
    torch.float16: "__half",
    torch.float32: "float",
}


# Short scans use smaller checkpoints to keep more CTAs resident.
_SHORT_SCAN = 14
_SHORT_CHECKPOINT_CHUNK = 4
_CHECKPOINT_CHUNK = 8

# Persistent workers queue beyond residency to cover the scan tail.
_SHORT_VSCAN_WAVES = 3
_VSCAN_WAVES = 8
_VSCAN_UNCACHED_FORWARD_WAVES = 4
_HSCAN_WAVES = 32
_LONG_HSCAN_WAVES = 16

# Long scans trade shared line storage against resident warps.
_LONG_SCAN = 48
_SHORT_COARSE_VSCAN = 64
_NARROW_HSCAN = 96
_WIDE_HSCAN = 128
_VSCAN_SCRATCH_TASKS_PER_SM = 4
_COARSE_VSCAN_SCRATCH_TASKS_PER_SM = 8
_VSCAN_TWO_WARP_TASKS_PER_SM = 16
_VSCAN_FEW_MODE_TILES = 4
_NARROW_HSCAN_ACTIVE_WARPS = 8
_NARROW_VSCAN_ACTIVE_WARPS = 3
_HSCAN_BACKWARD_TIEBREAK_WARPS = 12
_HSCAN_UNCACHED_FORWARD_PENALTY = 4
_HSCAN_UNCACHED_BACKWARD_PENALTY = 6
# Column caches compete with the resident warps that hide strided memory latency.
_TRAJECTORY_MIN_SMEM = 128 * 1024


class ScanPlan(NamedTuple):
    warps: int
    smem: int
    chunk: int = _CHECKPOINT_CHUNK
    cache_weights: bool = False
    global_checkpoints: bool = False
    cache_line: bool = False
    shared_acc: bool = False
    full_trajectory: bool = False
    cache_source: bool = False
    shared_path_rows: int = 0


@dataclass(frozen=True)
class Kernel:
    dtype: torch.dtype
    warps: int
    backward: bool
    vertical: bool = False
    coarse: bool = False
    hidden8: bool = False
    pure_vertical: bool = False
    index32: bool = True
    chunk: int = _CHECKPOINT_CHUNK
    cache_weights: bool = False
    global_checkpoints: bool = False
    cache_line: bool = False
    shared_acc: bool = False
    precise: bool = False
    projection: bool = False
    projection_reduction: bool = False
    reduction: bool = False
    compact: bool = False
    real_interleaved: bool = False
    imag_interleaved: bool = False
    cache_source: bool = False
    real_transposed: bool = False
    imag_transposed: bool = False
    vector_weights: bool = False
    shared_path_rows: int = 0


class Launch(NamedTuple):
    plan: ScanPlan
    key: Kernel
    workers: int
    partials: int


# CUDA kernels

_CUDA_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<class T> __device__ float read(T x) { return static_cast<float>(x); }
template<class T> __device__ T write(float x) { return static_cast<T>(x); }
template<> __device__ float read(__half x) { return __half2float(x); }
template<> __device__ float read(__nv_bfloat16 x) { return __bfloat162float(x); }
template<> __device__ __half write(float x) { return __float2half_rn(x); }
template<> __device__ __nv_bfloat16 write(float x) { return __float2bfloat16_rn(x); }
template<class T> __device__ float rounded(float x) { return read(write<T>(x)); }

enum ScalarType { FP32, FP16, BF16 };

template<class I>
__device__ float load_scalar(const void* p, I offset, int dtype) {
    if (dtype == FP32) return static_cast<const float*>(p)[offset];
    if (dtype == FP16) return read(static_cast<const __half*>(p)[offset]);
    return read(static_cast<const __nv_bfloat16*>(p)[offset]);
}

struct Complex {
    float r, i;
    __device__ Complex operator+(Complex z) const { return {r + z.r, i + z.i}; }
    __device__ Complex operator*(Complex z) const {
        return {r * z.r - i * z.i, r * z.i + i * z.r};
    }
    __device__ Complex conj() const { return {r, -i}; }
};

__device__ float4 advance(float4 state, float4 source, Complex a, Complex g) {
    Complex u = a * Complex{state.x, state.y} + g * Complex{source.x, source.y};
    Complex v = a * Complex{state.z, state.w} + g * Complex{source.z, source.w};
    return make_float4(u.r, u.i, v.r, v.i);
}

template<class T, class I>
__device__ Complex load_complex(const T* p, I offset, I stride) {
    return {read(p[offset]), read(p[offset + stride])};
}

template<class T, class I>
__device__ void store_complex(T* p, I offset, I stride, Complex z) {
    p[offset] = write<T>(z.r);
    p[offset + stride] = write<T>(z.i);
}

template<class T> struct alignas(2 * sizeof(T)) Pair {
    T r, i;
    __device__ static Pair from(T r, T i) { return {r, i}; }
    __device__ static Pair pack(Complex x) { return {write<T>(x.r), write<T>(x.i)}; }
    __device__ Complex unpack() const { return {read(r), read(i)}; }
};

template<> struct Pair<__half> {
    __half2 value;
    __device__ static Pair from(__half r, __half i) { return {__halves2half2(r, i)}; }
    __device__ static Pair pack(Complex x) { return {__floats2half2_rn(x.r, x.i)}; }
    __device__ Complex unpack() const {
        const auto x = __half22float2(value);
        return {x.x, x.y};
    }
};

template<> struct Pair<__nv_bfloat16> {
    __nv_bfloat162 value;
    __device__ static Pair from(__nv_bfloat16 r, __nv_bfloat16 i) {
        return {__halves2bfloat162(r, i)};
    }
    __device__ static Pair pack(Complex x) { return {__floats2bfloat162_rn(x.r, x.i)}; }
    __device__ Complex unpack() const {
        const auto x = __bfloat1622float2(value);
        return {x.x, x.y};
    }
};

template<class T, class I, bool CACHED>
struct PathWeights;

template<class T, class I>
struct PathWeights<T, I, false> {
    static constexpr bool vector_weights = false;
    const void* __restrict__ wi;
    const void* __restrict__ bi;
    const void* __restrict__ wo;
    int types, m, size;

    __device__ float input(I field) const {
        return rounded<T>(load_scalar(wi, I(m) * size * 8 + field, types & 3));
    }

    __device__ Complex input_pair(I pair) const {
        return {input(2 * pair), input(2 * pair + 1)};
    }

    __device__ float bias(I field) const {
        return rounded<T>(load_scalar(bi, I(m) * size + field, (types >> 2) & 3));
    }

    __device__ float output(I field) const {
        return rounded<T>(load_scalar(wo, I(m) * size * 2 + field, (types >> 4) & 3));
    }
};

template<class T, class I>
struct PathWeights<T, I, true> {
    static constexpr bool vector_weights = false;
    const Pair<T>* inputs;
    const T* cache;
    int size;

    __device__ Complex input_pair(I pair) const { return inputs[pair * 32].unpack(); }
    __device__ float bias(I field) const { return read(cache[(8 * I(size) + field) * 32]); }
    __device__ float output(I field) const { return read(cache[(9 * I(size) + field) * 32]); }
};

template<class T, class I, bool CACHED>
__device__ auto path_weights(const T* cache, const void* wi, const void* bi,
                             const void* wo, int types, int m, int size, int lane) {
    if constexpr (CACHED) {
        return PathWeights<T, I, true>{reinterpret_cast<const Pair<T>*>(cache) + lane,
                                       cache + lane, size};
    } else {
        return PathWeights<T, I, false>{wi, bi, wo, types, m, size};
    }
}

template<class T, class I = long long>
__device__ void load_path_weights(T* weights_shared, const void* wi, const void* bi,
                                  const void* wo, int types, int mode0, int M,
                                  int size, int threads) {
    // Load each mode-major matrix in contiguous order, then transpose it
    // into field-major shared storage for the scan.
    const I tile_modes = min(32, M - mode0);
    auto* packed_inputs = reinterpret_cast<Pair<T>*>(weights_shared);
    for (I idx = threadIdx.x; idx < tile_modes * 4 * size; idx += threads) {
        const I mode = idx / (4 * I(size)), pair = idx % (4 * I(size));
        const I offset = 2 * (I(mode0) * 4 * size + idx);
        packed_inputs[pair * 32 + mode] = Pair<T>::pack(
            {load_scalar(wi, offset, types & 3), load_scalar(wi, offset + 1, types & 3)});
    }
    #pragma unroll
    for (int matrix = 1; matrix < 3; ++matrix) {
        const int width = matrix * size;
        const I field0 = (7 + matrix) * I(size);
        const void* src = matrix == 1 ? bi : wo;
        const int dtype = (types >> (2 * matrix)) & 3;
        for (I idx = threadIdx.x; idx < tile_modes * width; idx += threads) {
            const I mode = idx / width, field = idx % width;
            weights_shared[(field0 + field) * 32 + mode] =
                write<T>(load_scalar(src, I(mode0) * width + idx, dtype));
        }
    }
}

__device__ void pole_grad(Complex adj, Complex prev, Complex source, Complex& da,
                          Complex& dg, bool reverse) {
    const Complex ca = adj * prev.conj(), cg = adj * source.conj();
    da = da + (reverse ? ca.conj() : ca);
    dg = dg + (reverse ? cg.conj() : cg);
}

template<class T, class I, int WARPS, bool BACKWARD, bool CACHE, bool COMPACT,
         bool REAL_INTERLEAVED, bool IMAG_INTERLEAVED, bool REAL_TRANSPOSED, bool IMAG_TRANSPOSED>
__device__ __forceinline__ void horizontal(
    const float* __restrict__ dr, const float* __restrict__ di,
    const float* __restrict__ decay_gr, const float* __restrict__ decay_gi,
    const T* __restrict__ sr, const T* __restrict__ si,
    const T* __restrict__ states, const T* __restrict__ grad, T* __restrict__ out,
    float* __restrict__ partial, Pair<float>* __restrict__ scratch,
    int B, int H, int W, int M, int workers
) {
    const I stride_r = I(M) * (1 + REAL_INTERLEAVED);
    const I stride_i = I(M) * (1 + IMAG_INTERLEAVED);
    constexpr int threads = 32 * WARPS;
    extern __shared__ float4 shared[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int tiles = (M - 1) / 32 + 1;
    const int mode0 = (blockIdx.x % tiles) * 32, m = mode0 + lane;
    const int group = blockIdx.x / tiles;
    const int worker = group * WARPS + warp;
    const bool active = m < M && worker < workers;
    const I plane = I(B) * H * W * M;
    Complex da{}, dg{};
    if (active) {
        const Complex a{dr[m], di[m]}, g{decay_gr[m], decay_gi[m]};
        for (I line = worker; line < I(B) * H; line += workers) {
            const I base = line * W * M + m;
            const I original = (line / H) * W * H + line % H;
            const I base_r = (REAL_TRANSPOSED ? original : line * W) * stride_r + m;
            const I base_i = (IMAG_TRANSPOSED ? original : line * W) * stride_i + m;
            const I step_r = REAL_TRANSPOSED ? H * stride_r : stride_r;
            const I step_i = IMAG_TRANSPOSED ? H * stride_i : stride_i;
            auto source = [&](int x) -> Complex {
                return {read(sr[base_r + I(x) * step_r]), read(si[base_i + I(x) * step_i])};
            };
            if constexpr (!BACKWARD) {
                auto* cache = reinterpret_cast<Pair<T>*>(shared) + threadIdx.x;
                Complex pos{}, neg{};
                for (int x = 0; x < W; ++x) {
                    const auto input = Pair<T>::from(sr[base_r + I(x) * step_r],
                                                     si[base_i + I(x) * step_i]);
                    if constexpr (CACHE) cache[I(x) * threads] = input;
                    const auto s = input.unpack();
                    pos = a * pos + g * s;
                    store_complex(out, base + I(x) * M, plane, pos);
                }
                for (int x = W - 1; x >= 0; --x) {
                    const auto s = CACHE ? cache[I(x) * threads].unpack() : source(x);
                    neg = a.conj() * neg + g.conj() * s;
                    store_complex(out, 2 * plane + base + I(x) * M, plane, neg);
                }
            } else {
                const int saved = W / (1 + COMPACT);
                const I tile = I(worker) * M + mode0;
                Pair<float>* buf = CACHE ? reinterpret_cast<Pair<float>*>(shared) + threadIdx.x
                                        : scratch + tile * saved + lane;
                const I stride = CACHE ? threads : min(32, M - mode0);
                Complex right{}, left{}, neg{};
                const I grad_plane = COMPACT ? plane / 2 : plane;
                const I grad_base = COMPACT ? line * (W / 2) * M + m : base;
                #pragma unroll 4
                for (int x = 0; x < W; ++x) {
                    const I i = base + I(x) * M;
                    const I gi = COMPACT ? grad_base + I(x / 2) * M : i;
                    right = a * right + (COMPACT && (x & 1)
                        ? Complex{} : load_complex(grad, 2 * grad_plane + gi, grad_plane));
                    // Odd compact positions have no incoming right adjoint.
                    // Reconstruct them from the preceding even position.
                    if (!COMPACT || !(x & 1)) {
                        buf[I(x / (1 + COMPACT)) * stride] = Pair<float>::pack(right);
                    }
                }
                #pragma unroll 4
                for (int x = W - 1; x >= 0; --x) {
                    const I i = base + I(x) * M;
                    const I gi = COMPACT ? grad_base + I(x / 2) * M : i;
                    left = a.conj() * left + (COMPACT && !(x & 1)
                        ? Complex{} : load_complex(grad, gi, grad_plane));
                    auto right_adj = buf[I(x / (1 + COMPACT)) * stride].unpack();
                    if constexpr (COMPACT) {
                        if (x & 1) right_adj = a * right_adj;
                    }
                    const auto prev = x ? load_complex(states, i - M, plane) : Complex{};
                    const auto s = source(x);
                    const Complex next = [&]() -> Complex {
                        if constexpr (sizeof(T) == sizeof(float)) {
                            return x + 1 < W ? load_complex(states, 2 * plane + i + M, plane)
                                             : Complex{};
                        } else {
                            // This sweep already has the negative recurrence's input.
                            // Recreate its stored values without loading two more planes.
                            const auto stored = Pair<T>::pack(neg).unpack();
                            neg = a.conj() * neg + g.conj() * s;
                            return stored;
                        }
                    }();
                    da.r += left.r * prev.r + left.i * prev.i
                            + right_adj.r * next.r + right_adj.i * next.i;
                    da.i += left.i * prev.r - left.r * prev.i
                            + right_adj.r * next.i - right_adj.i * next.r;
                    const Complex sum = left + right_adj;
                    const Complex diff{left.r - right_adj.r, left.i - right_adj.i};
                    dg.r += sum.r * s.r + sum.i * s.i;
                    dg.i += diff.i * s.r - diff.r * s.i;
                    store_complex(out, 2 * (i - m) + m, I(M),
                                  {sum.r * g.r + diff.i * g.i, sum.i * g.r - diff.r * g.i});
                }
            }
        }
    }
    if constexpr (BACKWARD) {
        // Warps own separate lines for the same tile of modes.
        float4 sums = make_float4(da.r, da.i, dg.r, dg.i);
        if constexpr (WARPS > 1) {
            __syncthreads();
            shared[threadIdx.x] = sums;
            __syncthreads();
            if (warp == 0) {
                #pragma unroll
                for (int k = 1; k < WARPS; ++k) {
                    const float4 add = shared[k * 32 + lane];
                    sums.x += add.x;
                    sums.y += add.y;
                    sums.z += add.z;
                    sums.w += add.w;
                }
            }
        }
        if (warp == 0 && m < M) {
            const I row = 4 * I(group) * M + m;
            partial[row] = sums.x;
            partial[row + M] = sums.y;
            partial[row + 2 * I(M)] = sums.z;
            partial[row + 3 * I(M)] = sums.w;
        }
    }
}

template<bool PRECISE> struct Sum {
    float value = 0, correction = 0;
    __device__ void add(float x) {
        if constexpr (PRECISE) {
            const float adjusted = x - correction;
            const float updated = value + adjusted;
            correction = (updated - value) - adjusted;
            value = updated;
        } else {
            value += x;
        }
    }
};

template<class T> struct Quad {
    Pair<T> a, b;
    __device__ static Quad pack(float4 v) {
        return {Pair<T>::pack({v.x, v.y}), Pair<T>::pack({v.z, v.w})};
    }
    __device__ float4 unpack() const {
        auto x = a.unpack(), y = b.unpack();
        return make_float4(x.r, x.i, y.r, y.i);
    }
};

template<class T, class I>
struct Column {
    const T* states;
    const float* vy;
    I offset_a, offset_b, plane, source_plane, output, feature_plane;
    int H, W, M, OW, m;
    float vx_a, vx_b, eps;

    __device__ Quad<T> packed_source(int y) const {
        const I row = I(y) * W * M;
        const I ia = row + offset_a, ib = row + offset_b + 2 * source_plane;
        return {Pair<T>::from(states[ia], states[ia + source_plane]),
                Pair<T>::from(states[ib], states[ib + source_plane])};
    }

    __device__ float4 source(int y) const {
        const I row = I(y) * W * M;
        const Complex a = load_complex(states, row + offset_a, source_plane);
        const Complex b = load_complex(states, row + offset_b + 2 * source_plane, source_plane);
        return make_float4(a.r, a.i, b.r, b.i);
    }

    __device__ float2 gain(int y, bool reverse) const {
        float v = vy[(I(reverse) * H + y) * M + m];
        return make_float2(rsqrtf(fmaxf(vx_a * v, eps)), rsqrtf(fmaxf(vx_b * v, eps)));
    }

    __device__ I cell(int y) const { return output + I(y) * OW * M; }
};

template<class T>
__device__ Quad<T> normalize_packed(float4 v, float2 gain) {
    const auto stored = Quad<T>::pack(v).unpack();
    return Quad<T>::pack(make_float4(stored.x * gain.x, stored.y * gain.x,
                                     stored.z * gain.y, stored.w * gain.y));
}

template<class T>
__device__ float4 normalize(float4 v, float2 gain) {
    return normalize_packed<T>(v, gain).unpack();
}

template<class T, class I, int HIDDEN, bool PRECISE, bool BACKWARD, int SHARED_HIDDEN = 0,
         class Weights>
__device__ __forceinline__ float2 path_mlp(
    float4 pos, float4 neg, Weights path, float2 bias,
    float2 dy, float* partial, I partial_stride, float* dp,
    float* shared_dwi = nullptr, int shared_stride = 0
) {
    const int size = HIDDEN ? HIDDEN : path.size;
    const float p[8] = {pos.x, pos.z, neg.x, neg.z, pos.y, pos.w, neg.y, neg.w};
    Sum<PRECISE> out_r, out_i;
    if constexpr (BACKWARD) {
        partial[(11 * I(size) - 8 * SHARED_HIDDEN) * partial_stride] += dy.x;
        partial[(11 * I(size) + 1 - 8 * SHARED_HIDDEN) * partial_stride] += dy.y;
    }
    #pragma unroll
    for (int j = 0; j < size; ++j) {
        float pre = 0;
        float weights[8];
        if constexpr (Weights::vector_weights) {
            #pragma unroll
            for (int k = 0; k < 8; k += 4) {
                const auto v = path.input_group(2 * j + k / 4);
                weights[k] = v.x;
                weights[k + 1] = v.y;
                weights[k + 2] = v.z;
                weights[k + 3] = v.w;
                pre = __fmaf_rn(v.x, p[k], pre);
                pre = __fmaf_rn(v.y, p[k + 1], pre);
                pre = __fmaf_rn(v.z, p[k + 2], pre);
                pre = __fmaf_rn(v.w, p[k + 3], pre);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; k += 2) {
                const auto pair = path.input_pair(4 * I(j) + k / 2);
                weights[k] = pair.r;
                weights[k + 1] = pair.i;
                pre = __fmaf_rn(pair.r, p[k], pre);
                pre = __fmaf_rn(pair.i, p[k + 1], pre);
            }
        }
        pre = rounded<T>(rounded<T>(pre) + path.bias(j));
        const float sig = __fdividef(1.0f, 1.0f + __expf(-pre));
        const float act = rounded<T>(pre * sig);
        const float wr = path.output(j), wj = path.output(I(size) + j);
        if constexpr (BACKWARD) {
            const float dz = rounded<T>(rounded<T>(dy.x * wr + dy.y * wj) *
                                         (sig * (1.0f + pre * (1.0f - sig))));
            partial[(8 * I(size) + j - 8 * SHARED_HIDDEN) * partial_stride] += dz;
            partial[(9 * I(size) + j - 8 * SHARED_HIDDEN) * partial_stride] += dy.x * act;
            partial[(10 * I(size) + j - 8 * SHARED_HIDDEN) * partial_stride] += dy.y * act;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                dp[k] += dz * weights[k];
                if (j < SHARED_HIDDEN) shared_dwi[(8 * j + k) * shared_stride] += dz * p[k];
                else partial[(8 * (I(j) - SHARED_HIDDEN) + k) * partial_stride] += dz * p[k];
            }
        } else {
            out_r.add(wr * act);
            out_i.add(wj * act);
        }
    }
    if constexpr (!BACKWARD) {
        return make_float2(rounded<T>(out_r.value) + bias.x,
                           rounded<T>(out_i.value) + bias.y);
    }
    return dy;
}

template<bool REVERSE>
__device__ __forceinline__ float4 column_adjoint_step(
    float4 prev, float4 source, float4 inject,
    Complex a, Complex g, Complex& adj_a, Complex& adj_b, Complex& da, Complex& dg
) {
    adj_a = a.conj() * adj_a + Complex{inject.x, inject.y};
    adj_b = a.conj() * adj_b + Complex{inject.z, inject.w};
    pole_grad(adj_a, {prev.x, prev.y}, {source.x, source.y}, da, dg, REVERSE);
    pole_grad(adj_b, {prev.z, prev.w}, {source.z, source.w}, da, dg, REVERSE);
    const auto ds_a = g.conj() * adj_a, ds_b = g.conj() * adj_b;
    return make_float4(ds_a.r, ds_a.i, ds_b.r, ds_b.i);
}

template<bool REVERSE, bool COMPACT, bool ACCUMULATE = false, class T, class I>
__device__ __forceinline__ void write_column_gradient(
    Column<T, I> c, int y, float4 ds, T* gs, Quad<T> positive = {}
) {
    Complex ds_a{ds.x, ds.y}, ds_b{ds.z, ds.w};
    const I row = I(y) * c.W * c.M;
    const I plane = COMPACT ? c.plane / 2 : c.plane;
    const I ia = COMPACT ? (row + c.offset_a - c.m - c.M) / 2 + c.m : row + c.offset_a;
    const I ib = (COMPACT ? ia : row + c.offset_b) + 2 * plane;
    if constexpr (REVERSE) {
        // The two directions each differentiate through their own storage cast.
        ds_a = Complex{rounded<T>(ds.x), rounded<T>(ds.y)}
            + (ACCUMULATE ? load_complex(gs, ia, plane) : positive.a.unpack());
        ds_b = Complex{rounded<T>(ds.z), rounded<T>(ds.w)}
            + (ACCUMULATE ? load_complex(gs, ib, plane) : positive.b.unpack());
    }
    store_complex(gs, ia, plane, ds_a);
    store_complex(gs, ib, plane, ds_b);
}

template<bool COMPACT, class T, class I>
__device__ __forceinline__ void clear_unused_columns(Column<T, I> c, int y, T* gs) {
    if constexpr (!COMPACT) {
        const I row = I(y) * c.W * c.M;
        store_complex(gs, row + c.offset_a - c.M, c.plane, {});
        store_complex(gs, row + c.offset_b + c.M + 2 * c.plane, c.plane, {});
    }
}

template<bool REVERSE, bool COMPACT, class T, class I>
__device__ __forceinline__ void column_adjoint(
    Column<T, I> c, int y, float4 prev, float4 source, float4 inject,
    Complex a, Complex g, Complex& adj_a, Complex& adj_b, Complex& da, Complex& dg,
    T* gs
) {
    const auto ds = column_adjoint_step<REVERSE>(
        prev, source, inject, a, g, adj_a, adj_b, da, dg);
    write_column_gradient<REVERSE, COMPACT, REVERSE>(c, y, ds, gs);
}

struct ProjectWeights {
    static constexpr bool vector_weights = true;
    const float* __restrict__ cache;
    int size, lane;

    __device__ float4 input_group(int group) const {
        return reinterpret_cast<const float4*>(cache)[group * 32 + lane];
    }
    __device__ float bias(int field) const { return cache[(8 * size + field) * 32 + lane]; }
    __device__ float output(int field) const { return cache[(9 * size + field) * 32 + lane]; }
    __device__ float final_bias(int field) const { return cache[(11 * size + field) * 32 + lane]; }
};

template<class T, int HIDDEN, int THREADS, bool BACKWARD>
__device__ void load_project_weights(
    float* __restrict__ weights, const void* __restrict__ wi, const void* __restrict__ bi,
    const void* __restrict__ wo, const void* __restrict__ bo, int types, int mode0, int M,
    int hidden
) {
    const int size = HIDDEN ? HIDDEN : hidden;
    const int modes = min(32, M - mode0);
    #pragma unroll
    for (int matrix = 0; matrix < (BACKWARD ? 3 : 4); ++matrix) {
        const int width = matrix == 0 ? 8 * size : matrix == 3 ? 2 : matrix * size;
        const int field0 = matrix == 0 ? 0 : (7 + matrix + (matrix == 3)) * size;
        const void* src = matrix == 0 ? wi : matrix == 1 ? bi : matrix == 2 ? wo : bo;
        const int dtype = (types >> (2 * matrix)) & 3;
        for (int idx = threadIdx.x; idx < modes * width; idx += THREADS) {
            const int mode = idx / width, field = idx % width;
            // Group four weights per lane so a warp can load them as float4.
            const int slot = matrix == 0
                ? (field / 4 * 32 + mode) * 4 + field % 4 : (field0 + field) * 32 + mode;
            weights[slot] =
                rounded<T>(load_scalar(src, static_cast<long long>(mode0) * width + idx, dtype));
        }
    }
}

template<class T, class I, bool COARSE, int CHUNK, int THREADS, int HIDDEN,
         bool COMPACT, bool REAL_INTERLEAVED, bool IMAG_INTERLEAVED, bool CACHE_SOURCE,
         bool PURE, int SHARED_HIDDEN, class Weights>
__device__ __forceinline__ void column_backward(
    Column<T, I> c, Complex a, Complex g, Weights path,
    const T* gr, const T* gi, T* gs, float* acc, I acc_stride,
    float4* seg, Quad<T>* seg_src, float4* ckpt, I ckpt_stride, Quad<T>* g2, I g2_stride,
    int chunks, Complex& da, Complex& dg, float* shared_dwi
) {
    const auto ar = a.conj(), gg = g.conj();
    // CHUNK == 0 keeps the whole trajectory and needs no boundaries.
    constexpr bool full = CHUNK == 0;
    static_assert(!full || CACHE_SOURCE);
    const int span = full ? c.H : CHUNK;
    float4 pos{}, neg{};
    for (int block = 0; block < chunks; ++block) {
        const int y0 = block * span;
        const int length = min(span, c.H - y0);
        const bool last = full || block == chunks - 1;
        for (int j = 0; j < length; ++j) {
            float4 source;
            if constexpr (CACHE_SOURCE) {
                const auto packed = c.packed_source(y0 + j);
                seg_src[(y0 + j) * THREADS] = packed;
                source = packed.unpack();
            } else {
                source = c.source(y0 + j);
            }
            pos = advance(pos, source, a, g);
            if (last) {
                seg[j * THREADS] = pos;
                if constexpr (COARSE && !CACHE_SOURCE) seg_src[j * THREADS] = Quad<T>::pack(source);
            }
        }
        if (!last) ckpt[I(block) * ckpt_stride] = pos;
    }

    Complex adj_a{}, adj_b{};
    for (int block = chunks - 1; block >= 0; --block) {
        const int y0 = block * span;
        const int length = min(span, c.H - y0);
        Quad<T>* source_cache = seg_src + (CACHE_SOURCE ? I(y0) * THREADS : 0);
        const float4 before = !full && block ? ckpt[I(block - 1) * ckpt_stride] : float4{};
        if (!full && block + 1 < chunks) {
            // The positive boundary at this slot was consumed in the next
            // chunk. Replace it with this chunk's negative entry state.
            ckpt[I(block) * ckpt_stride] = neg;
            pos = before;
            for (int j = 0; j < length; ++j) {
                const auto source = CACHE_SOURCE
                    ? source_cache[j * THREADS].unpack() : c.source(y0 + j);
                pos = advance(pos, source, a, g);
                seg[j * THREADS] = pos;
                if constexpr (COARSE && !CACHE_SOURCE) seg_src[j * THREADS] = Quad<T>::pack(source);
            }
        }

        for (int j = length - 1; j >= 0; j -= COARSE ? 2 : 1) {
            const int y = y0 + j;
            const auto s = (COARSE || CACHE_SOURCE)
                ? source_cache[j * THREADS].unpack() : c.source(y);
            const auto next = neg;
            neg = advance(neg, s, ar, gg);
            auto se = s;
            const auto next_even = neg;
            if constexpr (COARSE) {
                se = source_cache[(j - 1) * THREADS].unpack();
                neg = advance(neg, se, ar, gg);
            }
            const int yo = COARSE ? y / 2 : y;
            const I cell = c.cell(yo);
            const auto gp = c.gain(y, false), gn = c.gain(y - int(COARSE), true);
            float dp[8]{};
            if constexpr (PURE) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) dp[k] = read(gr[I(k) * c.feature_plane + cell]);
            } else path_mlp<T, I, HIDDEN, false, true, SHARED_HIDDEN>(
                normalize<T>(seg[j * THREADS], gp), normalize<T>(neg, gn),
                path, {},
                make_float2(read(gr[(cell - c.m) * (1 + REAL_INTERLEAVED) + c.m]),
                            read(gi[(cell - c.m) * (1 + IMAG_INTERLEAVED) + c.m])),
                acc, acc_stride, dp, shared_dwi, THREADS);
            const auto ip = normalize<T>(make_float4(dp[0], dp[4], dp[1], dp[5]), gp);
            g2[I(yo) * g2_stride] =
                normalize_packed<T>(make_float4(dp[2], dp[6], dp[3], dp[7]), gn);

            column_adjoint<false, COMPACT>(
                c, y, j ? seg[(j - 1) * THREADS] : before, s, ip, a, g,
                adj_a, adj_b, da, dg, gs);
            if constexpr (COARSE) {
                column_adjoint<false, COMPACT>(
                    c, y - 1, j > 1 ? seg[(j - 2) * THREADS] : before, se, {}, a, g,
                    adj_a, adj_b, da, dg, gs);
            }
            if constexpr (full) {
                // The positive adjoint consumed these states. Reuse the slots
                // for the inputs of the negative recurrence.
                seg[j * THREADS] = next;
                if constexpr (COARSE) seg[(j - 1) * THREADS] = next_even;
            }
        }
    }

    adj_a = adj_b = {};
    for (int block = 0; block < chunks; ++block) {
        const int y0 = block * span;
        const int length = min(span, c.H - y0);
        Quad<T>* source_cache = seg_src + (CACHE_SOURCE ? I(y0) * THREADS : 0);
        if constexpr (!full) {
            neg = block + 1 < chunks ? ckpt[I(block) * ckpt_stride] : float4{};
            // Store the previous negative state for each adjoint update.
            for (int j = length - 1; j >= 0; --j) {
                const auto s = CACHE_SOURCE
                    ? source_cache[j * THREADS].unpack() : c.source(y0 + j);
                seg[j * THREADS] = neg;
                if constexpr (!CACHE_SOURCE) seg_src[j * THREADS] = Quad<T>::pack(s);
                neg = advance(neg, s, ar, gg);
            }
        }
        for (int j = 0; j < length; ++j) {
            const int y = y0 + j;
            const auto in = !COARSE || !(y & 1)
                ? g2[I(COARSE ? y / 2 : y) * g2_stride].unpack() : float4{};
            column_adjoint<true, COMPACT>(
                c, y, seg[j * THREADS], source_cache[j * THREADS].unpack(), in, ar, gg,
                adj_a, adj_b, da, dg, gs);
            if constexpr (COARSE) clear_unused_columns<COMPACT>(c, y, gs);
        }
    }
}

// Tile the CTA reduction so all scan warps can share one small staging buffer.
constexpr int vscan_reduce_tile = 32;

template<int WARPS, int FIELDS, int SHARED_WI, class I>
__device__ __forceinline__ void write_vscan_partial(
    float* shared, float* row, const float* gradients, Complex da, Complex dg, int m, int M,
    const float* shared_dwi = nullptr
) {
    constexpr int threads = WARPS * 32;
    const float pole[4] = {da.r, da.i, dg.r, dg.i};
    #pragma unroll
    for (int begin = 0; begin < FIELDS; begin += vscan_reduce_tile) {
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < vscan_reduce_tile && begin + j < FIELDS; ++j) {
            const int field = begin + j;
            if (field < 4) {
                shared[j * threads + threadIdx.x] = pole[field];
            } else if (field >= 4 + SHARED_WI) {
                shared[j * threads + threadIdx.x] = gradients[field - 4 - SHARED_WI];
            }
        }
        __syncthreads();
        if (threadIdx.x < 32 && m < M) {
            #pragma unroll
            for (int j = 0; j < vscan_reduce_tile && begin + j < FIELDS; ++j) {
                const int field = begin + j;
                const float* staged = field >= 4 && field < 4 + SHARED_WI
                    ? shared_dwi + (field - 4) * threads : shared + j * threads + threadIdx.x;
                float sum = staged[0];
                #pragma unroll
                for (int k = 1; k < WARPS; ++k) sum += staged[k * 32];
                row[I(begin + j) * M] = sum;
            }
        }
    }
}

template<class T, class I, bool COARSE, int WARPS, int CHUNK, int HIDDEN, bool BACKWARD,
         bool CACHE_WEIGHT, bool GLOBAL_CKPT, bool CACHE_LINE, bool SHARED_ACC, bool PRECISE,
         bool COMPACT, bool REAL_INTERLEAVED, bool IMAG_INTERLEAVED, bool CACHE_SOURCE, bool PURE,
         bool VECTOR_WEIGHTS, int SHARED_HIDDEN>
__device__ __forceinline__ void vertical(
    const float* __restrict__ dr, const float* __restrict__ di,
    const float* __restrict__ decay_gr, const float* __restrict__ decay_gi,
    const T* states, const float* vx, const float* vy,
    const void* __restrict__ wi, const void* __restrict__ bi,
    const void* __restrict__ wo, const void* __restrict__ bo,
    const T* gr, const T* gi,
    T* outr, T* outi, T* gs, float* partials,
    Quad<T>* scratch, float4* global_checkpoints, int B, int H, int W, int M,
    int workers, int hidden, int types, float eps, long long input_plane, long long grad_plane
) {
    constexpr int threads = WARPS * 32;
    extern __shared__ float4 shared[];
    const int tiles = (M - 1) / 32 + 1;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int mode0 = (blockIdx.x % tiles) * 32, m = mode0 + lane;
    const int group = blockIdx.x / tiles, worker = group * WARPS + warp;
    constexpr bool register_acc = BACKWARD && HIDDEN && !SHARED_ACC;
    constexpr bool reduce = (PURE || SHARED_ACC || register_acc) && WARPS > 1;
    const bool active = m < M && worker < workers;
    const int size = HIDDEN ? HIDDEN : hidden;
    const I fields = PURE ? 4 : 11 * I(size) + 6;
    const int ow = COARSE ? W / 2 : W, oh = COARSE ? H / 2 : H;
    constexpr bool full = CHUNK == 0;
    const int span = full ? H : CHUNK;
    const int chunks = full ? 1 : (H - 1) / span + 1;
    const I plane = I(B) * H * W * M;
    const I source_plane = PURE ? I(input_plane) : plane;
    const I target_plane = PURE && BACKWARD ? I(grad_plane) : plane;
    auto* weights_shared = reinterpret_cast<T*>(shared);
    const I weight_fields = 11 * I(size);
    if constexpr (VECTOR_WEIGHTS) {
        static_assert(CACHE_WEIGHT);
        load_project_weights<T, HIDDEN, threads, true>(
            reinterpret_cast<float*>(shared), wi, bi, wo, nullptr, types, mode0, M, size);
        __syncthreads();
    } else if constexpr (CACHE_WEIGHT) {
        load_path_weights<T, I>(weights_shared, wi, bi, wo, types, mode0, M, size, threads);
        __syncthreads();
    }
    const auto path = [&] {
        if constexpr (VECTOR_WEIGHTS) {
            return ProjectWeights{reinterpret_cast<const float*>(shared), size, lane};
        } else {
            return path_weights<T, I, CACHE_WEIGHT>(
                weights_shared, wi, bi, wo, types, m, size, lane);
        }
    }();
    constexpr int weight_itemsize = VECTOR_WEIGHTS ? sizeof(float) : sizeof(T);
    auto* seg = shared + (CACHE_WEIGHT ? weight_fields * 32 * weight_itemsize / 16 : 0);
    auto* ckpt = seg + (BACKWARD ? span * threads : 0);
    auto* seg_src = reinterpret_cast<Quad<T>*>(
        ckpt + (BACKWARD && !GLOBAL_CKPT ? I(chunks - 1) * threads : 0));
    auto* acc_base = reinterpret_cast<float*>(
        seg_src + (BACKWARD ? I(CACHE_SOURCE ? H : span) * threads : 0));
    auto* line_buf = reinterpret_cast<Quad<T>*>(acc_base + (SHARED_ACC ? fields * threads : 0));
    float* row = partials;
    constexpr int shared_wi = 8 * SHARED_HIDDEN;
    static_assert(!shared_wi || (register_acc && SHARED_HIDDEN <= HIDDEN));
    float local_acc[register_acc ? 11 * HIDDEN + 2 - shared_wi : 1]{};
    float* shared_dwi = nullptr;
    if constexpr (shared_wi) {
        const auto* storage_end = line_buf + (CACHE_LINE ? I(oh) * threads : 0);
        const I storage_floats = reinterpret_cast<const float*>(storage_end)
            - reinterpret_cast<float*>(shared);
        constexpr int reduce_floats = reduce ? vscan_reduce_tile * threads : 0;
        shared_dwi = reinterpret_cast<float*>(shared) + max(storage_floats, I(reduce_floats))
            + threadIdx.x;
        #pragma unroll
        for (int j = 0; j < shared_wi; ++j) shared_dwi[j * threads] = 0;
    }
    float* acc = acc_base + threadIdx.x;
    const I acc_stride = register_acc ? 1 : SHARED_ACC ? threads : M;
    if constexpr (BACKWARD) {
        row += I(reduce ? group : worker) * fields * M + m;
        if constexpr (register_acc) {
            acc = local_acc;
        } else if constexpr (!SHARED_ACC && !PURE) {
            acc = row + 4 * I(M);
            if (active) {
                for (I j = 0; j < fields - 4; ++j) acc[j * M] = 0;
            }
        } else if constexpr (SHARED_ACC) {
            for (I j = 4; j < fields; ++j) acc[j * threads] = 0;
            acc += 4 * threads;
        }
    }
    Complex da{}, dg{};
    if (active) {
        const Complex a{dr[m], di[m]}, g{decay_gr[m], decay_gi[m]};
        float2 bias{};
        if constexpr (!BACKWARD && !PURE) {
            bias = make_float2(rounded<T>(load_scalar(bo, 2 * I(m), types >> 6)),
                               rounded<T>(load_scalar(bo, 2 * I(m) + 1, types >> 6)));
        }
        for (I line = worker; line < I(B) * ow; line += workers) {
            const I b = line / ow;
            const int x = line % ow, xa = COARSE ? 2 * x + 1 : x, xb = COARSE ? 2 * x : x;
            const I base = b * H * W * M + m;
            Column<T, I> c{states, vy, base + I(xa) * M, base + I(xb) * M,
                           target_plane, source_plane, (b * oh * ow + x) * M + m,
                           I(B) * oh * ow * M,
                           H, W, M, ow, m,
                           vx[I(xa) * M + m], vx[(I(W) + xb) * M + m], eps};
            auto* buf = CACHE_LINE ? line_buf + threadIdx.x : scratch + I(worker) * M + m;
            const I stride = CACHE_LINE ? threads : I(workers) * M;
            if constexpr (BACKWARD) {
                auto* boundaries = GLOBAL_CKPT
                    ? global_checkpoints + I(worker) * M + m : ckpt + threadIdx.x;
                column_backward<T, I, COARSE, CHUNK, threads, HIDDEN, COMPACT,
                                REAL_INTERLEAVED, IMAG_INTERLEAVED, CACHE_SOURCE, PURE,
                                SHARED_HIDDEN>(
                    c, a, g, path, gr, gi, gs,
                    acc, acc_stride, seg + threadIdx.x, seg_src + threadIdx.x, boundaries,
                    GLOBAL_CKPT ? I(workers) * M : threads, buf, stride, chunks, da, dg,
                    shared_dwi);
            } else {
                float4 state{};
                for (int y = 0; y < H; ++y) {
                    state = advance(state, c.source(y), a, g);
                    if (!COARSE || (y & 1)) {
                        buf[I(COARSE ? y / 2 : y) * stride] =
                            normalize_packed<T>(state, c.gain(y, false));
                    }
                }
                state = {};
                for (int y = H - 1; y >= 0; --y) {
                    state = advance(state, c.source(y), a.conj(), g.conj());
                    if (COARSE && (y & 1)) continue;
                    const int yo = COARSE ? y / 2 : y;
                    const I cell = c.cell(yo);
                    if constexpr (PURE) {
                        const auto pos = buf[I(yo) * stride].unpack();
                        const auto neg = normalize<T>(state, c.gain(y, true));
                        const float p[8] = {pos.x, pos.z, neg.x, neg.z, pos.y, pos.w, neg.y, neg.w};
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            outr[I(k) * c.feature_plane + cell] = write<T>(p[k]);
                        }
                    } else {
                        const float2 out = path_mlp<T, I, HIDDEN, PRECISE, false>(
                            buf[I(yo) * stride].unpack(), normalize<T>(state, c.gain(y, true)),
                            path, bias,
                            {}, nullptr, 0, nullptr);
                        outr[(cell - m) * (1 + REAL_INTERLEAVED) + m] = write<T>(out.x);
                        outi[(cell - m) * (1 + IMAG_INTERLEAVED) + m] = write<T>(out.y);
                    }
                }
            }
        }
    }
    if constexpr (BACKWARD) {
        if constexpr (PURE && reduce) {
            write_vscan_partial<WARPS, 4, 0, I>(
                reinterpret_cast<float*>(shared), row, nullptr, da, dg, m, M);
        } else if constexpr (register_acc && reduce) {
            write_vscan_partial<WARPS, 11 * HIDDEN + 6, shared_wi, I>(
                reinterpret_cast<float*>(shared), row, local_acc, da, dg, m, M, shared_dwi);
        } else if constexpr (SHARED_ACC) {
            acc -= 4 * threads;
            acc[0] = da.r;
            acc[threads] = da.i;
            acc[2 * threads] = dg.r;
            acc[3 * threads] = dg.i;
            if constexpr (reduce) __syncthreads();
            if (m < M && (reduce ? warp == 0 : active)) {
                for (I j = 0; j < fields; ++j) {
                    float sum = acc[I(j) * threads];
                    if constexpr (reduce) {
                        #pragma unroll
                        for (int k = 1; k < WARPS; ++k) sum += acc[I(j) * threads + k * 32];
                    }
                    row[I(j) * M] = sum;
                }
            }
        } else if (active) {
            row[0] = da.r;
            row[M] = da.i;
            row[2 * I(M)] = dg.r;
            row[3 * I(M)] = dg.i;
            if constexpr (register_acc) {
                #pragma unroll
                for (int j = 0; j < shared_wi; ++j) row[(4 + I(j)) * M] = shared_dwi[j * threads];
                #pragma unroll
                for (int j = shared_wi; j < 11 * HIDDEN + 2; ++j) {
                    row[(4 + I(j)) * M] = local_acc[j - shared_wi];
                }
            }
        }
    }
}
template<class T, class I, int WARPS, int HIDDEN, bool CACHED, bool BACKWARD, bool PRECISE,
         bool REAL_INTERLEAVED, bool IMAG_INTERLEAVED, bool SHARED_ACC>
__device__ __forceinline__ void project_path(
    const T* __restrict__ features, const void* __restrict__ wi, const void* __restrict__ bi,
    const void* __restrict__ wo, const void* __restrict__ bo,
    const T* __restrict__ gr, const T* __restrict__ gi,
    T* __restrict__ real, T* __restrict__ imag, T* __restrict__ dp,
    float* __restrict__ partials, long long cells, long long feature_plane, int M, int hidden,
    int types
) {
    constexpr int threads = WARPS * 32;
    extern __shared__ float4 shared[];
    auto* accum = reinterpret_cast<float*>(shared);
    constexpr int SHARED_HIDDEN = SHARED_ACC ? HIDDEN / 2 : 0;
    constexpr int shared_fields = 8 * SHARED_HIDDEN;
    constexpr int register_fields = BACKWARD && HIDDEN ? 11 * HIDDEN + 2 - shared_fields : 1;
    auto* weights = accum + shared_fields * threads;
    const int size = HIDDEN ? HIDDEN : hidden;
    const int tiles = (M - 1) / 32 + 1;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int mode0 = (blockIdx.x % tiles) * 32, m = mode0 + lane;
    const int group = blockIdx.x / tiles, groups = gridDim.x / tiles;
    const int worker = group * WARPS + warp;
    const int fields = 11 * size + 2;
    if constexpr (CACHED) {
        load_project_weights<T, HIDDEN, threads, BACKWARD>(
            weights, wi, bi, wo, bo, types, mode0, M, size);
    }
    if constexpr (SHARED_HIDDEN) {
        #pragma unroll
        for (int j = 0; j < shared_fields; ++j) accum[j * threads + threadIdx.x] = 0;
    }
    if constexpr (CACHED) __syncthreads();
    const I plane = I(cells) * M, source_plane = I(feature_plane);
    float local_acc[register_fields]{};
    if (m < M) {
        const auto path = [&] {
            if constexpr (CACHED) return ProjectWeights{weights, size, lane};
            else return PathWeights<T, I, false>{wi, bi, wo, types, m, size};
        }();
        float* acc = local_acc;
        if constexpr (BACKWARD && !HIDDEN) {
            acc = partials + I(worker) * fields * M + m;
            for (int j = 0; j < fields; ++j) acc[I(j) * M] = 0;
        }
        float2 bias{};
        if constexpr (!BACKWARD) {
            if constexpr (CACHED) bias = make_float2(path.final_bias(0), path.final_bias(1));
            else bias = make_float2(rounded<T>(load_scalar(bo, 2 * I(m), types >> 6)),
                                    rounded<T>(load_scalar(bo, 2 * I(m) + 1, types >> 6)));
        }
        #pragma unroll 1
        for (I cell = worker; cell < I(cells); cell += groups * WARPS) {
            const I offset = cell * M + m;
            const I ri = cell * M * (1 + REAL_INTERLEAVED) + m;
            const I ii = cell * M * (1 + IMAG_INTERLEAVED) + m;
            float p[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) p[k] = read(features[I(k) * source_plane + offset]);
            const auto pos = make_float4(p[0], p[4], p[1], p[5]);
            const auto neg = make_float4(p[2], p[6], p[3], p[7]);
            float adj[8]{};
            float2 dy{};
            if constexpr (BACKWARD) dy = make_float2(read(gr[ri]), read(gi[ii]));
            const auto out = path_mlp<T, I, HIDDEN, PRECISE, BACKWARD, SHARED_HIDDEN>(
                pos, neg, path, bias, dy, acc, I(HIDDEN ? 1 : M), adj,
                accum + threadIdx.x, threads);
            if constexpr (BACKWARD) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) dp[I(k) * plane + offset] = write<T>(adj[k]);
            } else {
                real[ri] = write<T>(out.x);
                imag[ii] = write<T>(out.y);
            }
        }
    }
    if constexpr (BACKWARD && HIDDEN) {
        // Reuse the weight storage once all warps are done reading it.
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < register_fields; ++j) {
            accum[(shared_fields + j) * threads + threadIdx.x] = local_acc[j];
        }
        __syncthreads();
        if (warp == 0 && m < M) {
            auto* row = partials + I(group) * fields * M + m;
            for (int j = 0; j < fields; ++j) {
                float sum = accum[j * threads + lane];
                #pragma unroll
                for (int w = 1; w < WARPS; ++w) sum += accum[j * threads + lane + 32 * w];
                row[I(j) * M] = sum;
            }
        }
    }
}

__device__ void store_scalar(void* p, long long offset, int dtype, float value) {
    if (dtype == FP32) static_cast<float*>(p)[offset] = value;
    else if (dtype == FP16) static_cast<__half*>(p)[offset] = write<__half>(value);
    else static_cast<__nv_bfloat16*>(p)[offset] = write<__nv_bfloat16>(value);
}

template<bool VERTICAL, bool PROJECTION>
__device__ void reduce_partials(const float* partial, void* dr, void* di, void* gr, void* gi,
                                void* wi, void* bi, void* wo, void* bo,
                                int rows, int M, int hidden, int types) {
    // Horizontal reductions have only four fields. More mode tiles let them
    // distribute work across the GPU even when the mode count is small.
    constexpr int lanes = VERTICAL ? 16 : 8, groups = 128 / lanes;
    __shared__ float sums[groups][lanes];
    const int lane = threadIdx.x % lanes, group = threadIdx.x / lanes;
    const int tiles = (M - 1) / lanes + 1;
    const int m = (blockIdx.x % tiles) * lanes + lane;
    const int field = blockIdx.x / tiles;
    const long long stride = (VERTICAL ? 11LL * hidden + (PROJECTION ? 2 : 6) : 4) * M;
    float sum = 0;
    if (m < M) {
        const long long offset = static_cast<long long>(field) * M + m;
        for (int r = group; r < rows; r += groups) sum += partial[r * stride + offset];
    }
    sums[group][lane] = sum;
    __syncthreads();
    if (group != 0 || m >= M) return;
    #pragma unroll
    for (int r = 1; r < groups; ++r) sum += sums[r][lane];

    if (!PROJECTION && field == 0) store_scalar(dr, m, types & 3, sum);
    else if (!PROJECTION && field == 1) store_scalar(di, m, (types >> 2) & 3, sum);
    else if (!PROJECTION && field == 2) store_scalar(gr, m, (types >> 4) & 3, sum);
    else if (!PROJECTION && field == 3) store_scalar(gi, m, (types >> 6) & 3, sum);
    else if constexpr (VERTICAL) {
        int index = PROJECTION ? field : field - 4;
        const long long mode = m;
        if (index < 8 * hidden) {
            store_scalar(wi, mode * 8 * hidden + index, (types >> 8) & 3, sum);
        } else if ((index -= 8 * hidden) < hidden) {
            store_scalar(bi, mode * hidden + index, (types >> 10) & 3, sum);
        } else if ((index -= hidden) < 2 * hidden) {
            store_scalar(wo, mode * 2 * hidden + index, (types >> 12) & 3, sum);
        } else {
            store_scalar(bo, mode * 2 + index - 2 * hidden, (types >> 14) & 3, sum);
        }
    }
}

"""

# NVRTC compilation and driver launch

def _kernel_name(key: Kernel) -> str:
    if key.projection:
        return "path_mlp_backward" if key.backward else "path_mlp_forward"
    if key.projection_reduction:
        return "reduce_path_mlp_partials"
    if key.reduction:
        return "reduce_vscan_partials" if key.vertical else "reduce_hscan_partials"
    return "scan"


def _kernel_source(key: Kernel) -> str:
    """Instantiate a single entry point without modifying the device source."""
    if key.reduction:
        return _CUDA_SOURCE + f"""
extern "C" __global__ __launch_bounds__(128) void {_kernel_name(key)}(
    const float* partial, void* dr, void* di, void* gr, void* gi,
    void* wi, void* bi, void* wo, void* bo, int rows, int M, int hidden, int types
) {{
    reduce_partials<{str(key.vertical).lower()}, {str(key.projection_reduction).lower()}>(
        partial, dr, di, gr, gi, wi, bi, wo, bo, rows, M, hidden, types);
}}
"""
    io = _DTYPES[key.dtype]
    if key.projection:
        flags = ", ".join(str(flag).lower() for flag in (
            key.cache_weights, key.backward, key.precise,
            key.real_interleaved, key.imag_interleaved, key.shared_acc,
        ))
        hidden = _REGISTER_HIDDEN if key.hidden8 else 0
        index = "unsigned int" if key.index32 else "long long"
        return _CUDA_SOURCE + f"""
extern "C" __global__ __launch_bounds__({key.warps * 32}) void {_kernel_name(key)}(
    const {io}* __restrict__ features, const void* __restrict__ wi, const void* __restrict__ bi,
    const void* __restrict__ wo, const void* __restrict__ bo,
    const {io}* __restrict__ gr, const {io}* __restrict__ gi,
    {io}* __restrict__ real, {io}* __restrict__ imag, {io}* __restrict__ dp,
    float* __restrict__ partials, long long cells, long long feature_plane, int M, int hidden,
    int types
) {{
    project_path<{io}, {index}, {key.warps}, {hidden}, {flags}>(
        features, wi, bi, wo, bo, gr, gi, real, imag, dp, partials,
        cells, feature_plane, M, hidden, types);
}}
"""
    index = "int" if key.index32 else "long long"
    backward, coarse, cache_weights, global_ckpt, cache_line, shared_acc, precise = (
        str(flag).lower()
        for flag in (
            key.backward, key.coarse, key.cache_weights, key.global_checkpoints,
            key.cache_line, key.shared_acc, key.precise,
        )
    )
    bounds = f'extern "C" __global__ __launch_bounds__({key.warps * _WARP_SIZE}) void scan'
    compact = str(key.compact).lower()
    layout = f"{str(key.real_interleaved).lower()}, {str(key.imag_interleaved).lower()}"
    if key.vertical:
        hidden = _REGISTER_HIDDEN if key.hidden8 else 0
        cache_source = str(key.cache_source).lower()
        targs = (
            f"{io}, {index}, {coarse}, {key.warps}, {key.chunk}, {hidden}, "
            f"{backward}, {cache_weights}, {global_ckpt}, {cache_line}, "
            f"{shared_acc}, {precise}, {compact}, {layout}, {cache_source}, "
            f"{str(key.pure_vertical).lower()}, {str(key.vector_weights).lower()}, "
            f"{key.shared_path_rows}"
        )
        wrapper = f"""
{bounds}(
    const float* __restrict__ dr, const float* __restrict__ di,
    const float* __restrict__ decay_gr, const float* __restrict__ decay_gi,
    const {io}* __restrict__ states,
    const float* __restrict__ vx, const float* __restrict__ vy,
    const void* __restrict__ wi, const void* __restrict__ bi,
    const void* __restrict__ wo, const void* __restrict__ bo,
    const {io}* __restrict__ gr, const {io}* __restrict__ gi,
    {io}* __restrict__ outr, {io}* __restrict__ outi, {io}* __restrict__ gs,
    float* __restrict__ partials, Quad<{io}>* __restrict__ scratch,
    float4* __restrict__ checkpoints,
    int B, int H, int W, int M, int workers, int hidden, int types, float eps,
    long long input_plane, long long grad_plane
) {{
    vertical<{targs}>(dr, di, decay_gr, decay_gi, states, vx, vy, wi, bi, wo, bo, gr, gi,
                       outr, outi,
                       gs, partials, scratch, checkpoints, B, H, W, M, workers, hidden, types, eps,
                       input_plane, grad_plane);
}}
"""
    else:
        transposed = f"{str(key.real_transposed).lower()}, {str(key.imag_transposed).lower()}"
        wrapper = f"""
{bounds}(
    const float* __restrict__ dr, const float* __restrict__ di,
    const float* __restrict__ decay_gr, const float* __restrict__ decay_gi,
    const {io}* __restrict__ sr, const {io}* __restrict__ si,
    const {io}* __restrict__ states,
    const {io}* __restrict__ grad, {io}* __restrict__ out,
    float* __restrict__ partials, Pair<float>* __restrict__ scratch,
    int B, int H, int W, int M, int workers
) {{
    horizontal<{io}, {index}, {key.warps}, {backward}, {cache_line},
               {compact}, {layout}, {transposed}>(
        dr, di, decay_gr, decay_gi, sr, si, states, grad, out, partials, scratch,
        B, H, W, M, workers);
}}
"""
    return _CUDA_SOURCE + wrapper


def _compile_ptx(key: Kernel, capability: tuple[int, int]) -> ct.Array:
    nvrtc = _nvrtc()
    program = ct.c_void_p()
    source = _kernel_source(key).encode()
    _check_nvrtc(
        nvrtc,
        nvrtc.nvrtcCreateProgram(
            ct.byref(program),
            source,
            b"general_cuda_scans.cu",
            0,
            None,
            None,
        ),
    )
    fmad = (not key.vertical or key.backward or key.vector_weights) and not (
        key.projection and key.precise
    )
    flags = [
        f"--gpu-architecture=compute_{capability[0]}{capability[1]}".encode(),
        b"--std=c++17",
        b"--fmad=true" if fmad else b"--fmad=false",
        *(f"--include-path={path}".encode() for path in _include_dirs()),
    ]
    options = (ct.c_char_p * len(flags))(*flags)
    try:
        _check_nvrtc(nvrtc, nvrtc.nvrtcCompileProgram(program, len(flags), options), program)
        size = ct.c_size_t()
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), program)
        image = ct.create_string_buffer(size.value)
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTX(program, image), program)
    finally:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
    return image


@lru_cache
def device_limits(device: int) -> tuple[int, int, int]:
    props = torch.cuda.get_device_properties(device)
    limit = ct.c_int()
    driver = _load_cuda_driver()
    _check_cu(
        driver.cuDeviceGetAttribute(ct.byref(limit), 97, device),
        "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)",
    )
    max_threads = ct.c_int()
    _check_cu(
        driver.cuDeviceGetAttribute(ct.byref(max_threads), 39, device),
        "cuDeviceGetAttribute(MAX_THREADS_PER_MULTIPROCESSOR)",
    )
    return props.multi_processor_count, limit.value, max_threads.value // _WARP_SIZE


@lru_cache
def _l2_size(device: int) -> int:
    l2 = ct.c_int()
    driver = _load_cuda_driver()
    _check_cu(
        driver.cuDeviceGetAttribute(ct.byref(l2), 38, device),
        "cuDeviceGetAttribute(L2_CACHE_SIZE)",
    )
    return l2.value


@lru_cache
def _extra_workspace_budget(device: int) -> int:
    if not torch.cuda.is_available():
        return 0
    total = torch.cuda.get_device_properties(device).total_memory
    if total <= _BASE_BUDGET_VRAM:
        return 0
    # Bound additional scratch by both the cache size and a fraction of VRAM.
    return min(_l2_size(device), (total - _BASE_BUDGET_VRAM) // 1024)


@lru_cache
def _horizontal_cache_budget(device: int) -> int:
    # Leave room for current input and output cache lines.
    return min(_MAX_SCRATCH_BYTES, _l2_size(device) * 7 // 8)


@cache
def _function(device: int, key: Kernel) -> tuple[ct.CDLL, ct.c_void_p]:
    image = _compile_ptx(key, torch.cuda.get_device_capability(device))
    driver = _load_cuda_driver()
    module, func = ct.c_void_p(), ct.c_void_p()
    with torch.cuda.device(device):
        _ensure_context(driver, device)
        _check_cu(driver.cuModuleLoadData(ct.byref(module), image), "cuModuleLoadData")
        try:
            _check_cu(
                driver.cuModuleGetFunction(ct.byref(func), module, _kernel_name(key).encode()),
                "cuModuleGetFunction",
            )
            if not key.reduction:
                _configure_smem(driver, func, device_limits(device)[1])
        except RuntimeError:
            driver.cuModuleUnload(module)
            raise
    return driver, func


def _configure_smem(driver: ct.CDLL, function: ct.c_void_p, smem: int) -> None:
    if smem > 48 * 1024:
        _check_cu(
            driver.cuFuncSetAttribute(function, 8, smem),
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
        )


@lru_cache(maxsize=256)
def _resident(device: int, key: Kernel, smem: int) -> int:
    driver, func = _function(device, key)
    blocks = ct.c_int()
    with torch.cuda.device(device):
        _ensure_context(driver, device)
        _check_cu(
            driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ct.byref(blocks),
                func,
                key.warps * _WARP_SIZE,
                ct.c_size_t(smem),
            ),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )
    return blocks.value


def _dispatch(source: Tensor, launch: Launch | None, args: tuple) -> None:
    if launch is None:
        raise RuntimeError("CUDA scan is ineligible on this device")
    plan, key, workers, _ = launch
    grid = _ceil(source.shape[-1], _WARP_SIZE) * _ceil(workers, plan.warps)
    _launch_kernel(source, key, grid, plan.smem, args)


def _launch_kernel(source: Tensor, key: Kernel, grid: int, smem: int, args: tuple) -> None:
    device = source.device.index or 0
    with torch.cuda.device(device):
        driver, fn = _function(device, key)
        _ensure_context(driver, device)
        stream = torch.cuda.current_stream(source.device).cuda_stream
        argv = (ct.c_void_p * len(args))(*(ct.addressof(arg) for arg in args))
        _check_cu(
            driver.cuLaunchKernel(
                fn,
                grid,
                1,
                1,
                key.warps * _WARP_SIZE,
                1,
                1,
                smem,
                ct.c_void_p(stream),
                argv,
                None,
            ),
            f"cuLaunchKernel({_kernel_name(key)})",
        )


# Hardware-aware launch planner

def _ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def _trajectories(
    length: int, chunk: int, cache_trajectories: bool
) -> list[tuple[int, bool, bool]]:
    plans = [(chunk, False, False)]
    if cache_trajectories:
        plans.append((length, True, True))
        plans.extend((span, False, True) for span in (chunk, 2 * chunk) if span < length)
    return plans


def _scan_plans(
    length: int,
    limit: int,
    *,
    vertical: bool,
    backward: bool,
    coarse: bool,
    hidden: int,
    itemsize: int,
    compact: bool = False,
    cache_trajectories: bool = True,
    pure_vertical: bool = False,
) -> list[ScanPlan]:
    plans = []
    chunk = _SHORT_CHECKPOINT_CHUNK if length <= _SHORT_SCAN else _CHECKPOINT_CHUNK
    trajectories = _trajectories(length, chunk, vertical and backward and cache_trajectories)
    for warps in _SCAN_WARPS:
        threads = warps * _WARP_SIZE
        if not vertical:
            stored = length // 2 if backward and compact else length
            line = stored * threads * (8 if backward else 2 * itemsize)
            minimum = threads * 16 if backward and warps > 1 else 0
            plans.extend(
                ScanPlan(warps, max(minimum, line if cached else 0), cache_line=cached)
                for cached in (True, False)
                if max(minimum, line if cached else 0) <= limit
            )
            continue

        weights = 11 * hidden * _WARP_SIZE * itemsize
        line = (length // 2 if coarse else length) * threads * 4 * itemsize
        acc = (11 * hidden + 6) * threads * 4
        cache_options = (True, False) if hidden <= _MAX_CACHED_HIDDEN else (False,)
        for cache_weights in ((False,) if pure_vertical else cache_options):
            for shared_acc in ((True, False) if backward and not pure_vertical else (False,)):
                fixed = (weights if cache_weights else 0) + (acc if shared_acc else 0)
                reduction = (
                    (4 if pure_vertical else _WARP_SIZE) * threads * 4
                    if backward and (pure_vertical or hidden == _REGISTER_HIDDEN) and warps > 1
                    else 0
                )
                for span, full, cache_source in trajectories:
                    source_rows = length if cache_source else span
                    seg = threads * (16 * span + 4 * itemsize * source_rows) if backward else 0
                    boundaries = ((length - 1) // span) * threads * 16 if backward else 0
                    global_ckpt = bool(boundaries) and fixed + seg + boundaries > limit
                    storage = fixed + seg + (0 if global_ckpt else boundaries)
                    for cache_line in (True, False):
                        smem = max(reduction, storage + (line if cache_line else 0))
                        if smem <= limit:
                            plans.append(ScanPlan(
                                warps, smem, span, cache_weights, global_ckpt, cache_line,
                                shared_acc, full, cache_source,
                            ))
    return plans


def _workers(
    lines: int, modes: int, fields: int, warps: int, wave: int, budget: int = _MAX_PARTIAL_BYTES
) -> int:
    if fields == 0:
        return lines
    target = _ceil(wave, _ceil(modes, _WARP_SIZE)) * warps
    max_workers = budget // (4 * modes * fields)
    return min(lines, target, max(1, max_workers))


def _shape_supported(shape: tuple[int, ...]) -> bool:
    if len(shape) != 4 or any(not 0 < size < 2**31 for size in shape):
        return False
    b, h, w, m = shape
    return max(b * h, b * w) * _ceil(m, _WARP_SIZE) < 2**31


def _scratch_per_worker(plan: ScanPlan, length: int, *, vertical: bool, coarse: bool,
                        backward: bool, itemsize: int, compact: bool = False) -> int:
    """Return global scratch storage per worker and mode, in bytes."""
    if not vertical:
        return length // (1 + int(compact)) * 8 if backward and not plan.cache_line else 0
    outputs = length // (1 + int(coarse))
    line = 0 if plan.cache_line else outputs * 4 * itemsize
    checkpoints = (length - 1) // plan.chunk * 16 if plan.global_checkpoints else 0
    return line + checkpoints


def _horizontal_lines_fit_l2(
    device: int, length: int, modes: int, itemsize: int, *,
    backward: bool, compact: bool, tasks: int, sms: int,
) -> bool:
    # Uncached forward reloads both components. Backward stores FP32 adjoints.
    line_bytes = length * 2 * itemsize
    if backward:
        line_bytes = length // (1 + int(compact)) * 8
    cache_lines = _horizontal_cache_budget(device) // max(1, modes * line_bytes)
    return cache_lines * _ceil(modes, _WARP_SIZE) >= min(tasks, sms)


def _warp_preference(
    plan: ScanPlan,
    length: int,
    active_warps: int,
    achievable: int,
    *,
    vertical: bool,
    backward: bool,
    compact: bool,
    prefer_two_warps: bool,
) -> int:
    # At equal occupancy, narrower CTAs spread work and reduce synchronization.
    if prefer_two_warps:
        return -abs(plan.warps - 2)
    if not vertical and backward and achievable >= _HSCAN_BACKWARD_TIEBREAK_WARPS:
        if length < _SHORT_SCAN:
            return -abs(plan.warps - 2)
        return -plan.warps if compact and length >= _LONG_SCAN else plan.warps
    if vertical:
        narrow = (
            not backward and length > _SHORT_COARSE_VSCAN
            and active_warps <= _NARROW_VSCAN_ACTIVE_WARPS
        )
    else:
        narrow = plan.cache_line and (
            length >= _NARROW_HSCAN or (backward and active_warps < _NARROW_HSCAN_ACTIVE_WARPS)
        )
    return -plan.warps if narrow else plan.warps


def _with_vector_path_weights(
    device: int,
    launch: Launch | None,
    *,
    modes: int,
    hidden: int,
    wave_limit: int,
) -> Launch | None:
    if launch is None:
        return None
    plan, key, workers, _ = launch
    if (
        not key.vertical or key.pure_vertical or key.dtype.itemsize != 2
        or hidden != _REGISTER_HIDDEN or modes < _WARP_SIZE
        or not plan.cache_weights or plan.shared_acc
    ):
        return launch
    # Packing saves SMEM for uncached coarse forward and for partial mode tiles
    # in backward, where fewer lanes amortize the FP32 cache.
    if (not key.backward and key.coarse and not plan.cache_line) or (
        key.backward and modes % _WARP_SIZE
    ):
        return launch
    sms, limit, max_warps = device_limits(device)
    added_bytes = 11 * hidden * _WARP_SIZE * (4 - key.dtype.itemsize)
    smem = plan.smem + added_bytes
    if limit < _TRAJECTORY_MIN_SMEM or smem > limit:
        return launch
    vector = replace(key, vector_weights=True, precise=False)
    resident = _resident(device, vector, smem)
    original = _resident(device, key, plan.smem)
    enough_blocks = min(_ceil(max_warps // 4, plan.warps), _ceil(3 * original, 4))
    if resident < max(1, enough_blocks):
        return launch

    waves = wave_limit
    if not key.backward:
        waves = (
            _VSCAN_WAVES if plan.cache_line or key.coarse else _VSCAN_UNCACHED_FORWARD_WAVES
        )
    tiles = _ceil(modes, _WARP_SIZE)
    workers = min(workers, _ceil(waves * sms * resident, tiles) * plan.warps)
    partials = _ceil(workers, plan.warps) if key.backward else workers
    return Launch(plan._replace(smem=smem), vector, workers, partials)


def _kernel_local_bytes(device: int, key: Kernel) -> int:
    driver, function = _function(device, key)
    local_bytes = ct.c_int()
    with torch.cuda.device(device):
        _ensure_context(driver, device)
        _check_cu(
            driver.cuFuncGetAttribute(ct.byref(local_bytes), 3, function),
            "cuFuncGetAttribute(LOCAL_SIZE_BYTES)",
        )
    return local_bytes.value


@lru_cache(maxsize=256)
def _with_shared_path_gradients(device: int, launch: Launch, policy: str) -> Launch:
    plan, key, _, _ = launch
    if (
        policy not in ("on", "auto") or not key.vertical or not key.backward
        or key.pure_vertical or not key.hidden8 or plan.shared_acc or not plan.cache_weights
    ):
        return launch

    # Keep the CTA grid and all CUDA allocations fixed. Trade some of the
    # FP32 dW1 registers for a lane-interleaved shared tile.
    threads = plan.warps * _WARP_SIZE
    accumulator_bytes = 8 * _STAGED_PATH_ROWS * threads * 4
    smem = plan.smem + accumulator_bytes
    _, limit, _ = device_limits(device)
    if smem > limit:
        return launch
    staged = replace(key, shared_path_rows=_STAGED_PATH_ROWS)
    resident = _resident(device, staged, smem)
    if not resident or (policy == "auto" and resident <= _resident(device, key, plan.smem)):
        return launch
    if _kernel_local_bytes(device, staged):
        return launch
    return launch._replace(
        plan=plan._replace(smem=smem, shared_path_rows=_STAGED_PATH_ROWS), key=staged,
    )


@lru_cache(maxsize=256)
def _launch_for_shape(
    device: int,
    dtype: torch.dtype,
    shape: tuple[int, int, int, int],
    index_bound: int,
    *,
    backward: bool,
    vertical: bool,
    coarse: bool,
    hidden: int,
    precise: bool,
    compact: bool = False,
    interleaved: tuple[bool, bool] = (False, False),
    transposed: tuple[bool, bool] = (False, False),
    pure_vertical: bool = False,
) -> Launch | None:
    b, h, w, m = shape
    sms, limit, max_warps_per_sm = device_limits(device)
    length = h if vertical else w
    plans = _scan_plans(
        length, limit,
        vertical=vertical, backward=backward, coarse=coarse, hidden=hidden,
        itemsize=dtype.itemsize, compact=compact,
        cache_trajectories=limit >= _TRAJECTORY_MIN_SMEM,
        pure_vertical=pure_vertical,
    )
    fields = ((11 * hidden + 6) if vertical and not pure_vertical else 4) if backward else 0
    lines = b * ((w // 2 if coarse else w) if vertical else h)
    mode_tiles = _ceil(m, _WARP_SIZE)
    column_tasks = lines * mode_tiles
    extra_budget = _extra_workspace_budget(device)
    horizontal_scratch = (
        not vertical and _LONG_SCAN <= w < _WIDE_HSCAN
        and _horizontal_lines_fit_l2(
            device, w, m, dtype.itemsize, backward=backward, compact=compact,
            tasks=column_tasks, sms=sms,
        )
    )
    # Prefer global scratch when column tiles can keep the SMs busy.
    scratch_tasks_per_sm = (
        _COARSE_VSCAN_SCRATCH_TASKS_PER_SM if coarse and h <= _SHORT_COARSE_VSCAN
        else _VSCAN_SCRATCH_TASKS_PER_SM
    )
    forward_scratch = (
        vertical and not backward and h >= _LONG_SCAN
        and column_tasks >= scratch_tasks_per_sm * sms
        and hidden <= _MAX_CACHED_HIDDEN and dtype.itemsize == 2
    )
    many_column_tasks = column_tasks >= _VSCAN_TWO_WARP_TASKS_PER_SM * sms
    prefer_two_warps = (
        vertical and backward and hidden == _REGISTER_HIDDEN and h >= _LONG_SCAN
        and (
            (not coarse and many_column_tasks)
            or (h > _SHORT_COARSE_VSCAN and mode_tiles < _VSCAN_FEW_MODE_TILES
                and not many_column_tasks)
        )
    )
    # Hidden 8 can retain its unrolled path accumulators in registers. Other
    # widths use shared accumulators when available.
    def locality(plan: ScanPlan) -> tuple[bool, bool, bool]:
        acc_local = not backward or not vertical or plan.shared_acc != (hidden == _REGISTER_HIDDEN)
        preferred_line = (vertical and backward) or horizontal_scratch or plan.cache_line
        if forward_scratch:
            preferred_line = not plan.cache_line and plan.warps == 1
        if not vertical and w >= _WIDE_HSCAN:
            # A narrow CTA keeps long scan lines in shared memory.
            preferred_line = plan.cache_line and plan.warps == 1
        return acc_local, not vertical or plan.cache_weights, preferred_line

    wave_limit = (
        (_SHORT_VSCAN_WAVES if h <= _SHORT_SCAN else _VSCAN_WAVES) if vertical
        else (_LONG_HSCAN_WAVES if w >= _LONG_SCAN else _HSCAN_WAVES)
    )
    uncached_penalty = (
        _HSCAN_UNCACHED_BACKWARD_PENALTY if backward else _HSCAN_UNCACHED_FORWARD_PENALTY
    )
    best = best_score = None
    ranked = sorted(
        ((locality(plan), plan) for plan in plans), key=lambda item: item[0], reverse=True
    )
    for local, plan in ranked:
        if best_score is not None and local < best_score[0]:
            break
        reduce = backward and (
            not vertical or pure_vertical or plan.shared_acc or hidden == _REGISTER_HIDDEN
        )
        probe_workers = _workers(
            lines, m, fields, plan.warps, sms * max_warps_per_sm * wave_limit,
            _MAX_PARTIAL_BYTES + extra_budget,
        )
        bound = max(index_bound, fields * probe_workers * m, (11 * hidden + 6) * m)
        key = Kernel(
            dtype=dtype,
            warps=plan.warps,
            backward=backward,
            vertical=vertical,
            coarse=coarse,
            hidden8=hidden == _REGISTER_HIDDEN,
            pure_vertical=pure_vertical,
            index32=bound < 2**31,
            # This variant takes its full-column span from runtime H.
            chunk=0 if plan.full_trajectory else plan.chunk,
            cache_weights=plan.cache_weights,
            global_checkpoints=plan.global_checkpoints,
            cache_line=plan.cache_line,
            shared_acc=plan.shared_acc,
            precise=precise and vertical and not backward,
            compact=compact,
            real_interleaved=interleaved[0],
            imag_interleaved=interleaved[1],
            cache_source=plan.cache_source,
            real_transposed=transposed[0],
            imag_transposed=transposed[1],
        )
        resident = _resident(device, key, plan.smem)
        active_warps = resident * plan.warps
        workers = _workers(
            lines, m, fields, plan.warps, sms * resident * wave_limit,
            _MAX_PARTIAL_BYTES + extra_budget,
        )
        if not backward and vertical:
            # Coarse lines store half as many states and can queue more workers.
            waves = (
                _VSCAN_WAVES if plan.cache_line or coarse else _VSCAN_UNCACHED_FORWARD_WAVES
            )
            workers = min(lines, _ceil(waves * sms * resident, mode_tiles) * plan.warps)
        scratch = m * _scratch_per_worker(
            plan, length,
            vertical=vertical, coarse=coarse, backward=backward, itemsize=dtype.itemsize,
            compact=compact,
        )
        if scratch:
            workers = min(workers, (_MAX_SCRATCH_BYTES + extra_budget) // scratch)
        if not resident or not workers:
            continue
        groups = _ceil(workers, plan.warps)
        achievable = min(active_warps, _ceil(workers * mode_tiles, sms))
        # With few lines, wide CTAs can leave SMs idle at the same warp count.
        covered_sms = min(sms, groups * mode_tiles)
        warp_preference = _warp_preference(
            plan, length, active_warps, achievable,
            vertical=vertical, backward=backward, compact=compact,
            prefer_two_warps=prefer_two_warps,
        )
        # Uncached horizontal lines generate extra global transactions even
        # when the working set fits in L2.
        occupancy = (
            achievable if vertical or plan.cache_line
            else max(1, achievable // uncached_penalty)
        )
        # Source caching saves repeated global loads. The full trajectory also
        # saves both checkpoint replay sweeps.
        score = (
            local, occupancy * (6 if plan.full_trajectory else 5 if plan.cache_source else 4),
            covered_sms,
            plan.cache_source, plan.cache_line, not plan.global_checkpoints, warp_preference,
        )
        if best_score is None or score > best_score:
            best_score, best = score, Launch(plan, key, workers, groups if reduce else workers)
    return _with_vector_path_weights(device, best, modes=m, hidden=hidden, wave_limit=wave_limit)


def _plan_launch(
    source: Tensor,
    shape: tuple[int, int, int, int],
    *,
    backward: bool,
    vertical: bool,
    coarse: bool,
    hidden: int,
    precise: bool,
    compact: bool = False,
    interleaved: tuple[bool, bool] = (False, False),
    transposed: tuple[bool, bool] = (False, False),
    pure_vertical: bool = False,
) -> Launch | None:
    if not source.is_cuda or source.dtype not in _DTYPES:
        return None
    # Vertical input already includes every plane. Horizontal forward takes
    # one plane and writes four. Both bounds also cover the dense outputs.
    index_bound = source.numel() * (2 if pure_vertical else 1)
    if pure_vertical:
        index_bound = max(index_bound, source.stride(0) * 4)
    if not vertical:
        index_bound *= 4
    device = source.device.index or 0
    launch = _launch_for_shape(
        device, source.dtype, shape, index_bound,
        backward=backward, vertical=vertical, coarse=coarse, hidden=hidden, precise=precise,
        compact=compact, interleaved=interleaved, transposed=transposed,
        pure_vertical=pure_vertical,
    )

    if launch is not None and vertical and backward and not pure_vertical:
        launch = _with_shared_path_gradients(
            device, launch, os.environ.get("LNET_VSCAN_FUSED_OPT", "auto")
        )
    return launch


# Custom ops and public API

def _eligible(
    source: Tensor,
    shape: tuple[int, int, int, int],
    **options: bool | int | tuple[bool, bool],
) -> bool:
    if not source.is_cuda or source.dtype not in _DTYPES or not _shape_supported(shape):
        return False
    try:
        return all(
            _plan_launch(source, shape, backward=backward, **options) is not None
            for backward in (False, True)
        )
    except (RuntimeError, OSError, AttributeError):
        return False


def _matching(source: Tensor, values: list[Tensor]) -> bool:
    return all(t.device == source.device and t.dtype in _DTYPES for t in values)


def _pole_ok(pole: list[Tensor], source: Tensor) -> bool:
    return (
        len(pole) == 4
        and _matching(source, pole)
        and all(p.numel() == source.shape[-1] for p in pole)
    )


def _vertical_inputs_ok(
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    *,
    coarse: bool,
) -> bool:
    if states.ndim != 5 or states.shape[0] != 4 or len(auxiliary) != 2 or len(path) != 4:
        return False
    _, _, h, w, m = states.shape
    if (coarse and (h % 2 or w % 2)) or not _matching(states, [*auxiliary, *path]):
        return False
    if [t.shape for t in auxiliary] != [(2, w, m), (2, h, m)] or path[0].ndim != 3:
        return False
    hidden = path[0].shape[1]
    return 0 < hidden < 2**31 and [t.shape for t in path] == [
        (m, hidden, 8),
        (m, hidden),
        (m, 2, hidden),
        (m, 2),
    ]


def can_use_general_hscan(real: Tensor, imag: Tensor, pole: list[Tensor] | None = None) -> bool:
    return (
        real.shape == imag.shape
        and real.dtype == imag.dtype
        and real.device == imag.device
        and real.ndim == 4
        and (pole is None or _pole_ok(pole, real))
        and _eligible(
            real, tuple(real.shape), vertical=False, coarse=False, hidden=0, precise=False,
            **_horizontal_layout(real, imag),
        )
    )


def can_use_general_vscan(
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    *,
    coarse: bool = False,
    precise: bool = True,
    pole: list[Tensor] | None = None,
) -> bool:
    return (
        _vertical_inputs_ok(states, auxiliary, path, coarse=coarse)
        and (pole is None or _pole_ok(pole, states))
        and _eligible(
            states,
            tuple(states.shape[1:]),
            vertical=True,
            coarse=coarse,
            hidden=path[0].shape[1],
            precise=precise,
        )
    )


def _ptr(tensor: Tensor | None) -> ct.c_void_p:
    return ct.c_void_p(tensor.data_ptr() if tensor is not None else 0)


_SCALAR_TYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


def _scalar_types(tensors: list[Tensor]) -> int:
    return sum(_SCALAR_TYPES[t.dtype] << (2 * i) for i, t in enumerate(tensors))


def _reduce_partials(
    partial: Tensor, params: list[Tensor], hidden: int = 0, *, projection: bool = False,
) -> list[Tensor]:
    rows, fields, modes = partial.shape
    gradients = [p.new_empty(p.shape) for p in params]
    outputs = gradients if hidden else [*gradients, None, None, None, None]
    if projection:
        outputs = [None] * 4 + gradients
    types = _scalar_types(params) << (8 if projection else 0)
    key = Kernel(
        torch.float32, 4, backward=True, vertical=bool(hidden), reduction=True,
        projection_reduction=projection,
    )
    _launch_kernel(
        partial,
        key,
        _ceil(modes, 16 if hidden else 8) * fields,
        0,
        (
            *map(_ptr, (partial, *outputs)),
            *map(ct.c_int, (rows, modes, hidden, types)),
        ),
    )
    return gradients


def _is_interleaved(source: Tensor) -> bool:
    _, h, w, m = source.shape
    stride = 2 * m
    return not source.is_contiguous() and source.stride() == (h * w * stride, w * stride, stride, 1)


def _contiguous_or_interleaved(source: Tensor) -> Tensor:
    return source if _is_interleaved(source) else source.contiguous()


def _horizontal_source_layout(source: Tensor) -> tuple[bool, bool]:
    if source.is_contiguous():
        return False, False
    if _is_interleaved(source):
        return True, False
    # Columns of an NHWM tensor can use the same scan with a larger source step.
    original = source.transpose(1, 2)
    interleaved = _is_interleaved(original)
    return interleaved, interleaved or original.is_contiguous()


def _horizontal_layout(real: Tensor, imag: Tensor) -> dict[str, tuple[bool, bool]]:
    real_interleaved, real_transposed = _horizontal_source_layout(real)
    imag_interleaved, imag_transposed = _horizontal_source_layout(imag)
    return {
        "interleaved": (real_interleaved, imag_interleaved),
        "transposed": (real_transposed, imag_transposed),
    }


def _hscan(
    pole: list[Tensor],
    real: Tensor,
    imag: Tensor,
    states: Tensor | None = None,
    adjoints: Tensor | None = None,
    *,
    compact: bool = False,
) -> tuple[Tensor, list[Tensor] | None]:
    shape = tuple(real.shape)
    if (
        not _shape_supported(shape)
        or real.shape != imag.shape
        or imag.dtype != real.dtype
        or imag.device != real.device
        or not _pole_ok(pole, real)
    ):
        raise ValueError("Invalid horizontal scan inputs")
    backward = adjoints is not None
    b, h, w, m = shape
    adjoint_shape = (4, b, h, w // 2 if compact else w, m)
    if backward and (
        states is None
        or tuple(states.shape) != (4, b, h, w, m)
        or tuple(adjoints.shape) != adjoint_shape
        or (compact and w % 2)
        or states.dtype != real.dtype
        or adjoints.dtype != real.dtype
        or states.device != real.device
        or adjoints.device != real.device
    ):
        raise ValueError("Expected packed horizontal states and adjoints matching the source")
    layout = _horizontal_layout(real, imag)
    real_input, imag_input = (
        value if interleaved or transposed else value.contiguous()
        for value, interleaved, transposed in zip(
            (real, imag), layout["interleaved"], layout["transposed"], strict=True,
        )
    )
    launch = _plan_launch(
        real, shape, backward=backward, vertical=False, coarse=False, hidden=0, precise=False,
        compact=compact,
        **layout,
    )
    if launch is None:
        raise RuntimeError("Horizontal CUDA scan is ineligible")
    if backward:
        out = real.new_empty((b, h, w, 2 * m))
        partial = real.new_empty((launch.partials, 4, m), dtype=torch.float32)
        scratch = (
            None if launch.plan.cache_line else
            real.new_empty((w // (1 + int(compact)), launch.workers, m, 2), dtype=torch.float32)
        )
    else:
        out = real.new_empty((4, b, h, w, m))
        partial = scratch = None
    inputs = (
        *(p.float().contiguous() for p in pole),
        real_input,
        imag_input,
        states.contiguous() if states is not None else None,
        adjoints.contiguous() if adjoints is not None else None,
    )
    _dispatch(
        real,
        launch,
        (
            *map(_ptr, (*inputs, out, partial, scratch)),
            *map(ct.c_int, (*shape, launch.workers)),
        ),
    )
    if partial is not None:
        return out, _reduce_partials(partial, pole)
    return out, None


def _vertical_adjoints(
    grad: tuple[Tensor, Tensor], states: Tensor, shape: tuple[int, int, int, int]
) -> tuple[Tensor, Tensor]:
    if any(
        t.shape != shape or t.dtype != states.dtype or t.device != states.device for t in grad
    ):
        raise ValueError("Invalid vertical adjoints")
    return _contiguous_or_interleaved(grad[0]), _contiguous_or_interleaved(grad[1])


def _vscan(  # noqa: C901, PLR0912, PLR0915
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool,
    epsilon: float,
    precise: bool,
    grad: tuple[Tensor, Tensor] | None = None,
    *,
    packed: bool = False,
    planar: bool = False,
    compact: bool = False,
    features: Tensor | None = None,
    saved_features: Tensor | None = None,
    state_grad: Tensor | None = None,
    pure: bool = False,
) -> Tensor | tuple[Tensor, Tensor] | list[Tensor]:
    shape = tuple(states.shape[1:])
    if (
        not _shape_supported(shape)
        or not _vertical_inputs_ok(states, auxiliary, path, coarse=coarse)
        or not _pole_ok(pole, states)
        or not 0 < epsilon < float("inf")
        or (compact and not coarse)
    ):
        raise ValueError("Invalid vertical scan inputs")
    b, h, w, m = shape
    hidden = 0 if pure else path[0].shape[1]
    backward = grad is not None or features is not None
    oh, ow = (h // 2, w // 2) if coarse else (h, w)
    out_shape = (b, oh, ow, m)
    checkpoints = None
    output: Tensor | tuple[Tensor, Tensor] | None = None
    interleaved = (packed and not planar,) * 2
    if grad is not None:
        gr, gi = _vertical_adjoints(grad, states, out_shape)
        interleaved = (not gr.is_contiguous(), not gi.is_contiguous())
    if saved_features is not None and (
        grad is None
        or saved_features.shape != (8, b * oh * ow, m)
        or saved_features.device != states.device
        or saved_features.dtype != states.dtype
        or not saved_features.is_contiguous()
    ):
        raise ValueError("Invalid saved vertical features")
    if not pure:
        chunk = (
            _saved_backward_chunk(states, coarse)
            if saved_features is not None else _decoupled_chunk(states, coarse)
        )
        if chunk:
            with torch.autocast(device_type="cuda", enabled=False):
                return _decoupled_vscan(
                    pole, states, auxiliary, path, coarse, epsilon, precise,
                    (gr, gi) if grad is not None else None,
                    packed=packed, planar=planar, compact=compact, chunk=chunk,
                    saved_features=saved_features,
                )
    launch = _plan_launch(
        states,
        shape,
        backward=backward,
        vertical=True,
        coarse=coarse,
        hidden=hidden,
        precise=precise,
        compact=compact,
        interleaved=interleaved,
        pure_vertical=pure,
    )
    if launch is None:
        raise RuntimeError("Vertical CUDA scan is ineligible")
    if backward:
        if pure:
            gr, gi = features, None
        outr = outi = None
        # Coarse selects odd columns from positive and even columns from negative.
        gs = (
            states.new_empty((4, b, h, ow if compact else w, m))
            if state_grad is None else state_grad
        )
        fields = 4 if pure else 11 * hidden + 6
        partial = states.new_empty((launch.partials, fields, m), dtype=torch.float32)
        checkpoint_rows = _ceil(h, launch.plan.chunk) - 1
        if launch.plan.global_checkpoints and checkpoint_rows:
            checkpoints = states.new_empty(
                (checkpoint_rows, launch.workers, m, 4), dtype=torch.float32
            )
    else:
        if pure:
            output = outr = states.new_empty((8, b * oh * ow, m))
            outi = None
        elif packed and planar:
            output = states.new_empty((2, *out_shape))
            outr, outi = output.unbind(0)
        elif packed:
            output = states.new_empty((b, oh, ow, 2 * m))
            outr, outi = output.split(m, dim=-1)
        else:
            outr, outi = states.new_empty(out_shape), states.new_empty(out_shape)
            output = outr, outi
        gr = gi = gs = partial = None
    scratch = (
        None if launch.plan.cache_line else
        states.new_empty((oh, launch.workers, m, 4))
    )
    inputs = (
        *(p.float().contiguous() for p in pole),
        states if pure else states.contiguous(),
        *(t.float().contiguous() for t in auxiliary),
        *(None if pure else tensor.contiguous() for tensor in path),
    )
    _dispatch(
        states,
        launch,
        (
            *map(_ptr, (*inputs, gr, gi, outr, outi, gs, partial, scratch, checkpoints)),
            *map(ct.c_int, (
                *shape, launch.workers, hidden, _scalar_types(path),
            )),
            ct.c_float(epsilon),
            ct.c_longlong(states.stride(0) if pure else 0),
            ct.c_longlong(gs.stride(0) * (1 + int(compact)) if pure and gs is not None else 0),
        ),
    )
    if output is not None:
        return output
    if partial is None or gs is None:
        raise RuntimeError("Missing vertical gradients")
    gradients = _reduce_partials(partial, pole if pure else [*pole, *path], hidden)
    return [*gradients[:4], gs, *gradients[4:]]


def _decoupled_chunk(states: Tensor, coarse: bool) -> int:
    """Limit ephemeral features and adjoints to the hardware workspace budget."""
    policy = os.environ.get("LNET_VSCAN_DECOUPLED")
    if not states.is_cuda or policy not in {"chunk", "saved", "test_saved"}:
        return 0
    return _workspace_vscan_chunk(states, coarse)


def _workspace_vscan_chunk(states: Tensor, coarse: bool) -> int:
    _, batch, height, width, modes = states.shape
    step = 2 if coarse else 1
    per_sample = 16 * (height // step) * (width // step) * modes * states.element_size()
    vram = torch.cuda.get_device_properties(states.device).total_memory
    budget = min(2 * 1024**3, vram // 16)
    # Projection overlaps both feature planes with its partial reduction;
    # scan scratch overlaps only one plane.
    live_planes = min(budget - _MAX_PARTIAL_BYTES, 2 * (budget - 3 * _MAX_SCRATCH_BYTES))
    return min(batch, max(0, live_planes // per_sample))


def _saved_cache_chunk(states: Tensor, coarse: bool) -> int:
    _, batch, height, width, modes = states.shape
    step = 2 if coarse else 1
    dp_bytes = 8 * (height // step) * (width // step) * modes * states.element_size()
    cache_budget = 2 * _l2_size(states.device.index or 0)
    return min(batch, max(1, cache_budget // dp_bytes))


def _saved_backward_chunk(states: Tensor, coarse: bool) -> int:
    workspace = max(1, _workspace_vscan_chunk(states, coarse))
    cache_chunk = min(workspace, _saved_cache_chunk(states, coarse))
    # When L2-sized chunks would repeat the scan and reductions many times,
    # use the bounded workspace to keep those kernels large.
    return cache_chunk if states.shape[1] <= _MAX_SAVED_DP_CACHE_CHUNKS * cache_chunk else workspace


@torch.compiler.assume_constant_result
def _total_vram(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).total_memory


def _should_save_vscan(states: Tensor, coarse: bool) -> bool:
    policy = os.environ.get("LNET_VSCAN_DECOUPLED")
    if not states.is_cuda or policy not in {"saved", "test_saved"}:
        return False
    vram = _total_vram(states.device)
    if policy == "saved" and vram < 48 * 1024**3:
        return False
    # Reserve space for activations and all of the model's saved feature planes.
    budget = vram // 12
    _, b, h, w, m = states.shape
    step = 2 if coarse else 1
    return 8 * b * (h // step) * (w // step) * m * states.element_size() <= budget


@lru_cache(maxsize=512)
def _plan_projection(
    device: int, base: Kernel, cells: int, modes: int, hidden: int,
) -> tuple[Kernel, int, int]:
    tiles = _ceil(modes, _WARP_SIZE)
    sms, limit, _ = device_limits(device)
    fields = 11 * hidden + 2
    weights_bytes = (fields - 2 * base.backward) * _WARP_SIZE * 4
    cached = weights_bytes <= min(24 * 1024, limit)
    # Register accumulation measured faster than the shared accumulator on
    # both Ada and Hopper, even when sharing raised the active warp count.
    best = None
    for warps in ((2, 4, 8) if base.backward else (4, 8)):
        smem = max(
            weights_bytes if cached else 0,
            fields * warps * _WARP_SIZE * 4 if base.backward and base.hidden8 else 0,
        )
        if smem > limit:
            continue
        key = replace(base, warps=warps, cache_weights=cached)
        resident = _resident(device, key, smem)
        if not resident:
            continue
        groups = max(1, min(_ceil(cells, 16 * warps), _ceil(4 * sms * resident, tiles)))
        if base.backward:
            rows_per_group = 1 if base.hidden8 else warps
            bytes_per_group = rows_per_group * fields * modes * 4
            groups = min(groups, max(1, _MAX_PARTIAL_BYTES // bytes_per_group))
        score = min(sms * resident, groups * tiles) * warps
        if best is None or score > best[0]:
            best = score, key, groups, smem
    if best is None:
        raise RuntimeError("No CUDA path projection launch fits this device")
    return best[1:]


def _project_path(
    features: Tensor, path: list[Tensor], real: Tensor, imag: Tensor, *,
    backward: bool, precise: bool,
) -> tuple[Tensor, list[Tensor]] | None:
    _, cells, modes = features.shape
    hidden = path[0].shape[1]
    feature_plane = features.stride(0)
    base = Kernel(
        features.dtype, 0, backward, projection=True,
        precise=precise and not backward and features.dtype == torch.float32,
        hidden8=hidden == _REGISTER_HIDDEN, index32=8 * feature_plane < 2**32,
        real_interleaved=not real.is_contiguous(), imag_interleaved=not imag.is_contiguous(),
    )
    key, groups, smem = _plan_projection(features.device.index or 0, base, cells, modes, hidden)
    tiles = _ceil(modes, _WARP_SIZE)
    # Features can be a cell slice of the saved full-batch planes.
    # The pure scan consumes densely packed output planes.
    adjoints = features.new_empty(features.shape) if backward else None
    partial = None
    if backward:
        rows = groups if key.hidden8 else groups * key.warps
        partial = features.new_empty((rows, 11 * hidden + 2, modes), dtype=torch.float32)
    _launch_kernel(
        features, key, groups * tiles, smem,
        (
            *map(_ptr, (
                features, *path, real if backward else None, imag if backward else None,
                None if backward else real, None if backward else imag, adjoints, partial,
            )),
            ct.c_longlong(cells),
            ct.c_longlong(feature_plane),
            *map(ct.c_int, (modes, hidden, _scalar_types(path))),
        ),
    )
    if partial is not None and adjoints is not None:
        float_path = [parameter.to(dtype=torch.float32) for parameter in path]
        return adjoints, _reduce_partials(partial, float_path, hidden, projection=True)
    return None


def _decoupled_vscan(  # noqa: C901, PLR0912
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool,
    epsilon: float,
    precise: bool,
    grad: tuple[Tensor, Tensor] | None,
    *,
    packed: bool,
    planar: bool,
    compact: bool,
    chunk: int,
    saved_features: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor] | list[Tensor]:
    _, batch, height, width, modes = states.shape
    oh, ow = (height // 2, width // 2) if coarse else (height, width)
    weights = [parameter.contiguous() for parameter in path]
    if grad is None:
        shape = (batch, oh, ow, modes)
        if packed and planar:
            output = states.new_empty((2, *shape))
            real, imag = output.unbind(0)
        elif packed:
            output = states.new_empty((*shape[:-1], 2 * modes))
            real, imag = output.split(modes, dim=-1)
        else:
            real, imag = states.new_empty(shape), states.new_empty(shape)
            output = real, imag
    else:
        gs = states.new_empty((4, batch, height, ow if compact else width, modes))
        sums = [param.new_zeros(param.shape, dtype=torch.float32) for param in [*pole, *path]]
        float_pole = [param.float() for param in pole]

    states = states.contiguous()
    for begin in range(0, batch, chunk):
        part = slice(begin, begin + chunk)
        source = states[:, part]
        if saved_features is None:
            features = _vscan(pole, source, auxiliary, path, coarse, epsilon, precise, pure=True)
        else:
            area = oh * ow
            features = saved_features[:, begin * area : min(begin + chunk, batch) * area]
        if not isinstance(features, Tensor):
            raise TypeError("Expected normalized scan features")
        if grad is None:
            _project_path(
                features, weights, real[part], imag[part], backward=False, precise=precise,
            )
            del features
            continue

        projected = _project_path(
            features, weights, grad[0][part], grad[1][part], backward=True, precise=False,
        )
        del features
        if projected is None:
            raise TypeError("Expected path projection gradients")
        dp, path_grads = projected
        gradients = _vscan(
            float_pole, source, auxiliary, path, coarse, epsilon, precise,
            compact=compact, features=dp, state_grad=gs[:, part], pure=True,
        )
        if not isinstance(gradients, list):
            raise TypeError("Expected pure scan gradients")
        for total, value in zip(sums, (*gradients[:4], *path_grads), strict=True):
            total.add_(value.reshape_as(total))
        del projected, dp

    if grad is None:
        return output
    reduced = [total.to(param.dtype) for total, param in zip(sums, [*pole, *path], strict=True)]
    return [*reduced[:4], gs, *reduced[4:]]


def _reference_vertical(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    if not _vertical_inputs_ok(states, auxiliary, path, coarse=coarse):
        raise ValueError("Invalid vertical reference inputs")
    dr, di, gr, gi = (p.reshape(-1).float() for p in pole)
    vx, vy = (v.detach().float() for v in auxiliary)
    height = states.shape[2]
    dtype = states.dtype
    real, imag = [], []
    for reverse in (False, True):
        decay = torch.complex(dr, -di if reverse else di)
        gamma = torch.complex(gr, -gi if reverse else gi)
        weights = _causal_weights(decay, height)
        for side in (0, 1):
            source = torch.complex(states[2 * side].float(), states[2 * side + 1].float())
            if reverse:
                source = source.flip(1)
            value = torch.einsum("yjm,bjwm->bywm", weights, gamma * source)
            if reverse:
                value = value.flip(1)
            ys = slice(int(not reverse), None, 2) if coarse else slice(None)
            xs = slice(1 - side, None, 2) if coarse else slice(None)
            inverse = vy[int(reverse), ys, None, :] * vx[side, None, xs, :]
            inverse = inverse.clamp_min(epsilon).rsqrt()
            selected = value[:, ys, xs]
            real.append((selected.real.to(dtype).float() * inverse).to(dtype))
            imag.append((selected.imag.to(dtype).float() * inverse).to(dtype))
    packed = torch.cat((torch.stack(real, dim=-2), torch.stack(imag, dim=-2)), dim=-1)
    out = d4_grouped_path_collapse_packed_reference(packed, *path)
    modes = states.shape[-1]
    return out[..., :modes].contiguous().clone(), out[..., modes:].contiguous().clone()


@torch.library.custom_op("lnet::general_hscan_forward", mutates_args=())
def horizontal_forward(
    pole: list[Tensor],
    real: Tensor,
    imag: Tensor,
    mode: str = "cuda",
) -> Tensor:
    if mode == "reference":
        return torch.stack(bidirectional_product_scan_reference(tuple(pole), (real, imag)))
    if mode != "cuda":
        raise ValueError(f"Unknown scan forward mode {mode!r}")
    return _hscan(pole, real, imag)[0]


@horizontal_forward.register_fake
def _fake_hforward(
    pole: list[Tensor],  # noqa: ARG001
    real: Tensor,
    imag: Tensor,  # noqa: ARG001
    mode: str = "cuda",  # noqa: ARG001
) -> Tensor:
    return real.new_empty((4, *real.shape))


@torch.library.custom_op("lnet::general_hscan_backward", mutates_args=())
def horizontal_backward(
    pole: list[Tensor],
    source: list[Tensor],
    states: Tensor,
    adjoints: Tensor,
    compact: bool = False,
) -> list[Tensor]:
    out, gradients = _hscan(pole, *source, states, adjoints, compact=compact)
    if gradients is None:
        raise RuntimeError("Missing horizontal gradients")
    return [*gradients, out]


@horizontal_backward.register_fake
def _fake_hbackward(
    pole: list[Tensor],
    source: list[Tensor],
    states: Tensor,  # noqa: ARG001
    adjoints: Tensor,  # noqa: ARG001
    compact: bool = False,  # noqa: ARG001
) -> list[Tensor]:
    real = source[0]
    return [
        *(p.new_empty(p.shape) for p in pole),
        real.new_empty((*real.shape[:-1], 2 * real.shape[-1])),
    ]


@torch.library.custom_op("lnet::general_vscan_forward", mutates_args=())
def _vertical_forward(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool = False,
    epsilon: float = 1e-6,
    precise: bool = True,
    mode: str = "cuda",
) -> tuple[Tensor, Tensor]:
    if mode == "reference":
        return _reference_vertical(pole, states, auxiliary, path, coarse, epsilon)
    if mode != "cuda":
        raise ValueError(f"Unknown scan forward mode {mode!r}")
    out = _vscan(pole, states, auxiliary, path, coarse, epsilon, precise)
    if not isinstance(out, tuple):
        raise TypeError("Expected vertical outputs")
    return out


@_vertical_forward.register_fake
def _fake_vforward(
    pole: list[Tensor],  # noqa: ARG001
    states: Tensor,
    auxiliary: list[Tensor],  # noqa: ARG001
    path: list[Tensor],  # noqa: ARG001
    coarse: bool = False,
    epsilon: float = 1e-6,  # noqa: ARG001
    precise: bool = True,  # noqa: ARG001
    mode: str = "cuda",  # noqa: ARG001
) -> tuple[Tensor, Tensor]:
    _, b, h, w, m = states.shape
    step = 2 if coarse else 1
    shape = (b, h // step, w // step, m)
    return states.new_empty(shape), states.new_empty(shape)


@torch.library.custom_op("lnet::general_vscan_forward_packed", mutates_args=())
def vertical_forward_packed(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool,
    epsilon: float,
    planar: bool,
) -> Tensor:
    # Autograd belongs to the packed pipeline that calls this op.
    out = _vscan(pole, states, auxiliary, path, coarse, epsilon, True, packed=True, planar=planar)  # noqa: FBT003
    if not isinstance(out, Tensor):
        raise TypeError("Expected packed vertical output")
    return out


@vertical_forward_packed.register_fake
def _fake_vforward_packed(
    pole: list[Tensor],  # noqa: ARG001
    states: Tensor,
    auxiliary: list[Tensor],  # noqa: ARG001
    path: list[Tensor],  # noqa: ARG001
    coarse: bool,
    epsilon: float,  # noqa: ARG001
    planar: bool,
) -> Tensor:
    _, b, h, w, m = states.shape
    step = 2 if coarse else 1
    shape = (b, h // step, w // step)
    return states.new_empty((2, *shape, m) if planar else (*shape, 2 * m))


@torch.library.custom_op("lnet::general_vscan_backward", mutates_args=())
def vertical_backward(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    grad_real: Tensor,
    grad_imag: Tensor,
    coarse: bool = False,
    epsilon: float = 1e-6,
    compact: bool = False,
    saved_features: Tensor | None = None,
) -> list[Tensor]:
    out = _vscan(
        pole, states, auxiliary, path, coarse, epsilon, False, (grad_real, grad_imag),  # noqa: FBT003
        compact=compact, saved_features=saved_features,
    )
    if not isinstance(out, list):
        raise TypeError("Expected vertical gradients")
    return out


@vertical_backward.register_fake
def _fake_vbackward(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],  # noqa: ARG001
    path: list[Tensor],
    grad_real: Tensor,  # noqa: ARG001
    grad_imag: Tensor,  # noqa: ARG001
    coarse: bool = False,  # noqa: ARG001
    epsilon: float = 1e-6,  # noqa: ARG001
    compact: bool = False,
    saved_features: Tensor | None = None,  # noqa: ARG001
) -> list[Tensor]:
    _, b, h, w, m = states.shape
    return [
        *(p.new_empty(p.shape) for p in pole),
        states.new_empty((4, b, h, w // 2 if compact else w, m)),
        *(p.new_empty(p.shape) for p in path),
    ]


@torch.library.custom_op("lnet::general_vscan_forward_saved", mutates_args=())
def _vertical_forward_saved(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool,
    epsilon: float,
    planar: bool,
    precise: bool,
    separate: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    _, b, h, w, m = states.shape
    step = 2 if coarse else 1
    shape = (b, h // step, w // step, m)
    with torch.autocast(device_type="cuda", enabled=False):
        features = _vscan(
            pole, states.contiguous(), auxiliary, path, coarse, epsilon, precise, pure=True,
        )
        if not isinstance(features, Tensor):
            raise TypeError("Expected normalized scan features")
        if separate:
            output, secondary = states.new_empty(shape), states.new_empty(shape)
            real, imag = output, secondary
        else:
            output = states.new_empty((2, *shape) if planar else (*shape[:-1], 2 * m))
            real, imag = output.unbind(0) if planar else output.split(m, dim=-1)
            secondary = states.new_empty((0,))
        _project_path(
            features, [param.contiguous() for param in path], real, imag,
            backward=False, precise=precise,
        )
    return output, secondary, features


@_vertical_forward_saved.register_fake
def _fake_vertical_forward_saved(
    pole: list[Tensor],  # noqa: ARG001
    states: Tensor,
    auxiliary: list[Tensor],  # noqa: ARG001
    path: list[Tensor],  # noqa: ARG001
    coarse: bool,
    epsilon: float,  # noqa: ARG001
    planar: bool,
    precise: bool,  # noqa: ARG001
    separate: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    _, b, h, w, m = states.shape
    step = 2 if coarse else 1
    shape = (b, h // step, w // step, m)
    return (
        states.new_empty(shape if separate else ((2, *shape) if planar else (*shape[:-1], 2 * m))),
        states.new_empty(shape if separate else (0,)),
        states.new_empty((8, b * (h // step) * (w // step), m)),
    )


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    needs_input_grad: tuple[object, ...]
    coarse: bool
    epsilon: float
    separate: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...


def _save_horizontal(ctx: _AutogradContext, inputs: tuple, output: Tensor) -> None:
    pole, real, imag, _mode = inputs
    ctx.save_for_backward(*pole, real, imag, output)


def _differentiate_horizontal(ctx: _AutogradContext, adjoints: Tensor) -> tuple:
    *pole, real, imag, states = ctx.saved_tensors
    *parameters, source = horizontal_backward(pole, [real, imag], states, adjoints)
    modes = real.shape[-1]
    # The dispatcher strips trailing default arguments before applying autograd.
    return (parameters, source[..., :modes], source[..., modes:], None)[: len(ctx.needs_input_grad)]


def _save_vertical(
    ctx: _AutogradContext,
    inputs: tuple,
    output: tuple[Tensor, Tensor],  # noqa: ARG001
) -> None:
    pole, states, auxiliary, path, coarse, epsilon, _precise, _mode = inputs
    ctx.save_for_backward(*pole, states, *auxiliary, *path)
    ctx.coarse = coarse
    ctx.epsilon = epsilon


def _differentiate_vertical(ctx: _AutogradContext, grad_real: Tensor, grad_imag: Tensor) -> tuple:
    dr, di, gr, gi, states, vx, vy, *path = ctx.saved_tensors
    gradients = vertical_backward(
        [dr, di, gr, gi],
        states,
        [vx, vy],
        path,
        grad_real,
        grad_imag,
        ctx.coarse,
        ctx.epsilon,
    )
    result = (gradients[:4], gradients[4], [None, None], gradients[5:], None, None, None, None)
    return result[: len(ctx.needs_input_grad)]


def _save_vertical_features(
    ctx: _AutogradContext, inputs: tuple, output: tuple[Tensor, Tensor, Tensor],
) -> None:
    pole, states, auxiliary, path, ctx.coarse, ctx.epsilon, _, _, ctx.separate = inputs
    ctx.mark_non_differentiable(*(output[2:] if ctx.separate else output[1:]))
    ctx.save_for_backward(*pole, states, *auxiliary, *path, output[2])


def _differentiate_vertical_features(
    ctx: _AutogradContext, grad: Tensor, grad_secondary: Tensor, _grad_features: Tensor,
) -> tuple:
    dr, di, gr, gi, states, vx, vy, *path, features = ctx.saved_tensors
    if ctx.separate:
        real, imag = grad, grad_secondary
    else:
        real, imag = grad.unbind(0) if grad.ndim == 5 else grad.chunk(2, dim=-1)
    gradients = vertical_backward(
        [dr, di, gr, gi], states, [vx, vy], path, real, imag, ctx.coarse, ctx.epsilon,
        saved_features=features,
    )
    return gradients[:4], gradients[4], [None, None], gradients[5:], None, None, None, None, None


_vertical_forward_saved.register_autograd(
    _differentiate_vertical_features, setup_context=_save_vertical_features,
)


def vertical_forward(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool = False,
    epsilon: float = 1e-6,
    precise: bool = True,
    mode: str = "cuda",
) -> tuple[Tensor, Tensor]:
    if mode == "cuda" and _should_save_vscan(states, coarse):
        real, imag, _ = _vertical_forward_saved(
            pole, states, auxiliary, path, coarse, epsilon,
            planar=False, precise=precise, separate=True,
        )
        return real, imag
    return _vertical_forward(pole, states, auxiliary, path, coarse, epsilon, precise, mode)


horizontal_forward.register_autograd(_differentiate_horizontal, setup_context=_save_horizontal)
_vertical_forward.register_autograd(_differentiate_vertical, setup_context=_save_vertical)

__all__ = [
    "can_use_general_hscan",
    "can_use_general_vscan",
    "horizontal_backward",
    "horizontal_forward",
    "vertical_backward",
    "vertical_forward",
]


def general_scans_enabled(*, forward: bool = False) -> bool:
    return (
        os.environ.get("LNET_GENERAL_SCANS") != "0"
        and not (forward and os.environ.get("LNET_GENERAL_SCAN_REF_FWD") == "1")
        and not torch.are_deterministic_algorithms_enabled()
        and not inductor_config.emulate_precision_casts
    )


def try_general_horizontal_forward(
    pole: Sequence[Tensor], source: tuple[Tensor, Tensor],
) -> Tensor | None:
    if can_use_general_hscan(*source, list(pole)):
        return horizontal_forward(list(pole), *source)
    return None


def try_general_horizontal_backward(
    pole: Sequence[Tensor],
    source: tuple[Tensor, Tensor],
    states: Tensor | Sequence[Tensor],
    adjoints: Tensor | Sequence[Tensor],
) -> list[Tensor] | None:
    if not can_use_general_hscan(*source, list(pole)):
        return None
    if not isinstance(states, Tensor):
        states = torch.stack(tuple(states))
    if not isinstance(adjoints, Tensor):
        adjoints = torch.stack(tuple(adjoints))
    return horizontal_backward(list(pole), list(source), states, adjoints)


def _general_vertical_inputs(
    pole: Sequence[Tensor], states: Tensor | Sequence[Tensor],
    auxiliary: Sequence[Tensor], path: Sequence[Tensor],
    epilogue: str, kind: int, swiglu: bool,
) -> tuple[list[Tensor], Tensor, list[Tensor], list[Tensor], bool] | None:
    if epilogue not in ("full16", "coarse_path") or kind != 0 or swiglu or len(path) != 4:
        return None
    if not isinstance(states, Tensor):
        if not states[0].is_cuda:
            return None
        states = torch.stack(tuple(states))
    coarse = epilogue == "coarse_path"
    pole, auxiliary, path = list(pole), list(auxiliary[:2]), list(path)
    if not can_use_general_vscan(states, auxiliary, path, coarse=coarse, pole=pole):
        return None
    return pole, states, auxiliary, path, coarse


def try_general_vertical_forward(
    pole: Sequence[Tensor], states: Tensor | Sequence[Tensor],
    auxiliary: Sequence[Tensor], path: Sequence[Tensor],
    epilogue: str, epsilon: float, kind: int,
    packed: bool, planar: bool, swiglu: bool, squeeze: bool,
) -> tuple[Tensor, Tensor, Tensor] | None:
    inputs = _general_vertical_inputs(pole, states, auxiliary, path, epilogue, kind, swiglu)
    if inputs is None:
        return None
    real, imag = vertical_forward(*inputs, float(epsilon))
    empty = real.new_empty((0,))
    descriptor = real.new_empty((0,), dtype=torch.float32)
    if packed:
        out = torch.stack((real, imag)) if planar else torch.cat((real, imag), dim=-1)
        return out, empty, descriptor
    if not squeeze:
        real, imag = real.unsqueeze(-2), imag.unsqueeze(-2)
    return real, imag, descriptor


def try_general_vertical_backward(
    pole: Sequence[Tensor], states: Tensor | Sequence[Tensor],
    auxiliary: Sequence[Tensor], path: Sequence[Tensor],
    grad_real: Tensor, grad_imag: Tensor,
    epilogue: str, epsilon: float, kind: int, swiglu: bool,
) -> list[Tensor] | None:
    inputs = _general_vertical_inputs(pole, states, auxiliary, path, epilogue, kind, swiglu)
    if inputs is None:
        return None
    pole_list, states, auxiliary_list, path_list, coarse = inputs
    if grad_real.ndim == 5:
        grad_real, grad_imag = grad_real.squeeze(-2), grad_imag.squeeze(-2)
    return vertical_backward(
        pole_list, states, auxiliary_list, path_list, grad_real, grad_imag, coarse, float(epsilon),
    )
