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
import re
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
_MAX_CACHED_HIDDEN = 16
_MAX_SAVED_DP_CACHE_CHUNKS = 8

# Small devices reserve more VRAM for model state. Additional scan workspace
# on larger devices is useful up to one L2 working set.
_BASE_BUDGET_VRAM = 24 * 1024**3
_BASE_WORKSPACE_DIVISOR = 384
_DEFAULT_WORKSPACE_BYTES = _BASE_BUDGET_VRAM // _BASE_WORKSPACE_DIVISOR
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
    sgs_rows: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class _Wave53Options:
    # Keep the legacy positional constructor and vars(key) snapshot compatible.
    backward_batch4: bool = False
    backward_reload_source: bool = False
    partial_shared_acc: bool = False
    horizontal_stream: bool = False
    horizontal_vec2: bool = False
    path_bwd_pipe: int = 0
    horizontal_half_replay: bool = False
    horizontal_compact2: bool = False
    horizontal_bf16_buf: bool = False
    horizontal_wave53: bool = False


@dataclass(frozen=True)
class Kernel(_Wave53Options):
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
    batch_path: bool = False
    register_path: bool = False
    register_batch2: bool = False
    horizontal_unroll8: bool = False
    backward_batch2: bool = False
    backward_batch_tail: bool = False
    backward_readonly: bool = False
    saved_trajectory: int = 0
    sgs: int = 0  # 0: RMW, 1: full coverage
    backward_reduce_warps: int = 0
    odd_coarse_backward: bool = False
    odd_coarse_forward: bool = False
    odd_compact_width: bool = False
    forward_readonly: bool = False
    horizontal_readonly: bool = False
    horizontal_unroll: int = 0
    min_blocks: int = 0
    horizontal_span: int = 0
    backward_segment_source: bool = False
    fp32_path: bool = False
    tanh_gain: bool = False
    l2_single_write: bool = False
    dw1_tile: int = 0
    dw1_fp32: bool = False



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
         bool REAL_INTERLEAVED, bool IMAG_INTERLEAVED, bool REAL_TRANSPOSED, bool IMAG_TRANSPOSED,
         int BACKWARD_UNROLL = 4>
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
                #pragma unroll BACKWARD_UNROLL
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
                #pragma unroll BACKWARD_UNROLL
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
                            if constexpr (BACKWARD_UNROLL == 8 && WARPS == 3 && !COMPACT) {
                                const auto ar = a.conj(), gr = g.conj();
                                neg = {
                                    __fmaf_rn(ar.r, neg.r, -__fmul_rn(ar.i, neg.i)) +
                                        __fmaf_rn(gr.r, s.r, -__fmul_rn(gr.i, s.i)),
                                    __fmaf_rn(ar.i, neg.r, __fmul_rn(ar.r, neg.i)) +
                                        __fmaf_rn(gr.i, s.r, __fmul_rn(gr.r, s.i)),
                                };
                            } else {
                                neg = a.conj() * neg + g.conj() * s;
                            }
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

template<class T, class I, int HIDDEN, bool PRECISE, int K,
         class Weights>
__device__ __forceinline__ void path_mlp_batch(
    const float (&p)[K][8], Weights path, float2 bias, float2 (&out)[K]
) {
    const int size = HIDDEN ? HIDDEN : path.size;
    Sum<PRECISE> out_r[K], out_i[K];
    #pragma unroll
    for (int j = 0; j < size; ++j) {
        float weights[8];
        if constexpr (Weights::vector_weights) {
            #pragma unroll
            for (int k = 0; k < 8; k += 4) {
                const auto v = path.input_group(2 * j + k / 4);
                weights[k] = v.x;
                weights[k + 1] = v.y;
                weights[k + 2] = v.z;
                weights[k + 3] = v.w;
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; k += 2) {
                const auto pair = path.input_pair(4 * I(j) + k / 2);
                weights[k] = pair.r;
                weights[k + 1] = pair.i;
            }
        }
        const float b = path.bias(j);
        const float wr = path.output(j), wj = path.output(I(size) + j);
        #pragma unroll
        for (int t = 0; t < K; ++t) {
            float pre = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) pre = __fmaf_rn(weights[k], p[t][k], pre);
            pre = rounded<T>(rounded<T>(pre) + b);
            const float sig = __fdividef(1.0f, 1.0f + __expf(-pre));
            const float act = rounded<T>(pre * sig);
            out_r[t].add(wr * act);
            out_i[t].add(wj * act);
        }
    }
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        out[t] = make_float2(rounded<T>(out_r[t].value) + bias.x,
                             rounded<T>(out_i[t].value) + bias.y);
    }
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

template<class T, int HIDDEN, int THREADS, bool BACKWARD, class W>
__device__ void load_project_weights(
    W* __restrict__ weights, const void* __restrict__ wi, const void* __restrict__ bi,
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
            const float value = load_scalar(src, static_cast<long long>(mode0) * width + idx,
                                            dtype);
            if constexpr (sizeof(W) == sizeof(float)) weights[slot] = rounded<T>(value);
            else weights[slot] = write<W>(value);
        }
    }
}

#ifdef BACKWARD_BATCH2
template<class T, class I, class Weights>
__device__ __forceinline__ void path_adjoint_pair(
    const float (&p)[2][8], const float2 (&dy)[2], Weights path,
    float* acc, I stride, float (&dp)[2][8]
) {
    #pragma unroll
    for (int cell = 0; cell < 2; ++cell) {
        acc[88 * stride] += dy[cell].x;
        acc[89 * stride] += dy[cell].y;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float pre[2]{}, weights[8];
        if constexpr (Weights::vector_weights) {
            #pragma unroll
            for (int k = 0; k < 8; k += 4) {
                const auto v = path.input_group(2 * j + k / 4);
                weights[k] = v.x;
                weights[k + 1] = v.y;
                weights[k + 2] = v.z;
                weights[k + 3] = v.w;
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; k += 2) {
                const auto pair = path.input_pair(4 * I(j) + k / 2);
                weights[k] = pair.r;
                weights[k + 1] = pair.i;
            }
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            pre[0] = __fmaf_rn(weights[k], p[0][k], pre[0]);
            pre[1] = __fmaf_rn(weights[k], p[1][k], pre[1]);
        }
        // Keep each accumulator's updates in descending cell order.
        const float bias = path.bias(j), wr = path.output(j), wi = path.output(8 + j);
        #pragma unroll
        for (int cell = 0; cell < 2; ++cell) {
            const float z = rounded<T>(rounded<T>(pre[cell]) + bias);
            const float sig = __fdividef(1.0f, 1.0f + __expf(-z));
            const float act = rounded<T>(z * sig);
            const auto d = dy[cell];
            const float dz = rounded<T>(rounded<T>(d.x * wr + d.y * wi) *
                                         (sig * (1.0f + z * (1.0f - sig))));
            acc[(64 + j) * stride] += dz;
            acc[(72 + j) * stride] += d.x * act;
            acc[(80 + j) * stride] += d.y * act;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                dp[cell][k] += dz * weights[k];
                acc[(8 * j + k) * stride] += dz * p[cell][k];
            }
        }
    }
}
#endif

template<class T, class I, bool COARSE, int CHUNK, int THREADS, int HIDDEN,
         bool COMPACT, bool REAL_INTERLEAVED, bool IMAG_INTERLEAVED, bool CACHE_SOURCE,
         bool PURE, class Weights>
__device__ __forceinline__ void column_backward(
    Column<T, I> c, Complex a, Complex g, Weights path,
    const T* gr, const T* gi, T* gs, float* acc, I acc_stride,
    float4* seg, Quad<T>* seg_src, float4* ckpt, I ckpt_stride, Quad<T>* g2, I g2_stride,
    int chunks, Complex& da, Complex& dg
) {
    const auto ar = a.conj(), gg = g.conj();
    // CHUNK == 0 keeps the whole trajectory and needs no boundaries.
    constexpr bool full = CHUNK == 0;
    static_assert(!full || CACHE_SOURCE);
#define CACHED_SOURCE_AT(j) source_cache[(j) * THREADS].unpack()
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

#ifdef BACKWARD_BATCH2
        static_assert(HIDDEN == 8 && !PURE);
        constexpr int step = 1 + COARSE;
#ifdef BACKWARD_BATCH_TAIL
        // A short odd-sized segment can leave one cell for the scalar path.
        int end = length - 1;
        for (; end >= 2 * step - 1; end -= 2 * step) {
#else
        for (int end = length - 1; end >= 0; end -= 2 * step) {
#endif
            float p[2][8], dp[2][8]{};
            float2 dy[2];
            float4 next[2], next_even[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int j = end - i * step, y = y0 + j;
                next[i] = neg;
                neg = advance(neg, CACHED_SOURCE_AT(j), ar, gg);
                if constexpr (COARSE) {
                    next_even[i] = neg;
                    neg = advance(neg, CACHED_SOURCE_AT(j - 1), ar, gg);
                }
                const I cell = c.cell(COARSE ? y / 2 : y);
                dy[i] = make_float2(read(gr[(cell - c.m) * (1 + REAL_INTERLEAVED) + c.m]),
                                    read(gi[(cell - c.m) * (1 + IMAG_INTERLEAVED) + c.m]));
                const auto pp = normalize<T>(seg[j * THREADS], c.gain(y, false));
                const auto pn = normalize<T>(neg, c.gain(y - int(COARSE), true));
                p[i][0] = pp.x; p[i][1] = pp.z; p[i][2] = pn.x; p[i][3] = pn.z;
                p[i][4] = pp.y; p[i][5] = pp.w; p[i][6] = pn.y; p[i][7] = pn.w;
            }
            path_adjoint_pair<T, I>(p, dy, path, acc, acc_stride, dp);
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int j = end - i * step, y = y0 + j;
                const auto ip = normalize<T>(
                    make_float4(dp[i][0], dp[i][4], dp[i][1], dp[i][5]), c.gain(y, false));
                g2[I(COARSE ? y / 2 : y) * g2_stride] = normalize_packed<T>(
                    make_float4(dp[i][2], dp[i][6], dp[i][3], dp[i][7]),
                    c.gain(y - int(COARSE), true));
                column_adjoint<false, COMPACT>(
                    c, y, j ? seg[(j - 1) * THREADS] : before,
                    CACHED_SOURCE_AT(j), ip, a, g, adj_a, adj_b, da, dg, gs);
                if constexpr (COARSE) {
                    column_adjoint<false, COMPACT>(
                        c, y - 1, j > 1 ? seg[(j - 2) * THREADS] : before,
                        CACHED_SOURCE_AT(j - 1), {}, a, g,
                        adj_a, adj_b, da, dg, gs);
                }
                if constexpr (full) {
                    seg[j * THREADS] = next[i];
                    if constexpr (COARSE) seg[(j - 1) * THREADS] = next_even[i];
                }
            }
        }
#endif
#if !defined(BACKWARD_BATCH2) || defined(BACKWARD_BATCH_TAIL)
#ifdef BACKWARD_BATCH_TAIL
        for (int j = end; j >= 0; j -= step) {
#else
        for (int j = length - 1; j >= 0; j -= COARSE ? 2 : 1) {
#endif
            const int y = y0 + j;
            const auto s = (COARSE || CACHE_SOURCE)
                ? CACHED_SOURCE_AT(j) : c.source(y);
            const auto next = neg;
            neg = advance(neg, s, ar, gg);
            auto se = s;
            const auto next_even = neg;
            if constexpr (COARSE) {
                se = CACHED_SOURCE_AT(j - 1);
                neg = advance(neg, se, ar, gg);
            }
            const int yo = COARSE ? y / 2 : y;
            const I cell = c.cell(yo);
            const auto gp = c.gain(y, false), gn = c.gain(y - int(COARSE), true);
            float dp[8]{};
            if constexpr (PURE) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) dp[k] = read(gr[I(k) * c.feature_plane + cell]);
            } else path_mlp<T, I, HIDDEN, false, true>(
                normalize<T>(seg[j * THREADS], gp), normalize<T>(neg, gn),
                path, {},
                make_float2(read(gr[(cell - c.m) * (1 + REAL_INTERLEAVED) + c.m]),
                            read(gi[(cell - c.m) * (1 + IMAG_INTERLEAVED) + c.m])),
                acc, acc_stride, dp);
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
#endif
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
                c, y, seg[j * THREADS], CACHED_SOURCE_AT(j), in, ar, gg,
                adj_a, adj_b, da, dg, gs);
            if constexpr (COARSE) clear_unused_columns<COMPACT>(c, y, gs);
        }
    }
}

#undef CACHED_SOURCE_AT

// Tile the CTA reduction so all scan warps can share one small staging buffer.
constexpr int vscan_reduce_tile = 32;

template<int WARPS, int FIELDS, class I>
__device__ __forceinline__ void write_vscan_partial(
    float* shared, float* row, const float* gradients, Complex da, Complex dg, int m, int M
) {
    constexpr int threads = WARPS * 32;
    const float pole[4] = {da.r, da.i, dg.r, dg.i};
    #pragma unroll
    for (int begin = 0; begin < FIELDS; begin += vscan_reduce_tile) {
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < vscan_reduce_tile && begin + j < FIELDS; ++j) {
            const int field = begin + j;
            shared[j * threads + threadIdx.x] = field < 4 ? pole[field] : gradients[field - 4];
        }
        __syncthreads();
        if (threadIdx.x < 32 && m < M) {
            #pragma unroll
            for (int j = 0; j < vscan_reduce_tile && begin + j < FIELDS; ++j) {
                const float* staged = shared + j * threads + threadIdx.x;
                float sum = staged[0];
                #pragma unroll
                for (int k = 1; k < WARPS; ++k) sum += staged[k * 32];
                row[I(begin + j) * M] = sum;
            }
        }
    }
}

template<class T, bool VECTOR>
struct RegisterPathWeights {
    static constexpr bool vector_weights = VECTOR;
    int size = 8;
    Pair<T> pairs[44];

    __device__ Complex input_pair(int j) const { return pairs[j].unpack(); }
    __device__ float4 input_group(int j) const {
        const auto a = input_pair(2 * j), b = input_pair(2 * j + 1);
        return make_float4(a.r, a.i, b.r, b.i);
    }
    __device__ float bias(int j) const {
        const auto p = pairs[32 + j / 2].unpack();
        return j % 2 ? p.i : p.r;
    }
    __device__ float output(int j) const {
        const auto p = pairs[36 + j / 2].unpack();
        return j % 2 ? p.i : p.r;
    }
};

template<class T, class I, bool VECTOR>
__device__ __forceinline__ auto load_register_path_weights(
    const void* wi, const void* bi, const void* wo, int types, int m, bool active
) {
    const PathWeights<T, I, false> source{wi, bi, wo, types, m, 8};
    RegisterPathWeights<T, VECTOR> weights;
    if (active) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) weights.pairs[j] = Pair<T>::pack(source.input_pair(j));
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            weights.pairs[32 + j] = Pair<T>::pack({source.bias(2 * j), source.bias(2 * j + 1)});
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            weights.pairs[36 + j] = Pair<T>::pack({source.output(2 * j), source.output(2 * j + 1)});
    }
    return weights;
}

// Lock the real-product contraction used by the Wave 21C vector path.
template<bool REGISTER_VECTOR>
__device__ __forceinline__ float4 advance_forward(
    float4 state, float4 source, Complex a, Complex g
) {
    if constexpr (REGISTER_VECTOR) {
        return make_float4(
            __fmaf_rn(a.r, state.x, -__fmul_rn(a.i, state.y)) +
                __fmaf_rn(g.r, source.x, -__fmul_rn(g.i, source.y)),
            __fmaf_rn(a.r, state.y, __fmul_rn(a.i, state.x)) +
                __fmaf_rn(g.i, source.x, __fmul_rn(g.r, source.y)),
            __fmaf_rn(a.r, state.z, -__fmul_rn(a.i, state.w)) +
                __fmaf_rn(g.r, source.z, -__fmul_rn(g.i, source.w)),
            __fmaf_rn(a.r, state.w, __fmul_rn(a.i, state.z)) +
                __fmaf_rn(g.i, source.z, __fmul_rn(g.r, source.w)));
    } else {
        return advance(state, source, a, g);
    }
}

template<class T, class I, bool COARSE, int WARPS, int CHUNK, int HIDDEN, bool BACKWARD,
         bool CACHE_WEIGHT, bool GLOBAL_CKPT, bool CACHE_LINE, bool SHARED_ACC, bool PRECISE,
         bool COMPACT, bool REAL_INTERLEAVED, bool IMAG_INTERLEAVED, bool CACHE_SOURCE, bool PURE,
         bool VECTOR_WEIGHTS, bool BATCH_PATH, bool REGISTER_PATH = false, int BATCH_WIDTH = 4>
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
    if constexpr (VECTOR_WEIGHTS && !REGISTER_PATH) {
        static_assert(CACHE_WEIGHT);
        load_project_weights<T, HIDDEN, threads, true>(
            reinterpret_cast<float*>(shared), wi, bi, wo, nullptr, types, mode0, M, size);
        __syncthreads();
    } else if constexpr (CACHE_WEIGHT && !REGISTER_PATH) {
        load_path_weights<T, I>(weights_shared, wi, bi, wo, types, mode0, M, size, threads);
        __syncthreads();
    }
    const auto path = [&] {
        if constexpr (REGISTER_PATH) {
            static_assert(!BACKWARD && !PURE && HIDDEN == 8 && sizeof(T) == 2);
            return load_register_path_weights<T, I, VECTOR_WEIGHTS>(wi, bi, wo, types, m, active);
        } else if constexpr (VECTOR_WEIGHTS) {
            return ProjectWeights{reinterpret_cast<const float*>(shared), size, lane};
        } else {
            return path_weights<T, I, CACHE_WEIGHT>(
                weights_shared, wi, bi, wo, types, m, size, lane);
        }
    }();
    constexpr int weight_itemsize = VECTOR_WEIGHTS ? sizeof(float) : sizeof(T);
    auto* seg = shared + (CACHE_WEIGHT && !REGISTER_PATH
        ? weight_fields * 32 * weight_itemsize / 16 : 0);
    auto* ckpt = seg + (BACKWARD ? span * threads : 0);
    auto* seg_src = reinterpret_cast<Quad<T>*>(
        ckpt + (BACKWARD && !GLOBAL_CKPT ? I(chunks - 1) * threads : 0));
    auto* acc_base = reinterpret_cast<float*>(
        seg_src + (BACKWARD ? I(CACHE_SOURCE ? H : span) * threads : 0));
    auto* line_buf = reinterpret_cast<Quad<T>*>(acc_base + (SHARED_ACC ? fields * threads : 0));
    float* row = partials;
    float local_acc[register_acc ? 11 * HIDDEN + 2 : 1]{};
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
                                REAL_INTERLEAVED, IMAG_INTERLEAVED, CACHE_SOURCE, PURE>(
                    c, a, g, path, gr, gi, gs,
                    acc, acc_stride, (seg + threadIdx.x), seg_src + threadIdx.x, boundaries,
                    GLOBAL_CKPT ? I(workers) * M : threads, buf, stride, chunks, da, dg
                );
            } else {
                float4 state{};
                for (int y = 0; y < H; ++y) {
                    state = advance_forward<REGISTER_PATH && VECTOR_WEIGHTS>(
                        state, c.source(y), a, g);
                    if (!COARSE || (y & 1)) {
                        buf[I(COARSE ? y / 2 : y) * stride] =
                            normalize_packed<T>(state, c.gain(y, false));
                    }
                }
                state = {};
                int remaining = H;
                // The vertical recurrence depends on the source alone. Prepare
                // several cells before loading each row of path weights once.
                if constexpr (BATCH_PATH) {
                    for (; remaining >= BATCH_WIDTH * (1 + COARSE);
                         remaining -= BATCH_WIDTH * (1 + COARSE)) {
                        float p[BATCH_WIDTH][8];
                        #pragma unroll
                        for (int t = 0; t < BATCH_WIDTH; ++t) {
                            const int y = remaining - 1 - t * (1 + COARSE) - COARSE;
                            if constexpr (COARSE) {
                                state = advance_forward<REGISTER_PATH && VECTOR_WEIGHTS>(
                                    state, c.source(y + 1), a.conj(), g.conj());
                            }
                            state = advance_forward<REGISTER_PATH && VECTOR_WEIGHTS>(
                                state, c.source(y), a.conj(), g.conj());
                            const auto pos = buf[I(COARSE ? y / 2 : y) * stride].unpack();
                            const auto neg = normalize<T>(state, c.gain(y, true));
                            p[t][0] = pos.x;
                            p[t][1] = pos.z;
                            p[t][2] = neg.x;
                            p[t][3] = neg.z;
                            p[t][4] = pos.y;
                            p[t][5] = pos.w;
                            p[t][6] = neg.y;
                            p[t][7] = neg.w;
                        }
                        float2 out[BATCH_WIDTH];
                        path_mlp_batch<T, I, HIDDEN, PRECISE, BATCH_WIDTH>(p, path, bias, out);
                        #pragma unroll
                        for (int t = 0; t < BATCH_WIDTH; ++t) {
                            const int yo = remaining / (1 + COARSE) - 1 - t;
                            const I cell = c.cell(yo);
                            outr[(cell - m) * (1 + REAL_INTERLEAVED) + m] = write<T>(out[t].x);
                            outi[(cell - m) * (1 + IMAG_INTERLEAVED) + m] = write<T>(out[t].y);
                        }
                    }
                }
                for (int y = remaining - 1; y >= 0; --y) {
                    state = advance_forward<REGISTER_PATH && VECTOR_WEIGHTS>(
                        state, c.source(y), a.conj(), g.conj());
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
            write_vscan_partial<WARPS, 4, I>(
                reinterpret_cast<float*>(shared), row, nullptr, da, dg, m, M);
        } else if constexpr (register_acc && reduce) {
            write_vscan_partial<WARPS, 11 * HIDDEN + 6, I>(
                reinterpret_cast<float*>(shared), row, local_acc, da, dg, m, M);
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
                for (int j = 0; j < 11 * HIDDEN + 2; ++j) row[(4 + I(j)) * M] = local_acc[j];
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

_SAVED_QUAD_CUDA = r"""
__device__ unsigned int pair_bits(Pair<__half> p) {
    const __half2_raw raw = p.value;
    return raw.x | (static_cast<unsigned int>(raw.y) << 16);
}

__device__ unsigned int pair_bits(Pair<__nv_bfloat16> p) {
    const __nv_bfloat162_raw raw = p.value;
    return raw.x | (static_cast<unsigned int>(raw.y) << 16);
}

template<class T> __device__ Pair<T> unpack_pair_bits(unsigned int bits);
template<> __device__ Pair<__half> unpack_pair_bits(unsigned int bits) {
    return {__half2(__half2_raw{static_cast<unsigned short>(bits),
                               static_cast<unsigned short>(bits >> 16)})};
}
template<> __device__ Pair<__nv_bfloat16> unpack_pair_bits(unsigned int bits) {
    return {__nv_bfloat162(__nv_bfloat162_raw{static_cast<unsigned short>(bits),
                                            static_cast<unsigned short>(bits >> 16)})};
}

template<class T, class I>
__device__ __forceinline__ void store_saved_quad(void* base, I index, Quad<T> value) {
    if constexpr (sizeof(T) == 2) {
        reinterpret_cast<uint2*>(base)[index] = make_uint2(pair_bits(value.a), pair_bits(value.b));
    } else {
        reinterpret_cast<float4*>(base)[index] = value.unpack();
    }
}

template<class T, class I>
__device__ __forceinline__ float4 load_saved_quad(const void* base, I index) {
    if constexpr (sizeof(T) == 2) {
        const uint2 value = reinterpret_cast<const uint2*>(base)[index];
        return Quad<T>{unpack_pair_bits<T>(value.x), unpack_pair_bits<T>(value.y)}.unpack();
    } else {
        return reinterpret_cast<const float4*>(base)[index];
    }
}
"""

_SAVED_TRAJECTORY_CUDA = r"""
template<class T, class I, bool COARSE>
__device__ __forceinline__ Column<T, I> saved_column(
    const T* states, const float* vx, const float* vy,
    I line, int b, int h, int w, int mcount, int m, float eps
) {
    const int ow = w / (1 + COARSE), oh = h / (1 + COARSE);
    const I batch = line / ow;
    const int x = line % ow, xa = COARSE ? 2 * x + 1 : x, xb = COARSE ? 2 * x : x;
    const I base = batch * h * w * mcount + m;
    const I plane = I(b) * h * w * mcount;
    return {states, vy, base + I(xa) * mcount, base + I(xb) * mcount,
            plane, plane, (batch * oh * ow + x) * mcount + m, I(b) * oh * ow * mcount,
            h, w, mcount, ow, m, vx[I(xa) * mcount + m], vx[(I(w) + xb) * mcount + m], eps};
}

template<bool COARSE, bool RI, bool II, class T, class I, class Weights>
__device__ __forceinline__ void saved_path_cell(
    Column<T, I> c, int yo, const Quad<T>* trajectory, I start, I plane,
    const T* gr, const T* gi,
    Weights path, float* acc, float4& positive, Quad<T>& negative
) {
    const int yp = yo * (1 + COARSE) + COARSE, yn = yp - COARSE;
    const I offset = c.cell(yo) - c.m;
    const I stride = I(c.OW) * c.M;
    const float2 gp = c.gain(yp, false), gn = c.gain(yn, true);
    const auto p = load_saved_quad<T>(trajectory, start + yp * stride);
    const auto n = load_saved_quad<T>(trajectory, plane + start + yn * stride);
    const auto pos = Quad<T>::pack(
        make_float4(p.x * gp.x, p.y * gp.x, p.z * gp.y, p.w * gp.y)).unpack();
    const auto neg = Quad<T>::pack(
        make_float4(n.x * gn.x, n.y * gn.x, n.z * gn.y, n.w * gn.y)).unpack();
    float dp[8]{};
    path_mlp<T, I, 8, false, true>(
        pos, neg,
        path, {}, make_float2(read(gr[offset * (1 + RI) + c.m]),
                              read(gi[offset * (1 + II) + c.m])), acc, I(1), dp);
    positive = normalize<T>(make_float4(dp[0], dp[4], dp[1], dp[5]), gp);
    negative = normalize_packed<T>(make_float4(dp[2], dp[6], dp[3], dp[7]), gn);
}

template<class T, class I, bool COARSE, int WARPS, bool COMPACT, bool RI, bool II, int PASS>
__device__ __forceinline__ void saved_backward(
    const float* dr, const float* di, const float* gr_pole, const float* gi_pole,
    const T* states, const float* vx, const float* vy,
    const void* wi, const void* bi, const void* wo,
    const T* gr, const T* gi, T* gs, float* partial,
    const Quad<T>* trajectory, Quad<T>* dp,
    int B, int H, int W, int M, int workers, int types, float eps
) {
    constexpr int threads = WARPS * 32;
    constexpr int fields = PASS == 0 ? 94 : PASS == 1 ? 90 : 4;
    extern __shared__ float4 shared[];
    const int tiles = (M - 1) / 32 + 1, lane = threadIdx.x % 32;
    const int group = blockIdx.x / tiles, mode0 = (blockIdx.x % tiles) * 32;
    const int m = mode0 + lane, worker = group * WARPS + threadIdx.x / 32;
    auto* weights = reinterpret_cast<T*>(shared);
    if constexpr (PASS != 2) {
        load_path_weights<T, I>(weights, wi, bi, wo, types, mode0, M, 8, threads);
        __syncthreads();
    }
    float acc[PASS == 2 ? 1 : 90]{};
    Complex da{}, dg{};
    const int ow = W / (1 + COARSE), oh = H / (1 + COARSE);
    const I stride = I(ow) * M, plane = I(B) * H * stride;
    if (m < M && worker < workers) {
        const auto path = path_weights<T, I, true>(weights, wi, bi, wo, types, m, 8, lane);
        if constexpr (PASS == 1) {
            for (I cell = worker; cell < I(B) * oh * ow; cell += workers) {
                const I batch = cell / (oh * ow), x = cell % ow;
                const int yo = cell / ow % oh;
                const auto c = saved_column<T, I, COARSE>(
                    states, vx, vy, batch * ow + x, B, H, W, M, m, eps);
                float4 positive;
                Quad<T> negative;
                const I start = batch * H * stride + x * M + m;
                saved_path_cell<COARSE, RI, II>(
                    c, yo, trajectory, start, plane, gr, gi, path, acc,
                    positive, negative);
                store_saved_quad(dp, c.cell(yo), Quad<T>::pack(positive));
                store_saved_quad(dp, c.feature_plane + c.cell(yo), negative);
            }
        } else {
            const Complex a{dr[m], di[m]}, g{gr_pole[m], gi_pole[m]};
            auto* negative_cache = weights + 88 * 32;
            auto* cache = reinterpret_cast<Quad<T>*>(negative_cache) + threadIdx.x;
            for (I line = worker; line < I(B) * ow; line += workers) {
                const auto c = saved_column<T, I, COARSE>(
                    states, vx, vy, line, B, H, W, M, m, eps);
                const I start = (line / ow * H * ow + line % ow) * M + m;
                Complex adj_a{}, adj_b{};
                for (int y = H - 1; y >= 0; --y) {
                    float4 positive{};
                    if (!COARSE || (y & 1)) {
                        const int yo = y / (1 + COARSE);
                        if constexpr (PASS == 0) {
                            Quad<T> negative;
                            saved_path_cell<COARSE, RI, II>(
                                c, yo, trajectory, start, plane, gr, gi, path, acc,
                                positive, negative);
                            cache[yo * threads] = negative;
                        } else {
                            positive = load_saved_quad<T>(dp, c.cell(yo));
                        }
                    }
                    const auto prev = y
                        ? load_saved_quad<T>(trajectory, start + (y - 1) * stride)
                        : float4{};
                    column_adjoint<false, COMPACT>(
                        c, y, prev, c.source(y), positive, a, g, adj_a, adj_b, da, dg, gs);
                }
                adj_a = adj_b = {};
                for (int y = 0; y < H; ++y) {
                    float4 negative{};
                    if (y / (1 + COARSE) < oh && (!COARSE || !(y & 1))) {
                        const int yo = y / (1 + COARSE);
                        if constexpr (PASS == 0) negative = cache[yo * threads].unpack();
                        else negative = load_saved_quad<T>(dp, c.feature_plane + c.cell(yo));
                    }
                    const auto prev = y + 1 < H
                        ? load_saved_quad<T>(
                            trajectory, plane + start + (y + 1) * stride)
                        : float4{};
                    column_adjoint<true, COMPACT>(
                        c, y, prev, c.source(y), negative, a.conj(), g.conj(),
                        adj_a, adj_b, da, dg, gs);
                    if constexpr (COARSE) clear_unused_columns<COMPACT>(c, y, gs);
                }
            }
        }
    }
    // One independent partial row per warp avoids atomics and reduction scratch.
    if (m < M && worker < workers) {
        auto* row = partial + I(worker) * fields * M + m;
        if constexpr (PASS != 1) {
            row[0] = da.r; row[M] = da.i; row[2 * I(M)] = dg.r; row[3 * I(M)] = dg.i;
            if constexpr (PASS == 0) row += 4 * I(M);
        }
        if constexpr (PASS != 2) {
            #pragma unroll
            for (int j = 0; j < 90; ++j) row[I(j) * M] = acc[j];
        }
    }
}
"""

# NVRTC compilation and driver launch

def _kernel_name(key: Kernel) -> str:  # noqa: PLR0911
    if key.saved_trajectory:
        return ("", "vscan_fwd_saved", "vscan_bwd_saved", "vscan_path_bwd",
                "vscan_adj_bwd")[key.saved_trajectory]
    if key.sgs:
        return "vscan_sgs"
    if key.projection:
        return "path_mlp_backward" if key.backward else "path_mlp_forward"
    if key.projection_reduction:
        return "reduce_path_mlp_partials"
    if key.reduction:
        return "reduce_vscan_partials" if key.vertical else "reduce_hscan_partials"
    if key.forward_readonly or key.backward_reduce_warps or key.horizontal_readonly:
        return "general_vscan" if key.vertical else "general_hscan"
    # Wave 33A used one symbol for the default vertical and horizontal kernels.
    return "scan"


def _kernel_source(key: Kernel) -> str:  # noqa: C901, PLR0912, PLR0915
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
    minimum = f", {key.min_blocks}" if key.min_blocks else ""
    bounds = (
        f'extern "C" __global__ __launch_bounds__({key.warps * _WARP_SIZE}{minimum})'
        f" void {_kernel_name(key)}"
    )
    if key.saved_trajectory > 1:
        return _saved_backward_source(key, io, index, bounds)
    compact = str(key.compact).lower()
    layout = f"{str(key.real_interleaved).lower()}, {str(key.imag_interleaved).lower()}"
    if key.vertical:
        hidden = _REGISTER_HIDDEN if key.hidden8 else 0
        cache_source = str(key.cache_source).lower()
        register = (
            f", {str(key.register_path or not key.fp32_path).lower()}, 2" if key.register_batch2
            else ", true" if key.register_path else ""
        )
        targs = (
            f"{io}, {index}, {coarse}, {key.warps}, {key.chunk}, {hidden}, "
            f"{backward}, {cache_weights}, {global_ckpt}, {cache_line}, "
            f"{shared_acc}, {precise}, {compact}, {layout}, {cache_source}, "
            f"{str(key.pure_vertical).lower()}, {str(key.vector_weights).lower()}, "
            f"{str(key.batch_path).lower()}{register}"
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
        factor = key.horizontal_unroll or (8 if key.horizontal_unroll8 else 0)
        unroll = f", {factor}" if factor else ""
        if key.horizontal_wave53 and factor:
            # Ampere/Ada spill on some unrolled strided layouts.
            transposed_scalar = (
                (key.real_transposed and not key.real_interleaved)
                or (key.imag_transposed and not key.imag_interleaved)
            )
            older = 1 if transposed_scalar else 4
            unroll = f", (__CUDA_ARCH__ >= 900 ? {factor} : {older})"
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
               {compact}, {layout}, {transposed}{unroll}>(
        dr, di, decay_gr, decay_gi, sr, si, states, grad, out, partials, scratch,
        B, H, W, M, workers);
}}
"""
    source = _CUDA_SOURCE
    if key.backward_batch2:
        source = "#define BACKWARD_BATCH2\n" + source
        if key.backward_batch_tail:
            source = "#define BACKWARD_BATCH_TAIL\n" + source
    if key.backward_segment_source:
        # Dense backward also reads the segment source in the paired adjoint
        # and writes it during the negative replay.
        source = source.replace(
            "if constexpr (COARSE && !CACHE_SOURCE) seg_src",
            "if constexpr (!CACHE_SOURCE) seg_src",
        ).replace(
            "c.source(y0 + j)", "c.packed_source(y0 + j).unpack()",
        ).replace(
            "const auto s = (COARSE || CACHE_SOURCE)\n"
            "                ? CACHED_SOURCE_AT(j) : c.source(y);",
            "const auto s = CACHED_SOURCE_AT(j);",
        )
    if key.vertical and (key.backward_readonly or key.forward_readonly):
        source = (
            source.replace("states[ia]", "__ldg(states + ia)")
            .replace("states[ia + source_plane]", "__ldg(states + ia + source_plane)")
            .replace("states[ib]", "__ldg(states + ib)")
            .replace("states[ib + source_plane]", "__ldg(states + ib + source_plane)")
        )
    if not key.vertical and (key.horizontal_readonly or key.horizontal_unroll):
        source = _horizontal_candidate_source(source, key)
    if key.odd_coarse_forward:
        source = source.replace("int remaining = H;", r"""int remaining = H;
                if (H & 1) {
                    state = advance_forward<REGISTER_PATH && VECTOR_WEIGHTS>(
                        state, c.source(H - 1), a.conj(), g.conj());
                    --remaining;
                }""")
    if key.odd_coarse_backward:
        source = _odd_coarse_backward_source(source)
    if key.odd_compact_width:
        source = _odd_compact_source(source)
    if key.sgs or key.l2_single_write:
        source = _sgs_source(source, persistent=key.l2_single_write)
    if key.path_bwd_pipe and key.fp32_path and key.tanh_gain and key.backward_batch2 and key.hidden8:
        source = _pipelined_adjoint(source, key.path_bwd_pipe)
    if key.fp32_path:
        source = _fp32_path_source(source)
    if key.tanh_gain:
        source = _tanh_gain_source(source)
    if key.backward_reduce_warps:
        source = _grouped_vscan_source(source, key.backward_reduce_warps)
    if key.saved_trajectory:
        source = _with_saved_quads(_saved_forward_source(source))
    if key.dw1_tile:
        source = _tiled_dw1_source(source, key)
    if key.backward_reload_source:
        source = _vertical_wave54_source(source, key)
        if key.backward_batch4:
            source = _quad_adjoint_source(source)
    return source + wrapper


_FLUSH_DW1 = """
template<class T, class I, int THREADS>
__device__ __forceinline__ void flush_dw1(
    const Pair<T>* tile, int count, float* dw1, int M
) {
    // One 32-bit word per lane in every tile plane.
    #pragma unroll 1
    for (int j = 0; j < 8; ++j) {
        float acc[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] = dw1[I(8 * j + k) * M];
        #pragma unroll 1
        for (int t = 0; t < count; ++t) {
            const auto* row = tile + t * 8 * THREADS;
            const auto z = row[(4 + j / 2) * THREADS].unpack();
            const float dz = j & 1 ? z.i : z.r;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const auto p = row[k * THREADS].unpack();
                acc[2 * k] = __fmaf_rn(dz, p.r, acc[2 * k]);
                acc[2 * k + 1] = __fmaf_rn(dz, p.i, acc[2 * k + 1]);
            }
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) dw1[I(8 * j + k) * M] = acc[k];
    }
}
"""


def _tiled_dw1_source(source: str, key: Kernel) -> str:
    start = source.index("template<class T, class I, int HIDDEN, bool PRECISE, bool BACKWARD")
    end = source.index("template<class T, class I, int HIDDEN, bool PRECISE, int K", start)
    path = source[start:end].replace("int SHARED_HIDDEN = 0", "int THREADS").replace(
        "float2 path_mlp(", "float2 path_mlp_tiled(",
    ).replace(
        "float* shared_dwi = nullptr, int shared_stride = 0", "Pair<T>* tile",
    ).replace("8 * SHARED_HIDDEN", "64")
    path = path.replace(
        "    Sum<PRECISE> out_r, out_i;",
        """    #pragma unroll
    for (int k = 0; k < 4; ++k) tile[k * THREADS] = Pair<T>::pack({p[2*k], p[2*k+1]});
    float dz_even = 0;
    Sum<PRECISE> out_r, out_i;""",
    ).replace(
        "            #pragma unroll\n            for (int k = 0; k < 8; ++k) {",
        """            if (j & 1) tile[(4 + j / 2) * THREADS] = Pair<T>::pack({dz_even, dz});
            else dz_even = dz;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {""",
    ).replace(
        "                if (j < SHARED_HIDDEN) "
        "shared_dwi[(8 * j + k) * shared_stride] += dz * p[k];\n"
        "                else partial[(8 * (I(j) - SHARED_HIDDEN) + k) "
        "* partial_stride] += dz * p[k];",
        "",
    )
    if key.dw1_fp32:
        path = path.replace("rounded<T>", "rounded<float>")
    source = source[:end] + path + _FLUSH_DW1 + source[end:]
    source = _sgs_source(source)
    source = source.replace("T* gs, Quad<T>* pos_grad", "T*, Quad<T>* pos_grad, I pos_stride")
    source = source.replace("const Quad<T>* pos_grad", "const Quad<T>* pos_grad, I pos_stride")
    source = source.replace("pos_grad[I(y) * THREADS]", "pos_grad[I(y) * pos_stride]")
    source = source.replace("gs, pos_grad);", "gs, pos_grad, g2_stride);")
    source = source.replace(
        "Complex& dg, Quad<T>* pos_grad\n",
        "Complex& dg, Quad<T>* pos_grad, Pair<T>* tile, float* dw1\n",
    )
    source = source.replace(
        "line_buf + (CACHE_LINE ? I(oh) * threads : 0) + threadIdx.x",
        "buf + I(oh) * stride, reinterpret_cast<Pair<T>*>(seg_src) + threadIdx.x, row + 4 * I(M)",
    )
    before = source.index("void column_backward(")
    after = source.index("#undef CACHED_SOURCE_AT", before)
    column = source[before:after].replace(
        "source_cache[(j) * THREADS].unpack()", "c.source(y0 + (j))",
    ).replace(
        "        Quad<T>* source_cache = seg_src + (CACHE_SOURCE ? I(y0) * THREADS : 0);\n", "",
    ).replace("source_cache[j * THREADS].unpack()", "c.source(y0 + j)")
    column = "\n".join(
        line for line in column.split("\n")
        if "seg_src[" not in line
    ).replace(
        "#if !defined(BACKWARD_BATCH2) || defined(BACKWARD_BATCH_TAIL)",
        "#if !defined(BACKWARD_BATCH2) || defined(BACKWARD_BATCH_TAIL)"
        "\n        int tile_count = 0;",
    ).replace(
        "else path_mlp<T, I, HIDDEN, false, true>(",
        "else path_mlp_tiled<T, I, HIDDEN, false, true, THREADS>(",
    ).replace(
        "acc, acc_stride, dp);", "acc, acc_stride, dp, tile + tile_count * 8 * THREADS);",
    ).replace(
        "            if constexpr (full) {\n                // The positive adjoint",
        f"""            if (++tile_count == {key.dw1_tile}) {{
                flush_dw1<T, I, THREADS>(tile, tile_count, dw1, c.M);
                tile_count = 0;
            }}
            if constexpr (full) {{
                // The positive adjoint""",
    ).replace(
        "        }\n#endif\n    }\n\n    adj_a",
        "        }\n        if (tile_count) flush_dw1<T, I, THREADS>(tile, tile_count, dw1, c.M);"
        "\n#endif\n    }\n\n    adj_a",
    )
    source = source[:before] + column + source[after:]
    return source.replace(
        "constexpr bool reduce = (PURE || SHARED_ACC || register_acc) && WARPS > 1;",
        "constexpr bool reduce = false;",
    ).replace(
        "float local_acc[register_acc ? 11 * HIDDEN + 2 : 1]{};",
        "float local_acc[26]{};",
    ).replace(
        "            acc = local_acc;",
        """            acc = local_acc;
            if (active) {
                #pragma unroll 1
                for (int j = 0; j < 64; ++j) row[(4 + I(j)) * M] = 0;
            }""",
        1,
    ).replace(
        "for (int j = 0; j < 11 * HIDDEN + 2; ++j) row[(4 + I(j)) * M] = local_acc[j];",
        "for (int j = 0; j < 26; ++j) row[(68 + I(j)) * M] = local_acc[j];",
    )


def _with_saved_quads(source: str) -> str:
    start = source.index("template<class T, class I>\nstruct Column {")
    return source[:start] + _SAVED_QUAD_CUDA + source[start:]


def _saved_forward_source(source: str) -> str:
    start = source.index("    const auto path = [&]", source.index("void vertical("))
    end = source.index("template<class T, class I, int WARPS, int HIDDEN", start)
    forward = source[start:end]
    # Add one vector store after each recurrence update, including coarse
    # rows that are not sampled by the projection but feed the pole adjoint.
    forward, stores = re.subn(
        r"(state = advance_forward<REGISTER_PATH && VECTOR_WEIGHTS>\(\s*"
        r"state, c.source\(([^)]+)\), ([^;]+)\);)",
        lambda match: match[0] + "\n" + " " * 24
        + "store_saved_quad(global_checkpoints, "
        + ("I(B) * H * ow * M + " if "conj" in match[3] else "")
        + "((b * H + (" + match[2] + ")) * ow + x) * M + m, Quad<T>::pack(state));",
        forward,
    )
    # Read the positive trajectory as the reverse pass projects each cell.
    # The saved buffer replaces the temporary normalized line cache.
    forward, caches = re.subn(
        r"\s*if \(!COARSE \|\| \(y & 1\)\) \{\s*"
        r"buf\[I\(COARSE \? y / 2 : y\) \* stride\] =\s*"
        r"normalize_packed<T>\(state, c.gain\(y, false\)\);\s*\}",
        "",
        forward,
    )
    if stores not in (4, 5) or caches != 1:
        raise RuntimeError("Unexpected vertical forward trajectory template")
    positive = (
        "normalize<T>(load_saved_quad<T>(global_checkpoints, "
        "((b * H + y + int(COARSE)) * ow + x) * M + m), "
        "c.gain(y + int(COARSE), false))"
    )
    forward = forward.replace(
        "buf[I(COARSE ? y / 2 : y) * stride].unpack()", positive,
    ).replace("buf[I(yo) * stride].unpack()", positive)
    return source[:start] + forward + source[end:]


def _saved_backward_source(key: Kernel, io: str, index: str, bounds: str) -> str:
    flags = ", ".join(str(flag).lower() for flag in (
        key.compact, key.real_interleaved, key.imag_interleaved,
    ))
    source = _odd_compact_source(_CUDA_SOURCE) if key.odd_compact_width else _CUDA_SOURCE
    return _with_saved_quads(source) + _SAVED_TRAJECTORY_CUDA + f"""
{bounds}(
    const float* dr, const float* di, const float* decay_gr, const float* decay_gi,
    const {io}* states, const float* vx, const float* vy,
    const void* wi, const void* bi, const void* wo,
    const {io}* gr, const {io}* gi, {io}* gs, float* partial,
    const Quad<{io}>* trajectory, Quad<{io}>* dp,
    int B, int H, int W, int M, int workers, int types, float eps
) {{
    saved_backward<{io}, {index}, {str(key.coarse).lower()}, {key.warps}, {flags},
                   {key.saved_trajectory - 2}>(
        dr, di, decay_gr, decay_gi, states, vx, vy, wi, bi, wo, gr, gi, gs, partial,
        trajectory, dp, B, H, W, M, workers, types, eps);
}}
"""


def _quad_adjoint_source(source: str) -> str:
    start = source.index("void path_adjoint_pair(")
    end = source.index("#endif", source.index("if constexpr (full) {",
                                              source.index("path_adjoint_pair<T")))
    quad = source[start:end]
    for field in ("(&p)", "(&dy)", "(&dp)", "pre", " p", " dp", "dy", "gp_c",
                  "gn_c", "next", "next_even"):
        quad = quad.replace(field + "[2]", field + "[4]")
    quad = quad.replace("cell < 2", "cell < 4").replace("i < 2", "i < 4")
    quad = quad.replace("2 * step", "4 * step").replace(
        "pre[1] = __fmaf_rn(weights[k], p[1][k], pre[1]);",
        "\n".join(f"            pre[{i}] = __fmaf_rn(weights[k], p[{i}][k], pre[{i}]);"
                  for i in (1, 2, 3)),
    )
    return source[:start] + quad.replace("path_adjoint_pair", "path_adjoint_quad") + source[end:]


def _vertical_wave54_source(source: str, key: Kernel) -> str:
    source = source.replace("static_assert(!full || CACHE_SOURCE);", "").replace(
        "#define CACHED_SOURCE_AT(j) source_cache[(j) * THREADS].unpack()",
        "#define CACHED_SOURCE_AT(j) c.source(y0 + (j))",
    ).replace("I(CACHE_SOURCE ? H : span) * threads", "0")
    source = "\n".join(line for line in source.split("\n") if "seg_src[j * THREADS] =" not in line)
    start = source.index("    __device__ float4 source(int y) const {")
    end = source.index("    __device__ float2 gain", start)
    source = source[:start] + """    __device__ float4 source(int y) const {
        return packed_source(y).unpack();
    }

""" + source[end:]
    if not key.shared_acc:
        return source.replace("BACKWARD && HIDDEN && !SHARED_ACC;", "false;")
    head = 64 if key.partial_shared_acc else 90
    source = source.replace("SHARED_ACC ? fields * threads", f"SHARED_ACC ? {head} * threads")
    start = source.index("            for (I j = 4; j < fields;")
    end = source.index("\n        }", start)
    source = source[:start] + f"            for (int j=0; j<{head}; ++j) acc[j*threads]=0;" + source[end:]
    start = source.index("            acc -= 4 * threads;")
    end = source.index("        } else if (active)", start)
    # The weight tile is dead once every warp finishes its last line.
    reduction = """
            __syncthreads();
            float* poles = reinterpret_cast<float*>(shared) + threadIdx.x;
            poles[0]=da.r; poles[threads]=da.i; poles[2*threads]=dg.r; poles[3*threads]=dg.i;
            __syncthreads();
            if (warp == 0 && m < M) {
                for (int j=0; j<4+HEAD; ++j) {
                    float* field = j<4 ? poles+j*threads : acc+(j-4)*threads;
                    float sum = field[0];
                    for (int k=1; k<WARPS; ++k) sum += field[k*32];
                    row[I(j)*M] = sum;
                }
            }
"""
    if key.partial_shared_acc:
        reduction += """
            #pragma unroll
            for (int j=0; j<26; ++j) {
                __syncthreads();
                poles[0]=local_tail[j];
                __syncthreads();
                if (warp==0 && m<M) {
                    float sum=poles[0];
                    for (int k=1; k<WARPS; ++k) sum+=poles[k*32];
                    row[I(68+j)*M]=sum;
                }
            }
"""
    source = source[:start] + reduction.replace("HEAD", str(head)) + source[end:]
    if key.partial_shared_acc:
        source = source.replace("float* acc = acc_base", "float local_tail[26]{}; float* acc = acc_base")
        source = source.replace("T* gs, float* acc, I acc_stride,", "T* gs, float* acc, float* tail, I acc_stride,")
        source = source.replace("acc, acc_stride, (seg", "acc, local_tail, acc_stride, (seg")
        source = source.replace("float* acc, I stride, float (&dp)", "float* acc, float* tail, I stride, float (&dp)")
        source = source.replace("path, acc, acc_stride, dp);", "path, acc, tail, acc_stride, dp);")
        for offset in (64,72,80):
            source = source.replace(f"acc[({offset} + j) * stride]", f"tail[{offset-64}+j]")
        source = source.replace("acc[88 * stride]", "tail[24]").replace("acc[89 * stride]", "tail[25]")
        source = source.replace("else path_mlp<T, I, HIDDEN, false, true>(",
                                "else path_mlp<T, I, HIDDEN, false, true, 8>(")
        source = source.replace("                acc, acc_stride, dp);", "                tail, I(1), dp, acc, THREADS);")
    return source


_STREAM_STORES = r"""
template<class V> __device__ void stream_store(V* dst, V value) {
#if __CUDA_ARCH__ >= 700
    if constexpr (sizeof(V)==2)
        asm volatile("st.global.cs.b16 [%0], %1;" :: "l"(dst), "h"(__builtin_bit_cast(unsigned short,value)) : "memory");
    else
        asm volatile("st.global.cs.b32 [%0], %1;" :: "l"(dst), "r"(__builtin_bit_cast(unsigned,value)) : "memory");
#else
    *dst=value;
#endif
}
template<class T, class I>
__device__ void store_complex_cs(T* dst, I off, I plane, Complex z) {
    stream_store(dst+off,write<T>(z.r));
    stream_store(dst+off+plane,write<T>(z.i));
}
"""


def _horizontal_vec2_source(kernel: str) -> str:
    start = kernel.index("template<class T, class I, int WARPS")
    body = kernel.index("    const I stride_r", start)
    return _STREAM_STORES + kernel[start:body] + r"""
    constexpr int threads=32*WARPS;
    extern __shared__ float4 shared[];
    const int warp=threadIdx.x/32, lane=threadIdx.x%32, tiles=(M+63)/64;
    const int m=(blockIdx.x%tiles)*64+2*lane, group=blockIdx.x/tiles;
    const int worker=group*WARPS+warp;
    const I plane=I(B)*H*W*M, gp=plane/(1+COMPACT);
    float4 sums[2]{};
    using V=Complex;
    auto read2=[](const T* src) {
        unsigned bits=__ldg(reinterpret_cast<const unsigned*>(src));
        return reinterpret_cast<Pair<T>*>(&bits)->unpack();
    };
    auto store2=[](T* dst, float x, float y) {
        auto value=Pair<T>::pack({x,y});
        stream_store(reinterpret_cast<unsigned*>(dst),*reinterpret_cast<const unsigned*>(&value));
    };
    if (m<M && worker<workers) {
        Complex a[2]={{dr[m],di[m]},{dr[m+1],di[m+1]}};
        Complex g[2]={{decay_gr[m],decay_gi[m]},{decay_gr[m+1],decay_gi[m+1]}};
        for (I line=worker; line<I(B)*H; line+=workers) {
            const I base=line*W*M+m, gb=line*(W/(1+COMPACT))*M+m;
            auto inputs=[&](int x,V& r,V& i) {
                I tr=((line/H)*W+x)*H+line%H, ord=line*W+x;
                r=read2(sr+(REAL_TRANSPOSED?tr:ord)*M*(1+REAL_INTERLEAVED)+m);
                i=read2(si+(IMAG_TRANSPOSED?tr:ord)*M*(1+IMAG_INTERLEAVED)+m);
            };
            Complex right[2]{},left[2]{};
            #pragma unroll
            for (int pass=0;pass<2;++pass) {
                #pragma unroll 4
                for (int t=0;t<W;++t) {
                    int x=pass?W-1-t:t;
                    I off=base+I(x)*M, gi=gb+I(x/(1+COMPACT))*M;
                    V r{},i{},gr{},gim{},pr{},pi{},nr{},ni{};
                    if (!BACKWARD || pass) inputs(x,r,i);
                    if constexpr (BACKWARD) {
                        if (!COMPACT || ((x&1)==pass)) {
                            gr=read2(grad+(pass?0:2*gp)+gi);
                            gim=read2(grad+(pass?gp:3*gp)+gi);
                        }
                        if (pass && x) { pr=read2(states+off-M); pi=read2(states+plane+off-M); }
                        if (pass && x+1<W) {
                            nr=read2(states+2*plane+off+M); ni=read2(states+3*plane+off+M);
                        }
                    }
                    Complex output[2];
                    uint2 packed{};
                    auto* buf=reinterpret_cast<uint2*>(shared)+threadIdx.x+I(x/(1+COMPACT))*threads;
                    if constexpr (BACKWARD) { if (pass) packed=*buf; }
                    #pragma unroll
                    for (int q=0;q<2;++q) {
                        Complex s{read(q?r.i:r.r),read(q?i.i:i.r)};
                        if constexpr (!BACKWARD) {
                            auto& z=pass?left[q]:right[q];
                            z=(pass?a[q].conj():a[q])*z+(pass?g[q].conj():g[q])*s;
                            output[q]=z;
                        } else {
                            Complex adj{read(q?gr.i:gr.r),read(q?gim.i:gim.r)};
                            if (!pass) {
                                right[q]=a[q]*right[q]+adj;
                                auto stored=Pair<__nv_bfloat16>::pack(right[q]);
                                unsigned bits=*reinterpret_cast<unsigned*>(&stored);
                                if (q) packed.y=bits; else packed.x=bits;
                            } else {
                                left[q]=a[q].conj()*left[q]+adj;
                                unsigned bits=q?packed.y:packed.x;
                                auto ra=reinterpret_cast<Pair<__nv_bfloat16>*>(&bits)->unpack();
                                if (COMPACT && (x&1)) ra=a[q]*ra;
                                Complex prev{read(q?pr.i:pr.r),read(q?pi.i:pi.r)};
                                Complex next{read(q?nr.i:nr.r),read(q?ni.i:ni.r)};
                                Complex sum=left[q]+ra,diff{left[q].r-ra.r,left[q].i-ra.i};
                                sums[q].x+=left[q].r*prev.r+left[q].i*prev.i+ra.r*next.r+ra.i*next.i;
                                sums[q].y+=left[q].i*prev.r-left[q].r*prev.i+ra.r*next.i-ra.i*next.r;
                                sums[q].z+=sum.r*s.r+sum.i*s.i;
                                sums[q].w+=diff.i*s.r-diff.r*s.i;
                                output[q]={sum.r*g[q].r+diff.i*g[q].i,sum.i*g[q].r-diff.r*g[q].i};
                            }
                        }
                    }
                    if constexpr (BACKWARD) {
                        if (!pass && (!COMPACT || !(x&1))) *buf=packed;
                    }
                    if (!BACKWARD || pass) {
                        I dst=BACKWARD?2*(off-m)+m:off+2*pass*plane;
                        store2(out+dst,output[0].r,output[1].r);
                        store2(out+dst+(BACKWARD?I(M):plane),output[0].i,output[1].i);
                    }
                }
            }
        }
    }
    if constexpr (BACKWARD) {
        #pragma unroll
        for (int q=0;q<2;++q) {
            __syncthreads(); shared[threadIdx.x]=sums[q]; __syncthreads();
            if (warp==0 && m<M) {
                float4 z=shared[lane];
                for (int k=1;k<WARPS;++k) {
                    float4 t=shared[32*k+lane]; z.x+=t.x; z.y+=t.y; z.z+=t.z; z.w+=t.w;
                }
                float* row=partial+I(4)*group*M+m+q;
                row[0]=z.x;row[M]=z.y;row[2*I(M)]=z.z;row[3*I(M)]=z.w;
            }
        }
    }
}
"""


def _horizontal_candidate_source(source: str, key: Kernel) -> str:
    body = source.index("void horizontal(")
    start = source.rindex("template<", 0, body)
    end = source.index("template<bool PRECISE>", body)
    kernel = source[start:end]
    if key.horizontal_readonly:
        for name, index in (("sr", "base_r + I(x) * step_r"), ("si", "base_i + I(x) * step_i")):
            kernel = kernel.replace(f"{name}[{index}]", f"__ldg({name} + {index})")
        for name in ("grad", "states"):
            kernel = kernel.replace(f"load_complex({name},", f"load_complex_readonly({name},")
        helper = """template<class T, class I>
__device__ Complex load_complex_readonly(const T* p, I offset, I stride) {
    return {read(__ldg(p + offset)), read(__ldg(p + offset + stride))};
}
"""
        kernel = helper + kernel
    if key.horizontal_half_replay or key.horizontal_compact2:
        kernel = _horizontal_wave53_source(kernel, key)
    if key.horizontal_span:
        kernel = _horizontal_chunk_source(kernel, key.horizontal_span)
    if key.horizontal_stream:
        if key.horizontal_vec2:
            return source[:start] + _horizontal_vec2_source(kernel) + source[end:]
        kernel = _STREAM_STORES + kernel.replace("store_complex(out,", "store_complex_cs(out,")
        if key.horizontal_bf16_buf:
            kernel = kernel.replace("Pair<float>::pack(right)", "Pair<__nv_bfloat16>::pack(right)")
    if key.horizontal_unroll and not key.backward:
        for loop in ("for (int x = 0;", "for (int x = W - 1;"):
            pragma = "#pragma unroll BACKWARD_UNROLL\n                "
            kernel = kernel.replace(loop, pragma + loop, 1)
    return source[:start] + kernel + source[end:]


def _horizontal_wave53_source(kernel: str, key: Kernel) -> str:
    start = kernel.index("                #pragma unroll BACKWARD_UNROLL")
    end = kernel.index("\n            }\n        }", start)
    middle = kernel.index("                    const auto prev", start)
    body = kernel[middle:end].rsplit("\n                }", 1)[0]
    body = body.replace("source(x)", "Complex{read(__ldg(sr + ir)), read(__ldg(si + ii))}")
    body = body.replace("2 * (i - m) + m", "oi")
    consume = """
                I i = base + I(W-1) * M, oi = 2 * (i-m) + m;
                I ir = base_r + I(W-1) * step_r, ii = base_i + I(W-1) * step_i;
                auto consume = [&](int x, Complex right_adj) {
""" + body + """
                    i -= M; oi -= 2*M; ir -= step_r; ii -= step_i;
                };
"""
    if key.compact:
        # Compact adjoints are defined on even-width input pairs.
        loop = """
                #pragma unroll BACKWARD_UNROLL
                for (int k = 0; k < W/2; ++k) {
                    right = a * right + load_complex_readonly(grad,
                        2 * grad_plane + grad_base + I(k)*M, grad_plane);
                    buf[I(k)*stride] = Pair<float>::pack(right);
                    right = a * right;
                }
                I gi = grad_base + I(W/2-1)*M;
                #pragma unroll BACKWARD_UNROLL
                for (int k = W/2-1; k >= 0; --k, gi -= M) {
                    const auto even = buf[I(k)*stride].unpack();
                    left = a.conj() * left + load_complex_readonly(grad, gi, grad_plane);
                    consume(2*k+1, a*even);
                    left = a.conj()*left;
                    consume(2*k, even);
                }"""
    else:
        loop = """
                const int half = HALF_REPLAY ? W/2 : 0;
                #pragma unroll BACKWARD_UNROLL
                for (int x = 0; x < W; ++x) {
                    right = a * right + load_complex_readonly(grad,
                        2 * plane + base + I(x)*M, plane);
                    if (x >= half) buf[I(x-half)*stride] = Pair<BUF_TYPE>::pack(right);
                }
                auto descend = [&](int hi, int lo) {
                    #pragma unroll BACKWARD_UNROLL
                    for (int x = hi-1; x >= lo; --x) {
                        left = a.conj()*left + load_complex_readonly(grad, i, plane);
                        consume(x, buf[I(x-lo)*stride].unpack());
                    }
                };
                descend(W, half);
                if constexpr (HALF_REPLAY) {
                    right = {};
                    #pragma unroll BACKWARD_UNROLL
                    for (int x = 0; x < half; ++x) {
                        right = a*right + load_complex_readonly(grad,
                            2*plane + base + I(x)*M, plane);
                        buf[I(x)*stride] = Pair<BUF_TYPE>::pack(right);
                    }
                    descend(half, 0);
                }""".replace("HALF_REPLAY", str(key.horizontal_half_replay).lower())
    kernel = kernel[:start] + consume + loop + kernel[end:]
    dtype = "__nv_bfloat16" if key.horizontal_bf16_buf else "float"
    if key.horizontal_bf16_buf:
        begin = kernel.index("                Pair<float>* buf =")
        finish = kernel.index("                const I stride", begin)
        kernel = (kernel[:begin] + "                auto* buf = "
                  "reinterpret_cast<Pair<__nv_bfloat16>*>(shared) + threadIdx.x;\n"
                  + kernel[finish:])
    return kernel.replace("BUF_TYPE", dtype)


def _horizontal_chunk_source(kernel: str, span: int) -> str:
    # Save the right recurrence at segment boundaries and replay one segment
    # at a time. The left and negative recurrences keep their original order.
    start = kernel.index("                #pragma unroll BACKWARD_UNROLL")
    reverse = kernel.index("                #pragma unroll BACKWARD_UNROLL", start + 1)
    update = kernel[kernel.index("                    const I i", start):kernel.index(
        "                    // Odd compact", start,
    )]
    descending = kernel[reverse:].replace("x = W - 1; x >= 0;", "x = end - 1; x >= begin;", 1)
    descending = descending.replace(
        "buf[I(x / (1 + COMPACT))", "buf[I((x - begin) / (1 + COMPACT))",
    )
    descending = descending.replace(
        "                }\n            }",
        "                }\n                }\n            }", 1,
    )
    return kernel[:start] + f"""
                constexpr int span = {span};
                constexpr int slots = span / (1 + COMPACT);
                for (int x = 0; x < W; ++x) {{
{update}
                    if (x % span == span - 1 && x + 1 < W)
                        buf[I(slots + x / span) * stride] = Pair<float>::pack(right);
                }}
                for (int segment = (W - 1) / span; segment >= 0; --segment) {{
                    const int begin = segment * span;
                    const int end = begin + min(span, W - begin);
                    right = segment
                        ? buf[I(slots + segment - 1) * stride].unpack() : Complex{{}};
                    #pragma unroll BACKWARD_UNROLL
                    for (int x = begin; x < end; ++x) {{
{update}
                        if (!COMPACT || !(x & 1))
                            buf[I((x - begin) / (1 + COMPACT)) * stride] = Pair<float>::pack(right);
                    }}
""" + descending


_SGS_ADJOINT = r"""
template<class T, class I, bool COMPACT, int THREADS>
__device__ __forceinline__ void column_adjoint_positive(
    Column<T, I> c, int y, float4 prev, float4 source, float4 inject,
    Complex a, Complex g, Complex& adj_a, Complex& adj_b, Complex& da, Complex& dg,
    T* gs, Quad<T>* pos_grad
) {
    const auto ds = column_adjoint_step<false>(
        prev, source, inject, a, g, adj_a, adj_b, da, dg);
    pos_grad[I(y) * THREADS] = Quad<T>::pack(ds);
}

template<class T, class I, bool COMPACT, int THREADS>
__device__ __forceinline__ void column_adjoint_negative(
    Column<T, I> c, int y, float4 prev, float4 source, float4 inject,
    Complex a, Complex g, Complex& adj_a, Complex& adj_b, Complex& da, Complex& dg,
    T* gs, const Quad<T>* pos_grad
) {
    const auto ds = column_adjoint_step<true>(
        prev, source, inject, a, g, adj_a, adj_b, da, dg);
    write_column_gradient<true, COMPACT, false>(c, y, ds, gs, pos_grad[I(y) * THREADS]);
}
"""


def _odd_coarse_negative(source: str) -> str:
    source = source.replace("!COARSE || !(y & 1)", "!COARSE || (!(y & 1) && y + 1 < c.H)")
    return source.replace(
        "if constexpr (COARSE) clear_unused_columns<COMPACT>(c, y, gs);",
        r"""if constexpr (COARSE) {
                clear_unused_columns<COMPACT>(c, y, gs);
                if constexpr (!COMPACT) {
                    if ((c.W & 1) && (c.offset_b / c.M) % c.W == c.W - 3) {
                        const I tail = I(y) * c.W * c.M + c.offset_b + 2 * I(c.M);
                        store_complex(gs, tail, c.plane, {});
                        store_complex(gs, tail + 2 * c.plane, c.plane, {});
                    }
                }
            }""",
    )


def _odd_coarse_backward_source(source: str) -> str:
    before = source.index("#ifdef BACKWARD_BATCH2", source.index("void column_backward("))
    after = source.index("    adj_a = adj_b = {};", before)
    reverse = source[before:after].replace("length - 1", "scan_end")
    source = source[:before] + r"""
        int scan_end = length - 1;
        if constexpr (COARSE) {
            if ((y0 + scan_end) % 2 == 0) {
                const auto s = CACHED_SOURCE_AT(scan_end);
                const auto next = neg;
                neg = advance(neg, s, ar, gg);
                column_adjoint<false, COMPACT>(
                    c, y0 + scan_end, scan_end ? seg[(scan_end - 1) * THREADS] : before,
                    s, {}, a, g, adj_a, adj_b, da, dg, gs);
                if constexpr (full) seg[scan_end * THREADS] = next;
                --scan_end;
            }
        }
""" + reverse + source[after:]
    return _odd_coarse_negative(source)


def _pipelined_adjoint(source: str, pipe: int) -> str:
    start = source.index("    #pragma unroll\n    for (int j", source.index("void path_adjoint_pair("))
    end = source.index("\n}\n#endif", start)
    original = source[start:end]
    load = original[original.index("        if constexpr"):original.index("        #pragma unroll\n        for (int k = 0; k < 8; ++k)")]
    update = original[original.index("            const float act"):original.rindex("    }")]
    helpers = """
    auto load_w = [&](int j, float (&weights)[8]) {
""" + load + """
    };
    auto preactivate = [&](int j, const float (&w)[8], float (&z)[2]) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            float t[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                t[k] = __fmaf_rn(w[k+4], p[c][k+4], w[k] * p[c][k]);
            z[c] = ((t[0] + t[1]) + (t[2] + t[3])) + path.bias(j);
        }
    };
    auto apply = [&](int j, const float (&weights)[8], const float (&zs)[2], const float (&ss)[2]) {
        const float wr = path.output(j), wi = path.output(8+j);
        #pragma unroll
        for (int cell = 0; cell < 2; ++cell) {
            const float z = zs[cell], sig = ss[cell];
""" + update + "    };"
    if pipe == 1:
        loop = """
    float w[2][8], z[2][2], sig[2][2];
    load_w(0, w[0]);
    preactivate(0, w[0], z[0]);
    sig[0][0] = fast_sigmoid_tanh(z[0][0]);
    sig[0][1] = fast_sigmoid_tanh(z[0][1]);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int cur = j % 2, next = 1 - cur;
        if (j + 1 < 8) {
            load_w(j+1, w[next]);
            preactivate(j+1, w[next], z[next]);
            #pragma unroll
            for (int c = 0; c < 2; ++c) sig[next][c] = fast_sigmoid_tanh(z[next][c]);
        }
        apply(j, w[cur], z[cur], sig[cur]);
    }"""
    else:
        loop = """
    #pragma unroll
    for (int jb = 0; jb < 8; jb += 4) {
        float z[4][2], sig[4][2], w[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            load_w(jb+i, w);
            preactivate(jb+i, w, z[i]);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) sig[i][c] = fast_sigmoid_tanh(z[i][c]);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            load_w(jb+i, w);
            apply(jb+i, w, z[i], sig[i]);
        }
    }"""
    return source[:start] + helpers + loop + source[end:]


def _fp32_path_source(source: str) -> str:
    # Preserve the baseline CUDA source verbatim for all other keys.
    start = source.index("template<class T>\n__device__ Quad<T> normalize_packed")
    end = source.index("template<int WARPS, int FIELDS, class I>", start)
    path = source[start:end].replace(
        "const auto stored = Quad<T>::pack(v).unpack();",
        "const auto stored = v;",
    ).replace(
        "return normalize_packed<T>(v, gain).unpack();",
        "return make_float4(v.x * gain.x, v.y * gain.x, v.z * gain.y, v.w * gain.y);",
    )
    path = path.replace("rounded<T>(", "(").replace(
        "weights[slot] = (value);", "weights[slot] = rounded<T>(value);",
    )
    return source[:start] + path + source[end:]


def _tanh_gain_source(source: str) -> str:
    sigmoid = """__device__ __forceinline__ float fast_sigmoid_tanh(float z) {
#if __CUDA_ARCH__ >= 750
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * z));
    return __fmaf_rn(0.5f, t, 0.5f);
#else
    return __fdividef(1.0f, 1.0f + __expf(-z));
#endif
}

"""
    for value in ("pre", "z"):
        source = source.replace(
            f"const float sig = __fdividef(1.0f, 1.0f + __expf(-{value}));",
            f"const float sig = fast_sigmoid_tanh({value});",
        ).replace(
            f"(sig * (1.0f + {value} * (1.0f - sig)))",
            "__fmaf_rn(-act, sig, sig + act)",
        )
    source = source.replace("template<class T, class I>\nstruct Column", sigmoid + (
        "template<class T, class I>\nstruct Column"
    ))
    source = source.replace("float vx_a, vx_b, eps;", "float inv_vx_a, inv_vx_b, inv_eps;")
    # Keep zero variances at the epsilon cap via rsqrt(0) = +infinity.
    source = source.replace(
        "vx[I(xa) * M + m], vx[(I(W) + xb) * M + m], eps};",
        "rsqrtf(fmaxf(vx[I(xa) * M + m], 0.0f)),\n"
        "                           rsqrtf(fmaxf(vx[(I(W) + xb) * M + m], 0.0f)), rsqrtf(eps)};",
    )
    source = source.replace(
        "return make_float2(rsqrtf(fmaxf(vx_a * v, eps)), rsqrtf(fmaxf(vx_b * v, eps)));",
        "const float rv = rsqrtf(fmaxf(v, 0.0f));\n"
        "        return make_float2(fminf(inv_vx_a * rv, inv_eps), "
        "fminf(inv_vx_b * rv, inv_eps));",
    )
    source = source.replace(
        "const auto pp = normalize<T>(seg[j * THREADS], c.gain(y, false));",
        "gp_c[i] = c.gain(y, false);\n"
        "                gn_c[i] = c.gain(y - int(COARSE), true);\n"
        "                const auto pp = normalize<T>(seg[j * THREADS], gp_c[i]);",
    ).replace(
        "const auto pn = normalize<T>(neg, c.gain(y - int(COARSE), true));",
        "const auto pn = normalize<T>(neg, gn_c[i]);",
    ).replace(
        "make_float4(dp[i][0], dp[i][4], dp[i][1], dp[i][5]), c.gain(y, false));",
        "make_float4(dp[i][0], dp[i][4], dp[i][1], dp[i][5]), gp_c[i]);",
    ).replace(
        "                    c.gain(y - int(COARSE), true));", "                    gn_c[i]);",
    )
    return source.replace("float2 dy[2];", "float2 dy[2], gp_c[2], gn_c[2];", 1)


def _sgs_source(source: str, *, persistent: bool = False) -> str:
    if persistent:
        pos_grad = "buf + I(oh) * stride"
        stride_decl = ", I pos_stride"
        stride_arg = ", g2_stride"
        launch_stride = ", stride"
        adjoint = _SGS_ADJOINT.replace("Quad<T>* pos_grad", "Quad<T>* pos_grad, I pos_stride")
        adjoint = adjoint.replace("I(y) * THREADS", "I(y) * pos_stride")
    else:
        pos_grad = "line_buf + (CACHE_LINE ? I(oh) * threads : 0) + threadIdx.x"
        stride_decl = stride_arg = launch_stride = ""
        adjoint = _SGS_ADJOINT
    source = source.replace(
        "int chunks, Complex& da, Complex& dg",
        f"int chunks, Complex& da, Complex& dg, Quad<T>* pos_grad{stride_decl}",
    ).replace(
        "buf, stride, chunks, da, dg",
        f"buf, stride, chunks, da, dg, {pos_grad}{launch_stride}",
    )
    start = source.index("void column_backward(")
    end = source.index("#undef CACHED_SOURCE_AT", start)
    column = source[start:end].replace(
        "column_adjoint<false, COMPACT>(", "column_adjoint_positive<T, I, COMPACT, THREADS>(",
    ).replace("da, dg, gs);", f"da, dg, gs, pos_grad{stride_arg});").replace(
        "column_adjoint<true, COMPACT>(", "column_adjoint_negative<T, I, COMPACT, THREADS>(",
    )
    return (source[:start] + column + source[end:]).replace(
        "template<bool COMPACT, class T, class I>\n__device__ __forceinline__ void clear",
        adjoint
        + "template<bool COMPACT, class T, class I>\n__device__ __forceinline__ void clear",
    )


def _grouped_vscan_source(source: str, reduce_warps: int) -> str:
    # Keep baseline partial rows and worker stride so the final FP32
    # reduction sees the same operands in the same order.
    start = source.index("void write_vscan_partial(")
    end = source.index("struct RegisterPathWeights", start)
    reduction = source[start:end].replace(
        "threadIdx.x < 32", f"(threadIdx.x / 32) % {reduce_warps} == 0",
    ).replace("k < WARPS", f"k < {reduce_warps}").replace(
        "int m, int M", "int m, int M, bool active",
    ).replace("&& m < M", "&& active")
    source = source[:start] + reduction + source[end:]
    return source.replace(
        "row += I(reduce ? group : worker) * fields * M + m;",
        f"if (active) row += I(worker / {reduce_warps}) * fields * M + m;",
        1,
    ).replace(
        "row, nullptr, da, dg, m, M);", "row, nullptr, da, dg, m, M, active);",
    ).replace(
        "row, local_acc, da, dg, m, M);", "row, local_acc, da, dg, m, M, active);",
    )


def _odd_compact_source(source: str) -> str:
    return source.replace(
        "const I plane = COMPACT ? c.plane / 2 : c.plane;",
        "const I plane = COMPACT ? c.plane / c.W * c.OW : c.plane;",
    ).replace(
        "(row + c.offset_a - c.m - c.M) / 2 + c.m",
        "(I(y) + (c.output - c.m) / (I(c.H / 2) * c.OW * c.M) * c.H) * c.OW * c.M"
        " + (c.output - c.m) % (I(c.OW) * c.M) + c.m",
    )


def _effective_kernel(key: Kernel, capability: tuple[int, int]) -> Kernel:
    if key.dw1_tile and capability < (9, 0):
        # Ampere and Ada need more room than Hopper for the packed adjoint.
        return replace(key, min_blocks=min(key.min_blocks, 12 // key.warps))
    if (
        (key.horizontal_unroll8 or key.horizontal_unroll) and capability >= (9, 0)
        and key.warps == 1 and (key.real_interleaved or key.imag_interleaved)
    ):
        # Task 92 observed inexact forced unrolls on this layout.
        return replace(key, horizontal_unroll8=False, horizontal_unroll=0)
    return key


def _compile_ptx(key: Kernel, capability: tuple[int, int]) -> ct.Array:
    key = _effective_kernel(key, capability)
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
def _scan_workspace_budget(device: int) -> int:
    if not torch.cuda.is_available():
        return _DEFAULT_WORKSPACE_BYTES
    total = torch.cuda.get_device_properties(device).total_memory
    base = min(total, _BASE_BUDGET_VRAM) // _BASE_WORKSPACE_DIVISOR
    extra = max(0, total - _BASE_BUDGET_VRAM) // 1024
    return base + (min(_l2_size(device), extra) if extra else 0)


@lru_cache
def _horizontal_cache_budget(device: int) -> int:
    # Leave room for current input and output cache lines.
    return _l2_size(device) * 7 // 8


@cache
def _function(device: int, key: Kernel) -> tuple[ct.CDLL, ct.c_void_p]:
    capability = torch.cuda.get_device_capability(device)
    effective = _effective_kernel(key, capability)
    if key != effective:
        return _function(device, effective)
    image = _compile_ptx(key, capability)
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
        if (
            key.dw1_tile or key.sgs or key.l2_single_write or key.batch_path or key.register_path
            or key.horizontal_unroll8
            or key.backward_batch2 or key.backward_readonly
            or key.forward_readonly or key.horizontal_readonly or key.horizontal_unroll
            or (key.vertical and key.backward and key.chunk and key.cache_source)
        ):
            local = ct.c_int()
            _check_cu(
                driver.cuFuncGetAttribute(ct.byref(local), 3, func),
                "cuFuncGetAttribute(LOCAL_SIZE_BYTES)",
            )
            if local.value:
                return 0
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
    grid = _ceil(source.shape[-1], _WARP_SIZE * (1 + key.horizontal_vec2)) * _ceil(workers, plan.warps)
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
    lines: int, modes: int, fields: int, warps: int, wave: int,
    budget: int = _DEFAULT_WORKSPACE_BYTES,
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


def _checkpoint_rows(height: int, chunk: int) -> int:
    return (height - 1) // (chunk or height)


def _scratch_per_worker(plan: ScanPlan, length: int, *, vertical: bool, coarse: bool,
                        backward: bool, itemsize: int, compact: bool = False) -> int:
    """Return global scratch storage per worker and mode, in bytes."""
    if not vertical:
        return length // (1 + int(compact)) * 8 if backward and not plan.cache_line else 0
    outputs = length // (1 + int(coarse))
    line = 0 if plan.cache_line else outputs * 4 * itemsize
    checkpoints = _checkpoint_rows(length, plan.chunk) * 16 if plan.global_checkpoints else 0
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
    workspace = _scan_workspace_budget(device)
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
            workspace,
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
            workspace,
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
            workers = min(workers, (workspace) // scratch)
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


def _align_workers(
    launch: Launch, lines: int, tiles: int, wave: int, *, max_workers: int = 0,
) -> Launch:
    plan, key, workers, partials = launch
    groups = _ceil(workers, plan.warps)
    aligned = (groups * tiles // wave) * wave // tiles
    trim = 0 < aligned < groups and _ceil(lines, aligned * plan.warps) == _ceil(lines, workers)
    if not trim:
        if not max_workers:
            return launch
        aligned = _ceil(groups * tiles, wave) * wave // tiles
        if not workers < aligned * plan.warps <= min(lines, max_workers):
            return launch
    workers = aligned * plan.warps
    return Launch(plan, key, workers, aligned if key.backward and partials == groups else workers)


def _plan_cached_backward(
    device: int, launch: Launch, length: int, limit: int, resident: int,
) -> Launch:
    plan, key, _, _ = launch
    step = 1 + key.coarse
    pair_rows = 2 * step
    # Small coarse columns have too few complete pairs to amortize the tail.
    pair = length >= pair_rows and (not key.coarse or length >= 4 * pair_rows)
    tail = bool(length % pair_rows or plan.chunk % pair_rows)
    if pair:
        candidate = replace(key, backward_batch2=True, backward_batch_tail=tail)
        if _resident(device, candidate, plan.smem) == resident:
            key = candidate

    if not plan.full_trajectory:
        # Dense scans target four replay segments. Coarse scans perform one
        # path evaluation per two rows and can afford a longer segment.
        segments = 4 - key.coarse
        span = _ceil(length, segments * pair_rows) * pair_rows
        if plan.chunk < span < length:
            rows = span + (length - 1) // span - plan.chunk - (length - 1) // plan.chunk
            smem = plan.smem + rows * 16 * _WARP_SIZE * plan.warps
            candidate = replace(
                key, chunk=span,
                backward_batch_tail=key.backward_batch2 and bool(length % pair_rows),
            )
            if smem <= limit and _resident(device, candidate, smem) == resident:
                key, plan = candidate, plan._replace(chunk=span, smem=smem)
        if key.coarse and key.backward_batch_tail:
            key = replace(key, backward_batch2=False, backward_batch_tail=False)

    # Read-only source loads pay off when populating the cache for a long
    # replay or a dense column with enough rows to amortize the cache fill.
    readonly = (key.coarse and not plan.full_trajectory) or (
        not key.coarse and plan.full_trajectory and length >= 4 * pair_rows
    )
    if readonly:
        candidate = replace(key, backward_readonly=True)
        if _resident(device, candidate, plan.smem) == resident:
            key = candidate
    return launch._replace(plan=plan, key=key)


def _plan_horizontal_backward(
    device: int, launch: Launch, lines: int, width: int, modes: int, sms: int, resident: int,
) -> Launch:
    plan, key, _, _ = launch
    if not (key.backward and key.cache_line and width >= _LONG_SCAN):
        return launch
    # The single-warp interleaved SM 9.0+ kernel needs the compiler's
    # default unroll to preserve the recurrence's numerical behavior.
    interleaved = key.real_interleaved or key.imag_interleaved
    if modes % _WARP_SIZE == 0 and key.warps <= 2 + key.compact and not (
        interleaved and key.warps == 1 and torch.cuda.get_device_capability(device) >= (9, 0)
    ):
        candidate = replace(key, horizontal_unroll8=True)
        if _resident(device, candidate, plan.smem) >= resident:
            return launch._replace(key=candidate)
    if key.warps == max(_SCAN_WARPS):
        return _align_workers(launch, lines, _ceil(modes, _WARP_SIZE), sms * resident)
    return launch


def _plan_vertical_forward(
    device: int, launch: Launch, length: int, lines: int, modes: int, sms: int,
    resident: int, max_warps: int,
) -> Launch:
    plan, key, workers, _ = launch
    step = 1 + key.coarse
    tiles = _ceil(modes, _WARP_SIZE)
    if key.hidden8 and plan.cache_weights:
        if not plan.cache_line and key.warps == 1 and length >= 8 * step:
            candidate = replace(key, batch_path=True)
            paired = _resident(device, candidate, plan.smem)
            queued = _ceil(workers * tiles, sms)
            if paired >= min(max_warps // 4, resident) and queued >= 2 * paired:
                key, resident = candidate, paired
        elif key.vector_weights and not key.coarse and plan.cache_line and modes % _WARP_SIZE == 0:
            weight_bytes = 11 * _REGISTER_HIDDEN * _WARP_SIZE * 4
            smem = plan.smem - weight_bytes
            if smem >= 2 * weight_bytes and key.warps == max(_SCAN_WARPS):
                candidate = replace(
                    key, register_path=True, cache_weights=False, batch_path=True,
                    register_batch2=True,
                )
                paired = _resident(device, candidate, smem)
                if paired * key.warps >= 12:
                    key, resident = candidate, paired
                    plan = plan._replace(smem=smem, cache_weights=False)
    launch = launch._replace(plan=plan, key=key)
    if length // step < 3 * _SHORT_CHECKPOINT_CHUNK:
        return launch
    cap = 0
    if length >= _LONG_SCAN:
        scratch_per_worker = length // step * modes * 4 * key.dtype.itemsize
        # A round-up can consume one extra wave of scratch. Keep it bounded
        # by the workspace budget plus one L2 working set.
        cap = lines if plan.cache_line else min(
            (_scan_workspace_budget(device) + _l2_size(device)) // scratch_per_worker,
            (2**31 - 1) // (scratch_per_worker // key.dtype.itemsize),
        )
    return _align_workers(launch, lines, tiles, sms * resident, max_workers=cap)


def _plan_scan_auto(
    device: int, launch: Launch, shape: tuple[int, int, int, int],
) -> Launch:
    plan, key, _, _ = launch
    if key.dtype.itemsize != 2 or not key.index32 or key.pure_vertical:
        return launch
    b, h, w, m = shape
    sms, limit, max_warps = device_limits(device)
    resident = _resident(device, key, plan.smem)
    if not resident:
        return launch
    if not key.vertical:
        return _plan_horizontal_backward(device, launch, b * h, w, m, sms, resident)
    lines = b * (w // (1 + key.coarse))
    if not key.backward:
        return _plan_vertical_forward(device, launch, h, lines, m, sms, resident, max_warps)
    if key.hidden8 and plan.cache_weights and plan.cache_source and not (
        key.shared_acc or key.global_checkpoints
    ):
        launch = _align_workers(launch, lines, _ceil(m, _WARP_SIZE), sms * resident)
        return _plan_cached_backward(device, launch, h, limit, resident)
    return launch


W51A_MODES = ("off", "fp32", "l2", "fp32_l2", "fp32_hiocc")
W51B_MODES = ("off", "tiled_dw", "tiled_dw_fp32")
W52A_MODES = (
    "off", "tanh_gain", "tanh_gain_smemw", "l2_floor_g2", "l2_floor_sw", "combo_g2", "combo_sw",
)
SGS_MODES = ("off", "strict", "strict_ro", "strict_ro_hiocc", "strict_ro_seg2")
VFWD_MODES = ("off", "batch4", "batch4_w2", "batch4_hiocc")
W53A_MODES = ("off", "w52_best", "w52_best_swpipe", "w52_best_pipe4", "w53_lockin")
W54A_MODES = ("off", "v54_sacc_aw12", "v54_psacc_aw12", "v54_gacc_aw12", "v54_sacc_b4")
H54_MODES = ("off", "h54_fwd_nocache", "h54_fwd_vec2", "h54_bwd_bf16_all",
             "h54_bwd_bf16_half", "h54_bwd_vec2", "h54_combo_vec2")
H53_MODES = ("h53_ro_all", "h53_compact2", "h53_bf16_buf", "h53_l2_scratch", "h53_lockin")
HSCAN_MODES = ("off", "bwd_opt", "bwd_opt_ch4", "bwd_opt_w2") + H53_MODES


def _scan_mode(env: str, allowed: tuple[str, ...]) -> str:
    mode = os.environ.get(env, "off")
    if mode not in allowed:
        raise ValueError(f"{env}={mode!r}; allowed modes: {', '.join(allowed)}")
    return mode


def _wave52a_mode(itemsize: int) -> str:
    return _scan_mode("LNET_SCAN_W52A", W52A_MODES) if itemsize == 2 else "off"


@cache
def _sgs_device_limits(device: int) -> tuple[int, int, int]:
    driver = _load_cuda_driver()
    values = []
    for attribute in (81, 82, 111):
        value = ct.c_int()
        _check_cu(
            driver.cuDeviceGetAttribute(ct.byref(value), attribute, device),
            f"cuDeviceGetAttribute({attribute})",
        )
        values.append(value.value)
    return values[0], values[1], values[2]


def _sgs_smem(key: Kernel, height: int) -> int:
    threads = key.warps * _WARP_SIZE
    span = key.chunk or height
    checkpoints = 0 if key.global_checkpoints else (height - 1) // span
    source_rows = span if key.coarse or key.backward_segment_source else 0
    if key.cache_source:
        source_rows = height
    line_rows = height // (1 + key.coarse) if key.cache_line else 0
    itemsize = key.dtype.itemsize
    weights = 11 * _REGISTER_HIDDEN * _WARP_SIZE * (
        4 if key.vector_weights else itemsize
    ) if key.cache_weights else 0
    return weights + threads * (16 * (span + checkpoints) + 4 * itemsize * (
        source_rows + line_rows
    ))


def _sgs_cap(block_limit: int, shared: int, reserved: int, resident: int) -> int:
    return min(block_limit, ((shared // resident - reserved) // 256) * 256)


def _wider_readonly_layout(
    launch: Launch, shape: tuple[int, int, int, int], active_warps: int, sms: int,
    block_limit: int, shared: int, reserved: int,
) -> tuple[Kernel, int, int] | None:
    _, key, workers, _ = launch
    _, height, _, modes = shape
    # With an odd height, changing the span can move the scalar tail to
    # another row and change which rows run the paired adjoint.
    if not (
        key.chunk and not key.coarse and not key.global_checkpoints and key.cache_source
        and key.vector_weights and key.backward_batch2 and not (height % 2 or key.chunk % 2)
    ):
        return None
    best = None
    best_span = key.chunk
    for warps in _SCAN_WARPS:
        if warps <= key.warps or warps % key.warps or active_warps % warps:
            continue
        resident = active_warps // warps
        if _ceil(workers, warps) * _ceil(modes, _WARP_SIZE) < 2 * sms * resident:
            continue
        cap = _sgs_cap(block_limit, shared, reserved, resident)
        candidate = replace(
            key, warps=warps, backward_readonly=True, backward_reduce_warps=key.warps,
        )
        threads = warps * _WARP_SIZE
        max_span = min(height - 1, cap // (threads * 16)) & ~1
        for span in range(max_span, best_span, -2):
            candidate = replace(candidate, chunk=span)
            smem = max(threads * 128, _sgs_smem(candidate, height))
            if smem <= cap:
                best, best_span = (candidate, smem, resident), span
                break
    return best


@lru_cache(maxsize=512)
def _plan_sgs(
    device: int, launch: Launch, shape: tuple[int, int, int, int], mode: str,
) -> Launch:
    plan, key, _, _ = launch
    _, height, _, _ = shape
    if (
        mode not in SGS_MODES[1:] or key.shared_acc or not key.hidden8
        or key.dtype.itemsize != 2 or key.pure_vertical
    ):
        return launch
    strict = _plan_sgs(device, launch, shape, "strict") if mode != "strict" else launch
    full = not key.chunk and key.cache_source
    chunked = key.chunk and key.cache_source and not key.global_checkpoints
    eligible = full or (mode != "strict" and chunked)
    if not eligible:
        return launch
    target = _resident(device, key, plan.smem)
    if not target:
        return launch
    sms, block_limit, warp_limit = device_limits(device)
    shared, registers, reserved = _sgs_device_limits(device)
    active_warps = target * key.warps
    if mode == "strict":
        smem = plan.smem + height * key.warps * _WARP_SIZE * 8
        candidate = replace(key, sgs=1)
        if (
            active_warps <= min(warp_limit, registers // _WARP_SIZE)
            and smem <= _sgs_cap(block_limit, shared, reserved, target)
            and _resident(device, candidate, smem) == target
        ):
            launch = launch._replace(plan=plan._replace(smem=smem, sgs_rows=height), key=candidate)
        return launch

    readonly = replace(strict.key, backward_readonly=True)
    if not strict.key.backward_readonly and _resident(device, readonly, strict.plan.smem) == target:
        strict = strict._replace(key=readonly)
    wider = _wider_readonly_layout(launch, shape, active_warps, sms, block_limit, shared, reserved)
    if wider is not None:
        candidate, smem, resident = wider
        if _resident(device, candidate, smem) == resident:
            return launch._replace(
                plan=plan._replace(warps=candidate.warps, chunk=candidate.chunk, smem=smem),
                key=candidate,
            )
    return strict


def _probe_scan_candidate(device: int, baseline: Launch, candidate: Launch) -> int:
    target = baseline.key.warps * _resident(device, baseline.key, baseline.plan.smem)
    resident = _resident(device, candidate.key, candidate.plan.smem)
    return resident if target and resident * candidate.key.warps >= target else 0


@lru_cache(maxsize=512)
def _plan_vfwd(device: int, launch: Launch, height: int, mode: str) -> Launch:
    key = launch.key
    if mode == "off" or key.dtype.itemsize != 2 or not key.hidden8 or key.pure_vertical:
        return launch
    readonly = launch._replace(key=replace(key, forward_readonly=True))
    best = readonly if _probe_scan_candidate(device, launch, readonly) else launch
    rows = height // (1 + key.coarse)
    if rows < 2:
        return best
    weights = 11 * _REGISTER_HIDDEN * _WARP_SIZE * (4 if key.vector_weights else key.dtype.itemsize)
    plan = best.plan._replace(
        cache_weights=False, smem=best.plan.smem - weights * key.cache_weights,
    )
    for width in (4, 2):
        if rows < width:
            continue
        candidate = best._replace(
            plan=plan,
            key=replace(best.key, cache_weights=False, register_path=True,
                        batch_path=True, register_batch2=width == 2),
        )
        if _probe_scan_candidate(device, launch, candidate):
            return candidate
        if width == 4 and key.cache_weights:
            candidate = best._replace(key=replace(best.key, batch_path=True))
            if _probe_scan_candidate(device, launch, candidate):
                return candidate
    return best


@lru_cache(maxsize=512)
def _plan_hscan(
    device: int, launch: Launch, shape: tuple[int, int, int, int], mode: str,
) -> Launch:
    plan, key, _, _ = launch
    width = shape[2]
    if mode == "off" or not key.backward or key.warps > 3 or width <= 2 * plan.chunk:
        return launch
    base_occupancy = key.warps * _resident(device, key, plan.smem)
    if base_occupancy < 24:
        return launch
    candidate = replace(key, horizontal_readonly=True, horizontal_unroll=8)
    try:
        # The driver JIT assembles NVRTC PTX. _resident rejects local memory,
        # including stack slots used for register spills.
        occupancy = key.warps * _resident(device, candidate, plan.smem)
    except RuntimeError:
        return launch
    return launch._replace(key=candidate) if occupancy >= base_occupancy else launch



def _active_warps(device: int, launch: Launch) -> int:
    try:
        return launch.key.warps * _resident(device, launch.key, launch.plan.smem)
    except RuntimeError:
        return 0


def _structural_probe(device: int, launch: Launch, modes: int, minimum: int) -> int:
    sms, limit, warp_limit = device_limits(device)
    key = launch.key
    if launch.plan.smem > limit or minimum > warp_limit:
        return 0
    grid = _ceil(launch.workers, key.warps) * _ceil(modes, _WARP_SIZE)
    if grid < 2 * sms * _ceil(minimum, key.warps):
        return 0
    active = _active_warps(device, launch)
    return active if active >= minimum and grid >= 2 * sms * (active // key.warps) else 0


def _launch_bound(device: int, warps: int, smem: int, target: int) -> int:
    _, limit, warp_limit = device_limits(device)
    shared, registers, reserved = _sgs_device_limits(device)
    blocks = _ceil(target, warps)
    if (
        blocks * warps <= min(warp_limit, registers // _WARP_SIZE)
        and smem <= _sgs_cap(limit, shared, reserved, blocks)
    ):
        return blocks
    return 0


def _structural_sgs(  # noqa: C901
    device: int, base: Launch, shape: tuple[int, ...], mode: str,
) -> Launch:
    _, height, _, modes = shape
    key = base.key
    if not (
        key.hidden8 and key.backward_readonly and key.cache_source
        and not (key.global_checkpoints or key.shared_acc or key.odd_coarse_backward)
    ):
        return base
    occupancy = _active_warps(device, base)
    if not occupancy:
        return base
    group = key.backward_reduce_warps or key.warps
    span = key.chunk
    segment_mode = mode == "strict_ro_seg2"
    if segment_mode:
        pair_rows = (1 + key.coarse) * (1 + key.backward_batch2)
        span = _ceil(height, 2 * pair_rows) * pair_rows
        if not key.chunk or span <= key.chunk or span >= height or key.chunk % pair_rows:
            return base
    minimum = occupancy + (not segment_mode)
    best, score = base, None
    for warps in (2, 3, 4, 6, 8, 1):
        if warps % group:
            continue
        source_options = (True,) if key.cache_line or not key.chunk else (True, False)
        for cache_source in source_options:
            candidate = replace(
                key, warps=warps, chunk=span, cache_source=cache_source,
                backward_segment_source=not cache_source and not key.coarse,
                backward_reduce_warps=group if warps != group else 0,
            )
            smem = max(warps * _WARP_SIZE * 128, _sgs_smem(candidate, height))
            if key.sgs:
                smem += height * warps * _WARP_SIZE * 8
            trial = base._replace(
                key=candidate, plan=base.plan._replace(
                    warps=warps, chunk=span or base.plan.chunk,
                    smem=smem, cache_source=cache_source,
                ),
            )
            blocks = _launch_bound(device, warps, smem, minimum)
            if not blocks:
                continue
            active = 0 if trial == base else _structural_probe(device, trial, modes, minimum)
            if not segment_mode and not active:
                trial = trial._replace(key=replace(candidate, min_blocks=blocks))
                active = _structural_probe(device, trial, modes, minimum)
            if active and (score is None or (active, cache_source, -smem) > score):
                best, score = trial, (active, cache_source, -smem)
    return best


def _structural_vfwd(
    device: int, launch: Launch, base: Launch, shape: tuple[int, ...], mode: str,
) -> Launch:
    _, height, _, modes = shape
    capped = mode == "batch4_hiocc"
    seed = launch if launch.key.cache_weights else base
    key = seed.key
    register_batch4 = key.register_path and key.batch_path and not key.register_batch2
    if not (
        key.hidden8 and base.key.forward_readonly and height // (1 + key.coarse) >= 4
        and (key.cache_weights or (capped and register_batch4))
    ):
        return base
    occupancy = _active_warps(device, base)
    if not occupancy:
        return base
    minimum = occupancy + capped
    weight_itemsize = 4 if key.vector_weights else key.dtype.itemsize
    weights = 11 * _REGISTER_HIDDEN * _WARP_SIZE * weight_itemsize if key.cache_weights else 0
    best, score = base, None
    line = (seed.plan.smem - weights) // key.warps
    for warps in (2, 4):
        smem = weights + line * warps
        blocks = _launch_bound(device, warps, smem, minimum)
        if not blocks:
            continue
        candidate = replace(
            key, warps=warps, forward_readonly=True, batch_path=True,
            register_batch2=False,
        )
        trial = base._replace(
            key=candidate, plan=seed.plan._replace(warps=warps, smem=smem),
        )
        bounds = (0,)
        if capped:
            extra = _launch_bound(device, warps, smem, (blocks + 1) * warps)
            bounds = (extra, blocks) if extra else (blocks,)
        for bound in bounds:
            probe = trial._replace(key=replace(candidate, min_blocks=bound))
            active = _structural_probe(device, probe, modes, minimum)
            if active and (score is None or (active, warps) > score):
                best, score = probe, (active, warps)
    return best


def _structural_hscan(device: int, base: Launch, shape: tuple[int, ...], mode: str) -> Launch:
    _, _, width, modes = shape
    key = base.key
    if not (key.backward and key.cache_line and key.dtype.itemsize == 2 and width > 2 * key.chunk):
        return base
    if key.chunk // 2 < (1 + key.compact) or key.chunk // 2 % (1 + key.compact):
        return base
    occupancy = _active_warps(device, base)
    _, _, max_warps = device_limits(device)
    if not 0 < occupancy < max_warps * 3 // 8:
        return base
    # A smaller cache can help only when shared memory blocks the next CTA.
    if _launch_bound(device, key.warps, base.plan.smem, occupancy + 1):
        return base
    narrower = mode == "bwd_opt_w2"
    warps = 2 if narrower else key.warps
    minimum = occupancy + (not narrower)
    best, score = base, None
    for span in ((0, key.chunk // 2) if narrower else (key.chunk // 2,)):
        slots = (
            span // (1 + key.compact) + (width - 1) // span
            if span else width // (1 + key.compact)
        )
        smem = warps * _WARP_SIZE * max(16 if warps > 1 else 0, 8 * slots)
        for unroll in (4, 8):
            trial = base._replace(
                key=replace(key, warps=warps, horizontal_span=span, horizontal_readonly=True,
                            horizontal_unroll=unroll),
                plan=base.plan._replace(warps=warps, smem=smem),
                partials=_ceil(base.workers, warps),
            )
            active = _structural_probe(device, trial, modes, minimum)
            if active and (score is None or (active, not span) > score):
                best, score = trial, (active, not span)
    return best


@lru_cache(maxsize=512)
def _plan_structural(device: int, launch: Launch, shape: tuple[int, ...], mode: str) -> Launch:
    if not launch.key.vertical:
        base = _plan_hscan(device, launch, shape, "bwd_opt")
        return _structural_hscan(device, base, shape, mode)
    if launch.key.backward:
        base = _plan_sgs(device, launch, shape, "strict_ro")
        return _structural_sgs(device, base, shape, mode)
    base = _plan_vfwd(device, launch, shape[1], "batch4")
    return _structural_vfwd(device, launch, base, shape, mode)


def _persistent_workers(
    lines: int, rows: int, height: int, modes: int, itemsize: int,
    warps: int, sms: int, resident: int, budget: int, checkpoint_rows: int = 0,
) -> tuple[int, bool]:
    tiles = _ceil(modes, _WARP_SIZE)
    workers = min(_ceil(lines, warps), _ceil(sms * resident, tiles)) * warps
    quad_bytes = workers * modes * 4 * itemsize
    scan_bytes = rows * quad_bytes + checkpoint_rows * workers * modes * 16
    if scan_bytes > budget:
        return 0, False
    return workers, bool(height and scan_bytes + height * quad_bytes <= budget)


def _persistent_workers_floor(
    lines: int, rows: int, height: int, modes: int, itemsize: int,
    warps: int, sms: int, resident: int, budget: int, checkpoint_rows: int = 0,
) -> tuple[int, bool]:
    tiles = _ceil(modes, _WARP_SIZE)
    capacity = sms * resident
    groups = min(_ceil(lines, warps), max(1, capacity // tiles))
    if groups * tiles > capacity:
        return 0, False
    workers = groups * warps
    quad_bytes = workers * modes * 4 * itemsize
    scan_bytes = rows * quad_bytes + checkpoint_rows * workers * modes * 16
    if scan_bytes > budget:
        return 0, False
    return workers, bool(height and scan_bytes + height * quad_bytes <= budget)


@lru_cache(maxsize=512)
def _plan_w52a(device: int, launch: Launch, shape: tuple[int, int, int, int], mode: str) -> Launch:
    plan, key, old_workers, _ = launch
    b, h, w, m = shape
    key = replace(key, fp32_path=True, tanh_gain=mode.startswith(("tanh_gain", "combo")))
    smem_weights = mode in ("tanh_gain_smemw", "combo_g2", "combo_sw")
    if key.register_path and not key.backward and smem_weights:
        rows = h // (1 + key.coarse)
        line_bytes = rows * key.warps * _WARP_SIZE * 4 * key.dtype.itemsize * plan.cache_line
        weights = 11 * _REGISTER_HIDDEN * _WARP_SIZE * 4
        smem = plan.smem - line_bytes + weights
        candidate = replace(
            key, register_path=False, cache_weights=True, vector_weights=True, cache_line=False,
        )
        # Moving a line out of SMEM must also respect the global workspace budget.
        workers = old_workers
        if plan.cache_line:
            bytes_per_worker = rows * m * 4 * key.dtype.itemsize
            workers = min(workers, _scan_workspace_budget(device) // bytes_per_worker)
        if workers and _resident(device, candidate, smem):
            key = candidate
            plan = plan._replace(cache_weights=True, cache_line=False, smem=smem)
            old_workers = workers
            launch = launch._replace(workers=workers, partials=workers)
    launch = launch._replace(plan=plan, key=key)
    if not mode.startswith(("l2_floor", "combo")) or plan.cache_line:
        return launch
    resident = _resident(device, key, plan.smem)
    sms, _, _ = device_limits(device)
    step = 1 + key.coarse
    checkpoints = (h - 1) // plan.chunk if plan.global_checkpoints else 0
    workers, single = _persistent_workers_floor(
        b * (w // step), h // step, h if key.backward and not key.sgs else 0,
        m, key.dtype.itemsize, key.warps, sms, resident, _l2_size(device) * 3 // 4, checkpoints,
    )
    if not 0 < workers < old_workers:
        return launch
    if single and mode.endswith("_sw"):
        candidate = replace(key, l2_single_write=True)
        if _resident(device, candidate, plan.smem) >= resident:
            key = candidate
    group = key.backward_reduce_warps or key.warps
    reduce = key.backward and (key.shared_acc or key.hidden8)
    return Launch(plan, key, workers, workers // group if reduce else workers)


def _plan_w53a(device: int, launch: Launch, shape: tuple, mode: str) -> Launch:
    key = launch.key
    height = shape[1]
    floor = (
        (not key.backward and not key.coarse and height >= 48)
        or (key.backward and key.coarse and height >= 48)
        or (key.backward and not key.coarse and 24 <= height < 48)
    )
    smemw = not key.backward and not key.coarse and 24 <= height < 48
    winner = "combo_sw" if floor else "tanh_gain_smemw" if smemw else "tanh_gain"
    if mode == "w53_lockin" and key.backward:
        if (not key.coarse and height >= 48) or (key.coarse and height <= 16):
            winner = "fp32"
        elif 24 <= height < 48:
            mode = "w52_best_swpipe"
    launch = _plan_w52a(device, launch, shape, winner)
    if key.backward_batch2 and key.hidden8 and key.backward:
        pipe = {"w52_best_swpipe": 1, "w52_best_pipe4": 2}.get(mode, 0)
        launch = launch._replace(key=replace(launch.key, path_bwd_pipe=pipe))
    return launch


@lru_cache(maxsize=512)
def _plan_h53(device: int, launch: Launch, shape: tuple, mode: str) -> Launch:
    plan, key, old_workers, _ = launch
    if not key.backward:
        return launch
    b, h, width, modes = shape
    long_dense = width >= 48 and not key.compact
    if mode == "h53_lockin":
        if long_dense:
            # Reuse the measured dense pointer-stepping loop for both dtypes.
            mode = "h53_bf16_buf"
        elif (key.compact and width >= 48) or (not key.compact and 24 <= width < 48):
            return _plan_hscan(device, launch, shape, "bwd_opt")
        else:
            mode = "h53_ro_all"
    scratch = long_dense and mode == "h53_l2_scratch"
    bf16 = long_dense and key.cache_line and mode == "h53_bf16_buf" and key.dtype == torch.bfloat16
    half = long_dense and key.cache_line and not (scratch or bf16)
    warps = 2 if scratch else key.warps
    smem = plan.smem
    if scratch:
        smem = warps * 32 * 16
    elif half or bf16:
        smem = max(16, width * 4 if bf16 else (width - width // 2) * 8) * warps * 32
    candidate = replace(key, warps=warps, cache_line=key.cache_line and not scratch,
                        horizontal_wave53=True, horizontal_readonly=True, horizontal_half_replay=half,
                        horizontal_bf16_buf=bf16, horizontal_compact2=mode != "h53_ro_all")
    baseline = _active_warps(device, launch)
    for unroll in (8, 4):
        candidate = replace(candidate, horizontal_unroll=unroll)
        trial = launch._replace(key=candidate, plan=plan._replace(
            warps=warps, smem=smem, cache_line=candidate.cache_line))
        active = _active_warps(device, trial)
        if active and active >= baseline:
            break
    else:
        trial, active = launch, baseline
    if not active:
        return launch
    warps = trial.key.warps
    tiles = _ceil(modes, 32)
    lines = b * h
    wave = max(1, device_limits(device)[0] * (active // warps) // tiles) * warps
    limit = min(old_workers, lines, 2 * wave)
    if scratch and not trial.key.cache_line:
        limit = min(limit, (32 << 20) // (width * modes * 8))
    if limit <= 0:
        return launch
    if lines <= 2 * limit:
        workers = max(1, lines // _ceil(lines, limit) // warps) * warps
    elif limit >= wave:
        workers = limit // wave * wave
    else:
        workers = max(1, limit // warps) * warps
    workers = min(workers, limit)
    return trial._replace(workers=workers, partials=_ceil(workers, warps))


@lru_cache(maxsize=512)
def _plan_w54(device: int, launch: Launch, shape: tuple, mode: str) -> Launch:
    key = launch.key
    b, h, w, m = shape
    if (
        not (key.backward and key.hidden8 and key.index32)
        or key.pure_vertical
        or key.odd_coarse_backward
    ):
        return launch
    shared = mode != "v54_gacc_aw12"
    partial = mode == "v54_psacc_aw12"
    # The coarse four-cell adjoint needs 254–255 registers on SM90.
    quad = mode == "v54_sacc_b4" and not key.coarse
    step = 1 + key.coarse
    # The scalar tail of global accumulation exceeded the SM90 register budget.
    if not shared and h % (2 * step):
        return launch
    candidates = []
    for warps in ((4,) if quad else (4, 3)):
        spans = set(range(2 * step, min(h, 20) + 1, 2 * step))
        if h <= 14:
            spans.add(h)
        if quad:
            spans = {min(8, h)}
        for span in spans:
            chunk = 0 if span == h else span
            fields = (64 if partial else 90) if shared else 0
            smem = 11264 + warps * 32 * (16 * span + 4 * fields)
            if smem > (76458 if warps == 4 else 57088):
                continue
            k = replace(
                key,
                warps=warps,
                chunk=chunk,
                backward_reduce_warps=0,
                shared_acc=shared,
                partial_shared_acc=partial,
                backward_reload_source=True,
                backward_segment_source=False,
                global_checkpoints=True,
                cache_source=False,
                cache_line=False,
                cache_weights=True,
                vector_weights=True,
                backward_readonly=True,
                fp32_path=True,
                tanh_gain=True,
                backward_batch2=True,
                backward_batch4=quad,
                path_bwd_pipe=0,
                backward_batch_tail=bool(h % ((4 if quad else 2) * step)),
                sgs=0,
                l2_single_write=False,
                min_blocks=0,
            )
            plan = ScanPlan(warps, smem, chunk, True, True, False, shared, chunk == 0)
            candidates.append((span, warps, k, plan))
    for _, warps, k, plan in sorted(candidates, reverse=True):
        resident = _resident(device, k, plan.smem)
        if resident * warps < 12:
            continue
        wave = max(1, device_limits(device)[0] * resident // _ceil(m, 32)) * warps
        workers = min(b * (w // step), 2 * wave, launch.workers)
        if not shared:
            workers = min(workers, (36 << 20) // (94 * m * 4))
        if workers:
            return Launch(plan, k, workers, _ceil(workers, warps) if shared else workers)
    return launch


@lru_cache(maxsize=512)
def _plan_h54(device: int, launch: Launch, shape: tuple, mode: str, aligned: bool) -> Launch:
    base = _plan_h53(device, launch, shape, "h53_lockin")
    key = base.key
    if (
        key.backward
        and mode.startswith("h54_fwd")
        or not key.backward
        and mode.startswith("h54_bwd")
    ):
        return base
    _, _, width, m = shape
    vec = "vec2" in mode and aligned and m % 2 == 0
    half = key.backward and mode.endswith("half") and width >= 48 and not key.compact
    narrow = key.backward and (vec or width >= (48 if key.compact else 24))
    saved = (width + 1) // 2 if half else width // (1 + key.compact)
    smem = 128 * max(16, saved * (8 if vec or not narrow else 4)) if key.backward else 0
    k = replace(
        key,
        warps=4,
        horizontal_stream=True,
        horizontal_vec2=vec,
        horizontal_wave53=key.backward,
        horizontal_readonly=True,
        horizontal_unroll=4,
        horizontal_compact2=narrow and not vec,
        horizontal_half_replay=half,
        horizontal_bf16_buf=narrow,
        cache_line=key.backward,
    )
    if smem > device_limits(device)[1] or not _resident(device, k, smem):
        return base
    return Launch(
        base.plan._replace(warps=4, smem=smem, cache_line=key.backward),
        k,
        base.workers,
        _ceil(base.workers, 4),
    )


def _fp32_hiocc(
    device: int, launch: Launch, shape: tuple[int, int, int, int], budget: int,
) -> Launch:
    key = launch.key
    if key.backward or not key.hidden8 or not key.fp32_path:
        return launch
    b, height, width, modes = shape
    step = 1 + key.coarse
    rows, lines = height // step, b * (width // step)
    tiles = _ceil(modes, _WARP_SIZE)
    sms, block_limit, _ = device_limits(device)
    shared, _, reserved = _sgs_device_limits(device)
    best, score = launch, (0, 0, 0)
    weight_bytes = 11 * _REGISTER_HIDDEN * _WARP_SIZE * 4
    for warps in (2, 4):
        cta_bytes = rows * warps * modes * 4 * key.dtype.itemsize
        max_groups = budget // cta_bytes if cta_bytes else 0
        if not max_groups:
            continue
        # Reserve dynamic SMEM to bound residency only when all naturally
        # resident CTAs would overflow the L2 budget. Each resident CTA can
        # still receive work without oversubscribing scratch.
        cta_cap = max_groups * tiles // sms
        smem = weight_bytes
        if max_groups < _ceil(lines, warps):
            if not cta_cap:
                continue
            smem = max(smem, _ceil(shared // (cta_cap + 1) + 1 - reserved, 256) * 256)
        if smem > block_limit:
            continue
        for width in (2, 4):
            if rows < width:
                continue
            candidate = replace(
                key, warps=warps, cache_weights=True, cache_line=False,
                vector_weights=True, batch_path=True, register_path=False,
                register_batch2=width == 2, min_blocks=0,
            )
            plan = launch.plan._replace(
                warps=warps, smem=smem, cache_line=False, cache_weights=True,
            )
            trial = launch._replace(key=candidate, plan=plan)
            active = _active_warps(device, trial)
            rank = (active, width, warps)
            if active and rank > score:
                best, score = trial, rank
    return best


@lru_cache(maxsize=512)
def _plan_w51a(
    device: int, launch: Launch, shape: tuple[int, int, int, int], mode: str,
) -> Launch:
    if mode == "off" or not launch.key.vertical or launch.key.pure_vertical:
        return launch
    b, h, w, m = shape
    key = launch.key
    if mode.startswith("fp32") and key.dtype.itemsize == 2:
        launch = launch._replace(key=replace(key, fp32_path=True))
    if mode == "fp32":
        return launch
    budget = _l2_size(device) * 3 // 4
    fallback = launch
    if mode == "fp32_hiocc":
        launch = _fp32_hiocc(device, launch, shape, budget)
    plan, key, old_workers, _ = launch
    if plan.cache_line:
        return launch
    sms, _, _ = device_limits(device)
    resident = _resident(device, key, plan.smem)
    if not resident:
        return launch
    step = 1 + key.coarse
    checkpoint_rows = (h - 1) // plan.chunk if plan.global_checkpoints else 0
    workers, single = _persistent_workers(
        b * (w // step), h // step, h if key.backward and not key.sgs else 0,
        m, key.dtype.itemsize, key.warps, sms, resident, budget, checkpoint_rows,
    )
    if single:
        candidate = replace(key, l2_single_write=True)
        if _resident(device, candidate, plan.smem) >= resident:
            key = candidate
    if not workers or (
        launch is fallback and workers > _ceil(old_workers, key.warps) * key.warps
    ):
        return fallback
    group = key.backward_reduce_warps or key.warps
    reduce = key.backward and (key.shared_acc or key.hidden8)
    partials = workers // group if reduce else workers
    return Launch(plan, key, workers, partials)


def _w51b_key(
    base: Kernel, warps: int, chunk: int, tile: int, minimum: int, mode: str,
) -> Kernel:
    return replace(
        base, warps=warps, chunk=chunk, dw1_tile=tile, dw1_fp32=mode == "tiled_dw_fp32",
        min_blocks=minimum, cache_weights=True, vector_weights=True, cache_source=False,
        cache_line=False, global_checkpoints=False, shared_acc=False,
        backward_batch2=False, backward_batch_tail=False, backward_readonly=False,
        backward_segment_source=False, backward_reduce_warps=0,
    )


def _w51b_smem(key: Kernel, height: int) -> int:
    weights = 11 * 8 * 32 * 4
    boundaries = (height - 1) // key.chunk
    per_lane = 16 * (key.chunk + boundaries) + 32 * key.dw1_tile
    return weights + key.warps * 32 * per_lane


def _w51b_workspace(height: int, coarse: bool, modes: int) -> int:
    return modes * (94 * 4 + (height + height // (1 + coarse)) * 8)


@lru_cache(maxsize=128)
def _plan_w51b(
    device: int, base: Launch, shape: tuple[int, ...], mode: str,
) -> Launch:
    batch, height, width, modes = shape
    if mode == "off" or height < 8 or not base.key.hidden8 or base.key.dtype.itemsize != 2:
        return base
    sms, limit, _ = device_limits(device)
    tiles = _ceil(modes, 32)
    budget = _l2_size(device) * 7 // 8 // _w51b_workspace(height, base.key.coarse, modes)
    lines = batch * (width // (1 + base.key.coarse))
    for target in (16, 12):
        for warps, tile in ((4, 8), (4, 4), (8, 4)):
            key = _w51b_key(base.key, warps, 8, tile, _ceil(target, warps), mode)
            smem = _w51b_smem(key, height)
            if smem > limit or not _launch_bound(device, warps, smem, target):
                continue
            resident = _resident(device, key, smem)
            if resident * warps < target:
                continue
            workers = min(lines, _ceil(sms * resident, tiles) * warps, budget)
            if workers <= 0:
                continue
            return Launch(ScanPlan(warps, smem, 8, cache_weights=True), key, workers, workers)
    return base


def _plan_launch(  # noqa: C901, PLR0912
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
    saved_trajectory: bool = False,
    aligned: bool = True,
) -> Launch | None:
    if vertical and backward:
        mode = _scan_mode("LNET_SCAN_SGS", SGS_MODES)
    elif vertical:
        mode = _scan_mode("LNET_SCAN_VFWD", VFWD_MODES)
    else:
        mode = _scan_mode("LNET_SCAN_HSCAN", HSCAN_MODES)
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

    if launch is None:
        return None
    if os.environ.get("LNET_GENERAL_SCAN_OPT", "auto") != "off":
        launch = _plan_scan_auto(device, launch, shape)
    _, height, width, _ = shape
    if vertical and coarse and not pure_vertical and (height % 2 or width % 2):
        launch = launch._replace(key=replace(
            launch.key, odd_coarse_forward=not backward and bool(height % 2),
            odd_coarse_backward=backward,
            odd_compact_width=backward and compact and bool(width % 2),
            backward_batch2=False, backward_batch_tail=False,
        ))
    wave51a = _scan_mode("LNET_SCAN_W51A", W51A_MODES) if vertical else "off"
    wave53a = (
        _scan_mode("LNET_SCAN_W53A", W53A_MODES)
        if vertical and source.dtype.itemsize == 2 and not (pure_vertical or saved_trajectory)
        else "off"
    )
    wave52a = (
        _wave52a_mode(source.dtype.itemsize)
        if vertical and not (pure_vertical or saved_trajectory) else "off"
    )
    w54 = _scan_mode("LNET_SCAN_W54A", W54A_MODES) if vertical else "off"
    if w54 != "off" and source.dtype.itemsize == 2 and not (saved_trajectory or pure_vertical):
        wave53a = "w53_lockin"
    if vertical and backward and not (pure_vertical or saved_trajectory) and mode in (
        "off", "strict_ro",
    ) and wave52a == wave53a == "off":
        wave51b = _scan_mode("LNET_SCAN_W51B", W51B_MODES)
        if wave51b != "off":
            candidate = _plan_w51b(device, launch, shape, wave51b)
            if candidate.key.dw1_tile:
                return candidate
            if wave51a == "fp32_hiocc":
                wave51a = "fp32"
    if not vertical and source.dtype.itemsize == 2:
        h54 = _scan_mode("LNET_SCAN_H54", H54_MODES)
        if h54 != "off":
            return _plan_h54(device, launch, shape, h54, aligned)
    if mode in H53_MODES:
        return _plan_h53(device, launch, shape, mode)
    if mode not in ("off", "strict", "strict_ro", "batch4", "bwd_opt"):
        launch = _plan_structural(device, launch, shape, mode)
    elif mode != "off":
        if not vertical:
            return _plan_hscan(device, launch, shape, mode)
        if backward:
            launch = _plan_sgs(device, launch, shape, mode)
        else:
            launch = _plan_vfwd(device, launch, height, mode)
    if wave53a != "off":
        launch = _plan_w53a(device, launch, shape, wave53a)
        return launch if w54 == "off" else _plan_w54(device, launch, shape, w54)
    if wave52a != "off":
        # Both waves tune the same fused kernel and scratch allocation.
        return _plan_w52a(device, launch, shape, wave52a)
    return launch if wave51a == "off" else _plan_w51a(device, launch, shape, wave51a)


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
    if coarse and (h % 2 or w % 2) and states.dtype.itemsize != 2:
        return False
    if not _matching(states, [*auxiliary, *path]):
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


def _vec2_aligned(*values: Tensor | None) -> bool:
    return all(value.data_ptr() % 4 == 0 and all(step % 2 == 0 for step in value.stride()[:-1])
               for value in values if value is not None)


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
        aligned=_vec2_aligned(real, imag, states, adjoints)
                if os.getenv("LNET_SCAN_H54", "off") != "off" else True,
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
    trajectory: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor] | list[Tensor]:
    shape = tuple(states.shape[1:])
    if (
        not _shape_supported(shape)
        or not _vertical_inputs_ok(states, auxiliary, path, coarse=coarse)
        or not _pole_ok(pole, states)
        or not 0 < epsilon < float("inf")
        or (compact and not coarse)
        or (pure and coarse and (shape[1] % 2 or shape[2] % 2))
    ):
        raise ValueError("Invalid vertical scan inputs")
    b, h, w, m = shape
    hidden = 0 if pure else path[0].shape[1]
    backward = grad is not None or features is not None
    mode = _scan_mode("LNET_SCAN_SGS", SGS_MODES) if backward else _scan_mode(
        "LNET_SCAN_VFWD", VFWD_MODES,
    )
    oh, ow = (h // 2, w // 2) if coarse else (h, w)
    out_shape = (b, oh, ow, m)
    checkpoints = None
    output: Tensor | tuple[Tensor, Tensor] | None = None
    interleaved = (packed and not planar,) * 2
    if grad is not None:
        gr, gi = _vertical_adjoints(grad, states, out_shape)
        interleaved = (not gr.is_contiguous(), not gi.is_contiguous())
    if coarse and not (oh and ow):
        if backward:
            grad_shape = (4, b, h, ow if compact else w, m)
            return [
                *[torch.zeros_like(t) for t in pole], states.new_zeros(grad_shape),
                *[torch.zeros_like(t) for t in path],
            ]
        if packed:
            return states.new_empty((2, *out_shape) if planar else (b, oh, ow, 2 * m))
        return states.new_empty(out_shape), states.new_empty(out_shape)
    if saved_features is not None and (
        grad is None
        or saved_features.shape != (8, b * oh * ow, m)
        or saved_features.device != states.device
        or saved_features.dtype != states.dtype
        or not saved_features.is_contiguous()
    ):
        raise ValueError("Invalid saved vertical features")
    wave51a = _scan_mode("LNET_SCAN_W51A", W51A_MODES)
    wave52a = "off" if pure else _wave52a_mode(states.element_size())
    if _scan_mode("LNET_SCAN_W54A", W54A_MODES) == "off" and wave51a == wave52a == "off" and trajectory is None and not pure and not (
        (coarse and (h % 2 or w % 2))
        or (backward and saved_features is None and (
            mode != "off"
            or (hidden == 8 and h >= 8 and states.element_size() == 2
                and _scan_mode("LNET_SCAN_W51B", W51B_MODES) != "off")
        ))
    ):
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
        saved_trajectory=trajectory is not None,
    )
    if launch is None:
        raise RuntimeError("Vertical CUDA scan is ineligible")
    if trajectory is not None:
        plan, base, _, _ = launch
        line_bytes = oh * plan.warps * 32 * 4 * states.element_size() if plan.cache_line else 0
        plan = plan._replace(cache_line=False, smem=plan.smem - line_bytes)
        key = replace(
            base, saved_trajectory=1, cache_line=False, l2_single_write=False, fp32_path=False,
            index32=base.index32 and trajectory.numel() // 4 < 2**31,
        )
        launch = launch._replace(plan=plan, key=key)
        checkpoints = trajectory
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
        checkpoint_rows = _checkpoint_rows(h, launch.plan.chunk)
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
    # Keep g2 and the optional positive adjoints in disjoint row ranges of
    # one workspace. A persistent worker reuses both ranges on its next line.
    positive_rows = h if launch.key.l2_single_write or launch.key.dw1_tile else 0
    scratch = (
        None if launch.plan.cache_line or trajectory is not None else
        states.new_empty((oh + positive_rows, launch.workers, m, 4))
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
            ct.c_longlong(
                gs.stride(0) * (1 + int(compact))
                if pure and gs is not None else 0,
            ),
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
    budget = vram // (16 if vram <= _BASE_BUDGET_VRAM else 32)
    scan_workspace = _scan_workspace_budget(states.device.index or 0)
    # Projection overlaps both feature planes with its partial reduction;
    # scan scratch overlaps only one plane.
    live_planes = min(budget - scan_workspace, 2 * (budget - 3 * scan_workspace))
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
    if _scan_mode("LNET_SCAN_W51A", W51A_MODES) != "off":
        return False
    if _wave52a_mode(states.element_size()) != "off":
        return False
    vram = _total_vram(states.device)
    if policy == "saved" and vram < 48 * 1024**3:
        return False
    # Reserve space for activations and all of the model's saved feature planes.
    budget = vram // 12
    _, b, h, w, m = states.shape
    if coarse and (h % 2 or w % 2):
        return False
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
            groups = min(groups, max(1, _DEFAULT_WORKSPACE_BYTES // bytes_per_group))
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
            ys = slice(int(not reverse), 2 * (height // 2), 2) if coarse else slice(None)
            xs = slice(1 - side, 2 * (states.shape[3] // 2), 2) if coarse else slice(None)
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


VSCAN_SAVED_MODES = ("off", "saved_fused", "saved_split", "auto_vram")


def _trajectory_shape(states: Tensor, coarse: bool, split: bool) -> tuple[int, ...]:
    _, b, h, w, m = states.shape
    shape = (2, b, h, w // (1 + coarse), m, 4)
    # Keep the selected backward schedule in the tensor metadata saved by autograd.
    return (1, *shape) if split else shape


def _vscan_saved_policy(
    states: Tensor, path: list[Tensor], coarse: bool, mode: str | None = None,
) -> str:
    if mode is None:
        mode = _scan_mode("LNET_SCAN_VSCAN_SAVED", VSCAN_SAVED_MODES)
    # #854: saved_fused/saved_split cost +13.03/+16.33 ms on H200 and failed BF16 decay-grad parity.
    if mode in ("off", "auto_vram") or not states.is_cuda or not path or path[0].shape[1] != 8:
        return "off"
    if _scan_mode("LNET_SCAN_W51A", W51A_MODES) != "off":
        return "off"
    if _wave52a_mode(states.element_size()) != "off":
        return "off"
    _, _, h, w, _ = states.shape
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    if not (oh and ow):
        return "off"
    return mode


def _vscan_trajectory_backward(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    grad: tuple[Tensor, Tensor],
    coarse: bool,
    epsilon: float,
    compact: bool,
    trajectory: Tensor,
) -> list[Tensor]:
    split = trajectory.ndim == 7
    if (
        not _vertical_inputs_ok(states, auxiliary, path, coarse=coarse)
        or not _pole_ok(pole, states)
        or (compact and not coarse) or not 0 < epsilon < float("inf")
        or trajectory.shape != _trajectory_shape(states, coarse, split)
        or trajectory.dtype != states.dtype or trajectory.device != states.device
        or not trajectory.is_contiguous() or path[0].shape[1] != 8
    ):
        raise ValueError("Invalid saved vertical trajectory")
    _, b, h, w, m = states.shape
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    gr, gi = _vertical_adjoints(grad, states, (b, oh, ow, m))
    gs = states.new_empty((4, b, h, ow if compact else w, m))
    dp = states.new_empty((2, b, oh, ow, m, 4)) if split else None
    inputs = (
        *(p.float().contiguous() for p in pole),
        states.contiguous(),
        *(t.float().contiguous() for t in auxiliary),
        *(p.contiguous() for p in path[:3]),
        gr, gi, gs,
    )
    device = states.device.index or 0
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    tiles = _ceil(m, 32)
    reduced = []
    narrow_index = max(states.numel(), trajectory.numel() // 4) < 2**31
    for variant, count, fields in (
        ((3, b * oh * ow, 90), (4, b * ow, 4)) if split else ((2, b * ow, 94),)
    ):
        # Wide coarse adjoints use the two-warp schedule to avoid local spills.
        warps = 2 if variant == 4 and coarse and not narrow_index else 4
        weight_bytes = 88 * 32 * states.element_size() if variant != 4 else 0
        smem = weight_bytes
        if variant == 2:
            per_warp = oh * 32 * 4 * states.element_size()
            while warps * per_warp + weight_bytes > device_limits(device)[1] and warps > 1:
                warps //= 2
            smem += warps * per_warp
            if smem > device_limits(device)[1]:
                return vertical_backward(
                    pole, states, auxiliary, path, gr, gi, coarse, epsilon, compact,
                )
        workers = min(count, _ceil(sms * 8, tiles) * warps)
        partial = states.new_empty((workers, fields, m), dtype=torch.float32)
        index32 = narrow_index and partial.numel() < 2**31
        key = Kernel(
            states.dtype, warps, backward=True, vertical=True, coarse=coarse,
            hidden8=True, compact=compact, index32=index32,
            min_blocks=3 if variant == 3 and index32 and not coarse else 0,
            real_interleaved=not gr.is_contiguous(), imag_interleaved=not gi.is_contiguous(),
            odd_compact_width=compact and bool(w % 2), saved_trajectory=variant,
        )
        _launch_kernel(
            states, key, tiles * _ceil(workers, warps), smem,
            (*map(_ptr, (*inputs, partial, trajectory, dp)),
             *map(ct.c_int, (b, h, w, m, workers, _scalar_types(path))), ct.c_float(epsilon)),
        )
        if variant == 2:
            reduced = _reduce_partials(partial, [*pole, *path], 8)
        elif variant == 3:
            reduced.extend(_reduce_partials(partial, path, 8, projection=True))
        else:
            reduced[:0] = _reduce_partials(partial, pole)
    return [*reduced[:4], gs, *reduced[4:]]


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
    if saved_features is not None and saved_features.ndim in (6, 7):
        return _vscan_trajectory_backward(
            pole, states, auxiliary, path, (grad_real, grad_imag), coarse, epsilon, compact,
            saved_features,
        )
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
    trajectory_policy: str = "off",
) -> tuple[Tensor, Tensor, Tensor]:
    _, b, h, w, m = states.shape
    step = 2 if coarse else 1
    shape = (b, h // step, w // step, m)
    if trajectory_policy != "off":
        if trajectory_policy not in ("saved_fused", "saved_split") or not (shape[1] and shape[2]):
            raise ValueError("Invalid vertical trajectory policy or shape")
        trajectory = states.new_empty(
            _trajectory_shape(states, coarse, trajectory_policy == "saved_split"),
        )
        out = _vscan(
            pole, states, auxiliary, path, coarse, epsilon, precise,
            packed=not separate, planar=planar, trajectory=trajectory,
        )
        if isinstance(out, tuple):
            return *out, trajectory
        if isinstance(out, Tensor):
            return out, states.new_empty((0,)), trajectory
        raise TypeError("Expected saved scan outputs")
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
    trajectory_policy: str = "off",
) -> tuple[Tensor, Tensor, Tensor]:
    _, b, h, w, m = states.shape
    step = 2 if coarse else 1
    shape = (b, h // step, w // step, m)
    return (
        states.new_empty(shape if separate else ((2, *shape) if planar else (*shape[:-1], 2 * m))),
        states.new_empty(shape if separate else (0,)),
        states.new_empty((8, b * (h // step) * (w // step), m)) if trajectory_policy == "off"
        else states.new_empty(
            _trajectory_shape(states, coarse, trajectory_policy == "saved_split"),
        ),
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
    pole, states, auxiliary, path, ctx.coarse, ctx.epsilon, _, _, ctx.separate, *_ = inputs
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
    result = gradients[:4], gradients[4], [None, None], gradients[5:], *(None,) * 7
    return result[:len(ctx.needs_input_grad)]


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
    policy = _vscan_saved_policy(states, path, coarse) if mode == "cuda" else "off"
    if mode == "cuda" and (policy != "off" or _should_save_vscan(states, coarse)):
        real, imag, _ = _vertical_forward_saved(
            pole, states, auxiliary, path, coarse, epsilon,
            planar=False, precise=precise, separate=True, trajectory_policy=policy,
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
    if coarse and states.ndim == 5 and (states.shape[2] % 2 or states.shape[3] % 2):
        return None
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
