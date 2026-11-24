"""Packed real and strict-complex spatial convolution for factorized readers."""

from __future__ import annotations

# Custom op schemas use positional flags.
# ruff: noqa: FBT001, FBT002
import ctypes
import os
from functools import cache
from threading import local
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional

from .cuda_nvrtc import _check_cu, _check_nvrtc, _CudaModule, _ensure_context, _include_dirs, _nvrtc

_CUDA_TYPES = {torch.float32: "float", torch.float16: "__half", torch.bfloat16: "__nv_bfloat16"}
_MAX_CUDA_GRID_DIM = 65535
_WARP_SIZE = 32
_DEFAULT_THREADS = 128
_STATIC_RANK_LIMIT = 4
_STATIC_SPATIAL_KERNEL = 3
_COMPACT_COLUMNS = 7
_DEFAULT_COLUMNS = 8
_DEFAULT_SPATIAL_GEOMETRY = (2, 4)
_COMPACT_SPATIAL_GEOMETRY = (4, 2)
_RESIDENT_FILTER_RANK_LIMIT = 2
_CENTERED_WEIGHT_TILE = 2
_VEC2_SPATIAL_GEOMETRY = (4, 2)
_DEFAULT_WEIGHT_GEOMETRY = (_CENTERED_WEIGHT_TILE, 2, 4)
_W35B_PARTIAL_BUDGET = 64 * 2**20
_WIDE_WEIGHT_WARPS = 8
_MAX_STREAMING_WEIGHT_WAVES = 40
_WEIGHT_PARALLEL_REDUCTION_WORKERS = 512
_SMALL_WEIGHT_PLANE_PIXELS = 256
_LARGE_WEIGHT_PLANE_PIXELS = 262144
_SPECIALIZE_MIN_PIXELS = 16384
_SPECIALIZE_MODE_ALIGNMENT = 16
_CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3


# PyTorch references

def gram_kernel_energy(
    point_real: Tensor,
    point_imag: Tensor,
    spatial_real: Tensor,
    spatial_imag: Tensor,
) -> Tensor:
    """Compute squared row norms via the two Hermitian Gram matrices in FP32."""

    def gram(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        real = real.float().flatten(2)
        imag = imag.float().flatten(2)
        # Explicit FP32 products avoid autocast and reduced-precision GEMMs.
        rr = (real[:, :, None] * real[:, None, :]).sum(dim=-1)
        ii = (imag[:, :, None] * imag[:, None, :]).sum(dim=-1)
        ir = (imag[:, :, None] * real[:, None, :]).sum(dim=-1)
        return rr + ii, ir - ir.transpose(-1, -2)

    pr, pi = gram(point_real, point_imag)
    sr, si = gram(spatial_real, spatial_imag)
    # The factors multiply without conjugation, so both Grams use the same order.
    return (pr * sr - pi * si).sum(dim=(-1, -2))


def real_gram_kernel_energy(point_weight: Tensor, spatial_weight: Tensor) -> Tensor:
    """Return the FP32 squared energy of each of the two rows per mode."""
    modes, rank, _ = point_weight.shape
    with torch.autocast(device_type=point_weight.device.type, enabled=False):
        point = point_weight.float()
        spatial = spatial_weight.float().reshape(modes, 2, rank, -1)
        point_gram = (point[:, :, None] * point[:, None, :]).sum(dim=-1)
        spatial_gram = (spatial[:, :, :, None] * spatial[:, :, None, :]).sum(dim=-1)
        return (point_gram[:, None] * spatial_gram).sum(dim=(-2, -1))


def _grouped_reference_inputs(
    projected: Tensor, real: Tensor, imag: Tensor, *, strict_complex: bool,
) -> tuple[Tensor, Tensor]:
    modes, rank, kernel, _ = real.shape
    # BLAS projection packs (component, rank, mode). Grouped convolution
    # packs (mode, component, rank).
    components = 2 if strict_complex else 1
    features = components * rank
    channels = torch.arange(features * modes, device=projected.device)
    source = channels.remainder(features) * modes + channels.div(features, rounding_mode="floor")
    grouped = projected.movedim(-1, 1).index_select(1, source)
    filters = (
        (torch.cat((real, -imag), dim=1), torch.cat((imag, real), dim=1))
        if strict_complex
        else (real, imag)
    )
    weights = torch.stack(filters, dim=1).reshape(2 * modes, features, kernel, kernel)
    return grouped.float(), weights.float()


def _spatial_reference(
    projected: Tensor, real: Tensor, imag: Tensor, *, strict_complex: bool = True,
) -> Tensor:
    modes, _, kernel, _ = real.shape
    grouped, weights = _grouped_reference_inputs(
        projected, real, imag, strict_complex=strict_complex,
    )
    with torch.autocast(device_type=projected.device.type, enabled=False):
        result = functional.conv2d(grouped, weights, padding=kernel // 2, groups=modes)
    channels = torch.arange(2 * modes, device=projected.device)
    source = channels.remainder(modes) * 2 + channels.div(modes, rounding_mode="floor")
    return result.movedim(1, -1).index_select(-1, source).to(projected.dtype)


def _spatial_reference_backward(
    grad: Tensor, projected: Tensor, real: Tensor, imag: Tensor, *, strict_complex: bool,
) -> tuple[Tensor, Tensor]:
    modes, rank, kernel, _ = real.shape
    grouped, weights = _grouped_reference_inputs(
        projected, real, imag, strict_complex=strict_complex,
    )
    grouped_grad = grad.unflatten(-1, (2, modes)).transpose(-1, -2).flatten(-2)
    padding = kernel // 2
    with torch.autocast(device_type=projected.device.type, enabled=False):
        dp, dw, _ = torch.ops.aten.convolution_backward(
            grouped_grad.movedim(-1, 1).float(), grouped, weights, None,
            (1, 1), (padding, padding), (1, 1), False, (0, 0), modes, (True, True, False),  # noqa: FBT003
        )
    dp = dp.movedim(1, -1).unflatten(-1, (modes, -1)).transpose(-1, -2).flatten(-2)
    dw = dw.unflatten(0, (modes, 2)).transpose(0, 1)
    if strict_complex:
        dr, di = dw.split(rank, dim=2)
        dw = torch.stack((dr[0] + di[1], dr[1] - di[0]))
    return dp.to(projected.dtype).contiguous(), dw.contiguous()


# CUDA kernel templates

_READER_CUDA_TYPES = r"""
#include <cuda_fp16.h>
#if READER_BFLOAT16
#include <cuda_bf16.h>
#endif

__device__ __forceinline__ float as_float(float value) { return value; }
__device__ __forceinline__ float as_float(__half value) { return __half2float(value); }
template <class T> __device__ __forceinline__ T from_float(float value);
template <> __device__ __forceinline__ float from_float(float value) { return value; }
template <> __device__ __forceinline__ __half from_float(float value) {
    return __float2half_rn(value);
}
#if READER_BFLOAT16
__device__ __forceinline__ float as_float(__nv_bfloat16 value) { return __bfloat162float(value); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
}
#endif

using Input = READER_INPUT;
using Weight = READER_WEIGHT;
using Grad = READER_GRAD;
"""


# Each warp streams a row tile with one group of adjacent modes per lane.
# Neighboring columns stay in registers, and rank components share the same CTA.
_READER_CUDA = r"""
template <int N> struct alignas(N * sizeof(float)) ModeValues {
    float value[N];
};

template <int N>
__device__ __forceinline__ ModeValues<N> fmaf(
    ModeValues<N> a, ModeValues<N> b, ModeValues<N> c
) {
    #pragma unroll
    for (int i = 0; i < N; ++i) c.value[i] = ::fmaf(a.value[i], b.value[i], c.value[i]);
    return c;
}

template <int N>
__device__ __forceinline__ ModeValues<N> operator-(ModeValues<N> a) {
    #pragma unroll
    for (int i = 0; i < N; ++i) a.value[i] = -a.value[i];
    return a;
}

template <class T> struct ModePair;
template <> struct ModePair<__half> {
    using Type = __half2;
    __device__ static float2 unpack(Type a) { return __half22float2(a); }
    __device__ static Type pack(float a, float b) { return __floats2half2_rn(a, b); }
};
#if READER_BFLOAT16
template <> struct ModePair<__nv_bfloat16> {
    using Type = __nv_bfloat162;
    __device__ static float2 unpack(Type a) { return __bfloat1622float2(a); }
    __device__ static Type pack(float a, float b) { return __floats2bfloat162_rn(a, b); }
};
#endif

template <int N, class T> struct PackedModes { using Type = T; };
template <class T> struct PackedModes<2, T> { using Type = typename ModePair<T>::Type; };
template <class T> struct PackedModes<4, T> { using Type = uint2; };

template <int N, bool ReadOnly, class T>
__device__ __forceinline__ auto packed_load(const T* ptr) {
    using Packed = typename PackedModes<N, T>::Type;
    const auto* address = reinterpret_cast<const Packed*>(ptr);
    return ReadOnly ? __ldg(address) : *address;
}

template <int N, class T>
__device__ __forceinline__ ModeValues<N> unpack_modes(typename PackedModes<N, T>::Type packed) {
    if constexpr (N == 1) {
        return {{as_float(packed)}};
    } else if constexpr (N == 2) {
        const float2 pair = ModePair<T>::unpack(packed);
        return {{pair.x, pair.y}};
    } else {
        using Pair = typename ModePair<T>::Type;
        const float2 low = ModePair<T>::unpack(reinterpret_cast<const Pair&>(packed.x));
        const float2 high = ModePair<T>::unpack(reinterpret_cast<const Pair&>(packed.y));
        return {{low.x, low.y, high.x, high.y}};
    }
}

template <int N, class T, bool ReadOnly>
__device__ __forceinline__ ModeValues<N> mode_load(const T* ptr) {
    return unpack_modes<N, T>(packed_load<N, ReadOnly>(ptr));
}

template <int N, class T>
__device__ __forceinline__ void mode_store(T* ptr, ModeValues<N> value) {
    if constexpr (N == 1) {
        *ptr = from_float<T>(value.value[0]);
    } else if constexpr (N == 2) {
        using Pair = typename ModePair<T>::Type;
        *reinterpret_cast<Pair*>(ptr) = ModePair<T>::pack(value.value[0], value.value[1]);
    } else {
        using Pair = typename ModePair<T>::Type;
        const Pair low = ModePair<T>::pack(value.value[0], value.value[1]);
        const Pair high = ModePair<T>::pack(value.value[2], value.value[3]);
        *reinterpret_cast<uint2*>(ptr) = make_uint2(
            reinterpret_cast<const unsigned int&>(low),
            reinterpret_cast<const unsigned int&>(high));
    }
}

template <int N, bool ReadOnly, bool Convert, class T>
__device__ __forceinline__ auto window_load(const T* ptr) {
    const auto packed = packed_load<N, ReadOnly>(ptr);
#if READER_W35B_WEIGHT || READER_W35B_SPATIAL
    if constexpr (Convert && (N == 2 || N == 4)) return unpack_modes<N, T>(packed);
    else
#endif
    if constexpr (N == 1 && Convert) return as_float(packed);
    else return packed;
}

template <int N, class T, class Packed>
__device__ __forceinline__ ModeValues<N> window_values(Packed packed) {
    if constexpr (N == 1) return {{as_float(packed)}};
    else return unpack_modes<N, T>(packed);
}

#if READER_W35B_WEIGHT || READER_W35B_SPATIAL
template <int N, class T>
__device__ __forceinline__ ModeValues<N> window_values(ModeValues<N> value) {
    return value;
}
#endif

template <class T, class W, class O, int R, bool Strict, bool Backward, int Vec>
__device__ __forceinline__ void sliding_spatial(
    const T* __restrict__ input,
    const W* __restrict__ real,
    const W* __restrict__ imag,
    O* __restrict__ output,
    int runtime_height, int runtime_width, int runtime_modes
) {
    const int height = READER_HEIGHT ? READER_HEIGHT : runtime_height;
    const int width = READER_WIDTH ? READER_WIDTH : runtime_width;
    const int modes = READER_MODES ? READER_MODES : runtime_modes;
    constexpr int Rows = READER_SPATIAL_ROWS, Columns = READER_COLUMNS;
    constexpr int Warps = READER_SPATIAL_WARPS;
    constexpr int Inputs = Backward ? 2 : R * (Strict ? 2 : 1);
    using V = ModeValues<Vec>;
#if READER_W35B_SPATIAL
    // Forward has a separate input plane per rank. Keep those planes packed
    // across columns and share one unpack across both output components.
    constexpr bool ConvertSpatial = Backward;
    using Window = decltype(window_load<Vec, READER_SPATIAL_READ_ONLY, ConvertSpatial>(input));
    const int mode = (blockIdx.x * 32 + threadIdx.x % 32) * Vec;
    const int xtiles = (width + Columns - 1) / Columns;
    const int tile = blockIdx.y * Warps + threadIdx.x / 32;
    const int y0 = tile / xtiles * Rows;
    const int x0 = tile % xtiles * Columns;
    const long long base = static_cast<long long>(blockIdx.z) * height * width;
    if (mode >= modes || y0 >= height) return;
#else
    constexpr bool ConvertSpatial = true;
    using Window = decltype(window_load<Vec, READER_SPATIAL_READ_ONLY, ConvertSpatial>(input));
    // Small real filters fit alongside the low-precision window in registers.
    // Reuse them across columns without reloading taps from shared memory.
    constexpr bool RegisterFilters = Vec == 1 && !Strict && (READER_REGISTER_FILTERS < 0
        ? ((R <= 2 && sizeof(Input) == 2) || (Rows == 4 && Warps == 2))
        : READER_REGISTER_FILTERS != 0);
    constexpr bool KeepReal = RegisterFilters ||
        (Vec == 1 && Backward &&
         (READER_KEEP_REAL < 0 ? READER_WIDTH >= 28 : READER_KEEP_REAL != 0));
    const int lane = threadIdx.x % 32;
    const int mode = (blockIdx.x * 32 + lane) * Vec;
    const int xtiles = (width + Columns * Warps - 1) / (Columns * Warps);
    const int y0 = blockIdx.y / xtiles * Rows;
    const int x0 = (blockIdx.y % xtiles * Warps + threadIdx.x / 32) * Columns;
    const long long base = static_cast<long long>(blockIdx.z) * height * width;
    __shared__ V filters[2][R][3][3][32];
    V resident[RegisterFilters ? 2 : 1][R][3][3];
    if constexpr (!RegisterFilters) {
        // Each tap is shared across warps and stored with its modes adjacent.
        for (int field = threadIdx.x / 32; field < 2 * R * 9; field += Warps) {
            const int component = field / (R * 9), r = field / 9 % R, tap = field % 9;
            if (mode < modes) {
                V values;
                #pragma unroll
                for (int v = 0; v < Vec; ++v) {
                    const int index = ((mode + v) * R + r) * 9 + (Backward ? 8 - tap : tap);
                    values.value[v] = as_float((component == 0 ? real : imag)[index]);
                }
                filters[component][r][tap / 3][tap % 3][lane] = values;
            }
        }
        __syncthreads();
    }
    if (mode >= modes || x0 >= width) return;
    if constexpr (KeepReal) {
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky)
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    if constexpr (RegisterFilters) {
                        const int tap = ky * 3 + kx;
                        const int index = (mode * R + r) * 9 + (Backward ? 8 - tap : tap);
                        resident[0][r][ky][kx] = {{as_float(real[index])}};
                        resident[1][r][ky][kx] = {{as_float(imag[index])}};
                    } else {
                        resident[0][r][ky][kx] = filters[0][r][ky][kx][lane];
                    }
                }
    }
#endif
    Window left[Rows + 2][Inputs] = {}, middle[Rows + 2][Inputs] = {};
    #pragma unroll
    for (int y = 0; y < Rows + 2; ++y) {
        const int py = y0 + y - 1;
        if (py < 0 || py >= height) continue;
        #pragma unroll
        for (int i = 0; i < Inputs; ++i) {
            const long long index = ((base + py * width + x0) * Inputs + i) * modes + mode;
            middle[y][i] = window_load<Vec, READER_SPATIAL_READ_ONLY, ConvertSpatial>(
                input + index);
            if (x0 > 0)
                left[y][i] = window_load<Vec, READER_SPATIAL_READ_ONLY, ConvertSpatial>(
                    input + index - Inputs * modes);
        }
    }
    const long long input_step = static_cast<long long>(Inputs) * modes;
    long long next = (base + x0 + 1) * input_step + mode;
    constexpr int Outputs = Backward ? R * (Strict ? 2 : 1) : 2;
    const long long output_step = static_cast<long long>(Outputs) * modes;
    long long destination = (base + x0) * output_step + mode;
    for (int x = x0; x < x0 + Columns && x < width;
         ++x, next += input_step, destination += output_step) {
        Window right[Rows + 2][Inputs] = {};
        #pragma unroll
        for (int y = 0; y < Rows + 2; ++y) {
            const int py = y0 + y - 1;
            if (py < 0 || py >= height || x + 1 == width) continue;
            #pragma unroll
            for (int i = 0; i < Inputs; ++i)
                right[y][i] = window_load<Vec, READER_SPATIAL_READ_ONLY, ConvertSpatial>(
                    input + (READER_UPDATED_SPATIAL && Vec == 1 && !Backward
                        ? ((base + py * width + x + 1) * Inputs + i) * modes + mode
                        : next + py * width * input_step + i * modes));
        }
        V ar[Rows] = {}, ai[Rows] = {};
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const int i = Backward ? 0 : r;
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                constexpr int Left = 0, Middle = 1, Right = 2;
#if READER_W35B_SPATIAL
                V f[2][3];
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const int tap = ky * 3 + kx;
                    #pragma unroll
                    for (int v = 0; v < Vec; ++v) {
                        const int index = ((mode + v) * R + r) * 9 + (Backward ? 8 - tap : tap);
                        f[0][kx].value[v] = as_float(real[index]);
                        f[1][kx].value[v] = as_float(imag[index]);
                    }
                }
#else
                const V f[2][3] = {
                    {KeepReal ? resident[0][r][ky][0] : filters[0][r][ky][0][lane],
                     KeepReal ? resident[0][r][ky][1] : filters[0][r][ky][1][lane],
                     KeepReal ? resident[0][r][ky][2] : filters[0][r][ky][2][lane]},
                    {RegisterFilters ? resident[1][r][ky][0] : filters[1][r][ky][0][lane],
                     RegisterFilters ? resident[1][r][ky][1] : filters[1][r][ky][1][lane],
                     RegisterFilters ? resident[1][r][ky][2] : filters[1][r][ky][2][lane]},
                };
#endif
                #pragma unroll
                for (int y = 0; y < Rows; ++y) {
                    ar[y] = fmaf(window_values<Vec, T>(left[y + ky][i]), f[0][Left], ar[y]);
                    ar[y] = fmaf(window_values<Vec, T>(middle[y + ky][i]), f[0][Middle], ar[y]);
                    ar[y] = fmaf(window_values<Vec, T>(right[y + ky][i]), f[0][Right], ar[y]);
                    if (Backward) {
                        ar[y] = fmaf(window_values<Vec, T>(left[y + ky][1]), f[1][Left], ar[y]);
                        ar[y] = fmaf(window_values<Vec, T>(middle[y + ky][1]), f[1][Middle], ar[y]);
                        ar[y] = fmaf(window_values<Vec, T>(right[y + ky][1]), f[1][Right], ar[y]);
                        if (Strict) {
                            ai[y] = fmaf(window_values<Vec, T>(left[y + ky][1]), f[0][Left], ai[y]);
                            ai[y] = fmaf(window_values<Vec, T>(middle[y + ky][1]),
                                f[0][Middle], ai[y]);
                            ai[y] = fmaf(window_values<Vec, T>(right[y + ky][1]),
                                f[0][Right], ai[y]);
                            ai[y] = fmaf(-window_values<Vec, T>(left[y + ky][0]),
                                f[1][Left], ai[y]);
                            ai[y] = fmaf(-window_values<Vec, T>(middle[y + ky][0]),
                                f[1][Middle], ai[y]);
                            ai[y] = fmaf(-window_values<Vec, T>(right[y + ky][0]),
                                f[1][Right], ai[y]);
                        }
                    } else {
                        if (Strict) {
                            ar[y] = fmaf(-window_values<Vec, T>(left[y + ky][R + r]),
                                f[1][Left], ar[y]);
                            ar[y] = fmaf(-window_values<Vec, T>(middle[y + ky][R + r]),
                                f[1][Middle], ar[y]);
                            ar[y] = fmaf(-window_values<Vec, T>(right[y + ky][R + r]),
                                f[1][Right], ar[y]);
                            ai[y] = fmaf(window_values<Vec, T>(left[y + ky][R + r]),
                                f[0][Left], ai[y]);
                            ai[y] = fmaf(window_values<Vec, T>(middle[y + ky][R + r]),
                                f[0][Middle], ai[y]);
                            ai[y] = fmaf(window_values<Vec, T>(right[y + ky][R + r]),
                                f[0][Right], ai[y]);
                        }
                        ai[y] = fmaf(window_values<Vec, T>(left[y + ky][i]), f[1][Left], ai[y]);
                        ai[y] = fmaf(window_values<Vec, T>(middle[y + ky][i]), f[1][Middle], ai[y]);
                        ai[y] = fmaf(window_values<Vec, T>(right[y + ky][i]), f[1][Right], ai[y]);
                    }
                }
            }
            if (Backward) {
                #pragma unroll
                for (int y = 0; y < Rows; ++y) {
                    if (y0 + y >= height) continue;
                    O* out = output + (destination + (y0 + y) * width * output_step + r * modes);
                    mode_store<Vec>(out, ar[y]);
                    if (Strict) mode_store<Vec>(out + R * modes, ai[y]);
                    ar[y] = ai[y] = {};
                }
            }
        }
        if (!Backward) {
            #pragma unroll
            for (int y = 0; y < Rows; ++y) {
                if (y0 + y >= height) continue;
                O* out = output + (READER_UPDATED_SPATIAL && Vec == 1
                    ? (base + (y0 + y) * width + x) * Outputs * modes + mode
                    : destination + (y0 + y) * width * output_step);
                mode_store<Vec>(out, ar[y]);
                mode_store<Vec>(out + modes, ai[y]);
            }
        }
        #pragma unroll
        for (int y = 0; y < Rows + 2; ++y) {
            #pragma unroll
            for (int i = 0; i < Inputs; ++i) {
                left[y][i] = middle[y][i];
                middle[y][i] = right[y][i];
            }
        }
    }
}

template <class T, class W, class O, int R, int K, bool Strict, bool Backward>
__device__ __forceinline__ void direct_spatial(
    const T* __restrict__ input, const W* __restrict__ real, const W* __restrict__ imag,
    O* __restrict__ output, int height, int width, int modes, int runtime_rank
) {
    const int rank = R ? R : runtime_rank;
    constexpr int Components = Strict ? 2 : 1;
    const int mode = blockIdx.x * 32 + threadIdx.x % 32;
    const int xtiles = (width + READER_COLUMNS * 4 - 1) / (READER_COLUMNS * 4);
    const int y0 = blockIdx.y / xtiles * 2;
    const int x0 = (blockIdx.y % xtiles * 4 + threadIdx.x / 32) * READER_COLUMNS;
    if (mode >= modes) return;
    const long long base = static_cast<long long>(blockIdx.z) * height * width;
    for (int y = y0; y < y0 + 2 && y < height; ++y) {
        for (int x = x0; x < x0 + READER_COLUMNS && x < width; ++x) {
            const long long pixel = base + y * width + x;
            float ar = 0, ai = 0;
            #pragma unroll (R ? R : 1)
            for (int r = 0; r < rank; ++r) {
                if (Backward) ar = ai = 0;
                for (int ky = 0; ky < K; ++ky) {
                    const int py = y + ky - K / 2;
                    if (py < 0 || py >= height) continue;
                    for (int kx = 0; kx < K; ++kx) {
                        const int px = x + kx - K / 2;
                        if (px < 0 || px >= width) continue;
                        const long long neighbor = base + py * width + px;
                        const int tap = ky * K + kx;
                        const long long w = (static_cast<long long>(mode) * rank + r) * K * K
                            + (Backward ? K * K - 1 - tap : tap);
                        const float wr = as_float(real[w]), wi = as_float(imag[w]);
                        if (Backward) {
                            const float gr = as_float(input[neighbor * 2 * modes + mode]);
                            const float gi = as_float(input[neighbor * 2 * modes + modes + mode]);
                            ar = fmaf(gr, wr, fmaf(gi, wi, ar));
                            if (Strict) ai = fmaf(gi, wr, fmaf(-gr, wi, ai));
                        } else {
                            const long long i = (neighbor * Components * rank + r) * modes + mode;
                            const float pr = as_float(input[i]);
                            ar = fmaf(pr, wr, ar);
                            ai = fmaf(pr, wi, ai);
                            if (Strict) {
                                const float pi = as_float(input[i + rank * modes]);
                                ar = fmaf(-pi, wi, ar);
                                ai = fmaf(pi, wr, ai);
                            }
                        }
                    }
                }
                if (Backward) {
                    const long long i = (pixel * Components * rank + r) * modes + mode;
                    output[i] = from_float<O>(ar);
                    if (Strict) output[i + rank * modes] = from_float<O>(ai);
                }
            }
            if (!Backward) {
                output[pixel * 2 * modes + mode] = from_float<O>(ar);
                output[pixel * 2 * modes + modes + mode] = from_float<O>(ai);
            }
        }
    }
}

template <class T, class G, int R, int K, bool Strict>
__device__ __forceinline__ void weight_gradient(
    const T* __restrict__ projected, const G* __restrict__ grad, float* __restrict__ partial,
    int runtime_height, int runtime_width, int runtime_modes, int runtime_batch, int runtime_rank
) {
    const int height = READER_HEIGHT ? READER_HEIGHT : runtime_height;
    const int width = READER_WIDTH ? READER_WIDTH : runtime_width;
    const int modes = READER_MODES ? READER_MODES : runtime_modes;
    const int batch = READER_BATCH ? READER_BATCH : runtime_batch;
    const int rank = R ? R : runtime_rank;
    constexpr int Warps = 4, Components = Strict ? 2 : 1;
    constexpr int RankTile = Strict || !R ? 2 : R;
    constexpr int Taps = K == 3 ? 3 : 1;
    constexpr int Fields = RankTile * Taps * 2;
    // Neighboring tap tiles consume the same gradient rows and overlapping
    // projected rows. Schedule them together so these reads can share L2.
    constexpr int TapTiles = Taps == 3 ? K : K * K;
    const int tap_tile = blockIdx.x % TapTiles;
    const int ky = tap_tile % K;
    const int kx0 = Taps == 3 ? 0 : tap_tile / K;
    const int r0 = blockIdx.z * RankTile;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int mode = blockIdx.x / TapTiles * 32 + lane;
    float sr[RankTile][Taps] = {}, si[RankTile][Taps] = {};
    __shared__ float sums[Warps][Fields][32];

    if (mode < modes) {
        for (int row = blockIdx.y * Warps + warp; row < batch * height;
             row += gridDim.y * Warps) {
            const int y = row % height, py = y + ky - K / 2;
            if (py < 0 || py >= height) continue;
            const long long base = static_cast<long long>(row / height) * height * width;
            const T* p = projected + ((base + py * width) * Components * rank + r0) * modes + mode;
            const G* g = grad + (base + y * width) * 2 * modes + mode;
            float left[Components][RankTile] = {}, middle[Components][RankTile] = {};
            if (Taps == 3) {
                #pragma unroll
                for (int c = 0; c < Components; ++c)
                    #pragma unroll
                    for (int r = 0; r < RankTile; ++r)
                        if (r0 + r < rank) middle[c][r] = as_float(p[(c * rank + r) * modes]);
            }
            for (int x = 0; x < width; ++x) {
                const float gr = as_float(g[x * 2 * modes]);
                const float gi = as_float(g[x * 2 * modes + modes]);
                float right[Components][RankTile] = {};
                const int px = x + (Taps == 3 ? 1 : kx0 - K / 2);
                if (px >= 0 && px < width) {
                    #pragma unroll
                    for (int c = 0; c < Components; ++c)
                        #pragma unroll
                        for (int r = 0; r < RankTile; ++r)
                            if (r0 + r < rank)
                                right[c][r] = as_float(
                                    p[((px * Components + c) * rank + r) * modes]);
                }
                #pragma unroll
                for (int r = 0; r < RankTile; ++r) {
                    #pragma unroll
                    for (int kx = 0; kx < Taps; ++kx) {
                        const float pr = kx == Taps - 1 ? right[0][r]
                            : (kx == 0 ? left[0][r] : middle[0][r]);
                        sr[r][kx] = fmaf(pr, gr, sr[r][kx]);
                        si[r][kx] = fmaf(pr, gi, si[r][kx]);
                        if (Strict) {
                            const float pi = kx == Taps - 1 ? right[1][r]
                                : (kx == 0 ? left[1][r] : middle[1][r]);
                            sr[r][kx] = fmaf(pi, gi, sr[r][kx]);
                            si[r][kx] = fmaf(-pi, gr, si[r][kx]);
                        }
                    }
                }
                if (Taps == 3) {
                    #pragma unroll
                    for (int c = 0; c < Components; ++c)
                        #pragma unroll
                        for (int r = 0; r < RankTile; ++r) {
                            left[c][r] = middle[c][r];
                            middle[c][r] = right[c][r];
                        }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < RankTile; ++r)
        #pragma unroll
        for (int kx = 0; kx < Taps; ++kx) {
            const int field = r * Taps + kx;
            sums[warp][field][lane] = sr[r][kx];
            sums[warp][field + Fields / 2][lane] = si[r][kx];
        }
    __syncthreads();

    if (mode >= modes) return;
    // Each warp writes a disjoint set of fields to the same worker row.
    for (int field = warp; field < Fields; field += Warps) {
        const int component = field / (Fields / 2);
        const int r = r0 + (field % (Fields / 2)) / Taps;
        const int kx = kx0 + field % Taps;
        if (r >= rank) continue;
        const long long offset = ((((static_cast<long long>(blockIdx.y) * 2 + component)
            * rank + r) * K + ky) * K + kx) * modes + mode;
        partial[offset] = (sums[0][field][lane] + sums[1][field][lane])
                        + (sums[2][field][lane] + sums[3][field][lane]);
    }
}

template <class T, class G, int R, int Vec = READER_WEIGHT_VEC>
__device__ __forceinline__ void real_weight_gradient_3x3(
    const T* __restrict__ projected, const G* __restrict__ grad, float* __restrict__ partial,
    int runtime_height, int runtime_width, int runtime_modes, int runtime_batch
) {
    const int height = READER_HEIGHT ? READER_HEIGHT : runtime_height;
    const int width = READER_WIDTH ? READER_WIDTH : runtime_width;
    const int modes = READER_MODES ? READER_MODES : runtime_modes;
    const int batch = READER_BATCH ? READER_BATCH : runtime_batch;
    constexpr int Warps = READER_WEIGHT_WARPS, Rows = READER_WEIGHT_ROWS;
    constexpr bool ReadOnly = Vec > 1 ? READER_WEIGHT_READ_ONLY
        : (READER_WIDTH >= 28 || Warps == 8);
#if READER_W35B_WEIGHT
    // Grad components are in separate mode planes. Convert each plane when
    // the column enters the window and carry its values across all taps.
    constexpr bool ConvertWindow = true;
#else
    constexpr bool ConvertWindow = R <= 2 && sizeof(G) == 2 && READER_WIDTH > 2 * READER_COLUMNS;
#endif
    using V = ModeValues<Vec>;
    using Window = decltype(window_load<Vec, ReadOnly, ConvertWindow>(grad));
    constexpr int RankTile = R < READER_WEIGHT_TILE ? R : READER_WEIGHT_TILE;
    constexpr int RankTiles = (R + RankTile - 1) / RankTile;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int mode = (blockIdx.x / RankTiles * 32 + lane) * Vec;
    const int r0 = blockIdx.x % RankTiles * RankTile;
    const int ytiles = (height + Rows - 1) / Rows;
    V sums_r[3][3][RankTile] = {}, sums_i[3][3][RankTile] = {};
    constexpr int ReduceRanks = 1;
    constexpr int Fields = ReduceRanks * 18;
    __shared__ float warp_sums[Warps][Fields][Vec][32];
    if (mode < modes) {
        for (int row = blockIdx.y * Warps + warp; row < batch * ytiles;
             row += gridDim.y * Warps) {
            const int y = row % ytiles * Rows;
            const long long base = static_cast<long long>(row / ytiles) * height * width;
            const T* p = projected + ((base + y * width) * R + r0) * modes + mode;
            // For long windows, convert on load so every overlapping tap can reuse FP32.
            Window left[Rows + 2][2] = {}, middle[Rows + 2][2] = {};
            #pragma unroll
            for (int i = 0; i < Rows + 2; ++i) {
                const int gy = y + i - 1;
                if (gy >= 0 && gy < height) {
                    const G* g = grad + (base + gy * width) * 2 * modes + mode;
                    middle[i][0] = window_load<Vec, ReadOnly, ConvertWindow>(g);
                    middle[i][1] = window_load<Vec, ReadOnly, ConvertWindow>(g + modes);
                }
            }
            for (int x = 0; x < width; ++x) {
                Window right[Rows + 2][2] = {};
                #pragma unroll
                for (int i = 0; i < Rows + 2; ++i) {
                    const int gy = y + i - 1;
                    if (gy >= 0 && gy < height && x + 1 < width) {
                        const G* g = grad + (base + gy * width + x + 1) * 2 * modes + mode;
                        right[i][0] = window_load<Vec, ReadOnly, ConvertWindow>(g);
                        right[i][1] = window_load<Vec, ReadOnly, ConvertWindow>(g + modes);
                    }
                }
                #pragma unroll
                for (int j = 0; j < Rows; ++j) {
                    if (y + j >= height) continue;
                    V v[RankTile];
                    #pragma unroll
                    for (int r = 0; r < RankTile; ++r) {
                        const T* current = p + ((j * width + x) * R + r) * modes;
                        v[r] = r0 + r < R ? mode_load<Vec, T, ReadOnly>(current) : V{};
                    }
                    #pragma unroll
                    for (int ky = 0; ky < 3; ++ky) {
                        const int i = j + 2 - ky;
                        const V l[] = {window_values<Vec, G>(left[i][0]),
                            window_values<Vec, G>(left[i][1])};
                        const V m[] = {window_values<Vec, G>(middle[i][0]),
                            window_values<Vec, G>(middle[i][1])};
                        const V g[] = {window_values<Vec, G>(right[i][0]),
                            window_values<Vec, G>(right[i][1])};
                        #pragma unroll
                        for (int r = 0; r < RankTile; ++r) {
                            sums_r[ky][0][r] = fmaf(v[r], g[0], sums_r[ky][0][r]);
                            sums_r[ky][1][r] = fmaf(v[r], m[0], sums_r[ky][1][r]);
                            sums_r[ky][2][r] = fmaf(v[r], l[0], sums_r[ky][2][r]);
                            sums_i[ky][0][r] = fmaf(v[r], g[1], sums_i[ky][0][r]);
                            sums_i[ky][1][r] = fmaf(v[r], m[1], sums_i[ky][1][r]);
                            sums_i[ky][2][r] = fmaf(v[r], l[1], sums_i[ky][2][r]);
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < Rows + 2; ++i) {
                    left[i][0] = middle[i][0];
                    left[i][1] = middle[i][1];
                    middle[i][0] = right[i][0];
                    middle[i][1] = right[i][1];
                }
            }
        }
    }
    // Stage a subset of ranks at a time so reduction scratch does not
    // determine the occupancy of the much longer spatial loop.
    #pragma unroll
    for (int tile = 0; tile < RankTile; tile += ReduceRanks) {
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky)
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx)
                #pragma unroll
                for (int r = 0; r < ReduceRanks; ++r) {
                    const int field = r * 9 + ky * 3 + kx;
                    #pragma unroll
                    for (int v = 0; v < Vec; ++v) {
                        warp_sums[warp][field][v][lane] = sums_r[ky][kx][tile + r].value[v];
                        warp_sums[warp][field + Fields / 2][v][lane] =
                            sums_i[ky][kx][tile + r].value[v];
                    }
                }
        __syncthreads();
        if (mode < modes) {
            for (int field = warp; field < Fields; field += Warps) {
                const int component = field / (Fields / 2);
                const int tap = field % 9, rank = tile + r0 + field % (Fields / 2) / 9;
                if (rank >= R) continue;
                const long long index = (
                    (static_cast<long long>(blockIdx.y) * 2 + component) * R * 9 + rank * 9 + tap
                ) * modes;
                #pragma unroll
                for (int v = 0; v < Vec; ++v) {
                    float sum = warp_sums[0][field][v][lane] + warp_sums[1][field][v][lane];
                    if constexpr (Warps >= 4)
                        sum += warp_sums[2][field][v][lane] + warp_sums[3][field][v][lane];
                    if constexpr (Warps == 8)
                        sum += (warp_sums[4][field][v][lane] + warp_sums[5][field][v][lane])
                             + (warp_sums[6][field][v][lane] + warp_sums[7][field][v][lane]);
                    partial[index + mode + v] = sum;
                }
            }
        }
        if (tile + ReduceRanks < RankTile) __syncthreads();
    }
}


constexpr int Rank = READER_RANK, Kernel = READER_KERNEL;
constexpr bool Strict = READER_STRICT;

extern "C" __global__ void reader_forward(
    const Input* input, const Weight* real, const Weight* imag, Input* output,
    int h, int w, int m, int rank
) {
    if constexpr (Rank && Kernel == 3)
        sliding_spatial<Input, Weight, Input, Rank, Strict, false, READER_SPATIAL_VEC>(
            input, real, imag, output, h, w, m);
    else
        direct_spatial<Input, Weight, Input, Rank, Kernel, Strict, false>(
            input, real, imag, output, h, w, m, rank);
}

extern "C" __global__ void reader_dinput(
    const Grad* input, const Weight* real, const Weight* imag, Input* output,
    int h, int w, int m, int rank
) {
    if constexpr (Rank && Kernel == 3)
        sliding_spatial<Grad, Weight, Input, Rank, Strict, true, READER_SPATIAL_VEC>(
            input, real, imag, output, h, w, m);
    else
        direct_spatial<Grad, Weight, Input, Rank, Kernel, Strict, true>(
            input, real, imag, output, h, w, m, rank);
}

extern "C" __global__
#if READER_W35B_WEIGHT && READER_WEIGHT_VEC == 2
__maxnreg__(160)
#endif
void reader_dweight(
    const Input* projected, const Grad* grad, float* partial,
    int h, int w, int m, int batch, int rank
) {
    if constexpr (Rank && Kernel == 3 && !Strict)
        real_weight_gradient_3x3<Input, Grad, Rank>(projected, grad, partial, h, w, m, batch);
    else
        weight_gradient<Input, Grad, Rank, Kernel, Strict>(
            projected, grad, partial, h, w, m, batch, rank);
}

extern "C" __global__ void reader_dweight_reduce(
    const float* __restrict__ partial, float* __restrict__ output,
    int modes, int rank, int workers
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int taps = Kernel * Kernel;
    const int fields = 2 * rank * taps;
    if (index >= fields * modes) return;
    const int mode = index % modes, field = index / modes;
    float sum = 0;
    for (int worker = 0; worker < workers; ++worker)
        sum += partial[(static_cast<long long>(worker) * fields + field) * modes + mode];
    const int component = field / (rank * taps);
    output[((component * modes + mode) * rank * taps) + field % (rank * taps)] = sum;
}

extern "C" __global__ void reader_dweight_reduce_parallel(
    const float* __restrict__ partial, float* __restrict__ output,
    int modes, int rank, int workers
) {
    constexpr int taps = Kernel * Kernel;
    constexpr int Warps = 4;
    __shared__ float sums[Warps][32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int index = blockIdx.x * 32 + lane;
    const int stride = 2 * rank * taps * modes;
    float sum = 0;
    if (index < stride) {
        for (int worker = warp; worker < workers; worker += Warps)
            sum += partial[static_cast<long long>(worker) * stride + index];
    }
    sums[warp][lane] = sum;
    __syncthreads();
    if (warp != 0 || index >= stride) return;
    sum = (sums[0][lane] + sums[1][lane]) + (sums[2][lane] + sums[3][lane]);
    const int mode = index % modes, field = index / modes;
    const int component = field / (rank * taps);
    output[((component * modes + mode) * rank * taps) + field % (rank * taps)] = sum;
}


"""


_GRAM_NORMALIZE_CUDA = r"""
struct PointStrides {
    long long mode, rank, channel;
};

struct FilterStrides {
    long long outer, inner, rank, row, col;

    __device__ long long field(int rank_index, int tap) const {
        return rank_index * rank + (tap / READER_KERNEL) * row + (tap % READER_KERNEL) * col;
    }
};

// One block owns both components of one mode, including its shared point
// gradient. Each warp computes a pair of ranks in the Gram matrices.
template <class P, class S, class N, bool Backward>
__device__ void gram_normalize(
    const P* point, const S* spatial, const N* gradient,
    N* output, P* dpoint, S* dspatial,
    int modes, float eps, PointStrides ps, FilterStrides ss, FilterStrides gs
) {
    constexpr int rank = READER_RANK, channels = READER_HEIGHT;
    constexpr int taps = READER_KERNEL * READER_KERNEL;
    constexpr int pairs = rank * rank, fields = rank * taps;
    __shared__ float pg[pairs], sg[2][pairs];
    __shared__ float inverse[2], energy_gradient[2], scratch[4];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int mode = blockIdx.x;
    point += mode * ps.mode;
    spatial += mode * ss.outer;
    for (int pair = warp; pair < pairs; pair += 4) {
        const int r = pair / rank, s = pair % rank;
        float p = 0, a = 0, b = 0;
        for (int c = lane; c < channels; c += 32)
            p += __fmul_rn(as_float(point[r * ps.rank + c * ps.channel]),
                           as_float(point[s * ps.rank + c * ps.channel]));
        for (int t = lane; t < taps; t += 32) {
            a += __fmul_rn(as_float(spatial[ss.field(r, t)]), as_float(spatial[ss.field(s, t)]));
            b += __fmul_rn(as_float(spatial[ss.inner + ss.field(r, t)]),
                           as_float(spatial[ss.inner + ss.field(s, t)]));
        }
        for (int offset = 16; offset; offset /= 2) {
            p += __shfl_down_sync(0xffffffff, p, offset);
            a += __shfl_down_sync(0xffffffff, a, offset);
            b += __shfl_down_sync(0xffffffff, b, offset);
        }
        if (lane == 0) {
            pg[pair] = p;
            sg[0][pair] = a;
            sg[1][pair] = b;
        }
    }
    __syncthreads();
    if (warp < 2) {
        float energy = 0;
        for (int pair = lane; pair < pairs; pair += 32)
            energy += __fmul_rn(pg[pair], sg[warp][pair]);
        for (int offset = 16; offset; offset /= 2)
            energy += __shfl_down_sync(0xffffffff, energy, offset);
        if (lane == 0) {
            const float inv = rsqrtf(energy < eps ? eps : energy);
            inverse[warp] = inv;
            energy_gradient[warp] = energy >= eps ? -0.5f * inv * inv * inv : 0;
        }
    }
    __syncthreads();
    if constexpr (!Backward) {
        for (int i = tid; i < 2 * fields; i += blockDim.x) {
            const int component = i / fields;
            const int field = i % fields;
            const auto offset = component * ss.inner + ss.field(field / taps, field % taps);
            output[(static_cast<long long>(component) * modes + mode) * fields + field] =
                from_float<N>(as_float(spatial[offset]) * inverse[component]);
        }
    } else {
        dpoint += static_cast<long long>(mode) * rank * channels;
        dspatial += static_cast<long long>(mode) * 2 * fields;
        for (int component = 0; component < 2; ++component) {
            const N* g = gradient + component * gs.outer + mode * gs.inner;
            const S* plane = spatial + component * ss.inner;
            float dot = 0;
            for (int i = tid; i < fields; i += blockDim.x) {
                const int r = i / taps, t = i % taps;
                dot += __fmul_rn(as_float(g[gs.field(r, t)]), as_float(plane[ss.field(r, t)]));
            }
            for (int offset = 16; offset; offset /= 2)
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            if (lane == 0) scratch[warp] = dot;
            __syncthreads();
            if (tid == 0)
                energy_gradient[component] *= scratch[0] + scratch[1] + scratch[2] + scratch[3];
            __syncthreads();
            for (int i = tid; i < fields; i += blockDim.x) {
                const int r = i / taps, t = i % taps;
                float sum = 0;
                for (int s = 0; s < rank; ++s)
                    sum += __fmul_rn(energy_gradient[component] * pg[r * rank + s],
                                     as_float(plane[ss.field(s, t)]));
                dspatial[component * fields + i] = from_float<S>(
                    __fmul_rn(as_float(g[gs.field(r, t)]), inverse[component]) + 2 * sum);
            }
        }
        __syncthreads();
        if (tid < pairs)
            pg[tid] = energy_gradient[0] * sg[0][tid] + energy_gradient[1] * sg[1][tid];
        __syncthreads();
        for (int i = tid; i < rank * channels; i += blockDim.x) {
            const int r = i / channels, c = i % channels;
            float sum = 0;
            #pragma unroll
            for (int s = 0; s < rank; ++s)
                sum += pg[r * rank + s] * as_float(point[s * ps.rank + c * ps.channel]);
            dpoint[i] = from_float<P>(2 * sum);
        }
    }
}

extern "C" __global__ void reader_gram_normalize(
    const Input* point, const Weight* spatial, Grad* output,
    int modes, float eps, PointStrides ps, FilterStrides ss
) {
    gram_normalize<Input, Weight, Grad, false>(
        point, spatial, nullptr, output, nullptr, nullptr, modes, eps, ps, ss, {});
}

extern "C" __global__ void reader_gram_normalize_backward(
    const Input* point, const Weight* spatial, const Grad* gradient, Input* dp, Weight* ds,
    int modes, float eps, PointStrides ps, FilterStrides ss, FilterStrides gs
) {
    gram_normalize<Input, Weight, Grad, true>(
        point, spatial, gradient, nullptr, dp, ds, modes, eps, ps, ss, gs);
}
"""


# NVRTC compilation

@cache
def _reader_cuda_module(
    device_index: int,
    shape: tuple[int, int, int, int],
    columns: int,
    rank: int,
    kernel: int,
    types: tuple[str, str, str],
    *,
    strict: bool,
    normalize: bool = False,
    spatial_geometry: tuple[int, int] = _DEFAULT_SPATIAL_GEOMETRY,
    weight_geometry: tuple[int, int, int] | None = None,
    register_filters: int = -1,
    keep_real: int = -1,
    weight_vec: int = 1,
    spatial_vec: int = 1,
    spatial_read_only: bool = False,
    weight_read_only: bool = True,
    updated_spatial: bool = False,
    w35b_weight: bool = False,
    w35b_spatial: bool = False,
) -> _CudaModule:
    weight_tile, weight_rows, weight_warps = weight_geometry or _DEFAULT_WEIGHT_GEOMETRY
    macros = {
        **dict(zip(("BATCH", "HEIGHT", "WIDTH", "MODES"), shape, strict=True)),
        "COLUMNS": columns, "RANK": rank, "KERNEL": kernel, "STRICT": int(strict),
        "UPDATED_SPATIAL": int(updated_spatial),
        "SPATIAL_ROWS": spatial_geometry[0], "SPATIAL_WARPS": spatial_geometry[1],
        "REGISTER_FILTERS": register_filters, "KEEP_REAL": keep_real,
        "BFLOAT16": int("__nv_bfloat16" in types),
        "WEIGHT_VEC": weight_vec, "SPATIAL_VEC": spatial_vec,
        "SPATIAL_READ_ONLY": int(spatial_read_only), "WEIGHT_READ_ONLY": int(weight_read_only),
        "WEIGHT_TILE": weight_tile, "WEIGHT_ROWS": weight_rows, "WEIGHT_WARPS": weight_warps,
        **dict(zip(("INPUT", "WEIGHT", "GRAD"), types, strict=True)),
    }
    if w35b_weight:
        macros["W35B_WEIGHT"] = 1
    if w35b_spatial:
        macros["W35B_SPATIAL"] = 1
    header = "".join(f"#define READER_{key} {value}\n" for key, value in macros.items())
    source, operations = (
        (_GRAM_NORMALIZE_CUDA, ("gram_normalize", "gram_normalize_backward"))
        if normalize else (
            _READER_CUDA,
            ("forward", "dinput", "dweight", "dweight_reduce", "dweight_reduce_parallel"),
        )
    )
    # Reuse the CUDA launch interface with Reader-specific PTX and entry points.
    nvrtc = _nvrtc()
    program = ctypes.c_void_p()
    _check_nvrtc(
        nvrtc,
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(program), (header + _READER_CUDA_TYPES + source).encode(),
            b"factorized_reader.cu", 0, None, None,
        ),
    )
    major, minor = torch.cuda.get_device_capability(device_index)
    flags = [
        f"--gpu-architecture=compute_{major}{minor}".encode(), b"--std=c++17",
        *(f"--include-path={path}".encode() for path in _include_dirs()),
    ]
    options = (ctypes.c_char_p * len(flags))(*flags)
    try:
        _check_nvrtc(nvrtc, nvrtc.nvrtcCompileProgram(program, len(flags), options), program)
        size = ctypes.c_size_t()
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), program)
        image = ctypes.create_string_buffer(size.value)
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTX(program, image), program)
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    module = _CudaModule(image, device_index)
    for operation in operations:
        module.load_function(f"reader_{operation}", 0)
    return module


# Geometry planning

def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _reader_columns(width: int) -> int:
    return _COMPACT_COLUMNS if width % _COMPACT_COLUMNS == 0 else _DEFAULT_COLUMNS


def _reader_grid_fits(shape: torch.Size, rank: int) -> bool:
    batch, height, width, _ = shape
    rows, warps = _DEFAULT_SPATIAL_GEOMETRY
    tiles = _ceildiv(height, rows) * _ceildiv(width, warps * _reader_columns(width))
    return (
        0 < batch <= _MAX_CUDA_GRID_DIM
        and 0 < tiles <= _MAX_CUDA_GRID_DIM
        and _ceildiv(rank, 2) <= _MAX_CUDA_GRID_DIM
    )


def _use_reader_cuda(
    projected: Tensor, real: Tensor, imag: Tensor, grad: Tensor | None = None,
) -> bool:
    if not projected.is_cuda:
        return False
    _, rank, kernel, _ = real.shape
    grad_dtype = grad.dtype if grad is not None else projected.dtype
    dtypes = projected.dtype, real.dtype, imag.dtype, grad_dtype
    if real.dtype != imag.dtype or any(dtype not in _CUDA_TYPES for dtype in dtypes):
        return False
    if rank < 1 or kernel % 2 != 1 or not _reader_grid_fits(projected.shape, rank):
        return False
    capability = torch.cuda.get_device_capability(projected.device)
    return capability >= ((8, 0) if torch.bfloat16 in dtypes else (7, 0))


def _spatial_geometry(
    width: int, rank: int, kernel: int, dtype: torch.dtype, *, strict: bool,
) -> tuple[int, int]:
    if not strict and kernel == _STATIC_SPATIAL_KERNEL and rank <= _RESIDENT_FILTER_RANK_LIMIT:
        columns = _ceildiv(width, _reader_columns(width))
        if columns <= _COMPACT_SPATIAL_GEOMETRY[1] and dtype != torch.float32:
            return _COMPACT_SPATIAL_GEOMETRY[0], columns
    _, compact_warps = _COMPACT_SPATIAL_GEOMETRY
    if (
        not strict and rank == _STATIC_RANK_LIMIT and kernel == _STATIC_SPATIAL_KERNEL
        and width == compact_warps * _reader_columns(width)
    ):
        return _COMPACT_SPATIAL_GEOMETRY
    return _DEFAULT_SPATIAL_GEOMETRY


def _weight_geometry(
    projected: Tensor, modes: int, rank: int, grad_dtype: torch.dtype,
) -> tuple[int, int, int]:
    batch, height, width, _ = projected.shape
    area = height * width
    if (
        rank <= _CENTERED_WEIGHT_TILE and grad_dtype == torch.bfloat16
        and width > 2 * _reader_columns(width) and batch * area >= _SPECIALIZE_MIN_PIXELS
    ):
        tile, rows, _ = _DEFAULT_WEIGHT_GEOMETRY
        sms = torch.cuda.get_device_properties(projected.device).multi_processor_count
        mode_tiles = _ceildiv(modes, _WARP_SIZE)
        blocks = _ceildiv(batch * _ceildiv(height, rows), _WIDE_WEIGHT_WARPS)
        blocks *= mode_tiles
        # Use larger CTAs when a long row amortizes the wider reduction or
        # when a narrower plane still supplies enough independent row tiles.
        minimum_waves = 1 if width >= _WARP_SIZE else _MAX_STREAMING_WEIGHT_WAVES // 2
        if blocks >= minimum_waves * sms:
            return tile, rows, _WIDE_WEIGHT_WARPS
        # For long rows, expose enough CTAs for a full device wave before
        # falling back to fewer warps. Shorter rows need more waves to pay
        # for the larger shared-memory reduction.
        if rank == tile and projected.dtype == grad_dtype and width >= _WARP_SIZE:
            single_row_blocks = _ceildiv(batch * height, _WIDE_WEIGHT_WARPS) * mode_tiles
            if single_row_blocks >= sms:
                return tile, 1, _WIDE_WEIGHT_WARPS
    if rank == _STATIC_RANK_LIMIT and grad_dtype == torch.bfloat16:
        if area <= _SMALL_WEIGHT_PLANE_PIXELS:
            return _STATIC_RANK_LIMIT, 2, 4
        if area >= _LARGE_WEIGHT_PLANE_PIXELS:
            return _STATIC_RANK_LIMIT, 2, 8
    return _DEFAULT_WEIGHT_GEOMETRY


def _reader_compile_shape(projected: Tensor, modes: int) -> tuple[int, int, int, int]:
    batch, height, width, _ = projected.shape
    # Large planes amortize the partial mode warp of a half-warp-aligned shape.
    alignment = (
        _SPECIALIZE_MODE_ALIGNMENT if height * width >= _LARGE_WEIGHT_PLANE_PIXELS else _WARP_SIZE
    )
    if batch * height * width >= _SPECIALIZE_MIN_PIXELS and modes % alignment == 0:
        return batch, height, width, modes
    return 0, 0, 0, 0


def _reader_vector_columns(width: int, vec: int) -> int:
    wide_columns = _COMPACT_COLUMNS * 2
    if vec == 2:
        return wide_columns if width % wide_columns == 0 else _DEFAULT_COLUMNS
    return _reader_columns(width)


class _VectorLaunch(NamedTuple):
    vec: int
    columns: int
    rows: int
    warps: int


class _WarpRowLaunch(_VectorLaunch):
    __slots__ = ()


@cache
def _reader_resident(module: _CudaModule, op: str, warps: int) -> int:
    blocks = ctypes.c_int()
    local_size = ctypes.c_int()
    function = module.functions[f"reader_{op}"]
    _check_cu(
        module.driver.cuFuncGetAttribute(
            ctypes.byref(local_size), _CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function,
        ),
        "cuFuncGetAttribute(LOCAL_SIZE_BYTES)",
    )
    if local_size.value:
        return 0
    _check_cu(
        module.driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks), function, warps * _WARP_SIZE, ctypes.c_size_t(0),
        ),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    )
    return blocks.value


class _WeightLaunch(NamedTuple):
    grid: tuple[int, int, int]
    warps: int
    reduce: str
    reduce_columns: int
    module: _CudaModule


def _reader_vector_width(
    projected: Tensor, real: Tensor, grad: Tensor | None, *, strict: bool,
) -> int:
    vec = {"auto": 2, "2": 2, "4": 4}.get(os.environ.get("LNET_READER_VEC", "auto"), 1)
    modes, rank, kernel, _ = real.shape
    dtype = projected.dtype
    if (
        vec == 1 or strict or kernel != _STATIC_SPATIAL_KERNEL or rank > _STATIC_RANK_LIMIT
        or modes % vec or dtype not in (torch.float16, torch.bfloat16) or real.dtype != dtype
        or (grad is not None and grad.dtype != dtype)
    ):
        return 1
    alignment = vec * projected.element_size()
    if projected.data_ptr() % alignment or (grad is not None and grad.data_ptr() % alignment):
        return 1
    props = torch.cuda.get_device_properties(projected.device)
    scratch = 2 * rank * kernel**2 * _WARP_SIZE * vec * 4
    return vec if props.major >= 8 and scratch <= props.shared_memory_per_block else 1


def _reader_module_for(
    projected: Tensor, real: Tensor, grad: Tensor | None, *, strict: bool,
    launch: _VectorLaunch, weight_geometry: tuple[int, int, int] | None = None,
    weight_vec: int = 1, updated: bool = False,
    w35b_weight: bool = False, w35b_spatial: bool = False,
) -> _CudaModule:
    modes, rank, kernel, _ = real.shape
    dtypes = (projected.dtype, real.dtype, grad.dtype if grad is not None else projected.dtype)
    filters = 0 if updated else -1
    return _reader_cuda_module(
        projected.device.index or 0, _reader_compile_shape(projected, modes), launch.columns,
        rank if rank <= _STATIC_RANK_LIMIT else 0, kernel,
        tuple(_CUDA_TYPES[dtype] for dtype in dtypes), strict=strict,
        spatial_geometry=(launch.rows, launch.warps), weight_geometry=weight_geometry,
        register_filters=filters, keep_real=filters, weight_vec=weight_vec, spatial_vec=launch.vec,
        updated_spatial=updated, weight_read_only=updated or weight_vec != 2,
        **({"w35b_weight": True} if w35b_weight else {}),
        **({"w35b_spatial": True} if w35b_spatial else {}),
    )


def _reader_vec_explicit() -> bool:
    return os.environ.get("LNET_READER_VEC", "auto") in ("2", "4")


def _reader_w35b() -> str:
    return os.environ.get("LNET_READER_W35B", "off")


def _reader_w35b_eligible(
    projected: Tensor, real: Tensor, grad: Tensor | None, *, strict: bool, vec: int,
) -> bool:
    modes, rank, kernel, _ = real.shape
    dtype = projected.dtype
    alignment = vec * projected.element_size()
    return (
        not strict and 1 <= rank <= _STATIC_RANK_LIMIT and kernel == _STATIC_SPATIAL_KERNEL
        and dtype in (torch.float16, torch.bfloat16) and real.dtype == dtype
        and modes % vec == 0 and projected.data_ptr() % alignment == 0
        and (grad is None or (grad.dtype == dtype and grad.data_ptr() % alignment == 0))
        and torch.cuda.get_device_properties(projected.device).major >= 8
    )


def _reader_auto(projected: Tensor, real: Tensor, grad: Tensor | None, *, strict: bool) -> bool:
    _, rank, kernel, _ = real.shape
    batch, height, width, _ = projected.shape
    return (
        os.environ.get("LNET_READER_OPT", "auto") == "auto"
        and not strict and 1 <= rank <= _STATIC_RANK_LIMIT and kernel == _STATIC_SPATIAL_KERNEL
        and projected.dtype == real.dtype and projected.dtype in (torch.float16, torch.bfloat16)
        and (grad is None or grad.dtype == projected.dtype)
        and (batch * height * width >= _SPECIALIZE_MIN_PIXELS or _reader_vec_explicit())
        and torch.cuda.get_device_properties(projected.device).major >= 8
    )


def _spatial_tile_candidates(
    height: int, width: int, modes: int, vec: int, *, backward: bool,
) -> tuple[_VectorLaunch, ...]:
    if backward:
        columns = 2 * _reader_columns(width)
        if width <= _reader_columns(width):
            columns = _DEFAULT_COLUMNS
        elif modes % (_WARP_SIZE * vec) and not _reader_vec_explicit():
            # Packing a partially occupied mode warp can use more loaded
            # sectors per active mode than the scalar adjoint.
            vec = 1
        warps = min(_DEFAULT_SPATIAL_GEOMETRY[1], _ceildiv(width, columns))
        if warps == 3:
            warps = 4
        rows = (min(height, _COMPACT_COLUMNS), min(height, _COMPACT_SPATIAL_GEOMETRY[0]))
        return tuple(_VectorLaunch(vec, columns, row, warps) for row in dict.fromkeys(rows))
    # The window slides one column at a time. Cap the live strip length so
    # very wide images can expose both column and batch parallelism.
    columns = min(max(width, _reader_vector_columns(width, vec)), _WARP_SIZE * vec)
    rows = tuple(dict.fromkeys(min(height, row) for row in (4, _COMPACT_COLUMNS, 8)))
    return tuple(_VectorLaunch(vec, columns, row, 1) for row in rows)


def _reader_spatial_plan(
    projected: Tensor, real: Tensor, grad: Tensor | None, *, strict: bool,
) -> tuple[_VectorLaunch, _CudaModule]:
    batch, height, width, _ = projected.shape
    modes, rank, kernel, _ = real.shape
    props = torch.cuda.get_device_properties(projected.device)
    backward = grad is not None
    op = "dinput" if backward else "forward"
    if (
        any(mode in _reader_w35b().split(",") for mode in ("auto", "fwd_dinput_vec4", "full_w35b"))
        and _reader_w35b_eligible(projected, real, grad, strict=strict, vec=4)
    ):
        # Distribute full-width row strips across warps to avoid horizontal halos.
        launch = _WarpRowLaunch(4, min(width, 128), 1, 4)
        grid = _reader_spatial_grid(batch, height, width, modes, launch.rows, launch.warps,
                                    launch.vec, launch.columns, warp_row_tiles=True)
        if grid[1] <= _MAX_CUDA_GRID_DIM:
            module = _reader_module_for(projected, real, grad, strict=False, launch=launch,
                                        w35b_spatial=True)
            if _reader_resident(module, op, launch.warps) * launch.warps >= 16:
                return launch, module
    if _reader_auto(projected, real, grad, strict=strict):
        vec = _reader_vector_width(projected, real, grad, strict=strict)
        best = None
        best_score = 0.0
        for launch in _spatial_tile_candidates(height, width, modes, vec, backward=backward):
            grid = _reader_spatial_grid(batch, height, width, modes, launch.rows, launch.warps,
                                        launch.vec, launch.columns)
            blocks = grid[0] * grid[1] * grid[2]
            if grid[1] > _MAX_CUDA_GRID_DIM or (
                blocks < props.multi_processor_count and not _reader_vec_explicit()
            ):
                continue
            module = _reader_module_for(projected, real, grad, strict=False, launch=launch,
                                        updated=True)
            resident = _reader_resident(module, op, launch.warps)
            sm_work = blocks / props.multi_processor_count
            row_tiles = _ceildiv(height, launch.rows)
            traffic = row_tiles * (launch.rows + kernel - 1) - (kernel - 1)
            if backward and width > _reader_columns(width):
                # Two-warp CTAs have room to double up on a scheduler.
                # Require enough work to exploit that residency before
                # paying for additional vertical halos.
                ready = min(resident, sm_work / 4) * launch.warps
                warp_capacity = props.max_threads_per_multi_processor // _WARP_SIZE
                target = warp_capacity // (2 if launch.warps == 2 else 4)
                score = min(target, ready) / traffic
            else:
                score = min(8, resident, sm_work) / traffic
            if score > best_score:
                best, best_score = (launch, module), score
        if best is not None:
            return best
    rows, warps = _spatial_geometry(width, rank, kernel, projected.dtype, strict=strict)
    launch = _VectorLaunch(1, _reader_columns(width), rows, warps)
    return launch, _reader_module_for(projected, real, grad, strict=strict, launch=launch)


def _weight_partial_limit(
    total: int, sms: int, stride: int, resident_threads: int, device: torch.device,
) -> int:
    # Limit the reduction's working set to the resident thread capacity,
    # and scale the allocation with device memory and allocator headroom.
    allocated = torch.cuda.memory_allocated(device)
    headroom = max(0, 7 * total // 8 - allocated)
    resident_bytes = sms * resident_threads * 256
    budget = min(headroom, total // 32, resident_bytes)
    return max(1, min(_MAX_CUDA_GRID_DIM, budget // (4 * stride)))


def _reader_weight_candidate(
    projected: Tensor, real: Tensor, grad: Tensor, spatial: _VectorLaunch, *, compact: bool,
) -> tuple[tuple[int, int, int], int, _CudaModule, int] | None:
    _, height, width, _ = projected.shape
    _, rank, _, _ = real.shape
    vec = _reader_vector_width(projected, real, grad, strict=False)
    if not _reader_vec_explicit() and width <= 2 * spatial.columns:
        vec = 1
    rank_tile = min(rank, _CENTERED_WEIGHT_TILE)
    if compact and grad.dtype == torch.bfloat16 and vec == 1:
        # Use the full-rank BF16 tile on short windows if it has no local memory.
        rank_tile = rank
    rows = min(height, 2 if compact else max(4, min(8, width // spatial.columns)))
    if vec == 4:
        rank_tile, rows = 1, 1
    if vec > 1:
        vector_rows, vector_warps = _VEC2_SPATIAL_GEOMETRY if vec == 2 else (1, 4)
        spatial = _VectorLaunch(vec, _reader_vector_columns(width, vec), vector_rows, vector_warps)
    geometry = (rank_tile, rows, _DEFAULT_SPATIAL_GEOMETRY[1])
    module = _reader_module_for(projected, real, grad, strict=False, launch=spatial,
                                weight_geometry=geometry, weight_vec=vec)
    resident = _reader_resident(module, "dweight", geometry[2])
    return (geometry, vec, module, resident) if resident else None


def _reader_w35b_weight_candidate(
    projected: Tensor, real: Tensor, grad: Tensor, spatial: _VectorLaunch, *, strict: bool,
) -> tuple[tuple[int, int, int], int, _CudaModule, int] | None:
    variant = next((mode for mode in _reader_w35b().split(",")
                    if mode in ("auto", "dweight_rt4", "dweight_rows4", "full_w35b")), None)
    if variant is None:
        return None
    vec = 2 if variant == "dweight_rows4" else 1
    if (
        8 * real.numel() > _W35B_PARTIAL_BUDGET
        or not _reader_w35b_eligible(projected, real, grad, strict=strict, vec=vec)
    ):
        return None
    height = projected.shape[1]
    rank = real.shape[1]
    tile = min(rank, 2 if vec == 2 else 4)
    candidates = (4, 3, 2) if vec == 2 else (3, 2)
    for rows in dict.fromkeys(min(height, row) for row in candidates):
        geometry = (tile, rows, 4)
        module = _reader_module_for(
            projected, real, grad, strict=False, launch=spatial,
            weight_geometry=geometry, weight_vec=vec, w35b_weight=True,
        )
        resident = _reader_resident(module, "dweight", geometry[2])
        if resident >= (3 if vec == 2 else 4):
            return geometry, vec, module, resident
    return None


def _reader_weight_plan(
    projected: Tensor, real: Tensor, grad: Tensor, *, strict: bool,
) -> _WeightLaunch:
    batch, height, width, _ = projected.shape
    modes, rank, kernel, _ = real.shape
    device = projected.device
    props = torch.cuda.get_device_properties(device)
    sms = props.multi_processor_count
    static = not strict and rank <= _STATIC_RANK_LIMIT and kernel == _STATIC_SPATIAL_KERNEL
    baseline = _weight_geometry(projected, modes, rank, grad.dtype) if static else None
    columns = _reader_columns(width)
    rows, warps = _spatial_geometry(width, rank, kernel, projected.dtype, strict=strict)
    spatial = _VectorLaunch(1, columns, rows, warps)
    stride = 2 * modes * rank * kernel**2
    geometry, weight_vec, module = baseline, 1, None
    cta_waves = 5 if static else 3
    resident = 1
    compact = width <= columns and height <= 2 * columns
    candidate = _reader_w35b_weight_candidate(projected, real, grad, spatial, strict=strict)
    w35b = candidate is not None
    if candidate is not None:
        geometry, weight_vec, module, resident = candidate
        cta_waves = 2 * resident
    elif _reader_auto(projected, real, grad, strict=strict):
        candidate = _reader_weight_candidate(projected, real, grad, spatial, compact=compact)
        if candidate is not None:
            geometry, weight_vec, module, resident = candidate
            # Longer row tiles have fewer bands. Keep their workers spread
            # over additional resident waves before reducing the partials.
            waves = max(2, geometry[1] // 2) if weight_vec > 1 else 2
            cta_waves = 5 if compact else resident * waves
    mode_tiles = _ceildiv(modes, _WARP_SIZE * weight_vec)
    if geometry is None:
        rank_tiles = _ceildiv(rank, 2 if strict or rank > _STATIC_RANK_LIMIT else rank)
        tap_tiles = 3 if kernel == _STATIC_SPATIAL_KERNEL else kernel**2
        weight_rows, weight_warps = 1, _DEFAULT_SPATIAL_GEOMETRY[1]
    else:
        tile, weight_rows, weight_warps = geometry
        rank_tiles, tap_tiles = 1, _ceildiv(rank, tile)
    ctas = mode_tiles * tap_tiles
    available = _ceildiv(batch * _ceildiv(height, weight_rows), weight_warps)
    if w35b:
        workers = _ceildiv(cta_waves * sms, ctas)
    elif module is not None and not compact:
        workers = max(1, cta_waves * sms // ctas)
    else:
        workers = _ceildiv(cta_waves * sms, mode_tiles)
    if module is None:
        module = _reader_module_for(projected, real, grad, strict=strict, launch=spatial,
                                    weight_geometry=geometry)
        resident = _reader_resident(module, "dweight", weight_warps)
    limit = _weight_partial_limit(
        props.total_memory, sms, stride, max(resident, 1) * weight_warps * _WARP_SIZE, device,
    )
    if w35b:
        limit = min(limit, _W35B_PARTIAL_BUDGET // (4 * stride))
    workers = min(available, workers, limit)
    if workers >= _WEIGHT_PARALLEL_REDUCTION_WORKERS:
        return _WeightLaunch((ctas, workers, rank_tiles), weight_warps, "dweight_reduce_parallel",
                             _WARP_SIZE, module)
    return _WeightLaunch((ctas, workers, rank_tiles), weight_warps, "dweight_reduce",
                         _DEFAULT_THREADS, module)


# CUDA launch state

class _PointStrides(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("mode", "rank", "channel")]


class _FilterStrides(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("outer", "inner", "rank", "row", "col")]


_CUDA_LAUNCH_STATE = local()
_CUDA_STRIDES = tuple[int, ...]


def _reader_stream(module: _CudaModule, device: torch.device) -> ctypes.c_void_p:
    _ensure_context(module.driver, device.index or 0)
    return ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)


def _launch_reader_cuda(
    module: _CudaModule, operation: str, grid: tuple[int, int, int],
    tensors: tuple[Tensor, ...], sizes: tuple[int | float | _CUDA_STRIDES, ...],
    *, stream: ctypes.c_void_p, threads: int = _DEFAULT_THREADS,
) -> None:
    driver = module.driver
    buffers = getattr(_CUDA_LAUNCH_STATE, "buffers", None)
    if buffers is None:
        buffers = _CUDA_LAUNCH_STATE.buffers = {}
    if operation not in buffers:
        values = [ctypes.c_void_p() for _ in tensors]
        for value in sizes:
            if isinstance(value, tuple):
                values.append(_PointStrides() if len(value) == 3 else _FilterStrides())
            else:
                values.append(ctypes.c_float() if isinstance(value, float) else ctypes.c_int())
        pointers = (ctypes.c_void_p * len(values))(*(ctypes.addressof(arg) for arg in values))
        buffers[operation] = values, pointers
    values, pointers = buffers[operation]
    for index, tensor in enumerate(tensors):
        values[index].value = tensor.data_ptr()
    for index, value in enumerate(sizes, len(tensors)):
        arg = values[index]
        if isinstance(value, tuple):
            for (name, _), stride in zip(arg._fields_, value, strict=True):
                setattr(arg, name, stride)
        else:
            arg.value = value
    # The driver consumes the argument values during this call, including during capture.
    name = f"reader_{operation}"
    _check_cu(
        driver.cuLaunchKernel(
            module.functions[name], *grid, threads, 1, 1, 0,
            stream, pointers, None,
        ),
        name,
    )


def _launch_gram_normalize(
    point: Tensor, spatial: Tensor, gradient: Tensor | None, eps: float, out_dtype: torch.dtype,
) -> Tensor | tuple[Tensor, Tensor]:
    modes, rank, channels = point.shape
    kernel = spatial.shape[-1]
    dtypes = (point.dtype, spatial.dtype, out_dtype)
    with torch.cuda.device(point.device):
        module = _reader_cuda_module(
            point.device.index or 0, (0, channels, 0, 0), 0, rank, kernel,
            tuple(_CUDA_TYPES[dtype] for dtype in dtypes), strict=False, normalize=True,
        )
        stream = _reader_stream(module, point.device)
        grid = (modes, 1, 1)
        sizes = (modes, float(eps), point.stride(), spatial.stride())
        if gradient is None:
            output = spatial.new_empty((2, modes, rank, kernel, kernel), dtype=out_dtype)
            _launch_reader_cuda(
                module, "gram_normalize", grid, (point, spatial, output), sizes, stream=stream,
            )
            return output
        dp, ds = point.new_empty(point.shape), spatial.new_empty(spatial.shape)
        _launch_reader_cuda(
            module, "gram_normalize_backward", grid,
            (point, spatial, gradient, dp, ds), (*sizes, gradient.stride()), stream=stream,
        )
        return dp, ds


# Custom ops and public dispatch

@torch.library.custom_op("va_reader::gram_normalize", mutates_args=(), device_types="cuda")
def _gram_normalize(
    point: Tensor, spatial: Tensor, eps: float, out_dtype: torch.dtype,
) -> Tensor:
    return _launch_gram_normalize(point, spatial, None, eps, out_dtype)


@_gram_normalize.register_fake
def _gram_normalize_fake(
    point: Tensor, spatial: Tensor, _eps: float, out_dtype: torch.dtype,
) -> Tensor:
    return spatial.new_empty((2, *point.shape[:2], *spatial.shape[-2:]), dtype=out_dtype)


@torch.library.custom_op("va_reader::gram_normalize_backward", mutates_args=(), device_types="cuda")
def _gram_normalize_backward(
    point: Tensor, spatial: Tensor, gradient: Tensor, eps: float,
) -> tuple[Tensor, Tensor]:
    return _launch_gram_normalize(point, spatial, gradient, eps, gradient.dtype)


@_gram_normalize_backward.register_fake
def _gram_normalize_backward_fake(
    point: Tensor, spatial: Tensor, _gradient: Tensor, _eps: float,
) -> tuple[Tensor, Tensor]:
    return point.new_empty(point.shape), spatial.new_empty(spatial.shape)


def _save_gram_normalize_inputs(ctx, inputs, output) -> None:  # noqa: ANN001, ARG001
    point, spatial, ctx.eps, _ = inputs
    ctx.save_for_backward(point, spatial)


def _backward_gram_normalize(ctx, gradient: Tensor) -> tuple[Tensor, Tensor, None, None]:  # noqa: ANN001
    point, spatial = ctx.saved_tensors
    dp, ds = _gram_normalize_backward(point, spatial, gradient, ctx.eps)
    return dp, ds, None, None


_gram_normalize.register_autograd(
    _backward_gram_normalize, setup_context=_save_gram_normalize_inputs,
)


def real_gram_normalize_spatial(
    point_weight: Tensor, spatial_weight: Tensor, eps: float, out_dtype: torch.dtype,
) -> Tensor:
    """Normalize the two real filter planes using their FP32 Gram energies."""
    modes, rank, channels = point_weight.shape
    kernel = spatial_weight.shape[-1]
    # Splitting the leading axis of (2*M, R, K, K) is a view even in channels-last.
    spatial = spatial_weight.reshape(modes, 2, rank, kernel, kernel)
    dtypes = (point_weight.dtype, spatial.dtype, out_dtype)
    if (
        point_weight.is_cuda and spatial.device == point_weight.device
        and modes > 0 and 1 <= rank <= 8 and channels > 0
        and all(dtype in _CUDA_TYPES for dtype in dtypes)
        and torch.cuda.get_device_capability(point_weight.device)
        >= ((8, 0) if torch.bfloat16 in dtypes else (7, 0))
    ):
        return _gram_normalize(point_weight, spatial, eps, out_dtype)
    energy = real_gram_kernel_energy(point_weight, spatial)
    inverse = energy.clamp_min(eps).rsqrt().transpose(0, 1)[..., None, None, None]
    return (spatial.float().transpose(0, 1) * inverse).to(out_dtype).contiguous()


def _reader_spatial_grid(
    batch: int, height: int, width: int, modes: int, rows: int, warps: int, vec: int = 1,
    columns: int | None = None, *, warp_row_tiles: bool = False,
) -> tuple[int, int, int]:
    if warp_row_tiles:
        strips = _ceildiv(height, rows) * _ceildiv(width, columns or width)
        return _ceildiv(modes, _WARP_SIZE * vec), _ceildiv(strips, warps), batch
    tiles = _ceildiv(height, rows) * _ceildiv(width, warps * (
        columns or _reader_vector_columns(width, vec)
    ))
    return _ceildiv(modes, _WARP_SIZE * vec), tiles, batch


@torch.library.custom_op("va_reader::packed_spatial", mutates_args=(), device_types="cuda")
def _packed_spatial(
    projected: Tensor, real: Tensor, imag: Tensor, strict_complex: bool = True,
) -> Tensor:
    batch, height, width, _ = projected.shape
    modes, rank, _, _ = real.shape
    if _use_reader_cuda(projected, real, imag):
        output = projected.new_empty((batch, height, width, 2 * modes))
        with torch.cuda.device(projected.device):
            launch, module = _reader_spatial_plan(
                projected, real, None, strict=strict_complex,
            )
            vec, columns, rows, warps = launch
            stream = _reader_stream(module, projected.device)
            _launch_reader_cuda(
                module, "forward",
                _reader_spatial_grid(batch, height, width, modes, rows, warps, vec, columns,
                                     warp_row_tiles=isinstance(launch, _WarpRowLaunch)),
                (projected, real, imag, output), (height, width, modes, rank),
                stream=stream, threads=warps * _WARP_SIZE,
            )
        return output
    return _spatial_reference(projected, real, imag, strict_complex=strict_complex)


@_packed_spatial.register_fake
def _packed_spatial_fake(
    projected: Tensor, real: Tensor, _imag: Tensor, _strict_complex: bool = True,
) -> Tensor:
    return projected.new_empty((*projected.shape[:-1], 2 * real.shape[0]))


@torch.library.custom_op("va_reader::packed_spatial_backward", mutates_args=(), device_types="cuda")
def _packed_spatial_backward(
    grad: Tensor,
    projected: Tensor,
    real: Tensor,
    imag: Tensor,
    strict_complex: bool = True,
) -> tuple[Tensor, Tensor]:
    batch, height, width, _ = projected.shape
    modes, rank, kernel, _ = real.shape
    if _use_reader_cuda(projected, real, imag, grad):
        grad = grad.contiguous()
        grad_projected = torch.empty_like(projected)
        with torch.cuda.device(projected.device):
            plan = _reader_weight_plan(projected, real, grad, strict=strict_complex)
            partial = projected.new_empty(
                (plan.grid[1], 2, rank, kernel, kernel, modes), dtype=torch.float32,
            )
            weight_grad = real.new_empty((2, *real.shape), dtype=torch.float32)
            stream = _reader_stream(plan.module, projected.device)
            _launch_reader_cuda(
                plan.module, "dweight", plan.grid, (projected, grad, partial),
                (height, width, modes, batch, rank), stream=stream, threads=plan.warps * _WARP_SIZE,
            )
            reduce_blocks = _ceildiv(weight_grad.numel(), plan.reduce_columns)
            _launch_reader_cuda(
                plan.module, plan.reduce, (reduce_blocks, 1, 1),
                (partial, weight_grad), (modes, rank, plan.grid[1]), stream=stream,
            )
            del partial
            launch, module = _reader_spatial_plan(
                projected, real, grad, strict=strict_complex,
            )
            vec, columns, spatial_rows, warps = launch
            stream = _reader_stream(module, projected.device)
            grid = _reader_spatial_grid(
                batch, height, width, modes, spatial_rows, warps, vec, columns,
                warp_row_tiles=isinstance(launch, _WarpRowLaunch),
            )
            _launch_reader_cuda(
                module, "dinput", grid, (grad, real, imag, grad_projected),
                (height, width, modes, rank), stream=stream, threads=warps * _WARP_SIZE,
            )
        return grad_projected, weight_grad

    return _spatial_reference_backward(
        grad, projected, real, imag, strict_complex=strict_complex,
    )


@_packed_spatial_backward.register_fake
def _packed_spatial_backward_fake(
    _grad: Tensor,
    projected: Tensor,
    real: Tensor,
    _imag: Tensor,
    _strict_complex: bool = True,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(projected), real.new_empty((2, *real.shape), dtype=torch.float32)


def _save_spatial_inputs(ctx, inputs, output) -> None:  # noqa: ANN001, ARG001
    projected, real, imag, ctx.strict_complex = inputs
    ctx.save_for_backward(projected, real, imag)


def _backward_spatial(ctx, grad: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:  # noqa: ANN001
    projected, real, imag = ctx.saved_tensors
    grad_projected, grad_weights = _packed_spatial_backward(
        grad, projected, real, imag, ctx.strict_complex,
    )
    return grad_projected, grad_weights[0].to(real.dtype), grad_weights[1].to(imag.dtype), None


_packed_spatial.register_autograd(_backward_spatial, setup_context=_save_spatial_inputs)


def packed_spatial(projected: Tensor, real: Tensor, imag: Tensor) -> Tensor:
    """Filter packed (component, rank, mode) projections with FP32 accumulation."""
    if projected.is_cuda and projected.numel() != 0:
        return _packed_spatial(projected.contiguous(), real.contiguous(), imag.contiguous())
    return _spatial_reference(projected, real, imag)


def real_packed_spatial(
    projected: Tensor, weights: Tensor, *, component_first: bool = False,
) -> Tensor:
    """Apply two independent filters per mode to a shared real projection."""
    real, imag = weights.unbind(dim=0 if component_first else 1)
    if projected.is_cuda and projected.numel() != 0:
        return _packed_spatial(
            projected.contiguous(), real.contiguous(), imag.contiguous(), strict_complex=False,
        )
    return _spatial_reference(projected, real, imag, strict_complex=False)
