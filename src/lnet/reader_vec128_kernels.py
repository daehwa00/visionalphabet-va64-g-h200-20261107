"""Shared-memory tiles for the real, rank-major packed Reader."""

MODE_TILE = 32
TILE_H = 8
TILE_W = 8
WARPS = 4


def source() -> str:
    return f"""
constexpr int Modes = {MODE_TILE}, TileH = {TILE_H}, TileW = {TILE_W};
constexpr int Threads = {WARPS * 32};
""" + _CUDA


_CUDA = r"""
constexpr int HaloH = TileH + 2, HaloW = TileW + 2;
constexpr int Vectors = Modes / 8;

template <class T>
__device__ __forceinline__ uint4 read128(const T* pointer) {
    uint4 value;
    asm("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w) : "l"(pointer));
    return value;
}

template <class T>
__device__ __forceinline__ void write128(T* pointer, uint4 value) {
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
        :: "l"(pointer), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
}

template <class T, int Channels, int Rows, int Columns, int Pad>
__device__ __forceinline__ void stage(
    uint4* tile, const T* input, long long image, int y0, int x0,
    int h, int w, int m, int m0
) {
    for (int i = threadIdx.x; i < Rows * Columns * Channels * Vectors; i += Threads) {
        const int mode = m0 + (i % Vectors) * 8;
        const int pixel = i / (Channels * Vectors);
        const int y = y0 + pixel / Columns - Pad;
        const int x = x0 + pixel % Columns - Pad;
        uint4 value{};
        if (mode < m && y >= 0 && y < h && x >= 0 && x < w) {
            const int channel = i / Vectors % Channels;
            value = read128(input + ((image + y * w + x) * Channels + channel) * m + mode);
        }
        tile[i] = value;
    }
}

template <bool Backward, class T>
__device__ __forceinline__ void spatial128(
    const T* input, const Weight* real, const Weight* imag, Input* output,
    int h, int w, int m
) {
    constexpr int Rank = READER_RANK;
    constexpr int In = Backward ? 2 : Rank, Out = Backward ? Rank : 2;
    __shared__ uint4 halo[HaloH * HaloW * In * Vectors];
    __shared__ uint4 result[TileH * TileW * Out * Vectors];
    __shared__ float filters[2 * Rank * 9][Modes];
    const int m0 = blockIdx.x * Modes;
    const int x_tiles = (w + TileW - 1) / TileW;
    const int y0 = blockIdx.y / x_tiles * TileH, x0 = blockIdx.y % x_tiles * TileW;
    const long long image = static_cast<long long>(blockIdx.z) * h * w;
    stage<T, In, HaloH, HaloW, 1>(halo, input, image, y0, x0, h, w, m, m0);
    for (int i = threadIdx.x; i < Modes * Rank * 9; i += Threads) {
        const int mode = m0 + i / (Rank * 9);
        const int field = i % (Rank * 9);
        if (mode < m) {
            const int index = mode * Rank * 9 + field;
            filters[field][mode - m0] = as_float(real[index]);
            filters[Rank * 9 + field][mode - m0] = as_float(imag[index]);
        }
    }
    __syncthreads();
    const T* patch = reinterpret_cast<const T*>(halo);
    Input* values = reinterpret_cast<Input*>(result);
    const int lane = threadIdx.x % Modes;
    if (m0 + lane < m) {
        #pragma unroll 1
        for (int pixel = threadIdx.x / Modes; pixel < TileH * TileW; pixel += Threads / Modes) {
            const int y = pixel / TileW, x = pixel % TileW;
            float out[Out] = {};
            #pragma unroll
            for (int r = 0; r < Rank; ++r) {
                #pragma unroll
                for (int tap = 0; tap < 9; ++tap) {
                    const int offset = ((y + tap / 3) * HaloW + x + tap % 3) * In * Modes + lane;
                    const int field = r * 9 + (Backward ? 8 - tap : tap);
                    const float a = filters[field][lane];
                    const float b = filters[Rank * 9 + field][lane];
                    if constexpr (Backward) {
                        out[r] = fmaf(as_float(patch[offset]), a, out[r]);
                        out[r] = fmaf(as_float(patch[offset + Modes]), b, out[r]);
                    } else {
                        const float p = as_float(patch[offset + r * Modes]);
                        out[0] = fmaf(p, a, out[0]);
                        out[1] = fmaf(p, b, out[1]);
                    }
                }
            }
            #pragma unroll
            for (int c = 0; c < Out; ++c)
                values[(pixel * Out + c) * Modes + lane] = from_float<Input>(out[c]);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TileH * TileW * Out * Vectors; i += Threads) {
        const int pixel = i / (Out * Vectors);
        const int mode = m0 + (i % Vectors) * 8;
        const int y = y0 + pixel / TileW, x = x0 + pixel % TileW;
        if (mode < m && y < h && x < w) {
            const int channel = i / Vectors % Out;
            write128(output + ((image + y * w + x) * Out + channel) * m + mode, result[i]);
        }
    }
}

extern "C" __global__ void reader_forward(
    const Input* input, const Weight* real, const Weight* imag,
    Input* output, int h, int w, int m, int rank
) {
    spatial128<false>(input, real, imag, output, h, w, m);
}

extern "C" __global__ void reader_dinput(
    const Grad* input, const Weight* real, const Weight* imag,
    Input* output, int h, int w, int m, int rank
) {
    spatial128<true>(input, real, imag, output, h, w, m);
}

extern "C" __global__ void reader_dweight(
    const Input* projected, const Grad* grad, float* partial,
    int h, int w, int m, int batch, int rank
) {
    constexpr int Rank = READER_RANK;
    __shared__ uint4 halo[HaloH * HaloW * Rank * Vectors];
    __shared__ uint4 center[TileH * TileW * 2 * Vectors];
    const Input* inputs = reinterpret_cast<const Input*>(halo);
    const Grad* gradients = reinterpret_cast<const Grad*>(center);
    const int m0 = blockIdx.x * Modes;
    const int lane = threadIdx.x % Modes, r = threadIdx.x / Modes;
    const int x_tiles = (w + TileW - 1) / TileW;
    const int tiles = x_tiles * ((h + TileH - 1) / TileH);
    const long long work = static_cast<long long>(batch) * tiles;
    const long long start = work * blockIdx.y / gridDim.y;
    const long long stop = work * (blockIdx.y + 1) / gridDim.y;
    float dr[9] = {}, di[9] = {};
    for (long long job = start; job < stop; ++job) {
        const long long image = (job / tiles) * h * w;
        const int tile = job % tiles;
        const int y0 = tile / x_tiles * TileH, x0 = tile % x_tiles * TileW;
        stage<Input, Rank, HaloH, HaloW, 1>(halo, projected, image, y0, x0, h, w, m, m0);
        stage<Grad, 2, TileH, TileW, 0>(center, grad, image, y0, x0, h, w, m, m0);
        __syncthreads();
        if (r < Rank && m0 + lane < m) {
            #pragma unroll 1
            for (int pixel = 0; pixel < TileH * TileW; ++pixel) {
                const int y = pixel / TileW, x = pixel % TileW;
                if (y0 + y >= h || x0 + x >= w) continue;
                const float a = as_float(gradients[pixel * 2 * Modes + lane]);
                const float b = as_float(gradients[pixel * 2 * Modes + Modes + lane]);
                #pragma unroll
                for (int tap = 0; tap < 9; ++tap) {
                    const int neighbor = (y + tap / 3) * HaloW + x + tap % 3;
                    const float p = as_float(inputs[(neighbor * Rank + r) * Modes + lane]);
                    dr[tap] = fmaf(p, a, dr[tap]);
                    di[tap] = fmaf(p, b, di[tap]);
                }
            }
        }
        __syncthreads();
    }
    if (r < Rank && m0 + lane < m) {
        const long long base = (static_cast<long long>(blockIdx.y) * 2 * Rank + r) * 9 * m;
        #pragma unroll
        for (int tap = 0; tap < 9; ++tap) {
            partial[base + tap * m + m0 + lane] = dr[tap];
            partial[base + (Rank * 9 + tap) * m + m0 + lane] = di[tap];
        }
    }
}
"""
