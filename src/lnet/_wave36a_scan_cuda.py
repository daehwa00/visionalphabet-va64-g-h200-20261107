"""NVRTC source for opt-in Wave 36A scan kernels."""

SOURCE = r"""
template<class T>
struct alignas(sizeof(Quad<T>)) SavedQuad {
    Quad<T> values;
    __device__ float4 unpack() const { return values.unpack(); }
    __device__ static SavedQuad pack(float4 v) { return {Quad<T>::pack(v)}; }
};

template<class T> struct SavedPairRaw;
template<> struct SavedPairRaw<__half> {
    using Type = __half2_raw;
    static constexpr bool fp16 = true;
};
template<> struct SavedPairRaw<__nv_bfloat16> {
    using Type = __nv_bfloat162_raw;
    static constexpr bool fp16 = false;
};

template<class T>
__device__ __forceinline__ float4 load_saved(const SavedQuad<T>* p) {
    const auto address = __cvta_generic_to_global(p);
    if constexpr (sizeof(T) == 2) {
        uint2 bits;
        asm("ld.global.v2.b32 {%0, %1}, [%2];"
            : "=r"(bits.x), "=r"(bits.y) : "l"(address) : "memory");
        using Raw = typename SavedPairRaw<T>::Type;
        const Pair<T> a{Raw{static_cast<unsigned short>(bits.x),
                            static_cast<unsigned short>(bits.x >> 16)}};
        const Pair<T> b{Raw{static_cast<unsigned short>(bits.y),
                            static_cast<unsigned short>(bits.y >> 16)}};
        const auto x = a.unpack(), y = b.unpack();
        return make_float4(x.r, x.i, y.r, y.i);
    } else {
        float4 v;
        asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
            : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(address) : "memory");
        return v;
    }
}

template<class T>
__device__ __forceinline__ void store_saved(SavedQuad<T>* p, SavedQuad<T> v) {
    const auto address = __cvta_generic_to_global(p);
    if constexpr (sizeof(T) == 2) {
        using Raw = typename SavedPairRaw<T>::Type;
        const Raw a = v.values.a.value, b = v.values.b.value;
        const unsigned x = unsigned(a.x) | (unsigned(a.y) << 16);
        const unsigned y = unsigned(b.x) | (unsigned(b.y) << 16);
        asm volatile("st.global.v2.b32 [%0], {%1, %2};"
                     :: "l"(address), "r"(x), "r"(y) : "memory");
    } else {
        const auto f = v.unpack();
        asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                     :: "l"(address), "f"(f.x), "f"(f.y), "f"(f.z), "f"(f.w) : "memory");
    }
}

template<class T, class I, int WARPS, bool COARSE, bool BWD, bool COMPACT, bool GS1>
__device__ __forceinline__ void saved_vscan(
    const float* dr, const float* di, const float* gr, const float* gi,
    const T* states, const float* vx, const float* vy, SavedQuad<T>* trajectory,
    T* features, T* gs, float* partial, int B, int H, int W, int M, float eps
) {
    extern __shared__ float4 shared[];
    constexpr int threads = 32 * WARPS;
    const int lane = threadIdx.x % 32;
    const int ow = W / (1 + COARSE), oh = H / (1 + COARSE);
    const I lines = I(B) * ow, plane = lines * H * M, fp = lines * oh * M;
    const int tiles = (M - 1) / 32 + 1, m = blockIdx.x % tiles * 32 + lane;
    const int group = blockIdx.x / tiles, groups = gridDim.x / tiles;
    const I worker = I(group) * WARPS + threadIdx.x / 32;
    Complex da{}, dg{};
    if (m < M) {
        const Complex a{dr[m], di[m]}, g{gr[m], gi[m]};
        const Complex ar = a.conj(), gg = g.conj();
        for (I line = worker; line < lines; line += I(groups) * WARPS) {
            const I b = line / ow, x = line % ow, stride = I(ow) * M;
            const int xa = COARSE ? 2 * x + 1 : x, xb = COARSE ? 2 * x : x;
            const I base = b * H * W * M + m, sp = I(B) * H * W * M;
            const Column<T, I> c{states, vy, base + I(xa) * M, base + I(xb) * M,
                sp, sp, (b * oh * ow + x) * M + m, fp, H, W, M, ow, m,
                vx[I(xa) * M + m], vx[(I(W) + xb) * M + m], eps};
            auto* pos = trajectory + b * H * stride + x * M + m;
            auto* neg = pos + plane;
            if constexpr (!BWD) {
                float4 p{}, n{};
                #pragma unroll 1
                for (int y = 0; y < H; ++y) {
                    p = advance(p, c.source(y), a, g);
                    const auto packed = SavedQuad<T>::pack(p);
                    store_saved(pos + I(y) * stride, packed);
                    if (!COARSE || (y & 1)) {
                        const auto v = packed.unpack();
                        const auto gain = c.gain(y, false);
                        const I i = c.cell(y / (1 + COARSE));
                        features[i] = write<T>(v.x * gain.x);
                        features[fp + i] = write<T>(v.z * gain.y);
                        features[4 * fp + i] = write<T>(v.y * gain.x);
                        features[5 * fp + i] = write<T>(v.w * gain.y);
                    }
                }
                #pragma unroll 1
                for (int y = H - 1; y >= 0; --y) {
                    n = advance(n, c.source(y), ar, gg);
                    const auto packed = SavedQuad<T>::pack(n);
                    store_saved(neg + I(y) * stride, packed);
                    if (!COARSE || !(y & 1)) {
                        const auto v = packed.unpack();
                        const auto gain = c.gain(y, true);
                        const I i = c.cell(y / (1 + COARSE));
                        features[2 * fp + i] = write<T>(v.x * gain.x);
                        features[3 * fp + i] = write<T>(v.z * gain.y);
                        features[6 * fp + i] = write<T>(v.y * gain.x);
                        features[7 * fp + i] = write<T>(v.w * gain.y);
                    }
                }
            } else {
                auto* cache = reinterpret_cast<Quad<T>*>(shared) + threadIdx.x;
                auto inject = [&](int y, bool reverse) {
                    const int k = 2 * reverse;
                    const I i = c.cell(y / (1 + COARSE)) + I(k) * fp;
                    const float2 gain = c.gain(y, reverse);
                    return make_float4(
                        rounded<T>(read(features[i]) * gain.x),
                        rounded<T>(read(features[i + 4 * fp]) * gain.x),
                        rounded<T>(read(features[i + fp]) * gain.y),
                        rounded<T>(read(features[i + 5 * fp]) * gain.y));
                };
                Complex adj_a{}, adj_b{};
                #pragma unroll 1
                for (int y = H - 1; y >= 0; --y) {
                    const auto d = !COARSE || (y & 1) ? inject(y, false) : float4{};
                    const auto prev = y ? load_saved(pos + I(y - 1) * stride) : float4{};
                    const auto ds = column_adjoint_step<false>(
                        prev, c.source(y), d, a, g, adj_a, adj_b, da, dg);
                    if constexpr (GS1) cache[I(y) * threads] = Quad<T>::pack(ds);
                    else write_column_gradient<false, COMPACT>(c, y, ds, gs);
                }
                adj_a = adj_b = {};
                #pragma unroll 1
                for (int y = 0; y < H; ++y) {
                    const auto d = !COARSE || !(y & 1) ? inject(y, true) : float4{};
                    const auto prev = y + 1 < H ? load_saved(neg + I(y + 1) * stride) : float4{};
                    const auto ds = column_adjoint_step<true>(
                        prev, c.source(y), d, ar, gg, adj_a, adj_b, da, dg);
                    if constexpr (GS1)
                        write_column_gradient<true, COMPACT>(c, y, ds, gs, cache[I(y) * threads]);
                    else write_column_gradient<true, COMPACT, true>(c, y, ds, gs);
                    if constexpr (COARSE) clear_unused_columns<COMPACT>(c, y, gs);
                }
            }
        }
    }
    if constexpr (BWD)
        write_vscan_partial<WARPS, 4, I>(reinterpret_cast<float*>(shared),
            partial + I(group) * 4 * M + m, nullptr, da, dg, m, M);
}

template<class T>
__device__ __forceinline__ void project_pair(
    Pair<T> packed, float x, float y, float u, float v, float& a, float& b
) {
    using Raw = typename SavedPairRaw<T>::Type;
    const Raw raw = packed.value;
    const unsigned bits = unsigned(raw.x) | (unsigned(raw.y) << 16);
#define PROJECT_PAIR_ASM(dtype) \
    asm volatile("{ .reg .b16 wx, wy; .reg .f32 fx, fy; " \
        "mov.b32 {wx, wy}, %2; cvt.f32." dtype " fx, wx; cvt.f32." dtype " fy, wy; " \
        "fma.rn.f32 %0, fx, %3, %0; fma.rn.f32 %0, fy, %4, %0; " \
        "fma.rn.f32 %1, fx, %5, %1; fma.rn.f32 %1, fy, %6, %1; }" \
        : "+f"(a), "+f"(b) : "r"(bits), "f"(x), "f"(y), "f"(u), "f"(v))
    if constexpr (SavedPairRaw<T>::fp16) {
        PROJECT_PAIR_ASM("f16");
    } else {
        PROJECT_PAIR_ASM("bf16");
    }
#undef PROJECT_PAIR_ASM
}

template<class T, class I, bool RI, bool II>
__device__ __forceinline__ void project_fwd_reg(
    const T* features, const void* wi, const void* bi, const void* wo, const void* bo,
    T* real, T* imag, I cells, I plane, int M, int types
) {
    const int m = blockIdx.x * 32 + threadIdx.x % 32;
    if (m >= M) return;
    const auto weights = load_register_path_weights<T, I, false>(wi, bi, wo, types, m, true);
    const float2 bias = make_float2(rounded<T>(load_scalar(bo, 2 * I(m), types >> 6)),
                                    rounded<T>(load_scalar(bo, 2 * I(m) + 1, types >> 6)));
    const I warp = I(blockIdx.y) * 4 + threadIdx.x / 32;
    auto load = [&](I cell, float (&p)[8]) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) p[k] = read(features[I(k) * plane + cell * M + m]);
    };
    auto emit = [&](I cell, float2 out) {
        real[cell * M * (1 + RI) + m] = write<T>(out.x);
        imag[cell * M * (1 + II) + m] = write<T>(out.y);
    };
    #pragma unroll 1
    for (I first = 2 * warp; first < cells; first += I(gridDim.y) * 8) {
        float p[2][8];
        const bool second = first + 1 < cells;
        load(first, p[0]);
        load(first + second, p[1]);
        float2 out[2]{};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float z[2]{};
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                project_pair(weights.pairs[4 * j + k], p[0][2 * k], p[0][2 * k + 1],
                             p[1][2 * k], p[1][2 * k + 1], z[0], z[1]);
            const float b = weights.bias(j), wr = weights.output(j), wi = weights.output(8 + j);
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const float pre = rounded<T>(rounded<T>(z[c]) + b);
                const float sig = __fdividef(1.0f, 1.0f + __expf(-pre));
                const float act = rounded<T>(pre * sig);
                out[c].x += wr * act;
                out[c].y += wi * act;
            }
        }
        emit(first, make_float2(rounded<T>(out[0].x) + bias.x, rounded<T>(out[0].y) + bias.y));
        if (second) {
            emit(first + 1,
                 make_float2(rounded<T>(out[1].x) + bias.x, rounded<T>(out[1].y) + bias.y));
        }
    }
}

template<class T, class I, bool RI, bool II, int K>
__device__ __forceinline__ void project_bwd_k8(
    const T* features, const void* wi, const void* bi, const void* wo,
    const T* gr, const T* gi, T* dp, float* partial, I cells, I plane, int M, int types
) {
    __shared__ T pbuf[K][8][32], dybuf[K][2][32];
    __shared__ float dbuf[4][K][8][32];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int m = blockIdx.x * 32 + lane;
    const int h0 = 2 * warp;
    Pair<T> weight[2][4];
    float bias[2]{}, wr[2]{}, wi_out[2]{};
    float dw[2][8]{}, db[2]{}, dwr[2]{}, dwi[2]{};
    float dbr = 0, dbi = 0;
    const PathWeights<T, I, false> path{wi, bi, wo, types, m, 8};
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            weight[j][k] = Pair<T>::pack(m < M ? path.input_pair(4 * (h0 + j) + k) : Complex{});
        if (m < M) {
            bias[j] = path.bias(h0 + j);
            wr[j] = path.output(h0 + j);
            wi_out[j] = path.output(8 + h0 + j);
        }
    }
    #pragma unroll 1
    for (I first = I(blockIdx.y) * K; first < cells; first += I(gridDim.y) * K) {
        #pragma unroll 1
        for (int u = 0; u < K / 4; ++u) {
            const int t = warp * (K / 4) + u;
            const I cell = first + t;
            if (cell < cells && m < M) {
                #pragma unroll
                for (int k = 0; k < 8; ++k)
                    pbuf[t][k][lane] = features[I(k) * plane + cell * M + m];
                dybuf[t][0][lane] = gr[cell * M * (1 + RI) + m];
                dybuf[t][1][lane] = gi[cell * M * (1 + II) + m];
            }
        }
        // All previous DP readers finish before this iteration writes DP.
        __syncthreads();
        if (m < M) {
            #pragma unroll 1
            for (int t = 0; t < K && first + t < cells; ++t) {
                float p[8], d[8]{};
                #pragma unroll
                for (int k = 0; k < 8; ++k) p[k] = read(pbuf[t][k][lane]);
                const float r = read(dybuf[t][0][lane]), i = read(dybuf[t][1][lane]);
                if (warp == 0) {
                    dbr += r;
                    dbi += i;
                }
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float z = 0;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const auto w = weight[j][k].unpack();
                        z = __fmaf_rn(w.r, p[2 * k], z);
                        z = __fmaf_rn(w.i, p[2 * k + 1], z);
                    }
                    z = rounded<T>(rounded<T>(z) + bias[j]);
                    const float sig = __fdividef(1.0f, 1.0f + __expf(-z));
                    const float act = rounded<T>(z * sig);
                    const float dz = rounded<T>(rounded<T>(r * wr[j] + i * wi_out[j]) *
                                                 (sig * (1.0f + z * (1.0f - sig))));
                    db[j] += dz;
                    dwr[j] += r * act;
                    dwi[j] += i * act;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const auto w = weight[j][k].unpack();
                        dw[j][2 * k] += dz * p[2 * k];
                        dw[j][2 * k + 1] += dz * p[2 * k + 1];
                        d[2 * k] += dz * w.r;
                        d[2 * k + 1] += dz * w.i;
                    }
                }
                #pragma unroll
                for (int k = 0; k < 8; ++k) dbuf[warp][t][k][lane] = d[k];
            }
        }
        // All P/DY readers finish before the next iteration writes P/DY.
        __syncthreads();
        #pragma unroll 1
        for (int u = 0; u < K / 4; ++u) {
            const int t = warp * (K / 4) + u;
            const I cell = first + t;
            if (cell < cells && m < M) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float d = dbuf[0][t][k][lane] + dbuf[1][t][k][lane]
                                  + dbuf[2][t][k][lane] + dbuf[3][t][k][lane];
                    dp[I(k) * plane + cell * M + m] = write<T>(d);
                }
            }
        }
    }
    if (m < M) {
        auto* row = partial + I(blockIdx.y) * 90 * M + m;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const I h = h0 + j;
            #pragma unroll
            for (int k = 0; k < 8; ++k) row[(8 * h + k) * M] = dw[j][k];
            row[(64 + h) * M] = db[j];
            row[(72 + h) * M] = dwr[j];
            row[(80 + h) * M] = dwi[j];
        }
        if (warp == 0) {
            row[88 * I(M)] = dbr;
            row[89 * I(M)] = dbi;
        }
    }
}
"""
