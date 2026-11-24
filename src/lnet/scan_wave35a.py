# ruff: noqa: E501
"""CUDA kernels selected by the Wave 35A planner."""

HORIZONTAL_VEC2 = r"""
struct ComplexModes {
    Complex lo, hi;
    __device__ ComplexModes operator+(ComplexModes z) const { return {lo + z.lo, hi + z.hi}; }
    __device__ ComplexModes operator*(ComplexModes z) const { return {lo * z.lo, hi * z.hi}; }
    __device__ ComplexModes conj() const { return {lo.conj(), hi.conj()}; }
};

template<class T, class I>
__device__ __forceinline__ ComplexModes load_modes(const T* base, I i, I stride) {
    const auto real = reinterpret_cast<const Pair<T>*>(base + i)->unpack();
    const auto imag = reinterpret_cast<const Pair<T>*>(base + i + stride)->unpack();
    return {{real.r, imag.r}, {real.i, imag.i}};
}

template<class T, class I>
__device__ __forceinline__ void store_modes(T* base, I i, I stride, ComplexModes z) {
    *reinterpret_cast<Pair<T>*>(base + i) = Pair<T>::pack({z.lo.r, z.hi.r});
    *reinterpret_cast<Pair<T>*>(base + i + stride) = Pair<T>::pack({z.lo.i, z.hi.i});
}

__device__ __forceinline__ Complex hscan_modes_grad(
    Complex left, Complex right, Complex prev, Complex next, Complex s, Complex g,
    Complex& da, Complex& dg
) {
    da.r += left.r * prev.r + left.i * prev.i + right.r * next.r + right.i * next.i;
    da.i += left.i * prev.r - left.r * prev.i + right.r * next.i - right.i * next.r;
    const Complex sum = left + right, diff{left.r - right.r, left.i - right.i};
    dg.r += sum.r * s.r + sum.i * s.i;
    dg.i += diff.i * s.r - diff.r * s.i;
    return {sum.r * g.r + diff.i * g.i, sum.i * g.r - diff.r * g.i};
}

template<class T, class I, int WARPS, bool BACKWARD, bool COMPACT, bool RI, bool II, int CHUNK>
__device__ __forceinline__ void horizontal_vec2(
    const float* dr, const float* di, const float* gr, const float* gi,
    const T* sr, const T* si, const T* states, const T* grad, T* out, float* partial,
    int B, int H, int W, int M, int workers
) {
    extern __shared__ float4 shared[];
    constexpr int threads = WARPS * 32;
    const int tiles = (M - 1) / 64 + 1;
    const int m = 64 * (blockIdx.x % tiles) + 2 * (threadIdx.x & 31);
    const int group = blockIdx.x / tiles, warp = threadIdx.x / 32;
    const int worker = group * WARPS + warp;
    const I plane = I(B) * H * W * M;
    ComplexModes da{}, dg{};
    if (m < M && worker < workers) {
        const ComplexModes a{{dr[m], di[m]}, {dr[m + 1], di[m + 1]}};
        const ComplexModes g{{gr[m], gi[m]}, {gr[m + 1], gi[m + 1]}};
        const auto ar = a.conj(), gg = g.conj();
        for (I line = worker; line < I(B) * H; line += workers) {
            const I base = line * W * M;
            auto source = [&](int x) {
                const I cell = base + I(x) * M;
                const auto r = reinterpret_cast<const Pair<T>*>(sr + cell * (1 + RI) + m)->unpack();
                const auto i = reinterpret_cast<const Pair<T>*>(si + cell * (1 + II) + m)->unpack();
                return ComplexModes{{r.r, i.r}, {r.i, i.i}};
            };
            if constexpr (!BACKWARD) {
                ComplexModes pos{}, neg{};
                for (int x = 0; x < W; ++x) {
                    pos = a * pos + g * source(x);
                    store_modes(out, base + I(x) * M + m, plane, pos);
                }
                for (int x = W - 1; x >= 0; --x) {
                    neg = ar * neg + gg * source(x);
                    store_modes(out, 2 * plane + base + I(x) * M + m, plane, neg);
                }
            } else {
                const I gp = plane / (1 + COMPACT);
                auto incoming = [&](int x, bool reverse) -> ComplexModes {
                    if (COMPACT && ((x & 1) == reverse)) return {};
                    const I i = (base / (1 + COMPACT)) + I(x / (1 + COMPACT)) * M + m;
                    return load_modes(grad, i + (reverse ? 2 * gp : 0), gp);
                };
                auto step = [&](ComplexModes right, int x) {
                    if constexpr (COMPACT) {
                        if (x & 1) return a * right;
                    }
                    return a * right + incoming(x, true);
                };
                // Store boundaries of the first three chunks in registers.
                ComplexModes c1{}, c2{}, c3{}, left{}, neg{};
                const int last = (W - 1) / CHUNK;
                ComplexModes prefix{};
                for (int x = 0; x < last * CHUNK; ++x) {
                    prefix = step(prefix, x);
                    if (x + 1 == CHUNK) c1 = prefix;
                    if (x + 1 == 2 * CHUNK) c2 = prefix;
                    if (x + 1 == 3 * CHUNK) c3 = prefix;
                }
                auto* trajectory = reinterpret_cast<ComplexModes*>(shared) + threadIdx.x;
                for (int block = last; block >= 0; --block) {
                    const int start = block * CHUNK, end = min(W, start + CHUNK);
                    auto right = block == 3 ? c3 : block == 2 ? c2 : block == 1 ? c1 : ComplexModes{};
                    for (int x = start; x < end; ++x) {
                        right = step(right, x);
                        trajectory[(x - start) * threads] = right;
                    }
                    for (int x = end - 1; x >= start; --x) {
                        const I i = base + I(x) * M + m;
                        left = ar * left + incoming(x, false);
                        right = trajectory[(x - start) * threads];
                        const auto prev = x ? load_modes(states, i - M, plane) : ComplexModes{};
                        const auto s = source(x);
                        const ComplexModes next{Pair<T>::pack(neg.lo).unpack(),
                                                Pair<T>::pack(neg.hi).unpack()};
                        neg = ar * neg + gg * s;
                        const ComplexModes ds{
                            hscan_modes_grad(left.lo, right.lo, prev.lo, next.lo, s.lo, g.lo, da.lo, dg.lo),
                            hscan_modes_grad(left.hi, right.hi, prev.hi, next.hi, s.hi, g.hi, da.hi, dg.hi)
                        };
                        store_modes(out, 2 * (i - m) + m, I(M), ds);
                    }
                }
            }
        }
    }
    if constexpr (BACKWARD) {
        auto* poles = reinterpret_cast<ComplexModes*>(shared);
        __syncthreads();
        poles[threadIdx.x] = da;
        poles[threads + threadIdx.x] = dg;
        __syncthreads();
        if (warp == 0 && m < M) {
            #pragma unroll
            for (int w = 1; w < WARPS; ++w) {
                da = da + poles[threadIdx.x + w * 32];
                dg = dg + poles[threads + threadIdx.x + w * 32];
            }
            const I row = 4 * I(group) * M + m;
            *reinterpret_cast<float2*>(partial + row) = make_float2(da.lo.r, da.hi.r);
            *reinterpret_cast<float2*>(partial + row + M) = make_float2(da.lo.i, da.hi.i);
            *reinterpret_cast<float2*>(partial + row + 2 * I(M)) = make_float2(dg.lo.r, dg.hi.r);
            *reinterpret_cast<float2*>(partial + row + 3 * I(M)) = make_float2(dg.lo.i, dg.hi.i);
        }
    }
}
"""

HIDDEN_SPLIT = r"""
template<class T> struct HiddenPair {
    Pair<T> weights[2][4];
    float bias[2], wr[2], wi[2];
    float dweights[2][8]{}, dbias[2]{}, dwr[2]{}, dwi[2]{};

    template<class Weights>
    __device__ __forceinline__ void load(Weights source, int warp) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int h = 2 * warp + j;
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                weights[j][k] = Pair<T>::pack(source.input_pair(4 * h + k));
            bias[j] = source.bias(h);
            wr[j] = source.output(h);
            wi[j] = source.output(8 + h);
        }
    }

    template<bool BACKWARD>
    __device__ __forceinline__ float2 apply(const float (&p)[8], float2 dy, float (&dp)[8]) {
        float2 out{};
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float pre = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const auto w = weights[j][k].unpack();
                pre = __fmaf_rn(w.r, p[2 * k], pre);
                pre = __fmaf_rn(w.i, p[2 * k + 1], pre);
            }
            const float z = rounded<T>(rounded<T>(pre) + bias[j]);
            const float sig = __fdividef(1.f, 1.f + __expf(-z));
            const float act = rounded<T>(z * sig);
            if constexpr (BACKWARD) {
                const float dz = rounded<T>(rounded<T>(dy.x * wr[j] + dy.y * wi[j]) *
                                            (sig * (1.f + z * (1.f - sig))));
                dbias[j] += dz;
                dwr[j] += dy.x * act;
                dwi[j] += dy.y * act;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const auto w = weights[j][k].unpack();
                    dp[2 * k] += dz * w.r;
                    dp[2 * k + 1] += dz * w.i;
                    dweights[j][2 * k] += dz * p[2 * k];
                    dweights[j][2 * k + 1] += dz * p[2 * k + 1];
                }
            } else {
                out.x += wr[j] * act;
                out.y += wi[j] * act;
            }
        }
        return out;
    }

    __device__ __forceinline__ void store(float* row, int M, int warp) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int h = 2 * warp + j;
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                row[(8 * h + k) * static_cast<long long>(M)] = dweights[j][k];
            row[(64 + h) * static_cast<long long>(M)] = dbias[j];
            row[(72 + h) * static_cast<long long>(M)] = dwr[j];
            row[(80 + h) * static_cast<long long>(M)] = dwi[j];
        }
    }
};

template<class I>
__device__ __forceinline__ I aosoa_offset(I cell, int m, int M, int k) {
    const int tiles = (M - 1) / 32 + 1;
    return ((cell * tiles + m / 32) * 8 + k) * 32 + (m & 31);
}

template<class T, class I, bool BACKWARD, bool RI, bool II>
__device__ __forceinline__ void project_path_aosoa(
    const T* features, const void* wi, const void* bi, const void* wo, const void* bo,
    const T* gr, const T* gi, T* real, T* imag, T* dp_out, float* partial,
    I cells, int M, int types
) {
    extern __shared__ float4 shared[];
    auto* stage = reinterpret_cast<float*>(shared);
    const int tiles = (M - 1) / 32 + 1;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int m = 32 * (blockIdx.x % tiles) + lane;
    const int group = blockIdx.x / tiles, groups = gridDim.x / tiles;
    auto* cached = reinterpret_cast<T*>(shared);
    const int mode0 = m - lane;
    load_path_weights<T, I>(cached, wi, bi, wo, types, mode0, M, 8, 128);
    __syncthreads();
    HiddenPair<T> path;
    if (m < M) path.load(path_weights<T, I, true>(cached, wi, bi, wo, types, m, 8, lane), warp);
    float2 bias{};
    if constexpr (!BACKWARD) {
        if (warp == 0 && m < M) {
            bias = make_float2(rounded<T>(load_scalar(bo, 2 * I(m), types >> 6)),
                               rounded<T>(load_scalar(bo, 2 * I(m) + 1, types >> 6)));
        }
    }
    __syncthreads();
    float2 db{};
    #pragma unroll (BACKWARD ? 1 : 4)
    for (I cell = group; cell < cells; cell += groups) {
        if (m < M) {
            float p[8], dp[8]{};
            #pragma unroll
            for (int k = 0; k < 8; ++k) p[k] = read(features[aosoa_offset(cell, m, M, k)]);
            float2 dy{};
            if constexpr (BACKWARD) {
                dy = make_float2(read(gr[cell * M * (1 + RI) + m]),
                                 read(gi[cell * M * (1 + II) + m]));
                if (warp == 0) { db.x += dy.x; db.y += dy.y; }
            }
            const auto out = path.template apply<BACKWARD>(p, dy, dp);
            if constexpr (BACKWARD) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) stage[(warp * 8 + k) * 32 + lane] = dp[k];
            } else {
                stage[warp * 64 + lane] = out.x;
                stage[warp * 64 + 32 + lane] = out.y;
            }
        }
        __syncthreads();
        if (m < M) {
            if constexpr (BACKWARD) {
                // Each warp drains two features, with adjacent lanes owning adjacent modes.
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int k = 2 * warp + j;
                    const auto* p = stage + k * 32 + lane;
                    dp_out[aosoa_offset(cell, m, M, k)] = write<T>(p[0] + p[256] + p[512] + p[768]);
                }
            } else if (warp == 0) {
                const auto* p = stage + lane;
                real[cell * M * (1 + RI) + m] = write<T>(rounded<T>(p[0] + p[64] + p[128] + p[192]) + bias.x);
                imag[cell * M * (1 + II) + m] =
                    write<T>(rounded<T>(p[32] + p[96] + p[160] + p[224]) + bias.y);
            }
        }
        __syncthreads();
    }
    if constexpr (BACKWARD) {
        if (m < M) {
            float* row = partial + I(group) * 90 * M + m;
            path.store(row, M, warp);
            if (warp == 0) {
                row[88 * I(M)] = db.x;
                row[89 * I(M)] = db.y;
            }
        }
    }
}
"""

COOPERATIVE_VERTICAL = r"""
template<class T, class I, bool COARSE, bool COMPACT, bool RI, bool II, int CHUNK>
__device__ __forceinline__ void vertical_cooperative(
    const float* dr, const float* di, const float* dgr, const float* dgi,
    const T* states, const float* vx, const float* vy,
    const void* wi, const void* bi, const void* wo,
    const T* gr, const T* gi, T* gs, float* partials,
    int B, int H, int W, int M, int workers, int types, float eps
) {
    extern __shared__ float4 shared[];
    constexpr int threads = 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, tid = threadIdx.x;
    const int tiles = (M - 1) / 32 + 1;
    const int m = 32 * (blockIdx.x % tiles) + lane, group = blockIdx.x / tiles;
    const int ow = W / (1 + COARSE), oh = H / (1 + COARSE);
    const int chunks = (H - 1) / CHUNK + 1;
    const I plane = I(B) * H * W * M, lines = I(B) * ow;
    auto* segment = shared;
    auto* boundaries = segment + CHUNK * threads;
    auto* inject = reinterpret_cast<Quad<T>*>(boundaries + (chunks - 1) * threads);
    auto* stage = reinterpret_cast<Pair<T>*>(inject + oh * threads);
    auto* parts = reinterpret_cast<float*>(stage + 5 * threads);
    const bool mode = m < M;
    Complex a{}, g{}, da{}, dg{};
    HiddenPair<T> path;
    if (mode) {
        a = {dr[m], di[m]};
        g = {dgr[m], dgi[m]};
        path.load(PathWeights<T, I, false>{wi, bi, wo, types, m, 8}, warp);
    }
    const auto ar = a.conj(), gg = g.conj();
    float2 db{};
    for (I first = 4 * I(group); first < lines; first += workers) {
        const I line = first + warp;
        const bool active = mode && line < lines;
        const I b = line / ow;
        const int x = line % ow, xa = COARSE ? 2 * x + 1 : x, xb = COARSE ? 2 * x : x;
        const I base = b * H * W * M + m;
        Column<T, I> c{states, vy, base + I(xa) * M, base + I(xb) * M,
                       plane, plane, (b * oh * ow + x) * M + m, 0,
                       H, W, M, ow, m, 0.f, 0.f, eps};
        float4 pos{}, neg{};
        if (active) {
            c.vx_a = vx[I(xa) * M + m];
            c.vx_b = vx[(I(W) + xb) * M + m];
            for (int y = 0; y < (chunks - 1) * CHUNK; ++y) {
                pos = advance(pos, c.source(y), a, g);
                if ((y + 1) % CHUNK == 0) boundaries[(y / CHUNK) * threads + tid] = pos;
            }
        }
        Complex adj_a{}, adj_b{};
        for (int block = chunks - 1; block >= 0; --block) {
            const int y0 = block * CHUNK, length = min(CHUNK, H - y0);
            float4 before{};
            if (active) {
                if (block) before = boundaries[(block - 1) * threads + tid];
                if (block + 1 < chunks) boundaries[block * threads + tid] = neg;
                pos = before;
                for (int j = 0; j < length; ++j) {
                    pos = advance(pos, c.source(y0 + j), a, g);
                    segment[j * threads + tid] = pos;
                }
            }
            for (int j = length - 1; j >= 0; j -= 1 + COARSE) {
                const int y = y0 + j, yo = y / (1 + COARSE);
                if (active) {
                    neg = advance(neg, c.source(y), ar, gg);
                    if constexpr (COARSE) neg = advance(neg, c.source(y - 1), ar, gg);
                    const auto pp = normalize_packed<T>(segment[j * threads + tid], c.gain(y, false));
                    const auto pn = normalize_packed<T>(neg, c.gain(y - int(COARSE), true));
                    const I cell = c.cell(yo);
                    stage[tid] = pp.a;
                    stage[threads + tid] = pp.b;
                    stage[2 * threads + tid] = pn.a;
                    stage[3 * threads + tid] = pn.b;
                    stage[4 * threads + tid] = Pair<T>::from(
                        gr[(cell - m) * (1 + RI) + m], gi[(cell - m) * (1 + II) + m]);
                }
                __syncthreads();
                if (mode) {
                    #pragma unroll 1
                    for (int col = 0; col < 4; ++col) {
                        if (first + col >= lines) break;
                        const int t = col * 32 + lane;
                        const auto p0 = stage[t].unpack(), p1 = stage[threads + t].unpack();
                        const auto p2 = stage[2 * threads + t].unpack();
                        const auto p3 = stage[3 * threads + t].unpack();
                        const auto d = stage[4 * threads + t].unpack();
                        if (warp == 0) { db.x += d.r; db.y += d.i; }
                        const float p[8] = {p0.r, p1.r, p2.r, p3.r, p0.i, p1.i, p2.i, p3.i};
                        float dp[8]{};
                        path.template apply<true>(p, make_float2(d.r, d.i), dp);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) parts[(warp * 8 + k) * threads + t] = dp[k];
                    }
                }
                __syncthreads();
                if (active) {
                    float dp[8];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const auto* p = parts + k * threads + tid;
                        dp[k] = p[0] + p[8 * threads] + p[16 * threads] + p[24 * threads];
                    }
                    const auto ip = normalize<T>(make_float4(dp[0], dp[4], dp[1], dp[5]),
                                                 c.gain(y, false));
                    inject[yo * threads + tid] = normalize_packed<T>(
                        make_float4(dp[2], dp[6], dp[3], dp[7]), c.gain(y - int(COARSE), true));
                    column_adjoint<false, COMPACT>(
                        c, y, j ? segment[(j - 1) * threads + tid] : before, c.source(y),
                        ip, a, g, adj_a, adj_b, da, dg, gs);
                    if constexpr (COARSE) {
                        column_adjoint<false, COMPACT>(
                            c, y - 1, j > 1 ? segment[(j - 2) * threads + tid] : before,
                            c.source(y - 1), {}, a, g, adj_a, adj_b, da, dg, gs);
                    }
                }
                // All readers must finish before the next row replaces the staging buffers.
                __syncthreads();
            }
        }
        if (active) {
            adj_a = adj_b = {};
            for (int block = 0; block < chunks; ++block) {
                const int y0 = block * CHUNK, length = min(CHUNK, H - y0);
                neg = block + 1 < chunks ? boundaries[block * threads + tid] : float4{};
                for (int j = length - 1; j >= 0; --j) {
                    segment[j * threads + tid] = neg;
                    neg = advance(neg, c.source(y0 + j), ar, gg);
                }
                for (int j = 0; j < length; ++j) {
                    const int y = y0 + j;
                    const auto in = !COARSE || !(y & 1)
                        ? inject[(y / (1 + COARSE)) * threads + tid].unpack() : float4{};
                    column_adjoint<true, COMPACT>(
                        c, y, segment[j * threads + tid], c.source(y), in,
                        ar, gg, adj_a, adj_b, da, dg, gs);
                    if constexpr (COARSE) clear_unused_columns<COMPACT>(c, y, gs);
                }
            }
        }
    }
    if (mode) {
        float* row = partials + I(group) * 94 * M + m;
        path.store(row + 4 * I(M), M, warp);
        if (warp == 0) {
            row[92 * I(M)] = db.x;
            row[93 * I(M)] = db.y;
        }
    }
    shared[tid] = make_float4(da.r, da.i, dg.r, dg.i);
    __syncthreads();
    if (mode && warp == 0) {
        auto v = shared[tid];
        #pragma unroll
        for (int w = 1; w < 4; ++w) {
            const auto q = shared[w * 32 + lane];
            v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
        }
        float* row = partials + I(group) * 94 * M + m;
        row[0] = v.x; row[M] = v.y; row[2 * I(M)] = v.z; row[3 * I(M)] = v.w;
    }
}
"""
