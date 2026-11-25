"""CUDA source for the opt-in four-warp VScan tile."""

CUDA = r"""
__device__ __forceinline__ unsigned w39_pack(__half lo, __half hi) {
    return __half_as_ushort(lo) | (unsigned(__half_as_ushort(hi)) << 16);
}

__device__ __forceinline__ unsigned w39_pack(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return __bfloat16_as_ushort(lo) | (unsigned(__bfloat16_as_ushort(hi)) << 16);
}

template<class T> constexpr bool w39_fp16 = false;
template<> constexpr bool w39_fp16<__half> = true;

#define W39_MMA8(DTYPE) \
    asm("mma.sync.aligned.m16n8k8.row.col.f32." DTYPE "." DTYPE ".f32 " \
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};" \
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w) : "r"(a0), "r"(a1), "r"(b))

template<class T>
__device__ __forceinline__ float4 w39_mma8(unsigned a0, unsigned a1, unsigned b) {
    float4 d{};
    if constexpr (w39_fp16<T>) { W39_MMA8("f16"); }
    else { W39_MMA8("bf16"); }
    return d;
}
#undef W39_MMA8

#define W39_MMA16(DTYPE) \
    asm("mma.sync.aligned.m16n8k16.row.col.f32." DTYPE "." DTYPE ".f32 " \
        "{%0,%1,%2,%3}, {%4,%6,%5,%6}, {%7,%8}, {%0,%1,%2,%3};" \
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w) \
        : "r"(a0), "r"(a1), "r"(0u), "r"(b0), "r"(b1))

template<class T>
__device__ __forceinline__ float4 w39_mma16(
    unsigned a0, unsigned a1, unsigned b0, unsigned b1, float2 prior
) {
    float4 d{prior.x, prior.y, 0, 0};
    if constexpr (w39_fp16<T>) { W39_MMA16("f16"); }
    else { W39_MMA16("bf16"); }
    return d;
}
#undef W39_MMA16

__device__ __forceinline__ float w39_sigmoid(float z) {
    // A flushed subnormal exponent cannot affect 1 + exp(-z). The
    // denominator is always >= 1 for finite z, so no divide fixup is needed.
    float e, sig;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(-1.4426950408889634f * z));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sig) : "f"(1.0f + e));
    return sig;
}

__device__ __forceinline__ unsigned w39_bits(Pair<__half> p) {
    return __builtin_bit_cast(unsigned, static_cast<__half2_raw>(p.value));
}

__device__ __forceinline__ unsigned w39_bits(Pair<__nv_bfloat16> p) {
    return __builtin_bit_cast(unsigned, static_cast<__nv_bfloat162_raw>(p.value));
}

__device__ __forceinline__ unsigned w39_transpose_pair(unsigned lo, unsigned hi, bool upper) {
    unsigned value;
    asm("prmt.b32 %0, %1, %2, %3;"
        : "=r"(value) : "r"(lo), "r"(hi), "r"(upper ? 0x7632 : 0x5410));
    return value;
}

template<class T, bool BWD> struct W39Tile {
    Pair<T> values[16][4][32];
    Pair<T> dy[BWD ? 16 : 1][32];
    T dz[BWD ? 4 : 1][8][16];

    // Scan lanes write one full bank word each. In an MMA fragment the
    // eight distinct rows advance four banks at a time; the pair picks
    // a distinct bank within each set of four.
    __device__ Pair<T>& p(int row, int pair, int mode) {
        return values[row][pair][(mode + 4 * row + pair) & 31];
    }

    __device__ void stage(int row, int lane, float4 pos, float4 neg) {
        p(row, 0, lane) = Pair<T>::pack({pos.x, pos.z});
        p(row, 1, lane) = Pair<T>::pack({neg.x, neg.z});
        p(row, 2, lane) = Pair<T>::pack({pos.y, pos.w});
        p(row, 3, lane) = Pair<T>::pack({neg.y, neg.w});
    }
};

__device__ __forceinline__ float w39_aux_pair(float lo, float hi, int lane_bit) {
    const bool upper = (threadIdx.x & lane_bit) != 0;
    const float local = upper ? hi : lo;
    const float remote = __shfl_xor_sync(0xffffffff, upper ? lo : hi, lane_bit);
    return local + remote;
}

struct W39Grad {
    float2 dw;
    float aux;
};

template<class T, bool BWD>
__device__ __forceinline__ void w39_path(
    W39Tile<T, BWD>& tile, int mode0, int M, const T* weights, W39Grad* grad
) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = lane >> 2, c = 2 * (lane & 3);
    const int pair = c / 2;
    const int row_bank = 4 * r + pair;
    const int transposed_bank = 4 * c + r / 2;
    #pragma unroll 1
    for (int mode = 0; mode < 8; ++mode) {
        const int m = 8 * warp + mode;
        W39Grad mode_grad{};
        if constexpr (BWD) mode_grad = grad[0];
        // Keep the loop uniform and rotate physical register slots after
        // each logical mode, including when the final mode tile is partial.
        if (mode0 + m < M) {
            const auto* w = weights + (BWD ? 88 : 90) * m;
            const int bank = (m + row_bank) & 31;
            auto& low = tile.values[r][pair][bank];
            auto& high = tile.values[r + 8][pair][bank];
            const auto bits = w39_pack(w[8 * r + c], w[8 * r + c + 1]);
            const auto pre = w39_mma8<T>(w39_bits(low), w39_bits(high), bits);
            const float bias[2] = {read(w[64 + c]), read(w[65 + c])};
            const float wr[2] = {read(w[72 + c]), read(w[73 + c])};
            const float wi[2] = {read(w[80 + c]), read(w[81 + c])};
            T dz[4];
            float2 db{}, dwr{}, dwi{}, dbo{};
            Complex upstream[2];
            if constexpr (BWD) {
                upstream[0] = tile.dy[r][m].unpack();
                upstream[1] = tile.dy[r + 8][m].unpack();
                dbo = make_float2(upstream[0].r + upstream[1].r, upstream[0].i + upstream[1].i);
            }
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = r + 8 * half;
                const float p[2] = {half ? pre.z : pre.x, half ? pre.w : pre.y};
                float out_r = 0, out_i = 0;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float z = rounded<T>(rounded<T>(p[j]) + bias[j]);
                    const float sig = w39_sigmoid(z);
                    const float act = rounded<T>(z * sig);
                    if constexpr (BWD) {
                        const auto dy = upstream[half];
                        const float d = rounded<T>(rounded<T>(dy.r * wr[j] + dy.i * wi[j]) *
                                                   (sig * (1.0f + z * (1.0f - sig))));
                        dz[2 * half + j] = write<T>(d);
                        tile.dz[warp][c + j][row ^ (c + j)] = dz[2 * half + j];
                        (j ? db.y : db.x) += d;
                        (j ? dwr.y : dwr.x) += dy.r * act;
                        (j ? dwi.y : dwi.x) += dy.i * act;
                    } else {
                        out_r += wr[j] * act;
                        out_i += wi[j] * act;
                    }
                }
                if constexpr (!BWD) {
                    out_r += __shfl_xor_sync(0xffffffff, out_r, 1);
                    out_i += __shfl_xor_sync(0xffffffff, out_i, 1);
                    out_r += __shfl_xor_sync(0xffffffff, out_r, 2);
                    out_i += __shfl_xor_sync(0xffffffff, out_i, 2);
                    if (c == 0) {
                        const float br = read(w[88]), bi = read(w[89]);
                        tile.p(row, 0, m) = Pair<T>::pack(
                            {rounded<T>(out_r) + br, rounded<T>(out_i) + bi});
                    }
                }
            }
            if constexpr (BWD) {
                // The butterfly both reduces the 16 cells and routes each
                // auxiliary field to the lane whose row number equals the field.
                const float bias_grad = w39_aux_pair(db.x, db.y, 4);
                const float real_grad = w39_aux_pair(dwr.x, dwr.y, 4);
                const float imag_grad = w39_aux_pair(dwi.x, dwi.y, 4);
                const float final_grad = w39_aux_pair(dbo.x, dbo.y, 4);
                mode_grad.aux += w39_aux_pair(w39_aux_pair(bias_grad, real_grad, 8),
                                               w39_aux_pair(imag_grad, final_grad, 8), 16);
                const auto dp = w39_mma8<T>(
                    w39_pack(dz[0], dz[1]), w39_pack(dz[2], dz[3]),
                    w39_pack(w[8 * c + r], w[8 * (c + 1) + r]));
                __syncwarp();
                const auto& dt = tile.dz[warp][r];
                const int col_bank0 = (m + transposed_bank) & 31;
                const int col_bank1 = (m + transposed_bank + 4) & 31;
                const auto dw = w39_mma16<T>(
                    w39_pack(dt[c ^ r], dt[(c + 1) ^ r]),
                    w39_pack(dt[(c + 8) ^ r], dt[(c + 9) ^ r]),
                    w39_transpose_pair(w39_bits(tile.values[c][r / 2][col_bank0]),
                        w39_bits(tile.values[c + 1][r / 2][col_bank1]), r & 1),
                    w39_transpose_pair(w39_bits(tile.values[c + 8][r / 2][col_bank0]),
                        w39_bits(tile.values[c + 9][r / 2][col_bank1]), r & 1), mode_grad.dw);
                mode_grad.dw = make_float2(dw.x, dw.y);
                __syncwarp();
                low = Pair<T>::pack({dp.x, dp.y});
                high = Pair<T>::pack({dp.z, dp.w});
            }
        }
        if constexpr (BWD) {
            #pragma unroll
            for (int slot = 0; slot < 7; ++slot) grad[slot] = grad[slot + 1];
            grad[7] = mode_grad;
        }
    }
}

template<class I>
__device__ __forceinline__ void w39_write_grad(
    const W39Grad* grad, float* partial, int mode0, int M
) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r = lane >> 2, c = 2 * (lane & 3);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = mode0 + 8 * warp + i;
        if (m >= M) continue;
        const auto g = grad[i];
        partial[I(4 + 8 * r + c) * M + m] = g.dw.x;
        partial[I(5 + 8 * r + c) * M + m] = g.dw.y;
        if (r < 6) {
            const int field = 68 + 8 * (r / 2) + c + (r & 1);
            partial[I(field) * M + m] = g.aux;
        } else if (c == 0) {
            partial[I(92 + r - 6) * M + m] = g.aux;
        }
    }
}

template<bool REVERSE, bool COMPACT, class T, class I>
__device__ __forceinline__ void w39_column_adjoint(
    Column<T, I> c, int y, float4 prev, float4 source, float4 inject,
    Complex a, Complex g, Complex& adj_a, Complex& adj_b, Complex& da, Complex& dg, T* gs
) {
    if constexpr (!COMPACT) {
        column_adjoint<REVERSE, false>(c, y, prev, source, inject, a, g, adj_a, adj_b, da, dg, gs);
    } else {
        const auto ds = column_adjoint_step<REVERSE>(
            prev, source, inject, a, g, adj_a, adj_b, da, dg);
        const I column = c.offset_b / c.M;
        const I batch = column / (I(c.H) * c.W);
        const I x = (column % c.W) / 2;
        const I plane = c.plane / c.W * c.OW;
        const I index = ((batch * c.H + y) * c.OW + x) * c.M + c.m;
        Complex left{ds.x, ds.y}, right{ds.z, ds.w};
        if constexpr (REVERSE) {
            left = Complex{rounded<T>(ds.x), rounded<T>(ds.y)} + load_complex(gs, index, plane);
            right = Complex{rounded<T>(ds.z), rounded<T>(ds.w)}
                + load_complex(gs, index + 2 * plane, plane);
        }
        store_complex(gs, index, plane, left);
        store_complex(gs, index + 2 * plane, plane, right);
    }
}

template<class T, class I, bool COARSE, int WARPS, int CHUNK, int HIDDEN, bool BWD,
         bool CACHE_WEIGHT, bool GLOBAL_CKPT, bool CACHE_LINE, bool SHARED_ACC, bool PRECISE,
         bool COMPACT, bool RI, bool II, bool CACHE_SOURCE, bool PURE, bool VECTOR_WEIGHTS,
         bool BATCH_PATH>
__device__ __forceinline__ void w39_vertical(
    const float* dr, const float* di, const float* gr_pole, const float* gi_pole,
    const T* states, const float* vx, const float* vy,
    const void* wi, const void* bi, const void* wo, const void* bo,
    const T* gr, const T* gi, T* outr, T* outi, T* gs, float* partials,
    Quad<T>*, float4* global_ckpt, int B, int H, int W, int M,
    int workers, int, int types, float eps, long long, long long
) {
    static_assert(WARPS == 4 && HIDDEN == 8 && CACHE_WEIGHT && CACHE_LINE);
    constexpr int threads = 128, step = 1 + COARSE;
    constexpr int weight_fields = BWD ? 88 : 90;
    extern __shared__ float4 shared[];
    auto& tile = *reinterpret_cast<W39Tile<T, BWD>*>(shared);
    static_assert(sizeof(tile) == (BWD ? 11264 : 8576));
    auto* weights = reinterpret_cast<T*>(&tile + 1);
    auto* seg = reinterpret_cast<float4*>(weights + weight_fields * 32);
    const int span = CHUNK ? CHUNK : H, chunks = (H - 1) / span + 1;
    auto* ckpt = seg + (BWD ? span * threads : 0);
    auto* source = reinterpret_cast<Quad<T>*>(
        ckpt + (BWD && !GLOBAL_CKPT ? (chunks - 1) * threads : 0));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    auto* g2 = source + (BWD && CACHE_SOURCE ? H * threads : 0) + threadIdx.x;
    seg += threadIdx.x;
    ckpt += threadIdx.x;
    source += threadIdx.x;
    const int tiles = (M - 1) / 32 + 1, mode0 = (blockIdx.x % tiles) * 32;
    const int m = mode0 + lane, group = blockIdx.x / tiles;
    const int ow = W / step, oh = H / step;
    const I plane = I(B) * H * W * M;
    const int valid_modes = min(32, M - mode0);
    for (int i = threadIdx.x; i < 64 * valid_modes; i += threads)
        weights[i / 64 * weight_fields + i % 64] = write<T>(
            load_scalar(wi, I(mode0) * 64 + i, types & 3));
    for (int i = threadIdx.x; i < 8 * valid_modes; i += threads)
        weights[i / 8 * weight_fields + 64 + i % 8] = write<T>(
            load_scalar(bi, I(mode0) * 8 + i, (types >> 2) & 3));
    for (int i = threadIdx.x; i < 16 * valid_modes; i += threads)
        weights[i / 16 * weight_fields + 72 + i % 16] = write<T>(
            load_scalar(wo, I(mode0) * 16 + i, (types >> 4) & 3));
    if constexpr (!BWD) {
        for (int i = threadIdx.x; i < 2 * valid_modes; i += threads)
            weights[i / 2 * weight_fields + 88 + i % 2] = write<T>(
                load_scalar(bo, I(mode0) * 2 + i, types >> 6));
    }
    __syncthreads();
    Complex a{}, g{}, da{}, dg{};
    if (m < M) {
        a = {dr[m], di[m]};
        g = {gr_pole[m], gi_pole[m]};
    }
    const auto ar = a.conj(), gg = g.conj();
    W39Grad grad[BWD ? 8 : 1]{};
    for (I first = I(group) * WARPS; first < I(B) * ow; first += workers) {
        const I line = first + warp;
        const bool active = m < M && group * WARPS + warp < workers && line < I(B) * ow;
        const I b = line / ow, x = line % ow;
        const I xa = COARSE ? 2 * x + 1 : x, xb = COARSE ? 2 * x : x;
        const I base = b * H * W * M + m;
        const Column<T, I> c{
            states, vy, base + xa * M, base + xb * M, plane, plane,
            (b * oh * ow + x) * M + m, I(B) * oh * ow * M,
            H, W, M, ow, m, active ? vx[xa * M + m] : 0,
            active ? vx[(I(W) + xb) * M + m] : 0, eps,
        };
        const I checkpoint_stride = GLOBAL_CKPT ? I(workers) * M : threads;
        auto* boundaries = GLOBAL_CKPT ? global_ckpt + (group * WARPS + warp) * I(M) + m : ckpt;
        const auto boundary = [&](int block) -> float4& {
            return boundaries[I(block) * checkpoint_stride];
        };
        float4 pos{}, neg{};
        const auto src = [&](int y) {
            if constexpr (BWD && CACHE_SOURCE) return source[y * threads].unpack();
            else return c.source(y);
        };
        if (active) {
            for (int y = 0; y < H; ++y) {
                float4 s;
                if constexpr (BWD && CACHE_SOURCE) {
                    const auto packed = c.packed_source(y);
                    source[y * threads] = packed;
                    s = packed.unpack();
                } else {
                    s = c.source(y);
                }
                if constexpr (BWD) {
                    pos = advance(pos, s, a, g);
                    if (y / span == chunks - 1) seg[(y % span) * threads] = pos;
                    else if (y % span == span - 1) boundary(y / span) = pos;
                } else {
                    pos = advance_forward<false>(pos, s, a, g);
                    if (!COARSE || (y & 1)) g2[(y / step) * threads] =
                        normalize_packed<T>(pos, c.gain(y, false));
                }
            }
        }
        Complex adj_a{}, adj_b{};
        for (int block = (BWD ? chunks : 1) - 1; block >= 0; --block) {
            const int y0 = BWD ? block * span : 0;
            const int length = BWD ? min(span, H - y0) : H;
            const auto before = BWD && active && block ? boundary(block - 1) : float4{};
            if constexpr (BWD && CHUNK) {
                if (active && block + 1 < chunks) {
                    boundary(block) = neg;
                    pos = before;
                    for (int j = 0; j < length; ++j) {
                        pos = advance(pos, src(y0 + j), a, g);
                        seg[j * threads] = pos;
                    }
                }
            }
            int end = length - 1;
            if constexpr (COARSE) {
                if ((y0 + end) % 2 == 0) {
                    if (active) {
                        const auto s = src(y0 + end);
                        neg = advance(neg, s, ar, gg);
                        if constexpr (BWD) {
                            w39_column_adjoint<false, COMPACT>(
                                c, y0 + end, end ? seg[(end - 1) * threads] : before,
                                s, {}, a, g, adj_a, adj_b, da, dg, gs);
                        }
                    }
                    --end;
                }
            }
            for (; end >= 0; end -= 4 * step) {
                const int count = min(4, (end + 1) / step);
                #pragma unroll 1
                for (int i = 0; i < 4; ++i) {
                    const int row = 4 * warp + i, j = end - i * step, y = y0 + j;
                    float4 p{}, n{};
                    Pair<T> dy{};
                    if (active && i < count) {
                        neg = advance(neg, src(y), ar, gg);
                        if constexpr (COARSE) {
                            neg = advance(neg, src(y - 1), ar, gg);
                        }
                        if constexpr (BWD) {
                            const I cell = c.cell(y / step);
                            dy = Pair<T>::from(gr[(cell - m) * (1 + RI) + m],
                                               gi[(cell - m) * (1 + II) + m]);
                            p = normalize<T>(seg[j * threads], c.gain(y, false));
                        } else {
                            p = g2[(y / step) * threads].unpack();
                        }
                        n = normalize<T>(neg, c.gain(y - COARSE, true));
                    }
                    tile.stage(row, lane, p, n);
                    if constexpr (BWD) tile.dy[row][lane] = dy;
                }
                __syncthreads();
                w39_path(tile, mode0, M, weights, grad);
                __syncthreads();
                if (active) {
                    #pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        if (i >= count) break;
                        const int row = 4 * warp + i, j = end - i * step, y = y0 + j;
                        if constexpr (BWD) {
                            const auto dp0 = tile.p(row, 0, lane).unpack();
                            const auto dp1 = tile.p(row, 1, lane).unpack();
                            const auto dp2 = tile.p(row, 2, lane).unpack();
                            const auto dp3 = tile.p(row, 3, lane).unpack();
                            const auto ip = normalize<T>(
                                make_float4(dp0.r, dp2.r, dp0.i, dp2.i), c.gain(y, false));
                            g2[(y / step) * threads] = normalize_packed<T>(
                                make_float4(dp1.r, dp3.r, dp1.i, dp3.i), c.gain(y - COARSE, true));
                            w39_column_adjoint<false, COMPACT>(
                                c, y, j ? seg[(j - 1) * threads] : before, src(y), ip,
                                a, g, adj_a, adj_b, da, dg, gs);
                            if constexpr (COARSE) w39_column_adjoint<false, COMPACT>(
                                c, y - 1, j > 1 ? seg[(j - 2) * threads] : before, src(y - 1),
                                {}, a, g, adj_a, adj_b, da, dg, gs);
                        } else {
                            const I cell = c.cell(y / step);
                            const auto out = tile.p(row, 0, lane).unpack();
                            outr[(cell - m) * (1 + RI) + m] = write<T>(out.r);
                            outi[(cell - m) * (1 + II) + m] = write<T>(out.i);
                        }
                    }
                }
            }
        }
        if constexpr (BWD) {
            if (active) {
                adj_a = adj_b = {};
                for (int block = 0; block < chunks; ++block) {
                    const int y0 = block * span, length = min(span, H - y0);
                    neg = block + 1 < chunks ? boundary(block) : float4{};
                    for (int j = length - 1; j >= 0; --j) {
                        seg[j * threads] = neg;
                        neg = advance(neg, src(y0 + j), ar, gg);
                    }
                    for (int j = 0; j < length; ++j) {
                        const int y = y0 + j;
                        const auto in = !COARSE || (!(y & 1) && y + 1 < H)
                            ? g2[(y / step) * threads].unpack() : float4{};
                        w39_column_adjoint<true, COMPACT>(c, y, seg[j * threads], src(y), in,
                            ar, gg, adj_a, adj_b, da, dg, gs);
                        if constexpr (COARSE && !COMPACT) {
                            clear_unused_columns<false>(c, y, gs);
                            if ((W & 1) && x + 1 == ow) {
                                const I tail = (b * H * W + I(y + 1) * W - 1) * M + m;
                                store_complex(gs, tail, plane, {});
                                store_complex(gs, tail + 2 * plane, plane, {});
                            }
                        }
                    }
                }
            }
        }
    }
    if constexpr (BWD) {
        seg[0] = make_float4(da.r, da.i, dg.r, dg.i);
        __syncthreads();
        auto* row = partials + I(group) * 94 * M;
        if (warp == 0 && m < M) {
            float4 sum{};
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                const auto p = seg[32 * w];
                sum.x += p.x;
                sum.y += p.y;
                sum.z += p.z;
                sum.w += p.w;
            }
            row[m] = sum.x;
            row[I(M) + m] = sum.y;
            row[2 * I(M) + m] = sum.z;
            row[3 * I(M) + m] = sum.w;
        }
        w39_write_grad<I>(grad, row, mode0, M);
    }
}
"""
