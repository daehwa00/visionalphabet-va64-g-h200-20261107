"""NVRTC device code for the mode-shared vertical scan adjoint."""

SOURCE = r"""
// PTX ISA, row.col: g = lane >> 2, t = lane & 3, i = scalar within fragment.
// A8:  (g + 8*(i/2), 2*t + i%2); B8: (2*t + i, g).
// A16: (g + 8*((i/2)%2), 2*t + i%2 + 8*(i/4));
// B16: (2*t + i%2 + 8*(i/2), g).
// C/D: (g + 8*(i/2), 2*t + i%2). Packed inputs store the first scalar low.
template<class T> constexpr bool tc_half = false;
template<> constexpr bool tc_half<__half> = true;
template<class T>
__device__ unsigned tc_pack(T a, T b) {
    if constexpr (tc_half<T>) {
        return __half_as_ushort(a) | (unsigned(__half_as_ushort(b)) << 16);
    } else {
        return __bfloat16_as_ushort(a) | (unsigned(__bfloat16_as_ushort(b)) << 16);
    }
}

template<class T>
__device__ void tc_mma8(float (&d)[4], unsigned a0, unsigned a1, unsigned b) {
#define TC_MMA8(TYPE) \
    asm("mma.sync.aligned.m16n8k8.row.col.f32." TYPE "." TYPE ".f32 " \
        "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};" \
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b))
    if constexpr (tc_half<T>) { TC_MMA8("f16"); }
    else { TC_MMA8("bf16"); }
#undef TC_MMA8
}

template<class T>
__device__ void tc_mma16(float (&d)[2], unsigned a0, unsigned a2, unsigned b0, unsigned b1) {
    // The bottom 8 output rows multiply zero; discard their accumulators.
#define TC_MMA16(TYPE) \
    asm("{ .reg .f32 unused0, unused1; " \
        "mma.sync.aligned.m16n8k16.row.col.f32." TYPE "." TYPE ".f32 " \
        "{%0,%1,unused0,unused1},{%2,%3,%4,%3},{%5,%6},{%0,%1,0f00000000,0f00000000}; }" \
        : "+f"(d[0]), "+f"(d[1]) : "r"(a0), "r"(0), "r"(a2), "r"(b0), "r"(b1))
    if constexpr (tc_half<T>) { TC_MMA16("f16"); }
    else { TC_MMA16("bf16"); }
#undef TC_MMA16
}

template<class T>
struct TcPathTile {
    // Odd strides distribute column-major fragment accesses over banks.
    T p[16][10], dz[16][10];
    float dp[16][9];
    float2 dy[16];
};

static_assert(sizeof(TcPathTile<__half>) == 1344);

struct TcAcc {
    float wi[2], bi[2], wr[2], wj[2], br, bj;
};

template<class T>
__device__ __forceinline__ float4 tc_path(
    TcPathTile<T>& tile, int lane, float4 pos, float4 neg, float2 dy,
    unsigned w_forward, unsigned w_backward, float2 bias, float2 wr, float2 wj,
    TcAcc& acc, float4& dp_neg
) {
    if (lane < 16) {
        const float p[8] = {pos.x, pos.z, neg.x, neg.z, pos.y, pos.w, neg.y, neg.w};
        #pragma unroll
        for (int i = 0; i < 8; ++i) tile.p[lane][i] = write<T>(p[i]);
        tile.dy[lane] = dy;
        acc.br += dy.x;
        acc.bj += dy.y;
    }
    __syncwarp();
    const int g = lane >> 2, k = 2 * (lane & 3);
    float dz[4]{};
    tc_mma8<T>(dz, tc_pack(tile.p[g][k], tile.p[g][k + 1]),
               tc_pack(tile.p[g + 8][k], tile.p[g + 8][k + 1]), w_forward);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = g + 8 * (i / 2), j = i % 2;
        const float real = j ? wr.y : wr.x, imag = j ? wj.y : wj.x;
        const float z = rounded<T>(rounded<T>(dz[i]) + (j ? bias.y : bias.x));
        const float sig = __fdividef(1.0f, 1.0f + __expf(-z));
        const float act = rounded<T>(z * sig);
        const auto d = tile.dy[row];
        const float v = rounded<T>(rounded<T>(d.x * real + d.y * imag)
                                  * (sig * (1.0f + z * (1.0f - sig))));
        dz[i] = v;
        tile.dz[row][k + j] = write<T>(v);
        acc.bi[j] += v;
        acc.wr[j] += d.x * act;
        acc.wj[j] += d.y * act;
    }
    float dp[4]{};
    tc_mma8<T>(dp, tc_pack(write<T>(dz[0]), write<T>(dz[1])),
               tc_pack(write<T>(dz[2]), write<T>(dz[3])), w_backward);
    __syncwarp();
    tc_mma16<T>(acc.wi,
        tc_pack(tile.dz[k][g], tile.dz[k + 1][g]),
        tc_pack(tile.dz[k + 8][g], tile.dz[k + 9][g]),
        tc_pack(tile.p[k][g], tile.p[k + 1][g]),
        tc_pack(tile.p[k + 8][g], tile.p[k + 9][g]));
    #pragma unroll
    for (int i = 0; i < 4; ++i) tile.dp[g + 8 * (i / 2)][k + i % 2] = dp[i];
    __syncwarp();
    float4 dp_pos{};
    if (lane < 16) {
        const float* p = tile.dp[lane];
        dp_pos = make_float4(p[0], p[4], p[1], p[5]);
        dp_neg = make_float4(p[2], p[6], p[3], p[7]);
    }
    __syncwarp();
    return dp_pos;
}

template<class T, class I, bool COARSE>
__device__ Column<T, I> tc_column(
    const T* states, const float* vy, I line, int mode, int B, int H, int W,
    int M, int ow, int oh, I plane, float eps
) {
    const I batch = line / ow;
    const int x = line % ow, xa = COARSE ? 2 * x + 1 : x, xb = COARSE ? 2 * x : x;
    const I base = batch * H * W * M + mode;
    return {states, vy, base + I(xa) * M, base + I(xb) * M, plane, plane,
            (batch * oh * ow + x) * M + mode, I(B) * oh * ow * M,
            H, W, M, ow, mode, 0, 0, eps};
}

template<class T, class I, bool COMPACT, bool COARSE, bool REVERSE>
__device__ void tc_store(Column<T, I> c, int y, Quad<T> gradient, T* gs) {
    write_column_gradient<REVERSE, COMPACT, REVERSE>(c, y, gradient.unpack(), gs);
    if constexpr (COARSE && !COMPACT && REVERSE) {
        clear_unused_columns<false>(c, y, gs);
        if ((c.W & 1) && (c.offset_b / c.M) % c.W == c.W - 3) {
            const I tail = I(y) * c.W * c.M + c.offset_b + 2 * I(c.M);
            store_complex(gs, tail, c.plane, {});
            store_complex(gs, tail + 2 * c.plane, c.plane, {});
        }
    }
}

template<int WARPS, class T, class I>
__device__ float4 tc_source(
    Quad<T>* stage, Column<T, I> input, bool valid, int y, int lane, int warp
) {
    const int index = threadIdx.x;
    if (index < 16 * WARPS)
        stage[(index / WARPS) * (WARPS + 1) + index % WARPS] =
            valid ? input.packed_source(y) : Quad<T>{};
    __syncthreads();
    float4 result{};
    if (lane < 16) result = stage[lane * (WARPS + 1) + warp].unpack();
    __syncthreads();
    return result;
}

template<int WARPS, class T, class I, bool COMPACT, bool COARSE, bool REVERSE>
__device__ void tc_write(
    Quad<T>* stage, Column<T, I> output, bool valid, int y, int lane, int warp, float4 ds, T* gs
) {
    if (lane < 16) stage[lane * (WARPS + 1) + warp] = Quad<T>::pack(ds);
    __syncthreads();
    const int index = threadIdx.x;
    if (index < 16 * WARPS && valid)
        tc_store<T, I, COMPACT, COARSE, REVERSE>(
            output, y, stage[(index / WARPS) * (WARPS + 1) + index % WARPS], gs);
    __syncthreads();
}

template<class T, class I, int WARPS, bool COARSE, bool COMPACT, bool GR, bool GI>
__device__ __forceinline__ void vscan_tc(
    const float* dr, const float* di, const float* decay_gr, const float* decay_gi,
    const T* states, const float* vx, const float* vy,
    const void* wi, const void* bi, const void* wo, const T* gr, const T* gi,
    T* gs, float* partials, int B, int H, int W, int M, int workers, int types, float eps,
    int span
) {
    constexpr int width = 16;
    extern __shared__ float4 shared[];
    const int tiles = 1 + (M - 1) / WARPS;
    const int mode0 = (blockIdx.x % tiles) * WARPS, group = blockIdx.x / tiles;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, m = mode0 + warp;
    const int ow = W / (1 + COARSE), oh = H / (1 + COARSE);
    const int chunks = 1 + (H - 1) / span;
    const I plane = I(B) * H * W * M, lines = I(B) * ow;
    auto* segment = shared + warp * width;
    auto* boundary = shared + (I(span) * WARPS + warp) * width;
    auto* injection = reinterpret_cast<Quad<T>*>(shared + I(span + chunks - 1) * WARPS * width);
    auto* stage = injection + I(oh) * WARPS * width;
    auto* grad_stage = reinterpret_cast<float2*>(stage);
    auto* blocks = reinterpret_cast<TcPathTile<T>*>(stage + width * (WARPS + 1));
    auto& path_tile = blocks[warp];
    constexpr int stride = width * WARPS;
    const int g = lane >> 2, k = 2 * (lane & 3);
    Complex a{}, decay{};
    float2 bias{}, wr{}, wj{};
    unsigned wf = 0, wb = 0;
    if (m < M) {
        a = {dr[m], di[m]};
        decay = {decay_gr[m], decay_gi[m]};
        const PathWeights<T, I, false> path{wi, bi, wo, types, m, 8};
        const auto w = path.input_pair(4 * g + k / 2);
        wf = tc_pack(write<T>(w.r), write<T>(w.i));
        wb = tc_pack(write<T>(load_scalar(wi, I(m) * 64 + k * 8 + g, types & 3)),
                     write<T>(load_scalar(wi, I(m) * 64 + (k + 1) * 8 + g, types & 3)));
        bias = make_float2(path.bias(k), path.bias(k + 1));
        wr = make_float2(path.output(k), path.output(k + 1));
        wj = make_float2(path.output(8 + k), path.output(9 + k));
    }
    const auto ar = a.conj(), grr = decay.conj();
    Complex da{}, dg{};
    TcAcc acc{};
    for (I first = I(group) * width; first < lines; first += I(workers) * width) {
        const I line = first + lane, io_line = first + threadIdx.x / WARPS;
        const int io_m = mode0 + threadIdx.x % WARPS;
        const bool active = lane < width && line < lines && m < M;
        const bool io_valid = threadIdx.x < width * WARPS && io_line < lines && io_m < M;
        auto c = tc_column<T, I, COARSE>(
            states, vy, active ? line : 0, m, B, H, W, M, ow, oh, plane, eps);
        auto io = tc_column<T, I, COARSE>(
            states, vy, io_valid ? io_line : 0, io_m, B, H, W, M, ow, oh, plane, eps);
        if (active) {
            const int x = line % ow, xa = COARSE ? 2 * x + 1 : x, xb = COARSE ? 2 * x : x;
            c.vx_a = vx[I(xa) * M + m];
            c.vx_b = vx[(I(W) + xb) * M + m];
        }
        auto load = [&](int y) { return tc_source<WARPS>(stage, io, io_valid, y, lane, warp); };
        auto save = [&](int y, float4 ds) {
            tc_write<WARPS, T, I, COMPACT, COARSE, false>(
                stage, io, io_valid, y, lane, warp, ds, gs);
        };
        auto* seg = segment + lane;
        auto* ckpt = boundary + lane;
        auto* g2 = injection + warp * width + lane;
        float4 pos{}, neg{};
        for (int b = 0; b < chunks; ++b) {
            const int y0 = b * span, len = min(span, H - y0);
            for (int j = 0; j < len; ++j) {
                pos = advance(pos, load(y0 + j), a, decay);
                if (lane < width && b + 1 == chunks) seg[j * stride] = pos;
            }
            if (lane < width && b + 1 < chunks) ckpt[b * stride] = pos;
        }
        Complex adj_a{}, adj_b{};
        for (int b = chunks - 1; b >= 0; --b) {
            const int y0 = b * span, len = min(span, H - y0);
            const auto before = lane < width && b ? ckpt[(b - 1) * stride] : float4{};
            if (b + 1 < chunks) {
                if (lane < width) ckpt[b * stride] = neg;
                pos = before;
                for (int j = 0; j < len; ++j) {
                    pos = advance(pos, load(y0 + j), a, decay);
                    if (lane < width) seg[j * stride] = pos;
                }
            }
            int j = len - 1;
            if constexpr (COARSE) {
                if (!((y0 + j) & 1)) {
                    const auto s = load(y0 + j), next = neg;
                    neg = advance(neg, s, ar, grr);
                    float4 ds{};
                    if (active) ds = column_adjoint_step<false>(
                        j ? seg[(j - 1) * stride] : before, s, {}, a, decay, adj_a, adj_b, da, dg);
                    if (lane < width) seg[j * stride] = next;
                    save(y0 + j, ds);
                    --j;
                }
            }
            for (; j >= 0; j -= 1 + COARSE) {
                const int y = y0 + j, yo = y / (1 + COARSE);
                const auto s = load(y), next = neg;
                neg = advance(neg, s, ar, grr);
                auto se = s;
                const auto next_even = neg;
                if constexpr (COARSE) {
                    se = load(y - 1);
                    neg = advance(neg, se, ar, grr);
                }
                const int idx = threadIdx.x;
                if (idx < width * WARPS) {
                    const I cell = io.cell(yo);
                    grad_stage[(idx / WARPS) * (WARPS + 1) + idx % WARPS] = io_valid ?
                        make_float2(read(gr[(cell - io_m) * (1 + GR) + io_m]),
                                    read(gi[(cell - io_m) * (1 + GI) + io_m])) : float2{};
                }
                __syncthreads();
                float2 gp{}, gn{}, dy{};
                float4 p{}, n{}, dpn{};
                if (active) {
                    gp = c.gain(y, false);
                    gn = c.gain(y - COARSE, true);
                    p = normalize<T>(seg[j * stride], gp);
                    n = normalize<T>(neg, gn);
                    dy = grad_stage[lane * (WARPS + 1) + warp];
                }
                const auto dpp = tc_path(path_tile, lane, p, n, dy, wf, wb, bias, wr, wj, acc, dpn);
                float4 ds{}, dse{};
                if (active) {
                    g2[yo * I(stride)] = normalize_packed<T>(dpn, gn);
                    ds = column_adjoint_step<false>(
                        j ? seg[(j - 1) * stride] : before, s, normalize<T>(dpp, gp),
                        a, decay, adj_a, adj_b, da, dg);
                    if constexpr (COARSE) dse = column_adjoint_step<false>(
                        j > 1 ? seg[(j - 2) * stride] : before, se, {},
                        a, decay, adj_a, adj_b, da, dg);
                }
                if (chunks == 1 && lane < width) {
                    seg[j * stride] = next;
                    if constexpr (COARSE) seg[(j - 1) * stride] = next_even;
                }
                __syncthreads();  // All readers finished with the gradient staging buffer.
                save(y, ds);
                if constexpr (COARSE) save(y - 1, dse);
            }
        }
        adj_a = adj_b = {};
        for (int b = 0; b < chunks; ++b) {
            const int y0 = b * span, len = min(span, H - y0);
            if (chunks > 1) {
                neg = lane < width && b + 1 < chunks ? ckpt[b * stride] : float4{};
                for (int j = len - 1; j >= 0; --j) {
                    const auto s = load(y0 + j);
                    if (lane < width) seg[j * stride] = neg;
                    neg = advance(neg, s, ar, grr);
                }
            }
            for (int j = 0; j < len; ++j) {
                const int y = y0 + j;
                const auto s = load(y);
                float4 ds{};
                if (active) {
                    const auto in = !COARSE || (!(y & 1) && y + 1 < H)
                        ? g2[I(y / (1 + COARSE)) * stride].unpack() : float4{};
                    ds = column_adjoint_step<true>(
                        seg[j * stride], s, in, ar, grr, adj_a, adj_b, da, dg);
                }
                tc_write<WARPS, T, I, COMPACT, COARSE, true>(
                    stage, io, io_valid, y, lane, warp, ds, gs);
            }
        }
    }
    #pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        da.r += __shfl_xor_sync(0xffffffff, da.r, offset);
        da.i += __shfl_xor_sync(0xffffffff, da.i, offset);
        dg.r += __shfl_xor_sync(0xffffffff, dg.r, offset);
        dg.i += __shfl_xor_sync(0xffffffff, dg.i, offset);
        acc.br += __shfl_xor_sync(0xffffffff, acc.br, offset);
        acc.bj += __shfl_xor_sync(0xffffffff, acc.bj, offset);
        if (offset >= 4) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                acc.bi[j] += __shfl_xor_sync(0xffffffff, acc.bi[j], offset);
                acc.wr[j] += __shfl_xor_sync(0xffffffff, acc.wr[j], offset);
                acc.wj[j] += __shfl_xor_sync(0xffffffff, acc.wj[j], offset);
            }
        }
    }
    if (m < M) {
        auto* row = partials + I(group) * 94 * M + m;
        row[I(4 + 8 * g + k) * M] = acc.wi[0];
        row[I(5 + 8 * g + k) * M] = acc.wi[1];
        if (lane < 4) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                row[I(68 + k + j) * M] = acc.bi[j];
                row[I(76 + k + j) * M] = acc.wr[j];
                row[I(84 + k + j) * M] = acc.wj[j];
            }
        }
        if (lane == 0) {
            row[0] = da.r;
            row[M] = da.i;
            row[2 * I(M)] = dg.r;
            row[3 * I(M)] = dg.i;
            row[92 * I(M)] = acc.br;
            row[93 * I(M)] = acc.bj;
        }
    }
}
"""
