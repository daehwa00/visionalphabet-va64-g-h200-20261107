// Appended to the Wave 13B CUDA source by the probe compiler.
template <int Height, int Width, int Fields, int Modes, bool Async>
__device__ __forceinline__ void stage_bf16(
    __nv_bfloat16* tile, const __nv_bfloat16* input, long long base,
    int y0, int x0, int h, int w, int modes, int mode0
) {
    constexpr int Pack = sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr int Packs = Modes / Pack;
    for (int index = threadIdx.x; index < Height * Width * Fields * Packs; index += blockDim.x) {
        int pixel = index / (Fields * Packs);
        int y = y0 + pixel / Width, x = x0 + pixel % Width;
        int field = index / Packs % Fields, mode = mode0 + index % Packs * Pack;
        bool valid = y >= 0 && y < h && x >= 0 && x < w && mode < modes;
        auto* dst = reinterpret_cast<uint4*>(tile) + index;
        auto* src = reinterpret_cast<const uint4*>(
            valid ? input + ((base + y * w + x) * Fields + field) * modes + mode : input
        );
        if constexpr (Async) {
            unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
            // A zero-byte source initializes halo and mode padding without a global read.
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                         :: "r"(smem), "l"(src), "r"(valid ? 16 : 0) : "memory");
        } else {
            *dst = valid ? *src : uint4{};
        }
    }
}

template <bool Async>
__device__ __forceinline__ void finish_stage() {
    if constexpr (Async) asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
}

template <int Vec>
__device__ __forceinline__ void store_bf16(__nv_bfloat16* dst, ModeValues<Vec> v) {
    if constexpr (Vec == 1) *dst = __float2bfloat16_rn(v.value[0]);
    else *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v.value[0], v.value[1]);
}

template <bool Backward, int Rows, int Columns, int Vec, bool Async, int YBlock>
__device__ __forceinline__ void tiled_spatial(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ real, const __nv_bfloat16* __restrict__ imag,
    __nv_bfloat16* __restrict__ output, int runtime_height, int runtime_width, int runtime_modes
) {
    constexpr int R = Rank, Inputs = Backward ? 2 : R, Outputs = Backward ? R : 2;
    constexpr int Modes = 32, Pitch = (Columns + 2) * Inputs * Modes, Lanes = Modes / Vec;
    using V = ModeValues<Vec>;
    const int h = READER_HEIGHT ? READER_HEIGHT : runtime_height;
    const int w = READER_WIDTH ? READER_WIDTH : runtime_width;
    const int modes = READER_MODES ? READER_MODES : runtime_modes;
    const int mode0 = blockIdx.x * Modes, local_mode = (threadIdx.x % Lanes) * Vec;
    const int xtiles = (w + Columns - 1) / Columns;
    const int y0 = blockIdx.y / xtiles * Rows, x0 = blockIdx.y % xtiles * Columns;
    const long long base = static_cast<long long>(blockIdx.z) * h * w;
    extern __shared__ __align__(16) unsigned char shared[];
    auto* tile = reinterpret_cast<__nv_bfloat16*>(shared);
    auto* filters = reinterpret_cast<float (*)[R][9][Modes]>(tile + (Rows + 2) * Pitch);
    stage_bf16<Rows + 2, Columns + 2, Inputs, Modes, Async>(
        tile, input, base, y0 - 1, x0 - 1, h, w, modes, mode0);
    for (int k = threadIdx.x; k < 2 * R * 9 * Modes; k += blockDim.x) {
        int mode = mode0 + k % Modes, tap = k / Modes % 9;
        int r = k / (Modes * 9) % R, component = k / (Modes * R * 9);
        const auto* weights = component ? imag : real;
        if (mode < modes)
            filters[component][r][tap][k % Modes] =
                as_float(weights[(mode * R + r) * 9 + (Backward ? 8 - tap : tap)]);
    }
    finish_stage<Async>();
    const int mode = mode0 + local_mode;
    if (mode >= modes) return;
    const int row_stride = blockDim.x / Lanes * YBlock;
    for (int row = threadIdx.x / Lanes * YBlock; row < Rows && y0 + row < h;
         row += row_stride) {
        for (int x = 0; x < Columns && x0 + x < w; ++x) {
            V a[YBlock]{}, b[YBlock]{};
            const auto* p = tile + row * Pitch + x * Inputs * Modes + local_mode;
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                #pragma unroll
                for (int ky = 0; ky < 3; ++ky) {
                    // Preserve the scalar FMA order within every output channel.
                    V av[3], bv[3];
                    #pragma unroll
                    for (int kx = 0; kx < 3; ++kx) {
                        #pragma unroll
                        for (int v = 0; v < Vec; ++v) {
                            av[kx].value[v] = filters[0][r][ky * 3 + kx][local_mode + v];
                            bv[kx].value[v] = filters[1][r][ky * 3 + kx][local_mode + v];
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < YBlock; ++j) {
                        V z[3];
                        #pragma unroll
                        for (int kx = 0; kx < 3; ++kx) {
                            const auto* q = p + (j + ky) * Pitch + kx * Inputs * Modes;
                            z[kx] = mode_load<Vec>(q + (Backward ? 0 : r * Modes));
                            a[j] = fmaf(z[kx], av[kx], a[j]);
                        }
                        #pragma unroll
                        for (int kx = 0; kx < 3; ++kx) {
                            const auto* q = p + (j + ky) * Pitch + kx * Inputs * Modes;
                            if constexpr (Backward) {
                                a[j] = fmaf(mode_load<Vec>(q + Modes), bv[kx], a[j]);
                            } else {
                                b[j] = fmaf(z[kx], bv[kx], b[j]);
                            }
                        }
                    }
                }
                if constexpr (Backward) {
                    #pragma unroll
                    for (int j = 0; j < YBlock; ++j) {
                        if (y0 + row + j < h) {
                            long long pixel = base + (y0 + row + j) * w + x0 + x;
                            store_bf16<Vec>(output + (pixel * Outputs + r) * modes + mode, a[j]);
                        }
                        a[j] = V{};
                    }
                }
            }
            if constexpr (!Backward) {
                #pragma unroll
                for (int j = 0; j < YBlock; ++j) {
                    if (y0 + row + j < h) {
                        long long pixel = base + (y0 + row + j) * w + x0 + x;
                        auto* out = output + pixel * Outputs * modes + mode;
                        store_bf16<Vec>(out, a[j]);
                        store_bf16<Vec>(out + modes, b[j]);
                    }
                }
            }
        }
    }
}

template <int Rows, int Columns, int Warps, int Vec, bool Async, bool Reuse>
__device__ __forceinline__ void tiled_dweight(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ grad,
    float* __restrict__ partial, int runtime_height, int runtime_width, int runtime_modes,
    int runtime_batch
) {
    constexpr int R = Rank, Modes = 32, Lanes = Modes / Vec;
    constexpr int Groups = Warps * 32 / (R * Lanes);
    constexpr int Pixels = Rows * Columns;
    using V = ModeValues<Vec>;
    const int h = READER_HEIGHT ? READER_HEIGHT : runtime_height;
    const int w = READER_WIDTH ? READER_WIDTH : runtime_width;
    const int modes = READER_MODES ? READER_MODES : runtime_modes;
    const int batch = READER_BATCH ? READER_BATCH : runtime_batch;
    const int xtiles = (w + Columns - 1) / Columns;
    const int tiles = ((h + Rows - 1) / Rows) * xtiles;
    const int group = threadIdx.x / (R * Lanes), r = threadIdx.x / Lanes % R;
    const int local_mode = threadIdx.x % Lanes * Vec, mode0 = blockIdx.x * Modes;
    constexpr int ReduceElems = Groups * R * 18 * Modes * sizeof(float) / sizeof(__nv_bfloat16);
    constexpr int ProjectedElems = Pixels * R * Modes;
    constexpr int ScratchElems = ReduceElems > ProjectedElems ? ReduceElems : ProjectedElems;
    extern __shared__ __align__(16) unsigned char shared[];
    auto* ps = reinterpret_cast<__nv_bfloat16*>(shared);
    auto* gs = ps + ScratchElems;
    V sum[2][9]{};
    for (int tile = blockIdx.y; tile < batch * tiles; tile += gridDim.y) {
        const int y0 = tile % tiles / xtiles * Rows, x0 = tile % xtiles * Columns;
        const long long base = static_cast<long long>(tile / tiles) * h * w;
        stage_bf16<Rows, Columns, R, Modes, Async>(
            ps, projected, base, y0, x0, h, w, modes, mode0);
        stage_bf16<Rows + 2, Columns + 2, 2, Modes, Async>(
            gs, grad, base, y0 - 1, x0 - 1, h, w, modes, mode0);
        finish_stage<Async>();
        if constexpr (Reuse) {
            for (int y = group; y < Rows; y += Groups) {
                const auto* g = gs + y * (Columns + 2) * 2 * Modes + local_mode;
                using Column = V[3][2];
                Column c0, c1, c2;
                auto read = [&](int x, Column& col) {
                    #pragma unroll
                    for (int ky = 0; ky < 3; ++ky)
                        #pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            const auto* q = g + ((ky * (Columns + 2) + x) * 2 + c) * Modes;
                            col[ky][c] = mode_load<Vec>(q);
                        }
                };
                auto step = [&](int x, const Column& l, const Column& m, Column& right) {
                    read(x + 2, right);
                    const V p = mode_load<Vec>(ps + ((y * Columns + x) * R + r) * Modes + local_mode);
                    #pragma unroll
                    for (int ky = 0; ky < 3; ++ky)
                        #pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            const int row = (2 - ky) * 3;
                            sum[c][row] = fmaf(p, right[ky][c], sum[c][row]);
                            sum[c][row + 1] = fmaf(p, m[ky][c], sum[c][row + 1]);
                            sum[c][row + 2] = fmaf(p, l[ky][c], sum[c][row + 2]);
                        }
                };
                read(0, c0);
                read(1, c1);
                int x = 0;
                #pragma unroll 1
                for (; x + 2 < Columns; x += 3) {
                    step(x, c0, c1, c2);
                    step(x + 1, c1, c2, c0);
                    step(x + 2, c2, c0, c1);
                }
                if (x < Columns) step(x, c0, c1, c2);
                if (x + 1 < Columns) step(x + 1, c1, c2, c0);
            }
        } else {
            for (int px = group; px < Pixels; px += Groups) {
                int y = px / Columns, x = px % Columns;
                const V p = mode_load<Vec>(ps + (px * R + r) * Modes + local_mode);
                const auto* g = gs + (y * (Columns + 2) + x) * 2 * Modes + local_mode;
                #pragma unroll
                for (int ky = 0; ky < 3; ++ky)
                    #pragma unroll
                    for (int kx = 0; kx < 3; ++kx)
                        #pragma unroll
                        for (int c = 0; c < 2; ++c) {
                            const int tap = 8 - (ky * 3 + kx);
                            const auto* neighbor = g + ((ky * (Columns + 2) + kx) * 2 + c) * Modes;
                            sum[c][tap] = fmaf(p, mode_load<Vec>(neighbor), sum[c][tap]);
                        }
            }
        }
        __syncthreads();
    }
    if constexpr (Groups == 1) {
        const auto mode = mode0 + local_mode;
        if (mode < modes)
            #pragma unroll
            for (int field = 0; field < 18; ++field) {
                long long offset = ((static_cast<long long>(blockIdx.y) * 2 + field / 9)
                                    * R * 9 + r * 9 + field % 9) * modes;
                #pragma unroll
                for (int v = 0; v < Vec; ++v) partial[offset + mode + v] = sum[field / 9][field % 9].value[v];
            }
    } else {
        // Reuse the input staging area after the last tile has finished.
        auto* scratch = reinterpret_cast<float*>(ps);
        #pragma unroll
        for (int field = 0; field < 18; ++field)
            #pragma unroll
            for (int v = 0; v < Vec; ++v)
                scratch[((group * R + r) * 18 + field) * Modes + local_mode + v] =
                    sum[field / 9][field % 9].value[v];
        __syncthreads();
        for (int i = threadIdx.x; i < R * 18 * Modes; i += blockDim.x) {
            int mode = mode0 + i % Modes;
            if (mode >= modes) continue;
            float accum = 0;
            #pragma unroll
            for (int g = 0; g < Groups; ++g) accum += scratch[g * R * 18 * Modes + i];
            int field = i / Modes % 18, rank = i / (18 * Modes);
            long long offset = ((static_cast<long long>(blockIdx.y) * 2 + field / 9)
                                * R * 9 + rank * 9 + field % 9) * modes;
            partial[offset + mode] = accum;
        }
    }
}
