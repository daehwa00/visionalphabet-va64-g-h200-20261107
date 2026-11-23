// Three inlined steps let each window slot keep the same register assignment.
template <int Rows, int Columns, int Warps, bool Backward>
__device__ __forceinline__ void ring_spatial(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ real, const __nv_bfloat16* __restrict__ imag,
    __nv_bfloat16* __restrict__ output, int runtime_height, int runtime_width, int runtime_modes
) {
    constexpr int R = Rank, Inputs = Backward ? 2 : R, Outputs = Backward ? R : 2;
    const int h = READER_HEIGHT ? READER_HEIGHT : runtime_height;
    const int w = READER_WIDTH ? READER_WIDTH : runtime_width;
    const int modes = READER_MODES ? READER_MODES : runtime_modes;
    const int mode = blockIdx.x * 32 + threadIdx.x % 32, lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int xtiles = (w + Warps * Columns - 1) / (Warps * Columns);
    const int y0 = blockIdx.y / xtiles * Rows, x0 = (blockIdx.y % xtiles * Warps + warp) * Columns;
    const long long base = static_cast<long long>(blockIdx.z) * h * w;
    __shared__ float f[2][R][3][3][32];
    for (int k = warp; k < 2 * R * 9; k += Warps) {
        int tap = k % 9, r = k / 9 % R, c = k / (R * 9);
        if (mode < modes) {
            f[c][r][tap / 3][tap % 3][lane] =
                as_float((c ? imag : real)[(mode * R + r) * 9 + (Backward ? 8 - tap : tap)]);
        }
    }
    __syncthreads();
    if (mode >= modes || x0 >= w) return;
    using Column = float[Rows + 2][Inputs];
    Column c0, c1, c2;
    auto read_column = [&](int x, Column& column) {
        #pragma unroll
        for (int j = 0; j < Rows + 2; ++j) {
            const int y = y0 + j - 1;
            #pragma unroll
            for (int i = 0; i < Inputs; ++i) {
                column[j][i] = (y >= 0 && y < h && x >= 0 && x < w)
                    ? as_float(input[((base + y * w + x) * Inputs + i) * modes + mode]) : 0;
            }
        }
    };
    auto step = [&](int x, const Column& l, const Column& m, Column& r) {
        read_column(x + 1, r);
        float a[Rows]{}, b[Rows]{};
        #pragma unroll
        for (int rank = 0; rank < R; ++rank) {
            const int i = Backward ? 0 : rank;
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                const float fl = f[0][rank][ky][0][lane], fm = f[0][rank][ky][1][lane];
                const float fr = f[0][rank][ky][2][lane], gl = f[1][rank][ky][0][lane];
                const float gm = f[1][rank][ky][1][lane], gr = f[1][rank][ky][2][lane];
                #pragma unroll
                for (int j = 0; j < Rows; ++j) {
                    a[j] = fmaf(l[j + ky][i], fl, a[j]);
                    a[j] = fmaf(m[j + ky][i], fm, a[j]);
                    a[j] = fmaf(r[j + ky][i], fr, a[j]);
                    if constexpr (Backward) {
                        a[j] = fmaf(l[j + ky][1], gl, a[j]);
                        a[j] = fmaf(m[j + ky][1], gm, a[j]);
                        a[j] = fmaf(r[j + ky][1], gr, a[j]);
                    } else {
                        b[j] = fmaf(l[j + ky][i], gl, b[j]);
                        b[j] = fmaf(m[j + ky][i], gm, b[j]);
                        b[j] = fmaf(r[j + ky][i], gr, b[j]);
                    }
                }
            }
            if constexpr (Backward) {
                #pragma unroll
                for (int j = 0; j < Rows; ++j) {
                    const long long pixel = base + (y0 + j) * w + x;
                    if (y0 + j < h)
                        output[(pixel * Outputs + rank) * modes + mode] = __float2bfloat16_rn(a[j]);
                    a[j] = 0;
                }
            }
        }
        if constexpr (!Backward) {
            #pragma unroll
            for (int j = 0; j < Rows; ++j) {
                const long long pixel = base + (y0 + j) * w + x;
                if (y0 + j < h) {
                    output[pixel * Outputs * modes + mode] = __float2bfloat16_rn(a[j]);
                    output[(pixel * Outputs + 1) * modes + mode] = __float2bfloat16_rn(b[j]);
                }
            }
        }
    };
    read_column(x0 - 1, c0);
    read_column(x0, c1);
    const int end = min(x0 + Columns, w);
    int x = x0;
    #pragma unroll 1
    for (; x + 2 < end; x += 3) {
        step(x, c0, c1, c2);
        step(x + 1, c1, c2, c0);
        step(x + 2, c2, c0, c1);
    }
    if (x < end) step(x, c0, c1, c2);
    if (x + 1 < end) step(x + 1, c1, c2, c0);
}
