__device__ __forceinline__ unsigned pack16(unsigned short a, unsigned short b) {
    return a | (static_cast<unsigned>(b) << 16);
}

__device__ __forceinline__ void mma(
    float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int R, int TH, int TW>
__device__ __forceinline__ void tensor_weight_gradient(
    const unsigned short* __restrict__ projected,
    const unsigned short* __restrict__ grad,
    float* __restrict__ partial,
    int height, int width, int modes, int batch
) {
    constexpr int M = 16, HaloH = TH + 2, HaloW = TW + 2, N = TH * TW;
    constexpr int Features = R * 9, Slots = (Features + 3) / 4;
    // The extra packed pair shifts consecutive pixel rows by a bank.
    __shared__ unsigned short p[HaloH * HaloW * R][M + 2];
    __shared__ unsigned short g[N * 2][M + 2];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int mb = blockIdx.x * M;
    const int group = lane / 4, q = lane % 4;
    const int xtiles = (width + TW - 1) / TW;
    const int ytiles = (height + TH - 1) / TH;
    float acc[Slots][4] = {};
    for (int tile = blockIdx.y; tile < batch * xtiles * ytiles; tile += gridDim.y) {
        const int x0 = tile % xtiles * TW, y0 = tile / xtiles % ytiles * TH;
        const long long base = static_cast<long long>(tile / (xtiles * ytiles)) * height * width;
        for (int i = tid / M; i < HaloH * HaloW * R; i += 128 / M) {
            const int y = y0 + i / (HaloW * R) - 1, x = x0 + i / R % HaloW - 1;
            const int mode = mb + tid % M;
            unsigned short value = 0;
            if (x >= 0 && x < width && y >= 0 && y < height && mode < modes)
                value = projected[((base + y * width + x) * R + i % R) * modes + mode];
            p[i][tid % M] = value;
        }
        for (int i = tid / M; i < N * 2; i += 128 / M) {
            const int y = y0 + i / (TW * 2), x = x0 + i / 2 % TW;
            const int mode = mb + tid % M;
            unsigned short value = 0;
            if (x < width && y < height && mode < modes)
                value = grad[((base + y * width + x) * 2 + i % 2) * modes + mode];
            g[i][tid % M] = value;
        }
        __syncthreads();
        #pragma unroll
        for (int pi = 0; pi < N; pi += 16) {
            const int k0 = pi + q * 2, k1 = k0 + 8;
            const int grad_mode = warp * 4 + group / 2;
            const unsigned b0 = pack16(g[k0 * 2 + group % 2][grad_mode],
                                       g[(k0 + 1) * 2 + group % 2][grad_mode]);
            const unsigned b1 = pack16(g[k1 * 2 + group % 2][grad_mode],
                                       g[(k1 + 1) * 2 + group % 2][grad_mode]);
            #pragma unroll
            for (int s = 0; s < Slots; ++s) {
                const int f = s * 4 + group % 4;
                unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                if (f < Features) {
                    const int r = f / 9, dy = f / 3 % 3, dx = f % 3;
                    const int p0 = ((k0 / TW + dy) * HaloW + k0 % TW + dx) * R + r;
                    const int p1 = ((k1 / TW + dy) * HaloW + k1 % TW + dx) * R + r;
                    const int m0 = warp * 4 + group / 4;
                    a0 = pack16(p[p0][m0], p[p0 + R][m0]);
                    a1 = pack16(p[p0][m0 + 2], p[p0 + R][m0 + 2]);
                    a2 = pack16(p[p1][m0], p[p1 + R][m0]);
                    a3 = pack16(p[p1][m0 + 2], p[p1 + R][m0 + 2]);
                }
                mma(acc[s], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }
    const int mode = mb + warp * 4 + q;
    if (mode >= modes || q % 2 != group / 4) return;
    #pragma unroll
    for (int s = 0; s < Slots; ++s) {
        const int f = s * 4 + group % 4;
        if (f >= Features) continue;
        const int offset = (blockIdx.y * 2 * Features + f) * modes + mode;
        partial[offset] = q < 2 ? acc[s][0] : acc[s][2];
        partial[offset + Features * modes] = q < 2 ? acc[s][1] : acc[s][3];
    }
}

