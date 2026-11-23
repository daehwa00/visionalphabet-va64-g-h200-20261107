#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>

template<bool BF16>
__device__ unsigned pack(int x, int y) {
    if constexpr (BF16)
        return __bfloat16_as_ushort(__float2bfloat16(x)) |
               (unsigned(__bfloat16_as_ushort(__float2bfloat16(y))) << 16);
    else
        return __half_as_ushort(__float2half(x)) |
               (unsigned(__half_as_ushort(__float2half(y))) << 16);
}

__host__ __device__ int a(int r, int k) { return (3*r + 5*k + r*k) % 17 - 8; }
__host__ __device__ int b(int k, int c) { return (7*k + 2*c + k*c) % 13 - 6; }

template<bool BF16, int K>
__global__ void fragments(float* out) {
    int group = threadIdx.x / 4, pair = threadIdx.x % 4, k = 2 * pair;
    unsigned a0 = pack<BF16>(a(group, k), a(group, k+1));
    unsigned a1 = pack<BF16>(a(group+8, k), a(group+8, k+1));
    unsigned b0 = pack<BF16>(b(k, group), b(k+1, group));
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#define OUTPUT "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
#define MMA8(TYPE) asm("mma.sync.aligned.m16n8k8.row.col.f32." TYPE "." TYPE ".f32 " \
    "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};" \
    : OUTPUT : "r"(a0), "r"(a1), "r"(b0))
#define MMA16(TYPE) asm("mma.sync.aligned.m16n8k16.row.col.f32." TYPE "." TYPE ".f32 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
    : OUTPUT : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
    if constexpr (K == 8) {
        if constexpr (BF16) MMA8("bf16"); else MMA8("f16");
    } else {
        unsigned a2 = pack<BF16>(a(group, k+8), a(group, k+9));
        unsigned a3 = pack<BF16>(a(group+8, k+8), a(group+8, k+9));
        unsigned b1 = pack<BF16>(b(k+8, group), b(k+9, group));
        if constexpr (BF16) MMA16("bf16"); else MMA16("f16");
    }
    out[group*8+k] = d0;
    out[group*8+k+1] = d1;
    out[(group+8)*8+k] = d2;
    out[(group+8)*8+k+1] = d3;
}

template<bool BF16, int K>
bool check(float* device) {
    fragments<BF16, K><<<1,32>>>(device);
    float host[128];
    auto status = cudaMemcpy(host, device, sizeof host, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        return false;
    }
    int mismatches = 0;
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 8; ++c) {
            int expected = 0;
            for (int k = 0; k < K; ++k) expected += a(r,k) * b(k,c);
            mismatches += host[r*8+c] != expected;
        }
    }
    std::printf("%s k=%d mismatches=%d/128\n", BF16 ? "bf16" : "f16", K, mismatches);
    return mismatches == 0;
}

int main() {
    float* device = nullptr;
    if (cudaMalloc(&device, 128 * sizeof(float)) != cudaSuccess) return 1;
    bool passed = check<true,8>(device) & check<false,8>(device) &
                  check<true,16>(device) & check<false,16>(device);
    cudaFree(device);
    return passed ? 0 : 1;
}
