"""SM90/SM89/SM120 NVRTC CUDA C++ kernels for vertical path-collapse scan.

Implements fused streaming vertical forward and backward scans for
(H, W) in {(56, 56), (28, 28), (14, 14)}, M % 32 == 0, packed_path_hidden == 8,
both direct (full16) and coarse (coarse_path) epilogues.
"""

# ruff: noqa: ARG001, E501, FBT001
from __future__ import annotations

import ctypes
import importlib.util
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

_CUDA_SOURCE = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    return __uint_as_float(static_cast<unsigned int>(bits) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16(float x) {
    unsigned short res;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(res) : "f"(x));
    return res;
}

__device__ __forceinline__ float round_bf16(float x) {
    return bf16_to_f32(f32_to_bf16(x));
}

__device__ __forceinline__ unsigned int pack_bf16x2(float lo, float hi) {
    return static_cast<unsigned int>(f32_to_bf16(lo))
         | (static_cast<unsigned int>(f32_to_bf16(hi)) << 16);
}

__device__ __forceinline__ void unpack_bf16x2(unsigned int packed, float& lo, float& hi) {
    lo = __uint_as_float(packed << 16);
    hi = __uint_as_float(packed & 0xFFFF0000u);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

template <int H, bool COARSE, int WARPS, int MIN_BLOCKS>
__device__ __forceinline__ void vscan_forward_impl(
    const float* __restrict__ decay_real,
    const float* __restrict__ decay_imag,
    const float* __restrict__ gamma_real,
    const float* __restrict__ gamma_imag,
    const unsigned short* __restrict__ src_ra,
    const unsigned short* __restrict__ src_ia,
    const unsigned short* __restrict__ src_rb,
    const unsigned short* __restrict__ src_ib,
    const float* __restrict__ variance_x,
    const float* __restrict__ variance_y,
    const float* __restrict__ path_in_w,
    const float* __restrict__ path_in_b,
    const float* __restrict__ path_out_w,
    const float* __restrict__ path_out_b,
    unsigned short* __restrict__ out_r,
    unsigned short* __restrict__ out_i,
    int B,
    int M,
    int x_chunks,
    float eps
) {
    constexpr int W = H;
    constexpr int OH = COARSE ? (H / 2) : H;
    constexpr int OW = COARSE ? (W / 2) : W;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int m = blockIdx.x * 32 + lane;
    const int b = blockIdx.y / x_chunks;
    const int x_chunk = blockIdx.y - b * x_chunks;

    extern __shared__ unsigned int smem[];
    unsigned int* win_smem = smem;
    unsigned int* wout_smem = win_smem + 8 * 4 * 32;
    unsigned int* bin_smem = wout_smem + 2 * 4 * 32;
    unsigned int* pos_buf = bin_smem + 4 * 32;

    if (m < M) {
        if (warp == 0) {
            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                const float* wrow = path_in_w + (m * 8 + h) * 8;
                #pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) {
                    win_smem[(h * 4 + k2) * 32 + lane] = pack_bf16x2(wrow[2 * k2], wrow[2 * k2 + 1]);
                }
            }
        }
        if (warp == (WARPS > 1 ? 1 : 0)) {
            const float* wo_r = path_out_w + m * 16;
            const float* wo_i = wo_r + 8;
            const float* bi = path_in_b + m * 8;
            #pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                wout_smem[(0 * 4 + k2) * 32 + lane] = pack_bf16x2(wo_r[2 * k2], wo_r[2 * k2 + 1]);
                wout_smem[(1 * 4 + k2) * 32 + lane] = pack_bf16x2(wo_i[2 * k2], wo_i[2 * k2 + 1]);
                bin_smem[k2 * 32 + lane] = pack_bf16x2(bi[2 * k2], bi[2 * k2 + 1]);
            }
        }
    }
    __syncthreads();

    if (m >= M || b >= B) return;

    const float ar = decay_real[m];
    const float ai = decay_imag[m];
    const float gr = gamma_real[m];
    const float gi = gamma_imag[m];
    const float bor = round_bf16(path_out_b[m * 2]);
    const float boi = round_bf16(path_out_b[m * 2 + 1]);

    unsigned int* my_pos = pos_buf + warp * (OH * 2 * 32);

    const int cols_per_chunk = (OW + x_chunks - 1) / x_chunks;
    const int x_start = x_chunk * cols_per_chunk;
    const int x_end = (x_start + cols_per_chunk < OW) ? (x_start + cols_per_chunk) : OW;

    const long long batch_in_base = static_cast<long long>(b) * H * W * M;
    const long long batch_out_base = static_cast<long long>(b) * OH * OW * M;
    const long long row_stride = static_cast<long long>(W) * M;
    const long long out_row_stride = static_cast<long long>(OW) * M;

    for (int x = x_start + warp; x < x_end; x += WARPS) {
        const int xa = COARSE ? (2 * x + 1) : x;
        const int xb = COARSE ? (2 * x) : x;

        const float vxp = variance_x[xa * M + m];
        const float vxn = variance_x[(W + xb) * M + m];

        const long long col_off_a = batch_in_base + static_cast<long long>(xa) * M + m;
        const long long col_off_b = batch_in_base + static_cast<long long>(xb) * M + m;

        float par = 0.0f, pai = 0.0f, pbr = 0.0f, pbi = 0.0f;
        for (int y = 0; y < H; ++y) {
            const long long oa = col_off_a + y * row_stride;
            const long long ob = col_off_b + y * row_stride;
            const float sar = bf16_to_f32(src_ra[oa]);
            const float sai = bf16_to_f32(src_ia[oa]);
            const float sbr = bf16_to_f32(src_rb[ob]);
            const float sbi = bf16_to_f32(src_ib[ob]);

            const float n_par = ar * par - ai * pai + (gr * sar - gi * sai);
            const float n_pai = ar * pai + ai * par + (gr * sai + gi * sar);
            const float n_pbr = ar * pbr - ai * pbi + (gr * sbr - gi * sbi);
            const float n_pbi = ar * pbi + ai * pbr + (gr * sbi + gi * sbr);
            par = n_par; pai = n_pai; pbr = n_pbr; pbi = n_pbi;

            if (!COARSE || (y & 1)) {
                const float vyp = variance_y[y * M + m];
                const float ga = rsqrtf(fmaxf(vxp * vyp, eps));
                const float gb = rsqrtf(fmaxf(vxn * vyp, eps));
                const int t = COARSE ? (y >> 1) : y;
                my_pos[(t * 2 + 0) * 32 + lane] = pack_bf16x2(par * ga, pbr * gb);
                my_pos[(t * 2 + 1) * 32 + lane] = pack_bf16x2(pai * ga, pbi * gb);
            }
        }

        float nar = 0.0f, nai = 0.0f, nbr = 0.0f, nbi = 0.0f;
        const long long out_col_off = batch_out_base + static_cast<long long>(x) * M + m;
        for (int y = H - 1; y >= 0; --y) {
            const long long oa = col_off_a + y * row_stride;
            const long long ob = col_off_b + y * row_stride;
            const float sar = bf16_to_f32(src_ra[oa]);
            const float sai = bf16_to_f32(src_ia[oa]);
            const float sbr = bf16_to_f32(src_rb[ob]);
            const float sbi = bf16_to_f32(src_ib[ob]);

            const float n_nar = ar * nar + ai * nai + (gr * sar + gi * sai);
            const float n_nai = ar * nai - ai * nar + (gr * sai - gi * sar);
            const float n_nbr = ar * nbr + ai * nbi + (gr * sbr + gi * sbi);
            const float n_nbi = ar * nbi - ai * nbr + (gr * sbi - gi * sbr);
            nar = n_nar; nai = n_nai; nbr = n_nbr; nbi = n_nbi;

            if (!COARSE || ((y & 1) == 0)) {
                const float vyn = variance_y[(H + y) * M + m];
                const float ga = rsqrtf(fmaxf(vxp * vyn, eps));
                const float gb = rsqrtf(fmaxf(vxn * vyn, eps));
                const int t = COARSE ? (y >> 1) : y;

                float p0, p1, p4, p5;
                unpack_bf16x2(my_pos[(t * 2 + 0) * 32 + lane], p0, p1);
                unpack_bf16x2(my_pos[(t * 2 + 1) * 32 + lane], p4, p5);
                const float p2 = round_bf16(nar * ga);
                const float p3 = round_bf16(nbr * gb);
                const float p6 = round_bf16(nai * ga);
                const float p7 = round_bf16(nbi * gb);

                float acc_r = bor;
                float acc_i = boi;
                #pragma unroll 1
                for (int h = 0; h < 8; ++h) {
                    float w0, w1, w2, w3, w4, w5, w6, w7;
                    unpack_bf16x2(win_smem[(h * 4 + 0) * 32 + lane], w0, w1);
                    unpack_bf16x2(win_smem[(h * 4 + 1) * 32 + lane], w2, w3);
                    unpack_bf16x2(win_smem[(h * 4 + 2) * 32 + lane], w4, w5);
                    unpack_bf16x2(win_smem[(h * 4 + 3) * 32 + lane], w6, w7);

                    float b_lo, b_hi;
                    unpack_bf16x2(bin_smem[(h >> 1) * 32 + lane], b_lo, b_hi);
                    float bias_h = (h & 1) ? b_hi : b_lo;

                    float pre = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3
                              + w4 * p4 + w5 * p5 + w6 * p6 + w7 * p7;
                    pre = round_bf16(pre + bias_h);
                    float hid = round_bf16(pre * fast_sigmoid(pre));

                    float wr_lo, wr_hi, wi_lo, wi_hi;
                    unpack_bf16x2(wout_smem[(0 * 4 + (h >> 1)) * 32 + lane], wr_lo, wr_hi);
                    unpack_bf16x2(wout_smem[(1 * 4 + (h >> 1)) * 32 + lane], wi_lo, wi_hi);
                    acc_r += hid * ((h & 1) ? wr_hi : wr_lo);
                    acc_i += hid * ((h & 1) ? wi_hi : wi_lo);
                }
                const long long oo = out_col_off + t * out_row_stride;
                out_r[oo] = f32_to_bf16(acc_r);
                out_i[oo] = f32_to_bf16(acc_i);
            }
        }
    }
}

template <int H, bool COARSE, int WARPS, int CHUNK, bool ACC_IN_SMEM>
__device__ __forceinline__ void vscan_backward_impl(
    const float* __restrict__ decay_real,
    const float* __restrict__ decay_imag,
    const float* __restrict__ gamma_real,
    const float* __restrict__ gamma_imag,
    const unsigned short* __restrict__ src_ra,
    const unsigned short* __restrict__ src_ia,
    const unsigned short* __restrict__ src_rb,
    const unsigned short* __restrict__ src_ib,
    const float* __restrict__ variance_x,
    const float* __restrict__ variance_y,
    const float* __restrict__ path_in_w,
    const float* __restrict__ path_in_b,
    const float* __restrict__ path_out_w,
    const unsigned short* __restrict__ grad_cr,
    const unsigned short* __restrict__ grad_ci,
    float* __restrict__ grad_win,
    float* __restrict__ grad_bin,
    float* __restrict__ grad_wout,
    float* __restrict__ grad_bout,
    float* __restrict__ grad_dr,
    float* __restrict__ grad_di,
    float* __restrict__ grad_gr,
    float* __restrict__ grad_gi,
    unsigned short* __restrict__ grad_src,
    int B,
    int M,
    int x_chunks,
    float eps
) {
    constexpr int W = H;
    constexpr int OH = COARSE ? (H / 2) : H;
    constexpr int OW = COARSE ? (W / 2) : W;
    constexpr int NCHUNKS = H / CHUNK;
    constexpr int CKPT_SLOTS = (NCHUNKS > 1) ? (NCHUNKS - 1) : 1;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int m = blockIdx.x * 32 + lane;
    const int b = blockIdx.y / x_chunks;
    const int x_chunk = blockIdx.y - b * x_chunks;

    extern __shared__ unsigned int smem[];
    unsigned int* win_smem = smem;                                 // 4 KB
    unsigned int* wout_smem = win_smem + 8 * 4 * 32;               // 1 KB
    unsigned int* bin_smem = wout_smem + 2 * 4 * 32;               // 0.5 KB
    unsigned int* g2_base = bin_smem + 4 * 32;                     // WARPS * OH * 2 * 32 uint32
    float* ckpt_base = reinterpret_cast<float*>(g2_base + WARPS * OH * 2 * 32);
    float* seg_base = ckpt_base + (NCHUNKS > 1 ? WARPS * CKPT_SLOTS * 4 * 32 : 0);
    float* acc_base = seg_base + WARPS * CHUNK * 4 * 32;

    if (m < M) {
        if (warp == 0) {
            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                const float* wrow = path_in_w + (m * 8 + h) * 8;
                #pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) {
                    win_smem[(h * 4 + k2) * 32 + lane] = pack_bf16x2(wrow[2 * k2], wrow[2 * k2 + 1]);
                }
            }
        }
        if (warp == (WARPS > 1 ? 1 : 0)) {
            const float* wo_r = path_out_w + m * 16;
            const float* wo_i = wo_r + 8;
            const float* bi = path_in_b + m * 8;
            #pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                wout_smem[(0 * 4 + k2) * 32 + lane] = pack_bf16x2(wo_r[2 * k2], wo_r[2 * k2 + 1]);
                wout_smem[(1 * 4 + k2) * 32 + lane] = pack_bf16x2(wo_i[2 * k2], wo_i[2 * k2 + 1]);
                bin_smem[k2 * 32 + lane] = pack_bf16x2(bi[2 * k2], bi[2 * k2 + 1]);
            }
        }
    }
    __syncthreads();

    if (m >= M || b >= B) return;

    const float ar = decay_real[m];
    const float ai = decay_imag[m];
    const float gr = gamma_real[m];
    const float gi = gamma_imag[m];

    unsigned int* my_g2 = g2_base + warp * (OH * 2 * 32);
    float* my_ckpt = ckpt_base + warp * (CKPT_SLOTS * 4 * 32);
    float* my_seg = seg_base + warp * (CHUNK * 4 * 32);
    float* my_acc = acc_base + warp * (26 * 32);

    float d_bin_reg[8];
    float d_wout_r_reg[8];
    float d_wout_i_reg[8];
    float d_bout_r_reg = 0.0f;
    float d_bout_i_reg = 0.0f;

    if constexpr (ACC_IN_SMEM) {
        for (int i = 0; i < 26; ++i) my_acc[i * 32 + lane] = 0.0f;
    } else {
        #pragma unroll
        for (int h = 0; h < 8; ++h) {
            d_bin_reg[h] = 0.0f;
            d_wout_r_reg[h] = 0.0f;
            d_wout_i_reg[h] = 0.0f;
        }
    }

    float d_win[8][8];
    #pragma unroll
    for (int h = 0; h < 8; ++h)
        #pragma unroll
        for (int k = 0; k < 8; ++k) d_win[h][k] = 0.0f;

    float d_dr = 0.0f, d_di = 0.0f, d_gr = 0.0f, d_gi = 0.0f;

    const int cols_per_chunk = (OW + x_chunks - 1) / x_chunks;
    const int x_start = x_chunk * cols_per_chunk;
    const int x_end = (x_start + cols_per_chunk < OW) ? (x_start + cols_per_chunk) : OW;

    const long long plane_size = static_cast<long long>(B) * H * W * M;
    const long long b_in = static_cast<long long>(b) * H * W * M + m;
    const long long b_out = static_cast<long long>(b) * OH * OW * M + m;
    const int row_stride = W * M;
    const int out_row_stride = OW * M;

    for (int x = x_start + warp; x < x_end; x += WARPS) {
        const int xa = COARSE ? (2 * x + 1) : x;
        const int xb = COARSE ? (2 * x) : x;
        const float vxp = variance_x[xa * M + m];
        const float vxn = variance_x[(W + xb) * M + m];
        const long long col_a = b_in + xa * M;
        const long long col_b = b_in + xb * M;
        const long long col_g = b_out + x * M;

        // PASS 1: Positive forward scan y = 0 .. H-1
        {
            float par = 0.0f, pai = 0.0f, pbr = 0.0f, pbi = 0.0f;
            for (int c = 0; c < NCHUNKS; ++c) {
                if constexpr (NCHUNKS > 1) {
                    if (c > 0) {
                        my_ckpt[((c - 1) * 4 + 0) * 32 + lane] = par;
                        my_ckpt[((c - 1) * 4 + 1) * 32 + lane] = pai;
                        my_ckpt[((c - 1) * 4 + 2) * 32 + lane] = pbr;
                        my_ckpt[((c - 1) * 4 + 3) * 32 + lane] = pbi;
                    }
                }
                const int y0 = c * CHUNK;
                for (int r = 0; r < CHUNK; ++r) {
                    const int y = y0 + r;
                    const long long oa = col_a + y * row_stride;
                    const long long ob = col_b + y * row_stride;
                    const float sar = bf16_to_f32(src_ra[oa]);
                    const float sai = bf16_to_f32(src_ia[oa]);
                    const float sbr = bf16_to_f32(src_rb[ob]);
                    const float sbi = bf16_to_f32(src_ib[ob]);
                    const float n_par = ar * par - ai * pai + (gr * sar - gi * sai);
                    const float n_pai = ar * pai + ai * par + (gr * sai + gi * sar);
                    const float n_pbr = ar * pbr - ai * pbi + (gr * sbr - gi * sbi);
                    const float n_pbi = ar * pbi + ai * pbr + (gr * sbi + gi * sbr);
                    par = n_par; pai = n_pai; pbr = n_pbr; pbi = n_pbi;
                    if (c == NCHUNKS - 1) {
                        my_seg[(r * 4 + 0) * 32 + lane] = par;
                        my_seg[(r * 4 + 1) * 32 + lane] = pai;
                        my_seg[(r * 4 + 2) * 32 + lane] = pbr;
                        my_seg[(r * 4 + 3) * 32 + lane] = pbi;
                    }
                }
            }
        }

        // PASS 2: Reverse chunks c = NCHUNKS-1 .. 0
        float nar = 0.0f, nai = 0.0f, nbr = 0.0f, nbi = 0.0f;
        float l0r = 0.0f, l0i = 0.0f, l1r = 0.0f, l1i = 0.0f;

        for (int c = NCHUNKS - 1; c >= 0; --c) {
            const int y0 = c * CHUNK;
            float p_prev_ar = 0.0f, p_prev_ai = 0.0f, p_prev_br = 0.0f, p_prev_bi = 0.0f;
            if constexpr (NCHUNKS > 1) {
                if (c > 0) {
                    p_prev_ar = my_ckpt[((c - 1) * 4 + 0) * 32 + lane];
                    p_prev_ai = my_ckpt[((c - 1) * 4 + 1) * 32 + lane];
                    p_prev_br = my_ckpt[((c - 1) * 4 + 2) * 32 + lane];
                    p_prev_bi = my_ckpt[((c - 1) * 4 + 3) * 32 + lane];
                }
                if (c < NCHUNKS - 1) {
                    my_ckpt[(c * 4 + 0) * 32 + lane] = nar;
                    my_ckpt[(c * 4 + 1) * 32 + lane] = nai;
                    my_ckpt[(c * 4 + 2) * 32 + lane] = nbr;
                    my_ckpt[(c * 4 + 3) * 32 + lane] = nbi;
                    float par = p_prev_ar, pai = p_prev_ai, pbr = p_prev_br, pbi = p_prev_bi;
                    for (int r = 0; r < CHUNK; ++r) {
                        const int y = y0 + r;
                        const long long oa = col_a + y * row_stride;
                        const long long ob = col_b + y * row_stride;
                        const float sar = bf16_to_f32(src_ra[oa]);
                        const float sai = bf16_to_f32(src_ia[oa]);
                        const float sbr = bf16_to_f32(src_rb[ob]);
                        const float sbi = bf16_to_f32(src_ib[ob]);
                        const float n_par = ar * par - ai * pai + (gr * sar - gi * sai);
                        const float n_pai = ar * pai + ai * par + (gr * sai + gi * sar);
                        const float n_pbr = ar * pbr - ai * pbi + (gr * sbr - gi * sbi);
                        const float n_pbi = ar * pbi + ai * pbr + (gr * sbi + gi * sbr);
                        par = n_par; pai = n_pai; pbr = n_pbr; pbi = n_pbi;
                        my_seg[(r * 4 + 0) * 32 + lane] = par;
                        my_seg[(r * 4 + 1) * 32 + lane] = pai;
                        my_seg[(r * 4 + 2) * 32 + lane] = pbr;
                        my_seg[(r * 4 + 3) * 32 + lane] = pbi;
                    }
                }
            }

            if constexpr (!COARSE) {
                for (int r = CHUNK - 1; r >= 0; --r) {
                    const int y = y0 + r;
                    const long long oa = col_a + y * row_stride;
                    const long long ob = col_b + y * row_stride;
                    const float sar = bf16_to_f32(src_ra[oa]);
                    const float sai = bf16_to_f32(src_ia[oa]);
                    const float sbr = bf16_to_f32(src_rb[ob]);
                    const float sbi = bf16_to_f32(src_ib[ob]);

                    {
                        const float n_nar = ar * nar + ai * nai + (gr * sar + gi * sai);
                        const float n_nai = ar * nai - ai * nar + (gr * sai - gi * sar);
                        const float n_nbr = ar * nbr + ai * nbi + (gr * sbr + gi * sbi);
                        const float n_nbi = ar * nbi - ai * nbr + (gr * sbi - gi * sbr);
                        nar = n_nar; nai = n_nai; nbr = n_nbr; nbi = n_nbi;
                    }

                    const float vyp = variance_y[y * M + m];
                    const float vyn = variance_y[(H + y) * M + m];
                    const float pa = rsqrtf(fmaxf(vxp * vyp, eps));
                    const float pb = rsqrtf(fmaxf(vxn * vyp, eps));
                    const float na = rsqrtf(fmaxf(vxp * vyn, eps));
                    const float nb = rsqrtf(fmaxf(vxn * vyn, eps));

                    const float p0 = round_bf16(my_seg[(r * 4 + 0) * 32 + lane] * pa);
                    const float p1 = round_bf16(my_seg[(r * 4 + 2) * 32 + lane] * pb);
                    const float p2 = round_bf16(nar * na);
                    const float p3 = round_bf16(nbr * nb);
                    const float p4 = round_bf16(my_seg[(r * 4 + 1) * 32 + lane] * pa);
                    const float p5 = round_bf16(my_seg[(r * 4 + 3) * 32 + lane] * pb);
                    const float p6 = round_bf16(nai * na);
                    const float p7 = round_bf16(nbi * nb);

                    const long long go = col_g + y * out_row_stride;
                    const float dyr = bf16_to_f32(grad_cr[go]);
                    const float dyi = bf16_to_f32(grad_ci[go]);
                    if constexpr (ACC_IN_SMEM) {
                        my_acc[24 * 32 + lane] += dyr;
                        my_acc[25 * 32 + lane] += dyi;
                    } else {
                        d_bout_r_reg += dyr;
                        d_bout_i_reg += dyi;
                    }

                    float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0, dp4 = 0, dp5 = 0, dp6 = 0, dp7 = 0;
                    #pragma unroll
                    for (int h = 0; h < 8; ++h) {
                        float w0, w1, w2, w3, w4, w5, w6, w7;
                        unpack_bf16x2(win_smem[(h * 4 + 0) * 32 + lane], w0, w1);
                        unpack_bf16x2(win_smem[(h * 4 + 1) * 32 + lane], w2, w3);
                        unpack_bf16x2(win_smem[(h * 4 + 2) * 32 + lane], w4, w5);
                        unpack_bf16x2(win_smem[(h * 4 + 3) * 32 + lane], w6, w7);

                        float b_lo, b_hi;
                        unpack_bf16x2(bin_smem[(h >> 1) * 32 + lane], b_lo, b_hi);
                        float pre = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3
                                  + w4 * p4 + w5 * p5 + w6 * p6 + w7 * p7;
                        pre = round_bf16(pre + ((h & 1) ? b_hi : b_lo));
                        float sig = fast_sigmoid(pre);
                        float hid = round_bf16(pre * sig);
                        float silu_d = sig * (1.0f + pre * (1.0f - sig));

                        if constexpr (ACC_IN_SMEM) {
                            my_acc[(8 + h) * 32 + lane] += dyr * hid;
                            my_acc[(16 + h) * 32 + lane] += dyi * hid;
                        } else {
                            d_wout_r_reg[h] += dyr * hid;
                            d_wout_i_reg[h] += dyi * hid;
                        }

                        float wr_lo, wr_hi, wi_lo, wi_hi;
                        unpack_bf16x2(wout_smem[(0 * 4 + (h >> 1)) * 32 + lane], wr_lo, wr_hi);
                        unpack_bf16x2(wout_smem[(1 * 4 + (h >> 1)) * 32 + lane], wi_lo, wi_hi);
                        float wr = (h & 1) ? wr_hi : wr_lo;
                        float wi = (h & 1) ? wi_hi : wi_lo;

                        float dpre = round_bf16((dyr * wr + dyi * wi) * silu_d);
                        if constexpr (ACC_IN_SMEM) {
                            my_acc[h * 32 + lane] += dpre;
                        } else {
                            d_bin_reg[h] += dpre;
                        }
                        d_win[h][0] += dpre * p0; d_win[h][1] += dpre * p1;
                        d_win[h][2] += dpre * p2; d_win[h][3] += dpre * p3;
                        d_win[h][4] += dpre * p4; d_win[h][5] += dpre * p5;
                        d_win[h][6] += dpre * p6; d_win[h][7] += dpre * p7;

                        dp0 += dpre * w0; dp1 += dpre * w1;
                        dp2 += dpre * w2; dp3 += dpre * w3;
                        dp4 += dpre * w4; dp5 += dpre * w5;
                        dp6 += dpre * w6; dp7 += dpre * w7;
                    }

                    my_g2[(y * 2 + 0) * 32 + lane] = pack_bf16x2(dp2, dp6);
                    my_g2[(y * 2 + 1) * 32 + lane] = pack_bf16x2(dp3, dp7);

                    {
                        const float n_l0r = ar * l0r + ai * l0i + dp0 * pa;
                        const float n_l0i = ar * l0i - ai * l0r + dp4 * pa;
                        const float n_l1r = ar * l1r + ai * l1i + dp1 * pb;
                        const float n_l1i = ar * l1i - ai * l1r + dp5 * pb;
                        l0r = n_l0r; l0i = n_l0i; l1r = n_l1r; l1i = n_l1i;
                    }

                    const float prev_par = (r > 0) ? my_seg[((r - 1) * 4 + 0) * 32 + lane] : p_prev_ar;
                    const float prev_pai = (r > 0) ? my_seg[((r - 1) * 4 + 1) * 32 + lane] : p_prev_ai;
                    const float prev_pbr = (r > 0) ? my_seg[((r - 1) * 4 + 2) * 32 + lane] : p_prev_br;
                    const float prev_pbi = (r > 0) ? my_seg[((r - 1) * 4 + 3) * 32 + lane] : p_prev_bi;

                    d_dr += l0r * prev_par + l0i * prev_pai + l1r * prev_pbr + l1i * prev_pbi;
                    d_di += -l0r * prev_pai + l0i * prev_par - l1r * prev_pbi + l1i * prev_pbr;
                    d_gr += l0r * sar + l0i * sai + l1r * sbr + l1i * sbi;
                    d_gi += -l0r * sai + l0i * sar - l1r * sbi + l1i * sbr;

                    grad_src[oa] = f32_to_bf16(gr * l0r + gi * l0i);
                    grad_src[plane_size + oa] = f32_to_bf16(-gi * l0r + gr * l0i);
                    grad_src[2 * plane_size + ob] = f32_to_bf16(gr * l1r + gi * l1i);
                    grad_src[3 * plane_size + ob] = f32_to_bf16(-gi * l1r + gr * l1i);
                }
            } else {
                // COARSE == true: CHUNK is even, walk pairs (r_odd, r_even)
                for (int r = CHUNK - 1; r >= 1; r -= 2) {
                    const int y_odd = y0 + r;
                    const int y_even = y_odd - 1;
                    const int cy = y_even >> 1;

                    {
                        const long long oa_o = col_a + y_odd * row_stride;
                        const long long ob_o = col_b + y_odd * row_stride;
                        const float sar_o = bf16_to_f32(src_ra[oa_o]);
                        const float sai_o = bf16_to_f32(src_ia[oa_o]);
                        const float sbr_o = bf16_to_f32(src_rb[ob_o]);
                        const float sbi_o = bf16_to_f32(src_ib[ob_o]);
                        const float n_nar = ar * nar + ai * nai + (gr * sar_o + gi * sai_o);
                        const float n_nai = ar * nai - ai * nar + (gr * sai_o - gi * sar_o);
                        const float n_nbr = ar * nbr + ai * nbi + (gr * sbr_o + gi * sbi_o);
                        const float n_nbi = ar * nbi - ai * nbr + (gr * sbi_o - gi * sbr_o);
                        nar = n_nar; nai = n_nai; nbr = n_nbr; nbi = n_nbi;
                    }
                    {
                        const long long oa_e = col_a + y_even * row_stride;
                        const long long ob_e = col_b + y_even * row_stride;
                        const float sar_e = bf16_to_f32(src_ra[oa_e]);
                        const float sai_e = bf16_to_f32(src_ia[oa_e]);
                        const float sbr_e = bf16_to_f32(src_rb[ob_e]);
                        const float sbi_e = bf16_to_f32(src_ib[ob_e]);
                        const float n_nar = ar * nar + ai * nai + (gr * sar_e + gi * sai_e);
                        const float n_nai = ar * nai - ai * nar + (gr * sai_e - gi * sar_e);
                        const float n_nbr = ar * nbr + ai * nbi + (gr * sbr_e + gi * sbi_e);
                        const float n_nbi = ar * nbi - ai * nbr + (gr * sbi_e - gi * sbr_e);
                        nar = n_nar; nai = n_nai; nbr = n_nbr; nbi = n_nbi;
                    }

                    const float vyp_o = variance_y[y_odd * M + m];
                    const float vyn_e = variance_y[(H + y_even) * M + m];
                    const float pa_o = rsqrtf(fmaxf(vxp * vyp_o, eps));
                    const float pb_o = rsqrtf(fmaxf(vxn * vyp_o, eps));
                    const float na_e = rsqrtf(fmaxf(vxp * vyn_e, eps));
                    const float nb_e = rsqrtf(fmaxf(vxn * vyn_e, eps));

                    const float p0 = round_bf16(my_seg[(r * 4 + 0) * 32 + lane] * pa_o);
                    const float p1 = round_bf16(my_seg[(r * 4 + 2) * 32 + lane] * pb_o);
                    const float p2 = round_bf16(nar * na_e);
                    const float p3 = round_bf16(nbr * nb_e);
                    const float p4 = round_bf16(my_seg[(r * 4 + 1) * 32 + lane] * pa_o);
                    const float p5 = round_bf16(my_seg[(r * 4 + 3) * 32 + lane] * pb_o);
                    const float p6 = round_bf16(nai * na_e);
                    const float p7 = round_bf16(nbi * nb_e);

                    const long long go = col_g + cy * out_row_stride;
                    const float dyr = bf16_to_f32(grad_cr[go]);
                    const float dyi = bf16_to_f32(grad_ci[go]);
                    if constexpr (ACC_IN_SMEM) {
                        my_acc[24 * 32 + lane] += dyr;
                        my_acc[25 * 32 + lane] += dyi;
                    } else {
                        d_bout_r_reg += dyr;
                        d_bout_i_reg += dyi;
                    }

                    float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0, dp4 = 0, dp5 = 0, dp6 = 0, dp7 = 0;
                    #pragma unroll
                    for (int h = 0; h < 8; ++h) {
                        float w0, w1, w2, w3, w4, w5, w6, w7;
                        unpack_bf16x2(win_smem[(h * 4 + 0) * 32 + lane], w0, w1);
                        unpack_bf16x2(win_smem[(h * 4 + 1) * 32 + lane], w2, w3);
                        unpack_bf16x2(win_smem[(h * 4 + 2) * 32 + lane], w4, w5);
                        unpack_bf16x2(win_smem[(h * 4 + 3) * 32 + lane], w6, w7);

                        float b_lo, b_hi;
                        unpack_bf16x2(bin_smem[(h >> 1) * 32 + lane], b_lo, b_hi);
                        float pre = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3
                                  + w4 * p4 + w5 * p5 + w6 * p6 + w7 * p7;
                        pre = round_bf16(pre + ((h & 1) ? b_hi : b_lo));
                        float sig = fast_sigmoid(pre);
                        float hid = round_bf16(pre * sig);
                        float silu_d = sig * (1.0f + pre * (1.0f - sig));

                        if constexpr (ACC_IN_SMEM) {
                            my_acc[(8 + h) * 32 + lane] += dyr * hid;
                            my_acc[(16 + h) * 32 + lane] += dyi * hid;
                        } else {
                            d_wout_r_reg[h] += dyr * hid;
                            d_wout_i_reg[h] += dyi * hid;
                        }

                        float wr_lo, wr_hi, wi_lo, wi_hi;
                        unpack_bf16x2(wout_smem[(0 * 4 + (h >> 1)) * 32 + lane], wr_lo, wr_hi);
                        unpack_bf16x2(wout_smem[(1 * 4 + (h >> 1)) * 32 + lane], wi_lo, wi_hi);
                        float wr = (h & 1) ? wr_hi : wr_lo;
                        float wi = (h & 1) ? wi_hi : wi_lo;

                        float dpre = round_bf16((dyr * wr + dyi * wi) * silu_d);
                        if constexpr (ACC_IN_SMEM) {
                            my_acc[h * 32 + lane] += dpre;
                        } else {
                            d_bin_reg[h] += dpre;
                        }
                        d_win[h][0] += dpre * p0; d_win[h][1] += dpre * p1;
                        d_win[h][2] += dpre * p2; d_win[h][3] += dpre * p3;
                        d_win[h][4] += dpre * p4; d_win[h][5] += dpre * p5;
                        d_win[h][6] += dpre * p6; d_win[h][7] += dpre * p7;

                        dp0 += dpre * w0; dp1 += dpre * w1;
                        dp2 += dpre * w2; dp3 += dpre * w3;
                        dp4 += dpre * w4; dp5 += dpre * w5;
                        dp6 += dpre * w6; dp7 += dpre * w7;
                    }

                    my_g2[(cy * 2 + 0) * 32 + lane] = pack_bf16x2(dp2, dp6);
                    my_g2[(cy * 2 + 1) * 32 + lane] = pack_bf16x2(dp3, dp7);

                    // Positive adjoint at y_odd (inject dp[0,4,1,5])
                    {
                        const long long oa_o = col_a + y_odd * row_stride;
                        const long long ob_o = col_b + y_odd * row_stride;
                        const float n_l0r = ar * l0r + ai * l0i + round_bf16(dp0) * pa_o;
                        const float n_l0i = ar * l0i - ai * l0r + round_bf16(dp4) * pa_o;
                        const float n_l1r = ar * l1r + ai * l1i + round_bf16(dp1) * pb_o;
                        const float n_l1i = ar * l1i - ai * l1r + round_bf16(dp5) * pb_o;
                        l0r = n_l0r; l0i = n_l0i; l1r = n_l1r; l1i = n_l1i;

                        const float par_e = my_seg[((r - 1) * 4 + 0) * 32 + lane];
                        const float pai_e = my_seg[((r - 1) * 4 + 1) * 32 + lane];
                        const float pbr_e = my_seg[((r - 1) * 4 + 2) * 32 + lane];
                        const float pbi_e = my_seg[((r - 1) * 4 + 3) * 32 + lane];
                        const float sar_o = bf16_to_f32(src_ra[oa_o]);
                        const float sai_o = bf16_to_f32(src_ia[oa_o]);
                        const float sbr_o = bf16_to_f32(src_rb[ob_o]);
                        const float sbi_o = bf16_to_f32(src_ib[ob_o]);

                        d_dr += l0r * par_e + l0i * pai_e + l1r * pbr_e + l1i * pbi_e;
                        d_di += -l0r * pai_e + l0i * par_e - l1r * pbi_e + l1i * pbr_e;
                        d_gr += l0r * sar_o + l0i * sai_o + l1r * sbr_o + l1i * sbi_o;
                        d_gi += -l0r * sai_o + l0i * sar_o - l1r * sbi_o + l1i * sbr_o;

                        grad_src[oa_o] = f32_to_bf16(gr * l0r + gi * l0i);
                        grad_src[plane_size + oa_o] = f32_to_bf16(-gi * l0r + gr * l0i);
                        grad_src[2 * plane_size + ob_o] = f32_to_bf16(gr * l1r + gi * l1i);
                        grad_src[3 * plane_size + ob_o] = f32_to_bf16(-gi * l1r + gr * l1i);
                    }
                    // Positive adjoint at y_even (inject 0)
                    {
                        const long long oa_e = col_a + y_even * row_stride;
                        const long long ob_e = col_b + y_even * row_stride;
                        const float n_l0r = ar * l0r + ai * l0i;
                        const float n_l0i = ar * l0i - ai * l0r;
                        const float n_l1r = ar * l1r + ai * l1i;
                        const float n_l1i = ar * l1i - ai * l1r;
                        l0r = n_l0r; l0i = n_l0i; l1r = n_l1r; l1i = n_l1i;

                        const float prev_par_e = (r > 1) ? my_seg[((r - 2) * 4 + 0) * 32 + lane] : p_prev_ar;
                        const float prev_pai_e = (r > 1) ? my_seg[((r - 2) * 4 + 1) * 32 + lane] : p_prev_ai;
                        const float prev_pbr_e = (r > 1) ? my_seg[((r - 2) * 4 + 2) * 32 + lane] : p_prev_br;
                        const float prev_pbi_e = (r > 1) ? my_seg[((r - 2) * 4 + 3) * 32 + lane] : p_prev_bi;
                        const float sar_e = bf16_to_f32(src_ra[oa_e]);
                        const float sai_e = bf16_to_f32(src_ia[oa_e]);
                        const float sbr_e = bf16_to_f32(src_rb[ob_e]);
                        const float sbi_e = bf16_to_f32(src_ib[ob_e]);

                        d_dr += l0r * prev_par_e + l0i * prev_pai_e + l1r * prev_pbr_e + l1i * prev_pbi_e;
                        d_di += -l0r * prev_pai_e + l0i * prev_par_e - l1r * prev_pbi_e + l1i * prev_pbr_e;
                        d_gr += l0r * sar_e + l0i * sai_e + l1r * sbr_e + l1i * sbi_e;
                        d_gi += -l0r * sai_e + l0i * sar_e - l1r * sbi_e + l1i * sbr_e;

                        grad_src[oa_e] = f32_to_bf16(gr * l0r + gi * l0i);
                        grad_src[plane_size + oa_e] = f32_to_bf16(-gi * l0r + gr * l0i);
                        grad_src[2 * plane_size + ob_e] = f32_to_bf16(gr * l1r + gi * l1i);
                        grad_src[3 * plane_size + ob_e] = f32_to_bf16(-gi * l1r + gr * l1i);
                    }
                }
            }
        }

        // PASS 3: Forward chunks c = 0 .. NCHUNKS-1
        float l2r = 0.0f, l2i = 0.0f, l3r = 0.0f, l3i = 0.0f;
        for (int c = 0; c < NCHUNKS; ++c) {
            const int y0 = c * CHUNK;
            float n_after_ar = 0.0f, n_after_ai = 0.0f, n_after_br = 0.0f, n_after_bi = 0.0f;
            if constexpr (NCHUNKS > 1) {
                if (c < NCHUNKS - 1) {
                    n_after_ar = my_ckpt[(c * 4 + 0) * 32 + lane];
                    n_after_ai = my_ckpt[(c * 4 + 1) * 32 + lane];
                    n_after_br = my_ckpt[(c * 4 + 2) * 32 + lane];
                    n_after_bi = my_ckpt[(c * 4 + 3) * 32 + lane];
                }
            }
            {
                float cnar = n_after_ar, cnai = n_after_ai, cnbr = n_after_br, cnbi = n_after_bi;
                for (int r = CHUNK - 1; r >= 0; --r) {
                    const int y = y0 + r;
                    const long long oa = col_a + y * row_stride;
                    const long long ob = col_b + y * row_stride;
                    const float sar = bf16_to_f32(src_ra[oa]);
                    const float sai = bf16_to_f32(src_ia[oa]);
                    const float sbr = bf16_to_f32(src_rb[ob]);
                    const float sbi = bf16_to_f32(src_ib[ob]);
                    const float n_nar = ar * cnar + ai * cnai + (gr * sar + gi * sai);
                    const float n_nai = ar * cnai - ai * cnar + (gr * sai - gi * sar);
                    const float n_nbr = ar * cnbr + ai * cnbi + (gr * sbr + gi * sbi);
                    const float n_nbi = ar * cnbi - ai * cnbr + (gr * sbi - gi * sbr);
                    cnar = n_nar; cnai = n_nai; cnbr = n_nbr; cnbi = n_nbi;
                    my_seg[(r * 4 + 0) * 32 + lane] = cnar;
                    my_seg[(r * 4 + 1) * 32 + lane] = cnai;
                    my_seg[(r * 4 + 2) * 32 + lane] = cnbr;
                    my_seg[(r * 4 + 3) * 32 + lane] = cnbi;
                }
            }
            for (int r = 0; r < CHUNK; ++r) {
                const int y = y0 + r;
                const long long oa = col_a + y * row_stride;
                const long long ob = col_b + y * row_stride;

                float g2v = 0.0f, g6v = 0.0f, g3v = 0.0f, g7v = 0.0f;
                if (!COARSE || ((y & 1) == 0)) {
                    const int t = COARSE ? (y >> 1) : y;
                    unpack_bf16x2(my_g2[(t * 2 + 0) * 32 + lane], g2v, g6v);
                    unpack_bf16x2(my_g2[(t * 2 + 1) * 32 + lane], g3v, g7v);
                    const float vyn = variance_y[(H + y) * M + m];
                    const float na = rsqrtf(fmaxf(vxp * vyn, eps));
                    const float nb = rsqrtf(fmaxf(vxn * vyn, eps));
                    g2v *= na; g6v *= na; g3v *= nb; g7v *= nb;
                }

                {
                    const float n_l2r = ar * l2r - ai * l2i + g2v;
                    const float n_l2i = ar * l2i + ai * l2r + g6v;
                    const float n_l3r = ar * l3r - ai * l3i + g3v;
                    const float n_l3i = ar * l3i + ai * l3r + g7v;
                    l2r = n_l2r; l2i = n_l2i; l3r = n_l3r; l3i = n_l3i;
                }

                const float prev_nar = (r + 1 < CHUNK) ? my_seg[((r + 1) * 4 + 0) * 32 + lane] : n_after_ar;
                const float prev_nai = (r + 1 < CHUNK) ? my_seg[((r + 1) * 4 + 1) * 32 + lane] : n_after_ai;
                const float prev_nbr = (r + 1 < CHUNK) ? my_seg[((r + 1) * 4 + 2) * 32 + lane] : n_after_br;
                const float prev_nbi = (r + 1 < CHUNK) ? my_seg[((r + 1) * 4 + 3) * 32 + lane] : n_after_bi;

                const float sar = bf16_to_f32(src_ra[oa]);
                const float sai = bf16_to_f32(src_ia[oa]);
                const float sbr = bf16_to_f32(src_rb[ob]);
                const float sbi = bf16_to_f32(src_ib[ob]);

                d_dr += l2r * prev_nar + l2i * prev_nai + l3r * prev_nbr + l3i * prev_nbi;
                d_di += l2r * prev_nai - l2i * prev_nar + l3r * prev_nbi - l3i * prev_nbr;
                d_gr += l2r * sar + l2i * sai + l3r * sbr + l3i * sbi;
                d_gi += l2r * sai - l2i * sar + l3r * sbi - l3i * sbr;

                grad_src[oa] = f32_to_bf16(bf16_to_f32(grad_src[oa]) + (gr * l2r - gi * l2i));
                grad_src[plane_size + oa] = f32_to_bf16(bf16_to_f32(grad_src[plane_size + oa]) + (gi * l2r + gr * l2i));
                grad_src[2 * plane_size + ob] = f32_to_bf16(bf16_to_f32(grad_src[2 * plane_size + ob]) + (gr * l3r - gi * l3i));
                grad_src[3 * plane_size + ob] = f32_to_bf16(bf16_to_f32(grad_src[3 * plane_size + ob]) + (gi * l3r + gr * l3i));
            }
        }
    }

    __syncthreads();
    const bool use_atomic = (x_chunks > 1);
    const long long b_m = static_cast<long long>(b) * M + m;

    if constexpr (ACC_IN_SMEM) {
        if (warp == 0) {
            auto red_acc = [&](int idx, float* dst) {
                float s = acc_base[idx * 32 + lane];
                #pragma unroll
                for (int w = 1; w < WARPS; ++w) s += acc_base[(w * 26 + idx) * 32 + lane];
                if (use_atomic) atomicAdd(dst, s);
                else *dst = s;
            };
            red_acc(24, grad_bout + b_m * 2 + 0);
            red_acc(25, grad_bout + b_m * 2 + 1);
            for (int h = 0; h < 8; ++h) {
                red_acc(h, grad_bin + b_m * 8 + h);
                red_acc(8 + h, grad_wout + b_m * 16 + h);
                red_acc(16 + h, grad_wout + b_m * 16 + 8 + h);
            }
        }
        __syncthreads();
    }

    float* red = reinterpret_cast<float*>(smem);
    float* my_r = red + warp * (32 * 32);

    my_r[0 * 32 + lane] = d_dr;
    my_r[1 * 32 + lane] = d_di;
    my_r[2 * 32 + lane] = d_gr;
    my_r[3 * 32 + lane] = d_gi;
    if constexpr (!ACC_IN_SMEM) {
        my_r[4 * 32 + lane] = d_bout_r_reg;
        my_r[5 * 32 + lane] = d_bout_i_reg;
        #pragma unroll
        for (int h = 0; h < 8; ++h) {
            my_r[(6 + h) * 32 + lane] = d_bin_reg[h];
            my_r[(14 + h) * 32 + lane] = d_wout_r_reg[h];
            my_r[(22 + h) * 32 + lane] = d_wout_i_reg[h];
        }
    }
    __syncthreads();
    if (warp == 0) {
        auto rw = [&](int slot, float* dst) {
            float s = red[slot * 32 + lane];
            #pragma unroll
            for (int w = 1; w < WARPS; ++w) s += red[(w * 32 + slot) * 32 + lane];
            if (use_atomic) atomicAdd(dst, s);
            else *dst = s;
        };
        rw(0, grad_dr + b_m);
        rw(1, grad_di + b_m);
        rw(2, grad_gr + b_m);
        rw(3, grad_gi + b_m);
        if constexpr (!ACC_IN_SMEM) {
            rw(4, grad_bout + b_m * 2 + 0);
            rw(5, grad_bout + b_m * 2 + 1);
            for (int h = 0; h < 8; ++h) {
                rw(6 + h, grad_bin + b_m * 8 + h);
                rw(14 + h, grad_wout + b_m * 16 + h);
                rw(22 + h, grad_wout + b_m * 16 + 8 + h);
            }
        }
    }
    __syncthreads();

    for (int h0 = 0; h0 < 8; h0 += 4) {
        #pragma unroll
        for (int dh = 0; dh < 4; ++dh)
            #pragma unroll
            for (int k = 0; k < 8; ++k) my_r[(dh * 8 + k) * 32 + lane] = d_win[h0 + dh][k];
        __syncthreads();
        if (warp == 0) {
            for (int dh = 0; dh < 4; ++dh) {
                float* dst_row = grad_win + (b_m * 8 + (h0 + dh)) * 8;
                for (int k = 0; k < 8; ++k) {
                    int slot = dh * 8 + k;
                    float s = red[slot * 32 + lane];
                    #pragma unroll
                    for (int w = 1; w < WARPS; ++w) s += red[(w * 32 + slot) * 32 + lane];
                    if (use_atomic) atomicAdd(dst_row + k, s);
                    else dst_row[k] = s;
                }
            }
        }
        __syncthreads();
    }
}

#define FWD_ARGS \
    const float* __restrict__ decay_real, \
    const float* __restrict__ decay_imag, \
    const float* __restrict__ gamma_real, \
    const float* __restrict__ gamma_imag, \
    const unsigned short* __restrict__ src_ra, \
    const unsigned short* __restrict__ src_ia, \
    const unsigned short* __restrict__ src_rb, \
    const unsigned short* __restrict__ src_ib, \
    const float* __restrict__ variance_x, \
    const float* __restrict__ variance_y, \
    const float* __restrict__ path_in_w, \
    const float* __restrict__ path_in_b, \
    const float* __restrict__ path_out_w, \
    const float* __restrict__ path_out_b, \
    unsigned short* __restrict__ out_r, \
    unsigned short* __restrict__ out_i, \
    int B, int M, int x_chunks, float eps

#define FWD_CALL \
    decay_real, decay_imag, gamma_real, gamma_imag, \
    src_ra, src_ia, src_rb, src_ib, \
    variance_x, variance_y, \
    path_in_w, path_in_b, path_out_w, path_out_b, \
    out_r, out_i, B, M, x_chunks, eps

#define BWD_ARGS \
    const float* __restrict__ decay_real, \
    const float* __restrict__ decay_imag, \
    const float* __restrict__ gamma_real, \
    const float* __restrict__ gamma_imag, \
    const unsigned short* __restrict__ src_ra, \
    const unsigned short* __restrict__ src_ia, \
    const unsigned short* __restrict__ src_rb, \
    const unsigned short* __restrict__ src_ib, \
    const float* __restrict__ variance_x, \
    const float* __restrict__ variance_y, \
    const float* __restrict__ path_in_w, \
    const float* __restrict__ path_in_b, \
    const float* __restrict__ path_out_w, \
    const unsigned short* __restrict__ grad_cr, \
    const unsigned short* __restrict__ grad_ci, \
    float* __restrict__ grad_win, \
    float* __restrict__ grad_bin, \
    float* __restrict__ grad_wout, \
    float* __restrict__ grad_bout, \
    float* __restrict__ grad_dr, \
    float* __restrict__ grad_di, \
    float* __restrict__ grad_gr, \
    float* __restrict__ grad_gi, \
    unsigned short* __restrict__ grad_src, \
    int B, int M, int x_chunks, float eps

#define BWD_CALL \
    decay_real, decay_imag, gamma_real, gamma_imag, \
    src_ra, src_ia, src_rb, src_ib, \
    variance_x, variance_y, \
    path_in_w, path_in_b, path_out_w, \
    grad_cr, grad_ci, \
    grad_win, grad_bin, grad_wout, grad_bout, \
    grad_dr, grad_di, grad_gr, grad_gi, \
    grad_src, B, M, x_chunks, eps

extern "C" {
__launch_bounds__(128, 8) __global__ void vscan_fwd_56_d(FWD_ARGS) { vscan_forward_impl<56, false, 4, 8>(FWD_CALL); }
__launch_bounds__(128, 8) __global__ void vscan_fwd_56_c(FWD_ARGS) { vscan_forward_impl<56, true,  4, 8>(FWD_CALL); }
__launch_bounds__(128, 8) __global__ void vscan_fwd_28_d(FWD_ARGS) { vscan_forward_impl<28, false, 4, 8>(FWD_CALL); }
__launch_bounds__(128, 8) __global__ void vscan_fwd_28_c(FWD_ARGS) { vscan_forward_impl<28, true,  4, 8>(FWD_CALL); }
__launch_bounds__(64,  8) __global__ void vscan_fwd_14_d(FWD_ARGS) { vscan_forward_impl<14, false, 2, 8>(FWD_CALL); }
__launch_bounds__(64,  8) __global__ void vscan_fwd_14_c(FWD_ARGS) { vscan_forward_impl<14, true,  2, 8>(FWD_CALL); }

__launch_bounds__(64,  4) __global__ void vscan_bwd_56_d(BWD_ARGS) { vscan_backward_impl<56, false, 2, 8,  false>(BWD_CALL); }
__launch_bounds__(128, 3) __global__ void vscan_bwd_56_c(BWD_ARGS) { vscan_backward_impl<56, true,  4, 8,  true >(BWD_CALL); }
__launch_bounds__(128, 3) __global__ void vscan_bwd_28_d(BWD_ARGS) { vscan_backward_impl<28, false, 4, 4,  true >(BWD_CALL); }
__launch_bounds__(128, 3) __global__ void vscan_bwd_28_c(BWD_ARGS) { vscan_backward_impl<28, true,  4, 4,  true >(BWD_CALL); }
__launch_bounds__(64,  6) __global__ void vscan_bwd_14_d(BWD_ARGS) { vscan_backward_impl<14, false, 2, 14, true >(BWD_CALL); }
__launch_bounds__(64,  6) __global__ void vscan_bwd_14_c(BWD_ARGS) { vscan_backward_impl<14, true,  2, 14, true >(BWD_CALL); }
}
"""

_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

# (height, coarse) -> (kernel_name, warps, chunk, acc_in_smem)
_FWD_SPECS: dict[tuple[int, bool], tuple[str, int]] = {
    (56, False): ("vscan_fwd_56_d", 4),
    (56, True): ("vscan_fwd_56_c", 4),
    (28, False): ("vscan_fwd_28_d", 4),
    (28, True): ("vscan_fwd_28_c", 4),
    (14, False): ("vscan_fwd_14_d", 2),
    (14, True): ("vscan_fwd_14_c", 2),
}

_BWD_SPECS: dict[tuple[int, bool], tuple[str, int, int, bool]] = {
    (56, False): ("vscan_bwd_56_d", 2, 8, False),
    (56, True): ("vscan_bwd_56_c", 4, 8, True),
    (28, False): ("vscan_bwd_28_d", 4, 4, True),
    (28, True): ("vscan_bwd_28_c", 4, 4, True),
    (14, False): ("vscan_bwd_14_d", 2, 14, True),
    (14, True): ("vscan_bwd_14_c", 2, 14, True),
}


def _fwd_smem_bytes(height: int, coarse: bool, warps: int) -> int:
    oh = height // 2 if coarse else height
    weights = (8 * 4 + 2 * 4 + 4) * 32 * 4  # 5632 B
    pos_buf = warps * oh * 2 * 32 * 4
    return weights + pos_buf


def _bwd_smem_bytes(height: int, coarse: bool, warps: int, chunk: int, acc_in_smem: bool) -> int:
    oh = height // 2 if coarse else height
    nchunks = height // chunk
    weights = (8 * 4 + 2 * 4 + 4) * 32 * 4  # 5632 B
    g2_buf = warps * oh * 2 * 32 * 4
    ckpt_buf = (warps * (nchunks - 1) * 4 * 32 * 4) if nchunks > 1 else 0
    seg_buf = warps * chunk * 4 * 32 * 4
    acc_buf = (warps * 26 * 32 * 4) if acc_in_smem else 0
    work = weights + g2_buf + ckpt_buf + seg_buf + acc_buf
    red = warps * 32 * 32 * 4  # Reduction buffer reuses smem at kernel epilogue
    return max(work, red)


def _find_nvrtc_lib() -> ctypes.CDLL:
    candidates: list[str] = []
    spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
    if spec is not None and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            lib_dir = Path(loc) / "lib"
            for name in ("libnvrtc.so.12", "libnvrtc.so"):
                p = lib_dir / name
                if p.exists():
                    candidates.append(str(p))
    candidates.extend(
        [
            "libnvrtc.so.12",
            "libnvrtc.so",
            "/usr/local/cuda/lib64/libnvrtc.so.12",
            "/usr/local/cuda/lib64/libnvrtc.so",
        ]
    )
    last_err: Exception | None = None
    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError as exc:
            last_err = exc
    raise RuntimeError(f"Unable to load libnvrtc: {last_err}")


def _load_cuda_driver() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1")


def _check_nvrtc(nvrtc: ctypes.CDLL, status: int, prog: ctypes.c_void_p | None = None) -> None:
    if status == 0:
        return
    log_msg = ""
    if prog is not None:
        log_size = ctypes.c_size_t(0)
        if nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size)) == 0 and log_size.value > 1:
            buf = ctypes.create_string_buffer(log_size.value)
            nvrtc.nvrtcGetProgramLog(prog, buf)
            log_msg = "\n" + buf.value.decode("utf-8", errors="replace")
    raise RuntimeError(f"NVRTC failed with status {status}{log_msg}")


def _check_cu(status: int, name: str) -> None:
    if status != 0:
        raise RuntimeError(f"CUDA driver call {name} failed with status {status}")


class _VScanCudaModule:
    def __init__(self, device_index: int) -> None:
        self._cu = _load_cuda_driver()
        nvrtc = _find_nvrtc_lib()
        major, minor = torch.cuda.get_device_capability(device_index)
        # Use compute_{major}{minor} PTX so the driver always JITs cleanly even on newer SMs
        arch_flag = f"--gpu-architecture=compute_{major}{minor}".encode()
        if (major, minor) == (9, 0):
            arch_flag = b"--gpu-architecture=sm_90a"
            use_cubin = True
        else:
            use_cubin = False

        prog = ctypes.c_void_p()
        _check_nvrtc(
            nvrtc,
            nvrtc.nvrtcCreateProgram(
                ctypes.byref(prog),
                _CUDA_SOURCE.encode("utf-8"),
                b"r11_cuda_vscan.cu",
                0,
                None,
                None,
            ),
        )
        opts = (ctypes.c_char_p * 3)(arch_flag, b"--use_fast_math", b"-default-device")
        try:
            _check_nvrtc(nvrtc, nvrtc.nvrtcCompileProgram(prog, 3, opts), prog)
            size = ctypes.c_size_t(0)
            if use_cubin:
                _check_nvrtc(nvrtc, nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size)), prog)
                image = ctypes.create_string_buffer(size.value)
                _check_nvrtc(nvrtc, nvrtc.nvrtcGetCUBIN(prog, image), prog)
            else:
                _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size)), prog)
                image = ctypes.create_string_buffer(size.value)
                _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTX(prog, image), prog)
        finally:
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        # Ensure PyTorch has initialized the CUDA context on this device
        torch.cuda.set_device(device_index)
        _ = torch.empty(1, device=f"cuda:{device_index}")

        self._module = ctypes.c_void_p()
        _check_cu(
            self._cu.cuModuleLoadData(ctypes.byref(self._module), image),
            "cuModuleLoadData",
        )
        self._funcs: dict[str, ctypes.c_void_p] = {}
        for (h, c), (name, warps) in _FWD_SPECS.items():
            self._load_func(name, _fwd_smem_bytes(h, c, warps))
        for (h, c), (name, warps, chunk, acc_smem) in _BWD_SPECS.items():
            self._load_func(name, _bwd_smem_bytes(h, c, warps, chunk, acc_smem))

    def _load_func(self, name: str, smem_bytes: int) -> None:
        func = ctypes.c_void_p()
        _check_cu(
            self._cu.cuModuleGetFunction(
                ctypes.byref(func), self._module, name.encode("utf-8")
            ),
            f"cuModuleGetFunction({name})",
        )
        if smem_bytes > 49152:
            _check_cu(
                self._cu.cuFuncSetAttribute(
                    func,
                    ctypes.c_int(_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
                    ctypes.c_int(smem_bytes),
                ),
                f"cuFuncSetAttribute({name}, {smem_bytes})",
            )
        self._funcs[name] = func

    def launch(
        self,
        name: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        smem_bytes: int,
        stream: int,
        args: list[ctypes._SimpleCData],
    ) -> None:
        ptrs = (ctypes.c_void_p * len(args))(*(ctypes.addressof(a) for a in args))
        _check_cu(
            self._cu.cuLaunchKernel(
                self._funcs[name],
                ctypes.c_uint(grid[0]),
                ctypes.c_uint(grid[1]),
                ctypes.c_uint(grid[2]),
                ctypes.c_uint(block[0]),
                ctypes.c_uint(block[1]),
                ctypes.c_uint(block[2]),
                ctypes.c_uint(smem_bytes),
                ctypes.c_void_p(stream),
                ptrs,
                ctypes.c_void_p(0),
            ),
            f"cuLaunchKernel({name})",
        )


@lru_cache(maxsize=16)
def _get_module(device_index: int) -> _VScanCudaModule:
    return _VScanCudaModule(device_index)


def _choose_x_chunks(batch: int, ow: int, modes: int, warps: int, *, is_backward: bool) -> int:
    m_blocks = (modes + 31) // 32
    base_ctas = batch * m_blocks
    if is_backward:
        # Prefer x_chunks == 1 (zero atomics!) whenever grid already fills SMs well
        if base_ctas >= 1536:
            return 1
        max_chunks = max(1, ow // warps)
        if base_ctas >= 512:
            return min(2, max_chunks)
        return min(4, max_chunks)
    # Forward has no atomics at all: split columns to saturate SMs with tiny CTAs
    max_chunks = max(1, ow // warps)
    if base_ctas >= 2560:
        return min(2, max_chunks)
    return min(4, max_chunks)


@torch.library.custom_op("lnet::r11_cuda_vscan_forward", mutates_args=())
def r11_cuda_vscan_forward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    coarse: bool,
    squeeze: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    b, h, w, m = source_real_a.shape
    oh = h // 2 if coarse else h
    ow = w // 2 if coarse else w
    out_shape = (b, oh, ow, m) if squeeze else (b, oh, ow, 1, m)
    out_r = source_real_a.new_empty(out_shape)
    out_i = source_real_a.new_empty(out_shape)

    dr = decay_real.contiguous()
    di = decay_imag.contiguous()
    gr = gamma_real.contiguous()
    gi = gamma_imag.contiguous()
    sra = source_real_a.contiguous()
    sia = source_imag_a.contiguous()
    srb = source_real_b.contiguous()
    sib = source_imag_b.contiguous()
    vx = variance_x.contiguous()
    vy = variance_y.contiguous()
    wi = path_input_weight.contiguous()
    bi = path_input_bias.contiguous()
    wo = path_output_weight.contiguous()
    bo = path_output_bias.contiguous()

    name, warps = _FWD_SPECS[(h, coarse)]
    x_chunks = _choose_x_chunks(b, ow, m, warps, is_backward=False)
    smem_bytes = _fwd_smem_bytes(h, coarse, warps)
    dev_idx = sra.device.index or 0
    mod = _get_module(dev_idx)
    stream = torch.cuda.current_stream(sra.device).cuda_stream

    args: list[ctypes._SimpleCData] = [
        ctypes.c_void_p(dr.data_ptr()),
        ctypes.c_void_p(di.data_ptr()),
        ctypes.c_void_p(gr.data_ptr()),
        ctypes.c_void_p(gi.data_ptr()),
        ctypes.c_void_p(sra.data_ptr()),
        ctypes.c_void_p(sia.data_ptr()),
        ctypes.c_void_p(srb.data_ptr()),
        ctypes.c_void_p(sib.data_ptr()),
        ctypes.c_void_p(vx.data_ptr()),
        ctypes.c_void_p(vy.data_ptr()),
        ctypes.c_void_p(wi.data_ptr()),
        ctypes.c_void_p(bi.data_ptr()),
        ctypes.c_void_p(wo.data_ptr()),
        ctypes.c_void_p(bo.data_ptr()),
        ctypes.c_void_p(out_r.data_ptr()),
        ctypes.c_void_p(out_i.data_ptr()),
        ctypes.c_int(b),
        ctypes.c_int(m),
        ctypes.c_int(x_chunks),
        ctypes.c_float(epsilon),
    ]
    mod.launch(
        name,
        ((m + 31) // 32, b * x_chunks, 1),
        (warps * 32, 1, 1),
        smem_bytes,
        stream,
        args,
    )
    return out_r, out_i


@r11_cuda_vscan_forward_op.register_fake
def _fake_vscan_forward(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    coarse: bool,
    squeeze: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    b, h, w, m = source_real_a.shape
    oh = h // 2 if coarse else h
    ow = w // 2 if coarse else w
    out_shape = (b, oh, ow, m) if squeeze else (b, oh, ow, 1, m)
    return source_real_a.new_empty(out_shape), source_real_a.new_empty(out_shape)


@torch.library.custom_op("lnet::r11_cuda_vscan_backward", mutates_args=())
def r11_cuda_vscan_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    grad_collapsed_real: Tensor,
    grad_collapsed_imag: Tensor,
    coarse: bool,
    epsilon: float,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    b, h, w, m = source_real_a.shape
    ow = w // 2 if coarse else w
    name, warps, chunk, acc_smem = _BWD_SPECS[(h, coarse)]
    x_chunks = _choose_x_chunks(b, ow, m, warps, is_backward=True)
    smem_bytes = _bwd_smem_bytes(h, coarse, warps, chunk, acc_smem)

    dr = decay_real.contiguous()
    di = decay_imag.contiguous()
    gr = gamma_real.contiguous()
    gi = gamma_imag.contiguous()
    sra = source_real_a.contiguous()
    sia = source_imag_a.contiguous()
    srb = source_real_b.contiguous()
    sib = source_imag_b.contiguous()
    vx = variance_x.contiguous()
    vy = variance_y.contiguous()
    wi = path_input_weight.contiguous()
    bi = path_input_bias.contiguous()
    wo = path_output_weight.contiguous()
    bo = path_output_bias.contiguous()
    gcr = grad_collapsed_real.contiguous()
    gci = grad_collapsed_imag.contiguous()

    alloc = (lambda t: t.new_empty((b, *t.shape))) if x_chunks == 1 else (lambda t: t.new_zeros((b, *t.shape)))
    g_dr_b = alloc(dr)
    g_di_b = alloc(di)
    g_gr_b = alloc(gr)
    g_gi_b = alloc(gi)
    g_win_b = alloc(wi)
    g_bin_b = alloc(bi)
    g_wout_b = alloc(wo)
    g_bout_b = alloc(bo)
    grad_src = sra.new_empty((4, b, h, w, m))

    dev_idx = sra.device.index or 0
    mod = _get_module(dev_idx)
    stream = torch.cuda.current_stream(sra.device).cuda_stream

    args: list[ctypes._SimpleCData] = [
        ctypes.c_void_p(dr.data_ptr()),
        ctypes.c_void_p(di.data_ptr()),
        ctypes.c_void_p(gr.data_ptr()),
        ctypes.c_void_p(gi.data_ptr()),
        ctypes.c_void_p(sra.data_ptr()),
        ctypes.c_void_p(sia.data_ptr()),
        ctypes.c_void_p(srb.data_ptr()),
        ctypes.c_void_p(sib.data_ptr()),
        ctypes.c_void_p(vx.data_ptr()),
        ctypes.c_void_p(vy.data_ptr()),
        ctypes.c_void_p(wi.data_ptr()),
        ctypes.c_void_p(bi.data_ptr()),
        ctypes.c_void_p(wo.data_ptr()),
        ctypes.c_void_p(gcr.data_ptr()),
        ctypes.c_void_p(gci.data_ptr()),
        ctypes.c_void_p(g_win_b.data_ptr()),
        ctypes.c_void_p(g_bin_b.data_ptr()),
        ctypes.c_void_p(g_wout_b.data_ptr()),
        ctypes.c_void_p(g_bout_b.data_ptr()),
        ctypes.c_void_p(g_dr_b.data_ptr()),
        ctypes.c_void_p(g_di_b.data_ptr()),
        ctypes.c_void_p(g_gr_b.data_ptr()),
        ctypes.c_void_p(g_gi_b.data_ptr()),
        ctypes.c_void_p(grad_src.data_ptr()),
        ctypes.c_int(b),
        ctypes.c_int(m),
        ctypes.c_int(x_chunks),
        ctypes.c_float(epsilon),
    ]
    mod.launch(
        name,
        ((m + 31) // 32, b * x_chunks, 1),
        (warps * 32, 1, 1),
        smem_bytes,
        stream,
        args,
    )
    return (
        g_dr_b.sum(0),
        g_di_b.sum(0),
        g_gr_b.sum(0),
        g_gi_b.sum(0),
        grad_src,
        g_win_b.sum(0),
        g_bin_b.sum(0),
        g_wout_b.sum(0),
        g_bout_b.sum(0),
    )


@r11_cuda_vscan_backward_op.register_fake
def _fake_vscan_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    grad_collapsed_real: Tensor,
    grad_collapsed_imag: Tensor,
    coarse: bool,
    epsilon: float,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    return (
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(gamma_real),
        torch.empty_like(gamma_imag),
        source_real_a.new_empty((4, *source_real_a.shape)),
        torch.empty_like(path_input_weight),
        torch.empty_like(path_input_bias),
        torch.empty_like(path_output_weight),
        torch.empty_like(path_output_bias),
    )


def r11_cuda_vscan_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    grad_collapsed_real: Tensor,
    grad_collapsed_imag: Tensor,
    *,
    coarse: bool,
    epsilon: float,
) -> tuple[Tensor, ...]:
    g_dr, g_di, g_gr, g_gi, grad_src, g_win, g_bin, g_wout, g_bout = r11_cuda_vscan_backward_op(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        path_input_weight,
        path_input_bias,
        path_output_weight,
        path_output_bias,
        grad_collapsed_real,
        grad_collapsed_imag,
        coarse,
        epsilon,
    )
    g_sra, g_sia, g_srb, g_sib = grad_src.unbind(0)
    return g_dr, g_di, g_gr, g_gi, g_sra, g_sia, g_srb, g_sib, g_win, g_bin, g_wout, g_bout


def can_use_cuda_vscan(
    source_real_a: Tensor,
    path_input_weight: Tensor,
    *,
    coarse: bool,
) -> bool:
    if not source_real_a.is_cuda or source_real_a.dtype != torch.bfloat16:
        return False
    b, h, w, m = source_real_a.shape
    if h != w or (h, coarse) not in _FWD_SPECS:
        return False
    if m <= 0 or m % 32 != 0 or b <= 0:
        return False
    return path_input_weight.shape == (m, 8, 8)
