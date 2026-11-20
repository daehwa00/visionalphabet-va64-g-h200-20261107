"""SM90/SM89/SM120 NVRTC CUDA C++ kernels for horizontal bidirectional complex scan.

Implements fused streaming horizontal forward and backward scans for
W in {56, 28, 14}, M % 32 == 0, both direct (full16) and coarse (even/odd masked)
adjoints, with zero global atomics at production batch sizes.
"""

# ruff: noqa: ARG001, E501, FBT001
from __future__ import annotations

import ctypes
from functools import lru_cache
# BEGIN R12 DISPATCH
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
# END R12 DISPATCH

import torch
from torch import Tensor

from .r11_cuda_vscan import (
    _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    _check_cu,
    _check_nvrtc,
    _find_nvrtc_lib,
    _load_cuda_driver,
)

_CUDA_SOURCE = r"""
__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    return __uint_as_float(static_cast<unsigned int>(bits) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16(float x) {
    unsigned short res;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(res) : "f"(x));
    return res;
}

template <int W, int WARPS>
__device__ __forceinline__ void hscan_forward_impl(
    const float* __restrict__ decay_real,
    const float* __restrict__ decay_imag,
    const float* __restrict__ gamma_real,
    const float* __restrict__ gamma_imag,
    const unsigned short* __restrict__ src_r,
    const unsigned short* __restrict__ src_i,
    unsigned short* __restrict__ states,
    int line_count,
    int M,
    int src_stride_r,
    int src_stride_i
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int m = blockIdx.x * 32 + lane;
    const int line = blockIdx.y * WARPS + warp;
    if (m >= M || line >= line_count) return;

    extern __shared__ unsigned int smem[];
    unsigned int* my_src = smem + warp * (W * 32);

    const float ar = decay_real[m];
    const float ai = decay_imag[m];
    const float gr = gamma_real[m];
    const float gi = gamma_imag[m];

    const long long plane_size = static_cast<long long>(line_count) * W * M;
    const long long base_sr = static_cast<long long>(line) * W * src_stride_r + m;
    const long long base_si = static_cast<long long>(line) * W * src_stride_i + m;
    const long long base_out = static_cast<long long>(line) * W * M + m;

    float pr = 0.0f, pi = 0.0f;
    for (int x = 0; x < W; ++x) {
        const unsigned short sr_bits = src_r[base_sr + static_cast<long long>(x) * src_stride_r];
        const unsigned short si_bits = src_i[base_si + static_cast<long long>(x) * src_stride_i];
        my_src[x * 32 + lane] = static_cast<unsigned int>(sr_bits)
                              | (static_cast<unsigned int>(si_bits) << 16);
        const float sr = bf16_to_f32(sr_bits);
        const float si = bf16_to_f32(si_bits);
        const float n_pr = ar * pr - ai * pi + (gr * sr - gi * si);
        const float n_pi = ar * pi + ai * pr + (gr * si + gi * sr);
        pr = n_pr;
        pi = n_pi;
        const long long oo = base_out + static_cast<long long>(x) * M;
        states[oo] = f32_to_bf16(pr);
        states[plane_size + oo] = f32_to_bf16(pi);
    }

    float nr = 0.0f, ni = 0.0f;
    for (int x = W - 1; x >= 0; --x) {
        const unsigned int packed = my_src[x * 32 + lane];
        const float sr = bf16_to_f32(static_cast<unsigned short>(packed & 0xFFFFu));
        const float si = bf16_to_f32(static_cast<unsigned short>(packed >> 16));
        const float n_nr = ar * nr + ai * ni + (gr * sr + gi * si);
        const float n_ni = ar * ni - ai * nr + (gr * si - gi * sr);
        nr = n_nr;
        ni = n_ni;
        const long long oo = base_out + static_cast<long long>(x) * M;
        states[2 * plane_size + oo] = f32_to_bf16(nr);
        states[3 * plane_size + oo] = f32_to_bf16(ni);
    }
}

template <int W, bool COARSE, int WARPS>
__device__ __forceinline__ void hscan_backward_impl(
    const float* __restrict__ decay_real,
    const float* __restrict__ decay_imag,
    const float* __restrict__ gamma_real,
    const float* __restrict__ gamma_imag,
    const unsigned short* __restrict__ src_r,
    const unsigned short* __restrict__ src_i,
    const unsigned short* __restrict__ st_pr,
    const unsigned short* __restrict__ st_pi,
    const unsigned short* __restrict__ st_nr,
    const unsigned short* __restrict__ st_ni,
    const unsigned short* __restrict__ grad_pr,
    const unsigned short* __restrict__ grad_pi,
    const unsigned short* __restrict__ grad_nr,
    const unsigned short* __restrict__ grad_ni,
    float* __restrict__ grad_dr,
    float* __restrict__ grad_di,
    float* __restrict__ grad_gr,
    float* __restrict__ grad_gi,
    unsigned short* __restrict__ grad_src_packed,
    int B,
    int H,
    int M,
    int src_stride_r,
    int src_stride_i,
    int y_chunks
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int m = blockIdx.x * 32 + lane;
    const int b = blockIdx.y / y_chunks;
    const int y_chunk = blockIdx.y - b * y_chunks;
    if (m >= M || b >= B) return;

    extern __shared__ float smem_f[];
    float* my_r = smem_f + warp * (W * 2 * 32);

    const float ar = decay_real[m];
    const float ai = decay_imag[m];
    const float gr = gamma_real[m];
    const float gi = gamma_imag[m];

    const int rows_per_chunk = (H + y_chunks - 1) / y_chunks;
    const int y_start = y_chunk * rows_per_chunk;
    const int y_end = (y_start + rows_per_chunk < H) ? (y_start + rows_per_chunk) : H;

    const long long ds_row_stride = static_cast<long long>(2) * M;

    float d_dr = 0.0f, d_di = 0.0f, d_gr = 0.0f, d_gi = 0.0f;

    for (int y = y_start + warp; y < y_end; y += WARPS) {
        const long long line = static_cast<long long>(b) * H + y;
        const long long base_off = line * W * M + m;
        const long long base_sr = line * W * src_stride_r + m;
        const long long base_si = line * W * src_stride_i + m;
        const long long base_ds = line * W * ds_row_stride + m;

        // Pass 1: negative adjoint forward x = 0 .. W-1
        float rr = 0.0f, ri = 0.0f;
        for (int x = 0; x < W; ++x) {
            float gnr = 0.0f, gni = 0.0f;
            if (!COARSE || ((x & 1) == 0)) {
                const long long off = base_off + static_cast<long long>(x) * M;
                gnr = bf16_to_f32(grad_nr[off]);
                gni = bf16_to_f32(grad_ni[off]);
            }
            const float n_rr = ar * rr - ai * ri + gnr;
            const float n_ri = ar * ri + ai * rr + gni;
            rr = n_rr;
            ri = n_ri;
            my_r[(x * 2 + 0) * 32 + lane] = rr;
            my_r[(x * 2 + 1) * 32 + lane] = ri;
        }

        // Pass 2: positive adjoint backward x = W-1 .. 0 + grads
        float lr = 0.0f, li = 0.0f;
        for (int x = W - 1; x >= 0; --x) {
            const long long off = base_off + static_cast<long long>(x) * M;
            float gpr = 0.0f, gpi = 0.0f;
            if (!COARSE || (x & 1)) {
                gpr = bf16_to_f32(grad_pr[off]);
                gpi = bf16_to_f32(grad_pi[off]);
            }
            const float n_lr = ar * lr + ai * li + gpr;
            const float n_li = ar * li - ai * lr + gpi;
            lr = n_lr;
            li = n_li;

            const float cr = my_r[(x * 2 + 0) * 32 + lane];
            const float ci = my_r[(x * 2 + 1) * 32 + lane];

            const float sr = bf16_to_f32(src_r[base_sr + static_cast<long long>(x) * src_stride_r]);
            const float si = bf16_to_f32(src_i[base_si + static_cast<long long>(x) * src_stride_i]);

            const float pr_prev = (x > 0) ? bf16_to_f32(st_pr[off - M]) : 0.0f;
            const float pi_prev = (x > 0) ? bf16_to_f32(st_pi[off - M]) : 0.0f;
            const float nr_next = (x + 1 < W) ? bf16_to_f32(st_nr[off + M]) : 0.0f;
            const float ni_next = (x + 1 < W) ? bf16_to_f32(st_ni[off + M]) : 0.0f;

            d_dr += lr * pr_prev + li * pi_prev + cr * nr_next + ci * ni_next;
            d_di += -lr * pi_prev + li * pr_prev + cr * ni_next - ci * nr_next;
            d_gr += lr * sr + li * si + cr * sr + ci * si;
            d_gi += -lr * si + li * sr + cr * si - ci * sr;

            const long long ds_off = base_ds + static_cast<long long>(x) * ds_row_stride;
            grad_src_packed[ds_off] = f32_to_bf16(lr * gr + li * gi + cr * gr - ci * gi);
            grad_src_packed[ds_off + M] = f32_to_bf16(-lr * gi + li * gr + cr * gi + ci * gr);
        }
    }

    __syncthreads();
    float* red = smem_f;
    red[(warp * 4 + 0) * 32 + lane] = d_dr;
    red[(warp * 4 + 1) * 32 + lane] = d_di;
    red[(warp * 4 + 2) * 32 + lane] = d_gr;
    red[(warp * 4 + 3) * 32 + lane] = d_gi;
    __syncthreads();

    if (warp == 0) {
        float s_dr = red[0 * 32 + lane];
        float s_di = red[1 * 32 + lane];
        float s_gr = red[2 * 32 + lane];
        float s_gi = red[3 * 32 + lane];
        #pragma unroll
        for (int w = 1; w < WARPS; ++w) {
            s_dr += red[(w * 4 + 0) * 32 + lane];
            s_di += red[(w * 4 + 1) * 32 + lane];
            s_gr += red[(w * 4 + 2) * 32 + lane];
            s_gi += red[(w * 4 + 3) * 32 + lane];
        }
        const long long b_m = static_cast<long long>(b) * M + m;
        if (y_chunks > 1) {
            atomicAdd(grad_dr + b_m, s_dr);
            atomicAdd(grad_di + b_m, s_di);
            atomicAdd(grad_gr + b_m, s_gr);
            atomicAdd(grad_gi + b_m, s_gi);
        } else {
            grad_dr[b_m] = s_dr;
            grad_di[b_m] = s_di;
            grad_gr[b_m] = s_gr;
            grad_gi[b_m] = s_gi;
        }
    }
}

#define HFWD_ARGS \
    const float* __restrict__ decay_real, \
    const float* __restrict__ decay_imag, \
    const float* __restrict__ gamma_real, \
    const float* __restrict__ gamma_imag, \
    const unsigned short* __restrict__ src_r, \
    const unsigned short* __restrict__ src_i, \
    unsigned short* __restrict__ states, \
    int line_count, int M, int src_stride_r, int src_stride_i

#define HFWD_CALL \
    decay_real, decay_imag, gamma_real, gamma_imag, \
    src_r, src_i, states, line_count, M, src_stride_r, src_stride_i

#define HBWD_ARGS \
    const float* __restrict__ decay_real, \
    const float* __restrict__ decay_imag, \
    const float* __restrict__ gamma_real, \
    const float* __restrict__ gamma_imag, \
    const unsigned short* __restrict__ src_r, \
    const unsigned short* __restrict__ src_i, \
    const unsigned short* __restrict__ st_pr, \
    const unsigned short* __restrict__ st_pi, \
    const unsigned short* __restrict__ st_nr, \
    const unsigned short* __restrict__ st_ni, \
    const unsigned short* __restrict__ grad_pr, \
    const unsigned short* __restrict__ grad_pi, \
    const unsigned short* __restrict__ grad_nr, \
    const unsigned short* __restrict__ grad_ni, \
    float* __restrict__ grad_dr, \
    float* __restrict__ grad_di, \
    float* __restrict__ grad_gr, \
    float* __restrict__ grad_gi, \
    unsigned short* __restrict__ grad_src_packed, \
    int B, int H, int M, int src_stride_r, int src_stride_i, int y_chunks

#define HBWD_CALL \
    decay_real, decay_imag, gamma_real, gamma_imag, \
    src_r, src_i, st_pr, st_pi, st_nr, st_ni, \
    grad_pr, grad_pi, grad_nr, grad_ni, \
    grad_dr, grad_di, grad_gr, grad_gi, grad_src_packed, \
    B, H, M, src_stride_r, src_stride_i, y_chunks

extern "C" {
__launch_bounds__(128, 8) __global__ void hscan_fwd_56(HFWD_ARGS) { hscan_forward_impl<56, 4>(HFWD_CALL); }
__launch_bounds__(128, 8) __global__ void hscan_fwd_28(HFWD_ARGS) { hscan_forward_impl<28, 4>(HFWD_CALL); }
__launch_bounds__(64,  8) __global__ void hscan_fwd_14(HFWD_ARGS) { hscan_forward_impl<14, 2>(HFWD_CALL); }

__launch_bounds__(128, 4) __global__ void hscan_bwd_56_d(HBWD_ARGS) { hscan_backward_impl<56, false, 4>(HBWD_CALL); }
__launch_bounds__(128, 4) __global__ void hscan_bwd_56_c(HBWD_ARGS) { hscan_backward_impl<56, true,  4>(HBWD_CALL); }
__launch_bounds__(128, 6) __global__ void hscan_bwd_28_d(HBWD_ARGS) { hscan_backward_impl<28, false, 4>(HBWD_CALL); }
__launch_bounds__(128, 6) __global__ void hscan_bwd_28_c(HBWD_ARGS) { hscan_backward_impl<28, true,  4>(HBWD_CALL); }
__launch_bounds__(64,  8) __global__ void hscan_bwd_14_d(HBWD_ARGS) { hscan_backward_impl<14, false, 2>(HBWD_CALL); }
__launch_bounds__(64,  8) __global__ void hscan_bwd_14_c(HBWD_ARGS) { hscan_backward_impl<14, true,  2>(HBWD_CALL); }
}
"""

_HFWD_SPECS: dict[int, tuple[str, int]] = {
    56: ("hscan_fwd_56", 4),
    28: ("hscan_fwd_28", 4),
    14: ("hscan_fwd_14", 2),
}

_HBWD_SPECS: dict[tuple[int, bool], tuple[str, int]] = {
    (56, False): ("hscan_bwd_56_d", 4),
    (56, True): ("hscan_bwd_56_c", 4),
    (28, False): ("hscan_bwd_28_d", 4),
    (28, True): ("hscan_bwd_28_c", 4),
    (14, False): ("hscan_bwd_14_d", 2),
    (14, True): ("hscan_bwd_14_c", 2),
}


def _hfwd_smem_bytes(width: int, warps: int) -> int:
    return warps * width * 32 * 4


def _hbwd_smem_bytes(width: int, warps: int) -> int:
    return max(warps * width * 2 * 32 * 4, warps * 4 * 32 * 4)


class _HScanCudaModule:
    def __init__(self, device_index: int) -> None:
        nvrtc = _find_nvrtc_lib()
        self._cu = _load_cuda_driver()
        major, minor = torch.cuda.get_device_capability(device_index)
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
                b"r11_cuda_hscan.cu",
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

        torch.cuda.set_device(device_index)
        _ = torch.empty(1, device=f"cuda:{device_index}")

        self._module = ctypes.c_void_p()
        _check_cu(
            self._cu.cuModuleLoadData(ctypes.byref(self._module), image),
            "cuModuleLoadData",
        )
        self._funcs: dict[str, ctypes.c_void_p] = {}
        for w, (name, warps) in _HFWD_SPECS.items():
            self._load_func(name, _hfwd_smem_bytes(w, warps))
        for (w, _coarse), (name, warps) in _HBWD_SPECS.items():
            self._load_func(name, _hbwd_smem_bytes(w, warps))

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
def _get_module(device_index: int) -> _HScanCudaModule:
    return _HScanCudaModule(device_index)


def _choose_y_chunks(batch: int, height: int, modes: int, warps: int) -> int:
    m_blocks = (modes + 31) // 32
    base_ctas = batch * m_blocks
    if base_ctas >= 1536:
        return 1
    max_chunks = max(1, height // warps)
    if base_ctas >= 512:
        return min(2, max_chunks)
    return min(4, max_chunks)


@torch.library.custom_op("lnet::r11_cuda_hscan_forward", mutates_args=())
def r11_cuda_hscan_forward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
) -> Tensor:
    b, h, w, m = source_real.shape
    line_count = b * h
    states = source_real.new_empty((4, b, h, w, m))
    name, warps = _HFWD_SPECS[w]
    smem_bytes = _hfwd_smem_bytes(w, warps)

    dr = decay_real.contiguous()
    di = decay_imag.contiguous()
    gr = gamma_real.contiguous()
    gi = gamma_imag.contiguous()

    dev_idx = source_real.device.index or 0
    mod = _get_module(dev_idx)
    stream = torch.cuda.current_stream(source_real.device).cuda_stream

    args: list[ctypes._SimpleCData] = [
        ctypes.c_void_p(dr.data_ptr()),
        ctypes.c_void_p(di.data_ptr()),
        ctypes.c_void_p(gr.data_ptr()),
        ctypes.c_void_p(gi.data_ptr()),
        ctypes.c_void_p(source_real.data_ptr()),
        ctypes.c_void_p(source_imag.data_ptr()),
        ctypes.c_void_p(states.data_ptr()),
        ctypes.c_int(line_count),
        ctypes.c_int(m),
        ctypes.c_int(source_real.stride(-2)),
        ctypes.c_int(source_imag.stride(-2)),
    ]
    mod.launch(
        name,
        ((m + 31) // 32, (line_count + warps - 1) // warps, 1),
        (warps * 32, 1, 1),
        smem_bytes,
        stream,
        args,
    )
    return states


@r11_cuda_hscan_forward_op.register_fake
def _fake_hscan_forward(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
) -> Tensor:
    return source_real.new_empty((4, *source_real.shape))


def _launch_hscan_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
    state_pos_real: Tensor,
    state_pos_imag: Tensor,
    state_neg_real: Tensor,
    state_neg_imag: Tensor,
    grad_pos_real: Tensor,
    grad_pos_imag: Tensor,
    grad_neg_real: Tensor,
    grad_neg_imag: Tensor,
    grad_src_packed: Tensor,
    coarse: bool,
    # BEGIN R12 DISPATCH
    *,
    reduce_gradients: Callable[[tuple[Tensor, ...]], tuple[Tensor, ...]] | None = None,
    # END R12 DISPATCH
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    b, h, w, m = source_real.shape
    name, warps = _HBWD_SPECS[(w, coarse)]
    y_chunks = _choose_y_chunks(b, h, m, warps)
    smem_bytes = _hbwd_smem_bytes(w, warps)

    dr = decay_real.contiguous()
    di = decay_imag.contiguous()
    gr = gamma_real.contiguous()
    gi = gamma_imag.contiguous()
    spr = state_pos_real.contiguous()
    spi = state_pos_imag.contiguous()
    snr = state_neg_real.contiguous()
    sni = state_neg_imag.contiguous()
    gpr = grad_pos_real.contiguous()
    gpi = grad_pos_imag.contiguous()
    gnr = grad_neg_real.contiguous()
    gni = grad_neg_imag.contiguous()

    alloc = (
        (lambda t: t.new_empty((b, *t.shape)))
        if y_chunks == 1
        else (lambda t: t.new_zeros((b, *t.shape)))
    )
    g_dr_b = alloc(dr)
    g_di_b = alloc(di)
    g_gr_b = alloc(gr)
    g_gi_b = alloc(gi)

    dev_idx = source_real.device.index or 0
    mod = _get_module(dev_idx)
    stream = torch.cuda.current_stream(source_real.device).cuda_stream

    args: list[ctypes._SimpleCData] = [
        ctypes.c_void_p(dr.data_ptr()),
        ctypes.c_void_p(di.data_ptr()),
        ctypes.c_void_p(gr.data_ptr()),
        ctypes.c_void_p(gi.data_ptr()),
        ctypes.c_void_p(source_real.data_ptr()),
        ctypes.c_void_p(source_imag.data_ptr()),
        ctypes.c_void_p(spr.data_ptr()),
        ctypes.c_void_p(spi.data_ptr()),
        ctypes.c_void_p(snr.data_ptr()),
        ctypes.c_void_p(sni.data_ptr()),
        ctypes.c_void_p(gpr.data_ptr()),
        ctypes.c_void_p(gpi.data_ptr()),
        ctypes.c_void_p(gnr.data_ptr()),
        ctypes.c_void_p(gni.data_ptr()),
        ctypes.c_void_p(g_dr_b.data_ptr()),
        ctypes.c_void_p(g_di_b.data_ptr()),
        ctypes.c_void_p(g_gr_b.data_ptr()),
        ctypes.c_void_p(g_gi_b.data_ptr()),
        ctypes.c_void_p(grad_src_packed.data_ptr()),
        ctypes.c_int(b),
        ctypes.c_int(h),
        ctypes.c_int(m),
        ctypes.c_int(source_real.stride(-2)),
        ctypes.c_int(source_imag.stride(-2)),
        ctypes.c_int(y_chunks),
    ]
    mod.launch(
        name,
        ((m + 31) // 32, b * y_chunks, 1),
        (warps * 32, 1, 1),
        smem_bytes,
        stream,
        args,
    )
    # BEGIN R12 DISPATCH
    if reduce_gradients is not None:
        dr_grad, di_grad, gr_grad, gi_grad = reduce_gradients((g_dr_b, g_di_b, g_gr_b, g_gi_b))
        return dr_grad, di_grad, gr_grad, gi_grad
    # END R12 DISPATCH
    return g_dr_b.sum(0), g_di_b.sum(0), g_gr_b.sum(0), g_gi_b.sum(0)


@torch.library.custom_op("lnet::r11_cuda_hscan_backward_into", mutates_args=("grad_src_packed",))
def r11_cuda_hscan_backward_into_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
    state_pos_real: Tensor,
    state_pos_imag: Tensor,
    state_neg_real: Tensor,
    state_neg_imag: Tensor,
    grad_pos_real: Tensor,
    grad_pos_imag: Tensor,
    grad_neg_real: Tensor,
    grad_neg_imag: Tensor,
    grad_src_packed: Tensor,
    coarse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _launch_hscan_backward(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real,
        source_imag,
        state_pos_real,
        state_pos_imag,
        state_neg_real,
        state_neg_imag,
        grad_pos_real,
        grad_pos_imag,
        grad_neg_real,
        grad_neg_imag,
        grad_src_packed,
        coarse,
    )


@r11_cuda_hscan_backward_into_op.register_fake
def _fake_hscan_backward_into(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
    state_pos_real: Tensor,
    state_pos_imag: Tensor,
    state_neg_real: Tensor,
    state_neg_imag: Tensor,
    grad_pos_real: Tensor,
    grad_pos_imag: Tensor,
    grad_neg_real: Tensor,
    grad_neg_imag: Tensor,
    grad_src_packed: Tensor,
    coarse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(gamma_real),
        torch.empty_like(gamma_imag),
    )


@torch.library.custom_op("lnet::r11_cuda_hscan_backward", mutates_args=())
def r11_cuda_hscan_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
    state_pos_real: Tensor,
    state_pos_imag: Tensor,
    state_neg_real: Tensor,
    state_neg_imag: Tensor,
    grad_pos_real: Tensor,
    grad_pos_imag: Tensor,
    grad_neg_real: Tensor,
    grad_neg_imag: Tensor,
    coarse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    b, h, w, m = source_real.shape
    grad_src_packed = source_real.new_empty((b, h, w, 2 * m))
    g_dr, g_di, g_gr, g_gi = _launch_hscan_backward(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real,
        source_imag,
        state_pos_real,
        state_pos_imag,
        state_neg_real,
        state_neg_imag,
        grad_pos_real,
        grad_pos_imag,
        grad_neg_real,
        grad_neg_imag,
        grad_src_packed,
        coarse,
    )
    return g_dr, g_di, g_gr, g_gi, grad_src_packed


@r11_cuda_hscan_backward_op.register_fake
def _fake_hscan_backward(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
    state_pos_real: Tensor,
    state_pos_imag: Tensor,
    state_neg_real: Tensor,
    state_neg_imag: Tensor,
    grad_pos_real: Tensor,
    grad_pos_imag: Tensor,
    grad_neg_real: Tensor,
    grad_neg_imag: Tensor,
    coarse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    b, h, w, m = source_real.shape
    return (
        torch.empty_like(decay_real),
        torch.empty_like(decay_imag),
        torch.empty_like(gamma_real),
        torch.empty_like(gamma_imag),
        source_real.new_empty((b, h, w, 2 * m)),
    )


def can_use_cuda_hscan(source_real: Tensor, source_imag: Tensor) -> bool:
    if (
        not source_real.is_cuda
        or source_real.dtype != torch.bfloat16
        or source_imag.dtype != torch.bfloat16
        or source_real.shape != source_imag.shape
        or source_real.ndim != 4
    ):
        return False
    b, h, w, m = source_real.shape
    if b <= 0 or h <= 0 or w not in _HFWD_SPECS or m <= 0 or m % 32 != 0:
        return False
    for v in (source_real, source_imag):
        s = v.stride(-2)
        if v.stride(-1) != 1 or s not in (m, 2 * m):
            return False
        if v.stride() != (h * w * s, w * s, s, 1):
            return False
    return True
