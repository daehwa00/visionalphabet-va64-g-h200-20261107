"""Experimental SM90 vertical scan adjoints with warp-owned path gradients."""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from functools import cache
import math

import torch
from torch import Tensor

from . import general_cuda_scans as scans
from .cuda_nvrtc import _CudaModule, _check_nvrtc, _include_dirs, _nvrtc

MODES = {
    "v58_tc_tm16_tw16_c28": (16, 28, False),
    "v58_tc_tm16_tw16_c16": (16, 16, False),
    "v58_tc_tm32_tw8_c28": (32, 28, False),
    "v58_tc_tm16_tw16_fulltc": (16, 16, True),
}


@dataclass(frozen=True)
class Plan:
    tm: int
    chunk: int
    fulltc: bool
    h: int
    coarse: bool
    dtype: torch.dtype
    odd_width: bool

    @property
    def tw(self):
        return 256 // self.tm

    @property
    def register_rows(self):
        if self.h != 56:
            return 0
        return 12 if self.tm == 32 and not self.coarse else 16

    @property
    def smem(self):
        # Shared positive trajectory becomes negative injection in-place.
        # The lane-private tail reduces shared storage on long columns.
        return (
            256 * 8 * (self.h - self.register_rows)
            + self.tm * (16 * 10 + 88) * 2
            + 2048 * (16 // self.tw) * (1 + self.coarse)
            + 4096 * ((self.h - 1) // self.chunk)
        )

    def grid(self, b, w, m):
        cols = b * (w // (1 + self.coarse))
        return ((cols + self.tw - 1) // self.tw, m // self.tm, 1)


def plan(mode, shape, dtype, coarse, compact=False):
    tm, chunk, fulltc = MODES[mode]
    b, h, w, m = shape
    if (
        min(shape) <= 0
        or 4 * b * h * w * m > 2**31 - 1
        or h > 56
        or m % tm
        or (coarse and min(h, w) < 2)
        or (compact and not coarse)
        or dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError("Unsupported Wave 58A scan")
    return Plan(tm, min(chunk, h), fulltc, h, coarse, dtype, bool(coarse and w % 2))


_KERNEL = r"""

using T = TYPE;
using I = int;
constexpr int TM = TILE_M, TW = 256 / TM, H = HEIGHT, CH = SPAN;
constexpr bool coarse = COARSE_FLAG, fulltc = FULL_FLAG;
constexpr int step = 1 + coarse, rows = 16 / TW;

__device__ unsigned pack2(T x, T y) {
    return unsigned(*reinterpret_cast<unsigned short *>(&x)) |
           (unsigned(*reinterpret_cast<unsigned short *>(&y)) << 16);
}
__device__ void mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                    unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.MMA_TYPE.MMA_TYPE.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A warp computes 16 cell rows x 8 hidden units or input features.
// Scalar variants keep the same ownership, including the accumulation order.
template <bool transpose> __device__ void project(const T *x, const T *w, float (&y)[4]) {
    int lane = threadIdx.x % 32, group = lane / 4, pair = 2 * (lane % 4);
    if constexpr (fulltc) {
        unsigned a0 = pack2(x[group * 8 + pair], x[group * 8 + pair + 1]);
        unsigned a1 = pack2(x[(group + 8) * 8 + pair], x[(group + 8) * 8 + pair + 1]);
        unsigned b = transpose ? pack2(w[group * 8 + pair], w[group * 8 + pair + 1])
                               : pack2(w[pair * 8 + group], w[(pair + 1) * 8 + group]);
        mma(y, a0, a1, 0, 0, b, 0);
    } else {
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            auto accum = Pair<T>::pack({0, 0}).value;
#pragma unroll 1
            for (int k = 0; k < 8; ++k) {
                T input = x[(group + 8 * row) * 8 + k];
                T w0 = w[transpose ? pair * 8 + k : k * 8 + pair];
                T w1 = w[transpose ? (pair + 1) * 8 + k : k * 8 + pair + 1];
                accum =
                    __hfma2(Pair<T>::from(input, input).value, Pair<T>::from(w0, w1).value, accum);
            }
            auto z = Pair<T>{accum}.unpack();
            y[2 * row] = z.r;
            y[2 * row + 1] = z.i;
        }
    }
}

// Keep a short tail of long trajectories in registers.
constexpr int tail = REGISTER_ROWS;
struct Trajectory {
    Quad<T> *base;
    Quad<T> last[tail ? tail : 1];
    __device__ Quad<T> get(int y) {
        if (y < H - tail)
            return base[y * 256];
        Quad<T> result{};
#pragma unroll
        for (int n = 0; n < tail; ++n)
            if (y == H - tail + n)
                result = last[n];
        return result;
    }
    __device__ void put(int y, Quad<T> v) {
        if (y < H - tail)
            base[y * 256] = v;
#pragma unroll
        for (int n = 0; n < tail; ++n)
            if (y == H - tail + n)
                last[n] = v;
    }
};

extern "C" __global__ __launch_bounds__(256, 2) void wave58(
    const float *dr, const float *di, const float *gr0, const float *gi0, const T *states,
    const float *vx, const float *vy, const void *wi, const void *bi, const void *wo, const T *gr,
    const T *gi, T *gs, float *partial, int B, int W, int M, int types, int rs, int is,
    bool compact, float eps) {
    extern __shared__ float4 shared[];
    auto *trace = reinterpret_cast<Quad<T> *>(shared);
    auto *tiles = reinterpret_cast<T *>(trace + (H - tail) * 256);
    auto *weight = tiles + TM * 16 * 10;
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, local = tid % TM, col = tid / TM;
    Trajectory history{trace + tid};
    int group = lane / 4, pair = 2 * (lane % 4);
    int mode0 = blockIdx.y * TM, m = mode0 + local, ow = W / step;
    I line = I(blockIdx.x) * TW + col;
    bool valid = line < I(B) * ow;
    I b = valid ? line / ow : 0;
    int x = valid ? line % ow : 0, xa = step * x + int(coarse), xb = step * x;
    I plane = I(B) * H * W * M, base = b * H * W * M + m;
    Column<T, I> c{states,
                   vy,
                   base + I(xa) * M,
                   base + I(xb) * M,
                   plane,
                   plane,
                   (b * (H / step) * ow + x) * M + m,
                   I(B) * (H / step) * ow * M,
                   H,
                   W,
                   M,
                   ow,
                   m,
                   vx[I(xa) * M + m],
                   vx[(I(W) + xb) * M + m],
                   eps};
    Complex a{dr[m], di[m]}, g{gr0[m], gi0[m]}, ar = a.conj(), gg = g.conj();
    for (int n = tid; n < TM * 88; n += 256) {
        int f = n % 88, mm = mode0 + n / 88;
        float v = f < 64   ? load_scalar(wi, mm * 64 + f, types & 3)
                  : f < 72 ? load_scalar(bi, mm * 8 + f - 64, (types >> 2) & 3)
                           : load_scalar(wo, mm * 16 + f - 72, (types >> 4) & 3);
        weight[n] = write<T>(v);
    }
    float4 pos{}, neg{};
    for (int y = 0; y < H; ++y) {
        pos = advance(pos, c.source(y), a, g);
        history.put(y, Quad<T>::pack(pos));
    }
    float dw[TM / 8][4]{}, aux[TM / 8][4]{};
    Complex aa{}, ab{}, da{}, dg{};
    // Save exact negative chunk entry states for the ascending adjoint.
    constexpr int blocks = (H + CH - 1) / CH;
    auto *seg_src = reinterpret_cast<Quad<T> *>(weight + TM * 88);
    auto *boundary = reinterpret_cast<float4 *>(seg_src + rows * step * 256);
    if constexpr (coarse && (H % 2)) {
        neg = advance(neg, c.source(H - 1), ar, gg);
        if (valid) {
            if (compact)
                write_column_gradient<false, true>(c, H - 1, {}, gs);
            else
                write_column_gradient<false, false>(c, H - 1, {}, gs);
        }
    }
    for (int end = (H / step) * step - 1; end >= 0; end -= rows * step) {
#pragma unroll 1
        for (int r = 0; r < rows; ++r) {
            int y = end - r * step, cell = r * TW + col;
            auto *t = tiles + local * 160;
            float4 pp{}, pn{};
            float2 dy{};
            if (y >= 0) {
                if (y % CH == CH - 1 && y < H - 1)
                    boundary[(y / CH) * 256 + tid] = neg;
#pragma unroll
                for (int k = 0; k < step; ++k) {
                    auto src = c.packed_source(y - k);
                    seg_src[(r * step + k) * 256 + tid] = src;
                    neg = advance(neg, src.unpack(), ar, gg);
                }
                if (valid) {
                    pp = normalize<T>(history.get(y).unpack(), c.gain(y, false));
                    pn = normalize<T>(neg, c.gain(y - int(coarse), true));
                    I i = c.cell(y / step) - m;
                    dy = make_float2(read(gr[i * rs + m]), read(gi[i * is + m]));
                }
            }
            float p[8] = {pp.x, pp.z, pn.x, pn.z, pp.y, pp.w, pn.y, pn.w};
#pragma unroll
            for (int k = 0; k < 8; ++k)
                t[cell * 8 + k] = write<T>(p[k]);
            t[128 + cell * 2] = write<T>(dy.x);
            t[129 + cell * 2] = write<T>(dy.y);
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < TM / 8; ++u) {
            int mm = warp + 8 * u;
            T *p = tiles + mm * 160;
            T *dz = p;
            const T *w = weight + mm * 88;
            float pre[4]{};
            project<true>(p, w, pre);
            unsigned b0 = pack2(p[pair * 8 + group], p[(pair + 1) * 8 + group]);
            unsigned b1 = pack2(p[(pair + 8) * 8 + group], p[(pair + 9) * 8 + group]);
            __syncwarp();
            aux[u][3] += read(p[128 + (lane % 16) * 2 + lane / 16]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int cell = group + 8 * (i / 2), j = pair + i % 2;
                float dy0 = read(p[128 + 2 * cell]), dy1 = read(p[129 + 2 * cell]);
                float z = rounded<T>(rounded<T>(pre[i]) + read(w[64 + j]));
                float sig = __fdividef(1.f, 1.f + __expf(-z)), act = rounded<T>(z * sig);
                float d = rounded<T>(rounded<T>(dy0 * read(w[72 + j]) + dy1 * read(w[80 + j])) *
                                     (sig * (1.f + z * (1.f - sig))));
                dz[cell * 8 + j] = write<T>(d);
                // Neighboring four-lane groups exchange their two hidden units.
                // Each lane retains the three fields for one hidden unit.
                float dd = __shfl_xor_sync(0xffffffff, d, 4);
                float v0 = dy0 * act, v1 = dy1 * act;
                float w0 = __shfl_xor_sync(0xffffffff, v0, 4);
                float w1 = __shfl_xor_sync(0xffffffff, v1, 4);
                if ((group % 2) == (i % 2)) {
                    aux[u][0] += d + dd;
                    aux[u][1] += v0 + w0;
                    aux[u][2] += v1 + w1;
                }
            }
            __syncwarp();
            unsigned a0 = pack2(dz[pair * 8 + group], dz[(pair + 1) * 8 + group]);
            unsigned a2 = pack2(dz[(pair + 8) * 8 + group], dz[(pair + 9) * 8 + group]);

            mma(dw[u], a0, 0, a2, 0, b0, b1);
            float dp[4]{};
            project<false>(dz, w, dp);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i)
                p[(group + 8 * (i / 2)) * 8 + pair + i % 2] = write<T>(dp[i]);
        }
        __syncthreads();
#pragma unroll 1
        for (int r = 0; r < rows; ++r) {
            int y = end - r * step;
            if (y >= 0) {
                T *p = tiles + local * 160 + (r * TW + col) * 8;
                auto ip = normalize<T>(make_float4(read(p[0]), read(p[4]), read(p[1]), read(p[5])),
                                       c.gain(y, false));
                auto in =
                    normalize_packed<T>(make_float4(read(p[2]), read(p[6]), read(p[3]), read(p[7])),
                                        c.gain(y - int(coarse), true));
                if (valid) {
#pragma unroll 1
                    for (int k = 0; k < step; ++k) {
                        int yy = y - k;
                        auto prev = yy ? history.get(yy - 1).unpack() : float4{};
                        auto ds = column_adjoint_step<false>(
                            prev, seg_src[(r * step + k) * 256 + tid].unpack(), k ? float4{} : ip,
                            a, g, aa, ab, da, dg);
                        if (compact)
                            write_column_gradient<false, true>(c, yy, ds, gs);
                        else
                            write_column_gradient<false, false>(c, yy, ds, gs);
                    }
                }
                history.put(y - int(coarse), in);
            }
        }
        __syncthreads();
    }
    float *out = partial + I(blockIdx.x) * 94 * M;
#pragma unroll
    for (int u = 0; u < TM / 8; ++u) {
        int mm = mode0 + warp + 8 * u;
        for (int i = 0; i < 2; ++i)
            out[(4 + group * 8 + pair + i) * M + mm] = dw[u][i];
        for (int f = 0; f < 3; ++f) {
            float sum = aux[u][f];
            sum += __shfl_down_sync(0xffffffff, sum, 16);
            sum += __shfl_down_sync(0xffffffff, sum, 8);
            if (group < 2)
                out[(68 + 8 * f + pair + group % 2) * M + mm] = sum;
        }
        float bias = aux[u][3];
        for (int d = 8; d; d /= 2)
            bias += __shfl_down_sync(0xffffffff, bias, d, 16);
        if (!(lane % 16))
            out[(92 + lane / 16) * M + mm] = bias;
    }
    aa = ab = {};
    if (valid) {
        for (int block = 0; block < blocks; ++block) {
            int begin = block * CH, limit = min(begin + CH, H);
            auto *seg = reinterpret_cast<Quad<T> *>(tiles);
            // The path tile is dead. Reuse it for packed sources;
            // the coarse scan reuses dead odd trajectory rows for snapshots.
            constexpr int shared_rows = TM * 248 * 2 / (256 * 8) + rows * step;
            constexpr int cached = CH < shared_rows ? CH : shared_rows;
            constexpr int free_rows = coarse ? (H - tail) / 2 : 0;
            constexpr int shared_prev = CH < free_rows ? CH : free_rows;
            Quad<T> prev_state[CH > shared_prev ? CH - shared_prev : 1]{};
#pragma unroll 1
            for (int j = CH - 1; j >= 0; --j) {
                int y = begin + j;
                if (j == CH - 1)
                    neg = block + 1 == blocks ? float4{} : boundary[block * 256 + tid];
                if (y < limit) {
                    auto src = c.packed_source(y);
                    if (j < cached)
                        seg[j * 256 + tid] = src;
                    if (j < shared_prev)
                        trace[(2 * j + 1) * 256 + tid] = Quad<T>::pack(neg);
#pragma unroll
                    for (int n = shared_prev; n < CH; ++n)
                        if (n == j)
                            prev_state[n - shared_prev] = Quad<T>::pack(neg);
                    neg = advance(neg, src.unpack(), ar, gg);
                }
            }
#pragma unroll 1
            for (int j = 0; j < CH; ++j) {
                int y = begin + j;
                if (y < limit) {
                    auto in =
                        (!coarse || (!(y & 1) && y + 1 < H)) ? history.get(y).unpack() : float4{};
                    auto src = j < cached ? seg[j * 256 + tid].unpack() : c.source(y);
                    Quad<T> previous = j < shared_prev ? trace[(2 * j + 1) * 256 + tid] : Quad<T>{};
#pragma unroll
                    for (int n = shared_prev; n < CH; ++n)
                        if (n == j)
                            previous = prev_state[n - shared_prev];
                    auto ds = column_adjoint_step<true>(previous.unpack(), src, in, ar, gg, aa, ab,
                                                        da, dg);
                    if (compact)
                        write_column_gradient<true, true, true>(c, y, ds, gs);
                    else {
                        write_column_gradient<true, false, true>(c, y, ds, gs);
                        if (coarse)
                            clear_unused_columns<false>(c, y, gs);
                        if (ODD_WIDTH && x == ow - 1) {
                            I offset = I(y) * W * M + c.offset_b + 2 * M;
                            store_complex(gs, offset, plane, {});
                            store_complex(gs, offset + 2 * plane, plane, {});
                        }
                    }
                }
            }
        }
    }
    auto *scratch = reinterpret_cast<float *>(trace);
    float pole[4] = {da.r, da.i, dg.r, dg.i};
    __syncthreads();
    for (int f = 0; f < 4; ++f) {
        scratch[tid] = pole[f];
        __syncthreads();
        if (col == 0) {
            float v = 0;
            for (int n = 0; n < TW; ++n)
                v += scratch[local + n * TM];
            out[f * M + m] = v;
        }
        __syncthreads();
    }
}
"""


def source(p):
    body = _KERNEL
    half = p.dtype == torch.float16
    for tag, value in (
        ("MMA_TYPE", "f16" if half else "bf16"),
        ("TYPE", "__half" if half else "__nv_bfloat16"),
        ("TILE_M", p.tm),
        ("HEIGHT", p.h),
        ("REGISTER_ROWS", p.register_rows),
        ("SPAN", p.chunk),
        ("ODD_WIDTH", int(p.odd_width)),
        ("COARSE_FLAG", int(p.coarse)),
        ("FULL_FLAG", int(p.fulltc)),
    ):
        body = body.replace(tag, str(value))
    prefix = scans._CUDA_SOURCE.split("struct ProjectWeights")[0]
    return (scans._odd_compact_source(prefix) if p.odd_width else prefix) + body


@cache
def module(device, p):
    compiler = _nvrtc()
    prog = ct.c_void_p()
    _check_nvrtc(
        compiler,
        compiler.nvrtcCreateProgram(
            ct.byref(prog), source(p).encode(), b"wave58.cu", 0, None, None
        ),
    )
    try:
        flags = [
            b"--gpu-architecture=compute_90",
            b"--std=c++17",
            *(f"--include-path={d}".encode() for d in _include_dirs()),
        ]
        options = (ct.c_char_p * len(flags))(*flags)
        _check_nvrtc(compiler, compiler.nvrtcCompileProgram(prog, len(flags), options), prog)
        size = ct.c_size_t()
        _check_nvrtc(compiler, compiler.nvrtcGetPTXSize(prog, ct.byref(size)), prog)
        image = ct.create_string_buffer(size.value)
        _check_nvrtc(compiler, compiler.nvrtcGetPTX(prog, image), prog)
    finally:
        compiler.nvrtcDestroyProgram(ct.byref(prog))
    mod = _CudaModule(image, device)
    mod.load_function("wave58", p.smem)
    return mod


@torch.library.custom_op("lnet::vscan_tc_bwd", mutates_args=())
def vscan_tc_bwd(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    gr: Tensor,
    gi: Tensor,
    coarse: bool,
    compact: bool,
    eps: float,
    mode: str,
) -> list[Tensor]:
    p = plan(mode, states.shape[1:], states.dtype, coarse, compact)
    if (
        not states.is_cuda
        or not scans._pole_ok(pole, states)
        or not scans._vertical_inputs_ok(states, auxiliary, path, coarse=coarse)
        or path[0].shape[1] != 8
        or not 0 < eps < math.inf
    ):
        raise ValueError("Invalid Wave 58A inputs")
    b, h, w, m = states.shape[1:]
    ow = w // (1 + coarse)
    gr, gi = scans._vertical_adjoints((gr, gi), states, (b, h // (1 + coarse), ow, m))
    grid = p.grid(b, w, m)
    gs = states.new_empty((4, b, h, ow if compact else w, m))
    partial = states.new_empty((grid[0], 94, m), dtype=torch.float32)
    inputs = [
        *(v.float().contiguous() for v in pole),
        states.contiguous(),
        *(v.float().contiguous() for v in auxiliary),
        *(v.contiguous() for v in path[:3]),
    ]
    with torch.cuda.device(states.device):
        mod = module(states.device.index, p)
        mod.launch(
            "wave58",
            grid,
            (256, 1, 1),
            p.smem,
            torch.cuda.current_stream().cuda_stream,
            [
                *map(scans._ptr, [*inputs, gr, gi, gs, partial]),
                *map(
                    ct.c_int,
                    [
                        b,
                        w,
                        m,
                        scans._scalar_types(path),
                        1 + int(not gr.is_contiguous()),
                        1 + int(not gi.is_contiguous()),
                    ],
                ),
                ct.c_bool(compact),
                ct.c_float(eps),
            ],
        )
    grad = scans._reduce_partials(partial, [*pole, *path], 8)
    return [*grad[:4], gs, *grad[4:]]


@vscan_tc_bwd.register_fake
def _fake(pole, states, auxiliary, path, gr, gi, coarse, compact, eps, mode):
    plan(mode, states.shape[1:], states.dtype, coarse, compact)
    _, b, h, w, m = states.shape
    return [
        *(torch.empty_like(v) for v in pole),
        states.new_empty((4, b, h, w // 2 if compact else w, m)),
        *(torch.empty_like(v) for v in path),
    ]
