"""Autotuned inference kernels for independent real feature coordinates.

The scan's two planes are storage, not a constraint on the real read matrix.
The two GEMM rounding boundaries are retained for checkpoint compatibility.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .pac_triton_real_spatial import _validate


@triton.autotune(
    configs=[
        triton.Config({"BM": m, "BN": n, "BK": k}, num_warps=w)
        for m, n, k, w in [
            (16, 32, 32, 4),
            (16, 64, 32, 4),
            (32, 64, 32, 4),
            (32, 128, 32, 4),
            (64, 64, 32, 4),
            (64, 128, 32, 8),
            (32, 64, 64, 4),
            (64, 64, 64, 4),
        ]
    ],
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def _real_projection(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    left = tl.full((BM, BN), 0, tl.float32)
    right = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        inner = start * BK + ks
        xm = (rows[:, None] < M) & (inner[None, :] < K)
        wm = (inner[:, None] < K) & (cols[None, :] < N)
        offsets = rows[:, None] * K + inner[None, :]
        x0 = tl.load(X + offsets, xm, 0)
        x1 = tl.load(X + M * K + offsets, xm, 0)
        offsets_w = cols[None, :] * (2 * K) + inner[:, None]
        w0 = tl.load(W + offsets_w, wm, 0).to(x0.dtype)
        w1 = tl.load(W + offsets_w + K, wm, 0).to(x0.dtype)
        left = tl.dot(x0, w0, left)
        right = tl.dot(x1, w1, right)
    value = left.to(Y.dtype.element_ty).to(tl.float32) + right.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(
        Y + rows[:, None] * N + cols[None, :], value, (rows[:, None] < M) & (cols[None, :] < N)
    )


@triton_op("lnet::real_inference_projection", mutates_args={})
def real_inference_projection(memory: Tensor, weight: Tensor) -> Tensor:
    if not memory.is_cuda or memory.dtype != torch.bfloat16:
        raise ValueError("Expected CUDA BF16 planar memory")
    if memory.ndim < 2 or weight.ndim != 2 or weight.device != memory.device:
        raise ValueError("Expected planar memory and a matrix on the same CUDA device")
    if min(*memory.shape, *weight.shape) <= 0 or weight.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError("Expected positive shapes and FP32/BF16 weights")
    if (
        not memory.is_contiguous()
        or not weight.is_contiguous()
        or memory.shape[0] != 2
        or weight.shape[-1] != 2 * memory.shape[-1]
    ):
        raise ValueError("Expected contiguous two-plane memory and full independent real matrix")
    k = memory.shape[-1]
    n = weight.shape[0]
    m = memory.numel() // (2 * k)
    result = memory.new_empty((*memory.shape[1:-1], n))
    wrap_triton(_real_projection)[
        lambda cfg: (triton.cdiv(m, cfg["BM"]), triton.cdiv(n, cfg["BN"]))
    ](memory, weight, result, m, n, k)
    return result


@triton.autotune(
    configs=[triton.Config({"PIXELS": p}, num_warps=w) for p in (1, 2, 4, 8) for w in (4, 8)],
    key=["N", "H", "W", "P", "R", "K", "C", "REF", "MATCH"],
    cache_results=True,
)
@triton.jit
def _real_spatial_rms(
    X,
    WG,
    SRC,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    REF: tl.constexpr,
    EPS: tl.constexpr,
    MATCH: tl.constexpr,
    POLES: tl.constexpr,
    CHANNELS: tl.constexpr,
    PIXELS: tl.constexpr,
):
    pixels = tl.program_id(0) * PIXELS + tl.arange(0, PIXELS)
    poles = tl.arange(0, POLES)
    row = pixels // W % H
    col = pixels % W
    batch = pixels // (H * W)
    out0 = tl.full((PIXELS, POLES), 0, tl.float32)
    out1 = tl.full((PIXELS, POLES), 0, tl.float32)
    for rank in tl.static_range(R):
        for ky in tl.static_range(K):
            for kx in tl.static_range(K):
                ih = row + ky - K // 2
                iw = col + kx - K // 2
                active = (pixels < N * H * W) & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                xp = ((batch * H + ih) * W + iw)[:, None] * (P * R) + poles[None, :] * R + rank
                value = tl.load(X + xp, active[:, None] & (poles[None, :] < P), 0).to(tl.float32)
                wp = ((poles * 2 * R + rank) * K + ky) * K + kx
                w0 = tl.load(WG + wp, poles < P, 0).to(tl.float32)
                w1 = tl.load(WG + wp + R * K * K, poles < P, 0).to(tl.float32)
                out0 = tl.fma(value, w0[None, :], out0)
                out1 = tl.fma(value, w1[None, :], out1)
    # The original spatial result is stored before RMS energy is computed.
    out0 = out0.to(Y.dtype.element_ty).to(tl.float32)
    out1 = out1.to(Y.dtype.element_ty).to(tl.float32)
    if MATCH:
        channels = tl.arange(0, CHANNELS)
        source = tl.load(
            SRC + pixels[:, None] * C + channels[None, :],
            (pixels[:, None] < N * H * W) & (channels[None, :] < C),
            0,
        ).to(tl.float32)
        input_energy = tl.sum(source * source, 1) / C
        output_energy = tl.sum(tl.where(poles[None, :] < REF, out0 * out0 + out1 * out1, 0), 1) / (
            2 * REF
        )
        scale = (
            tl.sqrt((input_energy + EPS) / (output_energy + EPS))
            .to(Y.dtype.element_ty)
            .to(tl.float32)
        )
        out0 *= scale[:, None]
        out1 *= scale[:, None]
    offset = pixels[:, None] * (2 * P) + poles[None, :]
    mask = (pixels[:, None] < N * H * W) & (poles[None, :] < P)
    tl.store(Y + offset, out0, mask)
    tl.store(Y + offset + P, out1, mask)


@triton_op("lnet::real_inference_spatial_rms", mutates_args={})
def real_inference_spatial_rms(
    x: Tensor, weight: Tensor, source: Tensor, reference: int, epsilon: float, match: bool
) -> Tensor:
    _validate(x, weight)
    if source.ndim != 4 or source.shape[:-1] != x.shape[:-1] or source.device != x.device:
        raise ValueError("Source and projected features must share NHW and CUDA device")
    if source.shape[-1] <= 0 or source.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Expected positive FP32/BF16 source channels")
    if not 0 < reference <= weight.shape[0] or not epsilon > 0:
        raise ValueError("Invalid RMS reference width or epsilon")
    n, h, w, _ = x.shape
    p, _, r, k, _ = weight.shape
    c = source.shape[-1]
    if not x.is_contiguous() or not weight.is_contiguous() or not source.is_contiguous():
        raise ValueError("Fused real Reader expects contiguous NHWC tensors")
    result = x.new_empty((n, h, w, 2 * p))
    wrap_triton(_real_spatial_rms)[lambda cfg: (triton.cdiv(n * h * w, cfg["PIXELS"]),)](
        x,
        weight,
        source,
        result,
        n,
        h,
        w,
        p,
        r,
        k,
        c,
        reference,
        epsilon,
        match,
        POLES=triton.next_power_of_2(p),
        CHANNELS=triton.next_power_of_2(c),
        enable_fp_fusion=False,
    )
    return result
