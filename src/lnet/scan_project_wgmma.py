"""Mode-local Tensor Core path projection for the Wave 34A vertical scan."""

# Triton infers pointer types and specializes uppercase compile-time arguments.
# ruff: noqa: ANN001, ANN202, N803

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _projection_tile(Trajectory, Vx, Vy, Wi, Bi, Wo,
                     m, n, H: tl.constexpr, W: tl.constexpr, M: tl.constexpr,
                     N: tl.constexpr, STEP: tl.constexpr, EPS: tl.constexpr):
    oh = H // STEP
    line = n // oh
    y = n % oh * STEP
    x = line % (W // STEP) * STEP
    k = tl.arange(0, 16)
    pos = ((k & 2) == 0)[None, :]
    component = ((k & 1) * 2 + (k & 4) // 4)[None, :]
    row = y[:, None] + tl.where(pos, STEP - 1, 0)
    address = ((((m * 2 + (~pos).to(tl.int32)) * (N // oh)
                 + line[:, None]) * H + row) * 4 + component)
    dtype = Trajectory.dtype.element_ty
    p = tl.load(Trajectory + address, (n[:, None] < N) & (k[None, :] < 8), 0)
    gx = tl.load(Vx + ((k[None, :] & 1) * W + x[:, None]
                       + tl.where((k[None, :] & 1) == 0, STEP - 1, 0)) * M + m,
                 n[:, None] < N, 0)
    gy = tl.load(Vy + (tl.where(pos, 0, H) + row) * M + m, n[:, None] < N, 0)
    p = (p.to(tl.float32) * tl.rsqrt(tl.maximum(gx * gy, EPS))).to(dtype)
    wi = tl.load(Wi + m * 64 + k[:, None] + k[None, :] * 8,
                 (k[:, None] < 8) & (k[None, :] < 8), 0)
    bias = tl.load(Bi + m * 8 + k, k < 8, 0)
    wo = tl.load(Wo + m * 16 + k[:, None] + k[None, :] * 8,
                 (k[:, None] < 8) & (k[None, :] < 2), 0)
    z = (tl.dot(p, wi).to(dtype).to(tl.float32) + bias[None, :]).to(dtype).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-z))
    act = (z * sigmoid).to(dtype)
    return p, wi, wo, z, sigmoid, act


@triton.jit
def _project_forward(Trajectory, Vx, Vy, Wi, Bi, Wo, Bo, Output,
                     H: tl.constexpr, W: tl.constexpr, M: tl.constexpr,
                     N: tl.constexpr, STEP: tl.constexpr, EPS: tl.constexpr,
                     BN: tl.constexpr):
    m = tl.program_id(1).to(tl.int64)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    _, _, wo, _, _, act = _projection_tile(Trajectory, Vx, Vy, Wi, Bi, Wo,
                                            m, n, H, W, M, N, STEP, EPS)
    k = tl.arange(0, 16)
    bias = tl.load(Bo + m * 2 + k, k < 2, 0)
    dtype = Output.dtype.element_ty
    out = tl.dot(act, wo).to(dtype).to(tl.float32) + bias[None, :]
    tl.store(Output + (m * 2 + k[None, :]) * N + n[:, None], out,
             (n[:, None] < N) & (k[None, :] < 2))


@triton.jit
def _project_backward(Trajectory, Vx, Vy, Wi, Bi, Wo, Dy, Dp, Partials,
                      H: tl.constexpr, W: tl.constexpr, M: tl.constexpr,
                      N: tl.constexpr, STEP: tl.constexpr, EPS: tl.constexpr,
                      BN: tl.constexpr, P: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    worker = tl.program_id(1)
    k = tl.arange(0, 16)
    dw1 = tl.full((16, 16), 0, tl.float32)
    dw2 = tl.full((16, 16), 0, tl.float32)
    db1 = tl.full((16,), 0, tl.float32)
    db2 = tl.full((16,), 0, tl.float32)
    dtype = Trajectory.dtype.element_ty
    for tile in range(worker, tl.cdiv(N, BN), P):
        n = tile * BN + tl.arange(0, BN)
        p, wi, wo, z, sigmoid, act = _projection_tile(
            Trajectory, Vx, Vy, Wi, Bi, Wo, m, n, H, W, M, N, STEP, EPS,
        )
        dy = tl.load(Dy + (m * 2 + k[None, :]) * N + n[:, None],
                     (n[:, None] < N) & (k[None, :] < 2), 0)
        da = tl.dot(dy, tl.trans(wo)).to(dtype).to(tl.float32)
        dz = (da * (sigmoid * (1.0 + z * (1.0 - sigmoid)))).to(dtype)
        dp = tl.dot(dz, tl.trans(wi))
        dw1 = tl.dot(tl.trans(dz), p, dw1)
        dw2 = tl.dot(tl.trans(dy), act, dw2)
        db1 += tl.sum(dz.to(tl.float32), 0)
        db2 += tl.sum(dy.to(tl.float32), 0)
        direction = (k & 2) // 2
        component = (k & 1) * 2 + (k & 4) // 4
        tl.store(Dp + ((m * 2 + direction[None, :]) * N + n[:, None]) * 4
                 + component[None, :], dp, (n[:, None] < N) & (k[None, :] < 8))
    base = Partials + (m * P + worker) * 90
    fields = k[:, None] * 8 + k[None, :]
    tl.store(base + fields, dw1, (k[:, None] < 8) & (k[None, :] < 8))
    tl.store(base + 64 + k, db1, k < 8)
    tl.store(base + 72 + fields, dw2, (k[:, None] < 2) & (k[None, :] < 8))
    tl.store(base + 88 + k, db2, k < 2)


@triton.jit
def _reduce_projection(Partials, Dwi, Dbi, Dwo, Dbo, P: tl.constexpr, BP: tl.constexpr):
    m = tl.program_id(0)
    k = tl.arange(0, 128)
    p = tl.arange(0, BP)
    acc = tl.load(Partials + (m * P + p[:, None]) * 90 + k[None, :],
                  (p[:, None] < P) & (k[None, :] < 90), 0)
    out = tl.sum(acc, 0)
    tl.store(Dwi + m * 64 + k, out, k < 64)
    tl.store(Dbi + m * 8 + k - 64, out, (k >= 64) & (k < 72))
    tl.store(Dwo + m * 16 + k - 72, out, (k >= 72) & (k < 88))
    tl.store(Dbo + m * 2 + k - 88, out, (k >= 88) & (k < 90))


@triton.jit
def _transpose_channels(Real, Imag, Stage,
                        N: tl.constexpr, M: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
                        RS: tl.constexpr, IS: tl.constexpr, BACKWARD: tl.constexpr):
    n = tl.program_id(0) * 32 + tl.arange(0, 32)
    m = (tl.program_id(1) * 32 + tl.arange(0, 32)).to(tl.int64)
    channel = tl.program_id(2)
    cell = ((n // OH // OW) * OH + n % OH) * OW + n // OH % OW
    stage = Stage + (m[:, None] * 2 + channel) * N + n[None, :]
    if channel == 0:
        output = Real + cell[None, :].to(tl.int64) * RS + m[:, None]
    else:
        output = Imag + cell[None, :].to(tl.int64) * IS + m[:, None]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if BACKWARD:
        tl.store(stage, tl.load(output, mask, 0), mask)
    else:
        tl.store(output, tl.load(stage, mask, 0), mask)


def _transpose(real: Tensor, imag: Tensor, stage: Tensor, *, backward: bool) -> None:
    _, oh, ow, modes = real.shape
    cells = stage.shape[-1]
    real_pitch = modes if real.is_contiguous() else real.stride(2)
    imag_pitch = modes if imag.is_contiguous() else imag.stride(2)
    _transpose_channels[(triton.cdiv(cells, 32), triton.cdiv(modes, 32), 2)](
        real, imag, stage, cells, modes, oh, ow, real_pitch, imag_pitch, backward,
    )


def project_modefirst(
    trajectory: Tensor, auxiliary: list[Tensor], path: list[Tensor],
    real: Tensor, imag: Tensor, height: int, width: int, step: int,
    epsilon: float, sms: int, *, partial_rows: int, backward: bool,
) -> tuple[Tensor, list[Tensor]] | None:
    modes = trajectory.shape[0]
    cells = real.numel() // modes
    stage = trajectory.new_empty((modes, 2, cells))
    vx, vy = (value.float().contiguous() for value in auxiliary)
    wi, bi, wo, bo = (value.to(trajectory.dtype).contiguous() for value in path)
    tiles = triton.cdiv(cells, 128)
    if not backward:
        _project_forward[(tiles, modes)](
            trajectory, vx, vy, wi, bi, wo, bo, stage,
            height, width, modes, cells, step, epsilon, 128, num_warps=4,
        )
        _transpose(real, imag, stage, backward=False)
        return None
    _transpose(real, imag, stage, backward=True)
    workers = min(tiles, 32, triton.cdiv(12 * sms, modes), partial_rows // modes)
    partials = trajectory.new_empty((modes, workers, 90), dtype=torch.float32)
    dp = trajectory.new_empty((modes, 2, 4 * cells))
    dwi, dbi, dwo, dbo = (value.new_empty(value.shape) for value in path)
    arch = torch.cuda.get_device_capability(trajectory.device)[0]
    _project_backward[(modes, workers)](
        trajectory, vx, vy, wi, bi, wo, stage, dp, partials,
        height, width, modes, cells, step, epsilon, 128, workers,
        num_warps=8 if arch >= 10 else 4, num_stages=2,
    )
    _reduce_projection[(modes,)](
        partials, dwi, dbi, dwo, dbo, workers, triton.next_power_of_2(workers),
    )
    return dp, [dwi, dbi, dwo, dbo]
