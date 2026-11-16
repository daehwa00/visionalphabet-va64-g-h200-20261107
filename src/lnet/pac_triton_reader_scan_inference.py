"""Inference candidates that consume Reader outputs inside the horizontal scan.

The full candidate removes both the raw spatial output and RMS-matched drive.
The smaller candidate retains the spatial output and removes the matched drive.
Both keep the Reader's BF16 spatial, scale, and multiplication boundaries.
These operators deliberately have no backward and must only be used on frozen
inference copies. They return the same four-plane layout as the packed pipeline.
"""

from __future__ import annotations

# Triton JIT functions describe pointer arguments and compile-time constants.
# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .pac_kernel_launch_config import autotuned, resolve
from .pac_product_scan_reference import bidirectional_product_scan_reference
from .pac_triton_bidirectional_product_scan import (
    FORWARD_LAUNCH_NAME,
    _bidirectional_forward_kernel,  # pyright: ignore[reportPrivateUsage]
    _compose_complex,  # pyright: ignore[reportPrivateUsage]
    _forward_planes_op,  # pyright: ignore[reportPrivateUsage]
    _scan_launch_scope,  # pyright: ignore[reportPrivateUsage]
)

Pole = tuple[Tensor, Tensor, Tensor, Tensor]
SCALED_FORWARD_LAUNCH_NAME = "reader_scaled_horizontal_forward"


@triton.jit
def _reader_scan_kernel(
    PROJECTED,
    WR,
    WI,
    RAW,
    ENERGY,
    AR,
    AI,
    GR,
    GI,
    OUTPUT,
    H: tl.constexpr,
    W: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    LINES: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    REFERENCE_MODES: tl.constexpr,
    EPSILON: tl.constexpr,
    MATCH_RMS: tl.constexpr,
    FUSE_SPATIAL: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
):
    line = tl.program_id(0)
    batch, row = line // H, line % H
    xs = tl.arange(0, BLOCK_WIDTH)
    x = xs[:, None]
    mode = tl.arange(0, BLOCK_MODES)[None, :]
    active = (x < W) & (mode < P)
    if FUSE_SPATIAL:
        real = tl.full((BLOCK_WIDTH, BLOCK_MODES), 0.0, tl.float32)
        imag = tl.full((BLOCK_WIDTH, BLOCK_MODES), 0.0, tl.float32)
        for rank in tl.static_range(R):  # pyright: ignore[reportGeneralTypeIssues]
            for ky in tl.static_range(K):  # pyright: ignore[reportGeneralTypeIssues]
                for kx in tl.static_range(K):  # pyright: ignore[reportGeneralTypeIssues]
                    ih, iw = row + ky - K // 2, x + kx - K // 2
                    valid = active & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                    offset = batch * S0 + (mode * R + rank) * S1 + ih * S2 + iw * S3
                    xr = tl.load(PROJECTED + offset, valid, 0).to(tl.float32)
                    xi = tl.load(PROJECTED + offset + P * R * S1, valid, 0).to(tl.float32)
                    weight = ((mode * R + rank) * K + ky) * K + kx
                    wr = tl.load(WR + weight, mode < P, 0).to(tl.float32)
                    wi = tl.load(WI + weight, mode < P, 0).to(tl.float32)
                    real += xr * wr - xi * wi
                    imag += xr * wi + xi * wr
        # The original spatial operator stores to this dtype before RMS matching.
        real = real.to(OUTPUT.dtype.element_ty).to(tl.float32)
        imag = imag.to(OUTPUT.dtype.element_ty).to(tl.float32)
    else:
        offset = (line * W + x) * (2 * P) + mode
        real = tl.load(RAW + offset, active, 0).to(tl.float32)
        imag = tl.load(RAW + offset + P, active, 0).to(tl.float32)
    if MATCH_RMS:
        reference = mode < REFERENCE_MODES
        real_energy = tl.sum(tl.where(reference, real * real, 0.0), axis=1)
        imag_energy = tl.sum(tl.where(reference, imag * imag, 0.0), axis=1)
        output_energy = (real_energy + imag_energy) / REFERENCE_MODES
        input_energy = tl.load(ENERGY + line * W + xs, xs < W, 0)
        scale = tl.sqrt((input_energy + EPSILON) / (output_energy + EPSILON))
        # Keep both the scale cast and the drive multiplication cast explicitly.
        scale = scale.to(OUTPUT.dtype.element_ty).to(tl.float32)
        real = (real * scale[:, None]).to(OUTPUT.dtype.element_ty).to(tl.float32)
        imag = (imag * scale[:, None]).to(OUTPUT.dtype.element_ty).to(tl.float32)
    ar = tl.load(AR + mode, mode < P, 0).to(tl.float32)
    ai = tl.load(AI + mode, mode < P, 0).to(tl.float32)
    gr = tl.load(GR + mode, mode < P, 0).to(tl.float32)
    gi = tl.load(GI + mode, mode < P, 0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    scan_ai = tl.where(active, ai, 0.0)
    positive = tl.associative_scan(
        (scan_ar, scan_ai, gr * real - gi * imag, gr * imag + gi * real),
        axis=0,
        combine_fn=_compose_complex,
    )
    negative = tl.associative_scan(
        (scan_ar, -scan_ai, gr * real + gi * imag, gr * imag - gi * real),
        axis=0,
        combine_fn=_compose_complex,
        reverse=True,
    )
    offset = (line * W + x) * P + mode
    plane_size = LINES * W * P
    tl.store(OUTPUT + offset, positive[2], active)
    tl.store(OUTPUT + plane_size + offset, positive[3], active)
    tl.store(OUTPUT + 2 * plane_size + offset, negative[2], active)
    tl.store(OUTPUT + 3 * plane_size + offset, negative[3], active)


@triton.jit
def _rms_scale_kernel(
    RAW,
    ENERGY,
    SCALE,
    PIXELS: tl.constexpr,
    P: tl.constexpr,
    REFERENCE_MODES: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_PIXELS: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
):
    pixel = tl.program_id(0) * BLOCK_PIXELS + tl.arange(0, BLOCK_PIXELS)
    mode = tl.arange(0, BLOCK_MODES)
    offset = pixel[:, None] * (2 * P) + mode[None, :]
    mask = (pixel[:, None] < PIXELS) & (mode[None, :] < REFERENCE_MODES)
    real = tl.load(RAW + offset, mask, 0).to(tl.float32)
    imag = tl.load(RAW + offset + P, mask, 0).to(tl.float32)
    energy = (tl.sum(real * real, axis=1) + tl.sum(imag * imag, axis=1)) / REFERENCE_MODES
    input_energy = tl.load(ENERGY + pixel, pixel < PIXELS, 0)
    scale = tl.sqrt((input_energy + EPSILON) / (energy + EPSILON))
    tl.store(SCALE + pixel, scale, pixel < PIXELS)


@triton.jit
def _scaled_scan_kernel(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    raw,
    scale,
    output,
    height: int,
    width: int,
    line_count: int,
    modes: int,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
):
    line = tl.program_id(0)
    batch, row = line // height, line % height
    x = tl.arange(0, BLOCK_WIDTH)[:, None]
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)[None, :]
    active = (line < line_count) & (x < width) & (mode < modes)
    offset = ((batch * height + row) * width + x) * (2 * modes) + mode
    real = tl.load(raw + offset, active, 0).to(tl.float32)
    imag = tl.load(raw + offset + modes, active, 0).to(tl.float32)
    gain = tl.load(scale + line * width + x, x < width, 0).to(tl.float32)
    # Multiplication crosses the same BF16 drive boundary as the original Reader.
    real = (real * gain).to(output.dtype.element_ty).to(tl.float32)
    imag = (imag * gain).to(output.dtype.element_ty).to(tl.float32)
    ar = tl.load(decay_real + mode, mode < modes, 0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mode < modes, 0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mode < modes, 0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mode < modes, 0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    scan_ai = tl.where(active, ai, 0.0)
    positive = tl.associative_scan(
        (scan_ar, scan_ai, gr * real - gi * imag, gr * imag + gi * real),
        axis=0,
        combine_fn=_compose_complex,
    )
    negative = tl.associative_scan(
        (scan_ar, -scan_ai, gr * real + gi * imag, gr * imag - gi * real),
        axis=0,
        combine_fn=_compose_complex,
        reverse=True,
    )
    out_offset = (line * width + x) * modes + mode
    plane_size = line_count * width * modes
    tl.store(output + out_offset, positive[2], active)
    tl.store(output + plane_size + out_offset, positive[3], active)
    tl.store(output + 2 * plane_size + out_offset, negative[2], active)
    tl.store(output + 3 * plane_size + out_offset, negative[3], active)


def _validate_input_energy(
    input_energy: Tensor, source: Tensor, shape: tuple[int, int, int, int],
) -> None:
    if input_energy.shape != (*shape[:3], 1):
        raise ValueError("reader scan input energy must have shape NHW1")
    if input_energy.device != source.device or not input_energy.is_contiguous():
        raise ValueError("reader scan input energy must be contiguous on the source device")
    if input_energy.dtype != torch.float32:
        raise TypeError("reader scan input energy must use FP32")


def _validate_common(
    source: Tensor,
    input_energy: Tensor,
    pole: Pole,
    shape: tuple[int, int, int, int],
    *,
    match_input_rms: bool,
    rms_reference_modes: int,
    epsilon: float,
) -> None:
    modes = shape[-1]
    if min(shape) < 1:
        raise ValueError("reader scan dimensions must be positive")
    if source.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("reader scan input must use FP32 or BF16")
    if not 1 <= rms_reference_modes <= modes:
        raise ValueError("reader scan RMS reference modes must be within the pole width")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("reader scan epsilon must be finite and positive")
    if any(value.shape != (1, 1, 1, modes) for value in pole):
        raise ValueError("reader scan requires compact 111P poles")
    if any(value.device != source.device or not value.is_contiguous() for value in pole):
        raise ValueError("reader scan poles must be contiguous and share the source device")
    if (
        any(value.dtype != pole[0].dtype for value in pole)
        or pole[0].dtype not in (torch.float32, torch.bfloat16)
    ):
        raise TypeError("reader scan poles must share one FP32 or BF16 dtype")
    if match_input_rms:
        _validate_input_energy(input_energy, source, shape)
    if torch.is_grad_enabled() and any(
        value.requires_grad for value in (source, input_energy, *pole)
    ):
        raise ValueError(
            "reader scan fusion is inference-only; use inference_mode or frozen tensors",
        )


def _match_and_scan_reference(
    raw: Tensor,
    input_energy: Tensor,
    pole: Pole,
    *,
    match_input_rms: bool,
    rms_reference_modes: int,
    epsilon: float,
) -> Tensor:
    real, imag = raw.chunk(2, dim=-1)
    if match_input_rms:
        output_energy = (
            real[..., :rms_reference_modes].float().square().sum(dim=-1, keepdim=True)
            + imag[..., :rms_reference_modes].float().square().sum(dim=-1, keepdim=True)
        ) / rms_reference_modes
        scale = torch.sqrt((input_energy + epsilon) / (output_energy + epsilon)).to(raw.dtype)
        real, imag = real * scale, imag * scale
    return torch.stack(bidirectional_product_scan_reference(pole, (real, imag)))


def _spatial_reference(projected: Tensor, real: Tensor, imag: Tensor) -> Tensor:
    """Scalar-tap FP32 accumulation, including the spatial operator's output cast."""
    batch, _, height, width = projected.shape
    modes, rank, kernel, _ = real.shape
    padded = torch.nn.functional.pad(projected.float(), (kernel // 2,) * 4)
    xr, xi = padded.unflatten(1, (2, modes, rank)).unbind(1)
    output_real = torch.zeros((batch, modes, height, width), device=projected.device)
    output_imag = torch.zeros_like(output_real)
    for r in range(rank):
        for y in range(kernel):
            for x in range(kernel):
                pr = xr[:, :, r, y:y + height, x:x + width]
                pi = xi[:, :, r, y:y + height, x:x + width]
                wr = real[:, r, y, x].float()[None, :, None, None]
                wi = imag[:, r, y, x].float()[None, :, None, None]
                output_real = output_real + (pr * wr - pi * wi)
                output_imag = output_imag + (pr * wi + pi * wr)
    return torch.cat((output_real, output_imag), dim=1).movedim(1, -1).to(projected.dtype)


def _launch(
    projected: Tensor,
    real: Tensor,
    imag: Tensor,
    raw: Tensor,
    input_energy: Tensor,
    pole: Pole,
    shape: tuple[int, int, int, int],
    *,
    match_input_rms: bool,
    rms_reference_modes: int,
    epsilon: float,
    fuse_spatial: bool,
) -> Tensor:
    batch, height, width, modes = shape
    rank, kernel = (real.shape[1], real.shape[2]) if fuse_spatial else (1, 1)
    output = projected.new_empty((4, batch, height, width, modes))
    wrap_triton(_reader_scan_kernel)[(batch * height,)](
        projected, real, imag, raw, input_energy, *pole, output,
        height, width, modes, rank, kernel, batch * height, *projected.stride(),
        REFERENCE_MODES=rms_reference_modes, EPSILON=epsilon,
        MATCH_RMS=match_input_rms, FUSE_SPATIAL=fuse_spatial,
        BLOCK_WIDTH=triton.next_power_of_2(width),
        BLOCK_MODES=triton.next_power_of_2(modes),
        num_warps=8 if width * modes >= 2048 else 4,
    )
    return output


@triton_op("lnet::reader_spatial_scan_inference", mutates_args={})
def _spatial_scan_op(
    projected: Tensor, real: Tensor, imag: Tensor, input_energy: Tensor,
    decay_real: Tensor, decay_imag: Tensor, gamma_real: Tensor, gamma_imag: Tensor,
    match_input_rms: bool, rms_reference_modes: int, epsilon: float,
) -> Tensor:
    if projected.ndim != 4 or real.ndim != 4 or real.shape != imag.shape:
        raise ValueError("reader spatial scan requires NCHW projection and PRKK factors")
    batch, channels, height, width = projected.shape
    modes, rank, kh, kw = real.shape
    if min(real.shape) < 1 or channels != 2 * modes * rank or kh != kw or kh % 2 != 1:
        raise ValueError("reader spatial scan channels or odd square kernel are incompatible")
    if not (
        projected.is_contiguous() or projected.is_contiguous(memory_format=torch.channels_last)
    ):
        raise ValueError("reader spatial scan requires a dense NCHW or NHWC projection")
    if any(value.device != projected.device or not value.is_contiguous() for value in (real, imag)):
        raise ValueError("reader spatial factors must be contiguous on the projection device")
    if real.dtype != projected.dtype or imag.dtype != projected.dtype:
        raise TypeError("reader spatial factors must share the projection dtype")
    if torch.is_grad_enabled() and (real.requires_grad or imag.requires_grad):
        raise ValueError("reader spatial scan fusion is inference-only")
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    shape = batch, height, width, modes
    _validate_common(
        projected, input_energy, pole, shape, match_input_rms=match_input_rms,
        rms_reference_modes=rms_reference_modes, epsilon=epsilon,
    )
    if not projected.is_cuda:
        return _match_and_scan_reference(
            _spatial_reference(projected, real, imag), input_energy, pole,
            match_input_rms=match_input_rms,
            rms_reference_modes=rms_reference_modes, epsilon=epsilon,
        )
    return _launch(
        projected, real, imag, projected, input_energy, pole, shape,
        match_input_rms=match_input_rms, rms_reference_modes=rms_reference_modes,
        epsilon=epsilon, fuse_spatial=True,
    )


@triton_op("lnet::reader_rms_scan_inference", mutates_args={})
def _rms_scan_op(
    raw: Tensor, input_energy: Tensor,
    decay_real: Tensor, decay_imag: Tensor, gamma_real: Tensor, gamma_imag: Tensor,
    match_input_rms: bool, rms_reference_modes: int, epsilon: float,
) -> Tensor:
    if raw.ndim != 4 or raw.shape[-1] % 2 or not raw.is_contiguous():
        raise ValueError("reader RMS scan requires contiguous NHW(2P) spatial output")
    batch, height, width, channels = raw.shape
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    shape = batch, height, width, channels // 2
    _validate_common(
        raw, input_energy, pole, shape, match_input_rms=match_input_rms,
        rms_reference_modes=rms_reference_modes, epsilon=epsilon,
    )
    if not raw.is_cuda:
        return _match_and_scan_reference(
            raw, input_energy, pole, match_input_rms=match_input_rms,
            rms_reference_modes=rms_reference_modes, epsilon=epsilon,
        )
    return _launch(
        raw, raw, raw, raw, input_energy, pole, shape,
        match_input_rms=match_input_rms, rms_reference_modes=rms_reference_modes,
        epsilon=epsilon, fuse_spatial=False,
    )


@triton_op("lnet::reader_scaled_scan_inference", mutates_args={})
def _scaled_scan_op(
    raw: Tensor, input_energy: Tensor,
    decay_real: Tensor, decay_imag: Tensor, gamma_real: Tensor, gamma_imag: Tensor,
    match_input_rms: bool, rms_reference_modes: int, epsilon: float,
) -> Tensor:
    if raw.ndim != 4 or raw.shape[-1] % 2 or not raw.is_contiguous():
        raise ValueError("reader scaled scan requires contiguous NHW(2P) spatial output")
    batch, height, width, channels = raw.shape
    modes = channels // 2
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    _validate_common(
        raw, input_energy, pole, (batch, height, width, modes),
        match_input_rms=match_input_rms, rms_reference_modes=rms_reference_modes,
        epsilon=epsilon,
    )
    if not raw.is_cuda:
        return _match_and_scan_reference(
            raw, input_energy, pole, match_input_rms=match_input_rms,
            rms_reference_modes=rms_reference_modes, epsilon=epsilon,
        )
    if not match_input_rms:
        return _forward_planes_op(*pole, *raw.chunk(2, -1))
    scale = raw.new_empty((batch, height, width, 1))
    pixels = batch * height * width
    wrap_triton(_rms_scale_kernel)[(triton.cdiv(pixels, 16),)](
        raw, input_energy, scale, pixels, modes, rms_reference_modes, epsilon,
        BLOCK_PIXELS=16, BLOCK_MODES=triton.next_power_of_2(rms_reference_modes),
        num_warps=4,
    )
    output = raw.new_empty((4, batch, height, width, modes))
    source_real = raw[..., :modes]
    # The common Autotuner cache is keyed by launch name, not by kernel object.
    # Inherit the old launch geometry without sharing its cached kernel callable.
    geometry = resolve(
        FORWARD_LAUNCH_NAME,
        scope=_scan_launch_scope(_bidirectional_forward_kernel, source_real),
    )
    kernel = autotuned(
        _scaled_scan_kernel, SCALED_FORWARD_LAUNCH_NAME,
        key=("height", "width", "line_count", "modes"),
        scope=_scan_launch_scope(_scaled_scan_kernel, source_real),
        geometry=geometry,
    )

    def grid(metadata: dict[str, int]) -> tuple[int, int]:
        return batch * height, int(triton.cdiv(modes, metadata["BLOCK_MODES"]))

    wrap_triton(kernel)[grid](
        *pole, raw, scale, output, height, width, batch * height, modes,
        BLOCK_WIDTH=triton.next_power_of_2(width),
    )
    return output


def reader_spatial_scan_inference(
    projected: Tensor,
    spatial_real: Tensor,
    spatial_imag: Tensor,
    input_energy: Tensor,
    pole_x: Pole,
    *,
    match_input_rms: bool = True,
    rms_reference_modes: int | None = None,
    epsilon: float = 1e-8,
) -> Tensor:
    """Fuse spatial Reader, RMS matching and scan into four contiguous planes.

    ``input_energy`` is the existing normalized packed input's FP32
    ``square().sum(-1, keepdim=True) / input_modes``. It may be an empty tensor
    when RMS matching is disabled. The projection has component/pole/rank
    channel order. Spatial factors use the projection dtype. The returned
    ``(4,N,H,W,P)`` dtype equals the projection dtype, preserving the existing
    horizontal-state storage boundary.
    """
    reference = spatial_real.shape[0] if rms_reference_modes is None else rms_reference_modes
    return _spatial_scan_op(
        projected, spatial_real, spatial_imag, input_energy, *pole_x,
        match_input_rms, reference, epsilon,
    )


def reader_rms_scan_inference(
    spatial_output: Tensor,
    input_energy: Tensor,
    pole_x: Pole,
    *,
    match_input_rms: bool = True,
    rms_reference_modes: int | None = None,
    epsilon: float = 1e-8,
) -> Tensor:
    """Fuse RMS matching into scan while retaining a packed raw spatial buffer."""
    reference = (
        spatial_output.shape[-1] // 2 if rms_reference_modes is None else rms_reference_modes
    )
    return _rms_scan_op(
        spatial_output, input_energy, *pole_x, match_input_rms, reference, epsilon,
    )


def reader_scaled_scan_inference(
    spatial_output: Tensor,
    input_energy: Tensor,
    pole_x: Pole,
    *,
    match_input_rms: bool = True,
    rms_reference_modes: int | None = None,
    epsilon: float = 1e-8,
) -> Tensor:
    """Keep a tiny BF16 scale buffer and apply it inside a mode-tiled scan.

    This preserves the usual horizontal parallelism over pole tiles while
    removing the NHW(2P) RMS-matched drive buffer. The raw spatial output and
    NHW1 scale are retained; the scale dtype follows the raw spatial dtype.
    """
    reference = (
        spatial_output.shape[-1] // 2 if rms_reference_modes is None else rms_reference_modes
    )
    return _scaled_scan_op(
        spatial_output, input_energy, *pole_x, match_input_rms, reference, epsilon,
    )


__all__ = [
    "reader_rms_scan_inference", "reader_scaled_scan_inference", "reader_spatial_scan_inference",
]
