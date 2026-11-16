"""Bounded whole-grid 2D product scan for frozen packed inference.

Each CTA holds a complete <=16x16 spatial grid for a mode tile. Horizontal
states cross the original BF16 boundary in registers before vertical scans.
No horizontal plane allocation or global-memory round trip is required.
The operator is an opt-in candidate until end-to-end GPU measurements select it.
"""

from __future__ import annotations

# Triton kernels describe pointer types and compile-time tensor layouts.
# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
# pyright: reportPrivateUsage=false, reportGeneralTypeIssues=false
import math
from typing import Literal, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice

from .pac_product_scan_contracts import DEFAULT_EPSILON, ProductGainNormalization, gain_kind
from .pac_product_scan_normalization import static_product_scan_auxiliary
from .pac_product_scan_pipeline import _packed_reference
from .pac_triton_bidirectional_product_scan import _compose_complex
from .pac_triton_product_scan_kernels import _compose_complex_pair

Pole = tuple[Tensor, Tensor, Tensor, Tensor]
Path = tuple[Tensor, Tensor, Tensor, Tensor]
_PATHS_TL = tl.constexpr(4)
_PACKED_PATH_INPUTS_TL = tl.constexpr(8)
_PACKED_PATH_OUTPUTS_TL = tl.constexpr(2)
_PATH_DOT_INPUTS_TL = tl.constexpr(16)
_PATH_DOT_OUTPUTS_TL = tl.constexpr(16)


@triton.jit
def _collapse_full_paths(
    r0, i0, r1, i1, r2, i2, r3, i3,
    path_input_weight, path_input_bias, path_output_weight, path_output_bias,
    mode_vector, modes: tl.constexpr, packed_path_hidden: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr, BLOCK_LINES: tl.constexpr, BLOCK_MODES: tl.constexpr,
    BLOCK_PATH_HIDDEN: tl.constexpr, BLOCK_SWIGLU_VALUE: tl.constexpr,
    SWIGLU_HIDDEN: tl.constexpr, PATH_SWIGLU: tl.constexpr,
):
    path_height: tl.constexpr = BLOCK_HEIGHT
    path_lines: tl.constexpr = BLOCK_LINES
    path_tokens: tl.constexpr = path_height * path_lines
    r0_flat = tl.reshape(
        tl.permute(r0, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    r1_flat = tl.reshape(
        tl.permute(r1, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    r2_flat = tl.reshape(
        tl.permute(r2, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    r3_flat = tl.reshape(
        tl.permute(r3, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    i0_flat = tl.reshape(
        tl.permute(i0, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    i1_flat = tl.reshape(
        tl.permute(i1, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    i2_flat = tl.reshape(
        tl.permute(i2, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    i3_flat = tl.reshape(
        tl.permute(i3, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    path_coordinate = tl.arange(0, _PATH_DOT_INPUTS_TL)
    path = path_coordinate % _PATHS_TL
    active_path = tl.where(
        path[None, None, :] == 0,
        r0_flat[:, :, None],
        tl.where(
            path[None, None, :] == 1,
            r1_flat[:, :, None],
            tl.where(
                path[None, None, :] == 2,
                r2_flat[:, :, None],
                r3_flat[:, :, None],
            ),
        ),
    )
    active_path_imag = tl.where(
        path[None, None, :] == 0,
        i0_flat[:, :, None],
        tl.where(
            path[None, None, :] == 1,
            i1_flat[:, :, None],
            tl.where(
                path[None, None, :] == 2,
                i2_flat[:, :, None],
                i3_flat[:, :, None],
            ),
        ),
    )
    packed_path = tl.where(
        path_coordinate[None, None, :] < _PATHS_TL,
        active_path,
        active_path_imag,
    ).to(tl.bfloat16)
    hidden_coordinate = tl.arange(0, BLOCK_PATH_HIDDEN)
    input_weight_offset = (
        mode_vector[:, None, None] * packed_path_hidden + hidden_coordinate[None, None, :]
    ) * _PACKED_PATH_INPUTS_TL + path_coordinate[None, :, None]
    active_input_weight = tl.load(
        path_input_weight + input_weight_offset,
        mask=(mode_vector[:, None, None] < modes)
        & (hidden_coordinate[None, None, :] < packed_path_hidden)
        & (path_coordinate[None, :, None] < _PACKED_PATH_INPUTS_TL),
        other=0.0,
    ).to(tl.bfloat16)
    preactivation = tl.dot(packed_path, active_input_weight)
    active_input_bias = tl.load(
        path_input_bias
        + mode_vector[:, None] * packed_path_hidden
        + hidden_coordinate[None, :],
        mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
        other=0.0,
    ).to(tl.bfloat16)
    preactivation = (preactivation + active_input_bias[:, None, :]).to(tl.bfloat16)
    output_coordinate = tl.arange(0, _PATH_DOT_OUTPUTS_TL)
    active_output_bias = tl.load(
        path_output_bias
        + mode_vector[:, None] * _PACKED_PATH_OUTPUTS_TL
        + output_coordinate[None, :],
        mask=(mode_vector[:, None] < modes)
        & (output_coordinate[None, :] < _PACKED_PATH_OUTPUTS_TL),
        other=0.0,
    ).to(tl.bfloat16)
    if PATH_SWIGLU:
        value_coordinate = tl.arange(0, BLOCK_SWIGLU_VALUE)
        value_indices = tl.broadcast_to(
            value_coordinate[None, None, :],
            (
                BLOCK_MODES,
                path_tokens,
                BLOCK_SWIGLU_VALUE,
            ),
        )
        gate_indices = tl.broadcast_to(
            (2 * SWIGLU_HIDDEN + value_coordinate % SWIGLU_HIDDEN)[None, None, :],
            (
                BLOCK_MODES,
                path_tokens,
                BLOCK_SWIGLU_VALUE,
            ),
        )
        value = tl.gather(preactivation, value_indices, axis=2).to(tl.bfloat16)
        gate_logits = tl.gather(preactivation, gate_indices, axis=2).to(tl.float32)
        gate = (gate_logits * tl.sigmoid(gate_logits)).to(tl.bfloat16)
        gated_value = tl.where(
            value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN,
            value * gate,
            0.0,
        ).to(tl.bfloat16)
        output_weight_offset = (
            mode_vector[:, None, None] * _PACKED_PATH_OUTPUTS_TL
            + output_coordinate[None, None, :]
        ) * (2 * SWIGLU_HIDDEN) + value_coordinate[None, :, None]
        active_output_weight = tl.load(
            path_output_weight + output_weight_offset,
            mask=(mode_vector[:, None, None] < modes)
            & (value_coordinate[None, :, None] < 2 * SWIGLU_HIDDEN)
            & (output_coordinate[None, None, :] < _PACKED_PATH_OUTPUTS_TL),
            other=0.0,
        ).to(tl.bfloat16)
        correction = tl.dot(gated_value, active_output_weight)
        path_output = (correction + active_output_bias[:, None, :]).to(tl.bfloat16)
    else:
        hidden_float = preactivation.to(tl.float32)
        hidden = (hidden_float * tl.sigmoid(hidden_float)).to(tl.bfloat16)
        output_weight_offset = (
            mode_vector[:, None, None] * _PACKED_PATH_OUTPUTS_TL
            + output_coordinate[None, None, :]
        ) * packed_path_hidden + hidden_coordinate[None, :, None]
        active_output_weight = tl.load(
            path_output_weight + output_weight_offset,
            mask=(mode_vector[:, None, None] < modes)
            & (hidden_coordinate[None, :, None] < packed_path_hidden)
            & (output_coordinate[None, None, :] < _PACKED_PATH_OUTPUTS_TL),
            other=0.0,
        ).to(tl.bfloat16)
        path_output = tl.dot(hidden, active_output_weight)
        path_output = (path_output + active_output_bias[:, None, :]).to(tl.bfloat16)
    real = tl.sum(tl.where(output_coordinate[None, None, :] == 0, path_output, 0.0), axis=2)
    imag = tl.sum(tl.where(output_coordinate[None, None, :] == 1, path_output, 0.0), axis=2)
    return (
        tl.permute(tl.reshape(real, (BLOCK_MODES, BLOCK_HEIGHT, BLOCK_LINES)), (1, 2, 0)),
        tl.permute(tl.reshape(imag, (BLOCK_MODES, BLOCK_HEIGHT, BLOCK_LINES)), (1, 2, 0)),
    )


@triton.jit
def _scan2d_kernel(
    DRIVE, AXR, AXI, GXR, GXI, AYR, AYI, GYR, GYI, VX, VY, GLOBAL_GAIN,
    W1, B1, W2, B2, OUTPUT,
    H: tl.constexpr, W: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
    KIND: tl.constexpr, EPSILON: tl.constexpr, EPILOGUE: tl.constexpr,
    PLANAR: tl.constexpr, PATH_SWIGLU: tl.constexpr, HIDDEN: tl.constexpr,
    BH: tl.constexpr, BW: tl.constexpr, BM: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr, BLOCK_VALUE: tl.constexpr, VALUE: tl.constexpr,
):
    batch = tl.program_id(0)
    modes = tl.program_id(1) * BM + tl.arange(0, BM)
    mode = modes[None, None, :]
    y = tl.arange(0, BH)[:, None, None]
    x = tl.arange(0, BW)[None, :, None]
    active = (y < H) & (x < W) & (mode < P)
    offset = ((batch * H + y) * W + x) * (2 * P) + mode
    real = tl.load(DRIVE + offset, active, 0).to(tl.float32)
    imag = tl.load(DRIVE + offset + P, active, 0).to(tl.float32)
    axr = tl.load(AXR + mode, mode < P, 0).to(tl.float32)
    axi = tl.load(AXI + mode, mode < P, 0).to(tl.float32)
    gxr = tl.load(GXR + mode, mode < P, 0).to(tl.float32)
    gxi = tl.load(GXI + mode, mode < P, 0).to(tl.float32)
    xr, xi = tl.where(active, axr, 1.0), tl.where(active, axi, 0.0)
    xp = tl.associative_scan(
        (xr, xi, gxr * real - gxi * imag, gxr * imag + gxi * real),
        axis=1, combine_fn=_compose_complex,
    )
    xn = tl.associative_scan(
        (xr, -xi, gxr * real + gxi * imag, gxr * imag - gxi * real),
        axis=1, combine_fn=_compose_complex, reverse=True,
    )
    # The two-pass implementation stores these four states in the drive dtype.
    # Preserve that BF16 rounding even though the intermediate remains on-chip.
    ar = xp[2].to(DRIVE.dtype.element_ty).to(tl.float32)
    ai = xp[3].to(DRIVE.dtype.element_ty).to(tl.float32)
    br = xn[2].to(DRIVE.dtype.element_ty).to(tl.float32)
    bi = xn[3].to(DRIVE.dtype.element_ty).to(tl.float32)
    ayr = tl.load(AYR + mode, mode < P, 0).to(tl.float32)
    ayi = tl.load(AYI + mode, mode < P, 0).to(tl.float32)
    gyr = tl.load(GYR + mode, mode < P, 0).to(tl.float32)
    gyi = tl.load(GYI + mode, mode < P, 0).to(tl.float32)
    yr, yi = tl.where(active, ayr, 1.0), tl.where(active, ayi, 0.0)
    yp = tl.associative_scan(
        (yr, yi, gyr * ar - gyi * ai, gyr * ai + gyi * ar,
         gyr * br - gyi * bi, gyr * bi + gyi * br),
        axis=0, combine_fn=_compose_complex_pair,
    )
    yn = tl.associative_scan(
        (yr, -yi, gyr * ar + gyi * ai, gyr * ai - gyi * ar,
         gyr * br + gyi * bi, gyr * bi - gyi * br),
        axis=0, combine_fn=_compose_complex_pair, reverse=True,
    )
    if KIND == 1:
        inv0 = tl.load(GLOBAL_GAIN + mode, mode < P, 0)
        inv1, inv2, inv3 = inv0, inv0, inv0
    else:
        vx0 = tl.load(VX + x * P + mode, (x < W) & (mode < P), 0)
        vx1 = tl.load(VX + W * P + x * P + mode, (x < W) & (mode < P), 0)
        vy0 = tl.load(VY + y * P + mode, (y < H) & (mode < P), 0)
        vy1 = tl.load(VY + H * P + y * P + mode, (y < H) & (mode < P), 0)
        inv0 = tl.rsqrt(tl.maximum(vx0 * vy0, EPSILON))
        inv1 = tl.rsqrt(tl.maximum(vx1 * vy0, EPSILON))
        inv2 = tl.rsqrt(tl.maximum(vx0 * vy1, EPSILON))
        inv3 = tl.rsqrt(tl.maximum(vx1 * vy1, EPSILON))
    r0, i0 = tl.where(active, yp[2] * inv0, 0.0), tl.where(active, yp[3] * inv0, 0.0)
    r1, i1 = tl.where(active, yp[4] * inv1, 0.0), tl.where(active, yp[5] * inv1, 0.0)
    r2, i2 = tl.where(active, yn[2] * inv2, 0.0), tl.where(active, yn[3] * inv2, 0.0)
    r3, i3 = tl.where(active, yn[4] * inv3, 0.0), tl.where(active, yn[5] * inv3, 0.0)
    if EPILOGUE == "path":
        out_r, out_i = _collapse_full_paths(
            r0, i0, r1, i1, r2, i2, r3, i3, W1, B1, W2, B2,
            modes, P, HIDDEN, BH, BW, BM, BLOCK_HIDDEN, BLOCK_VALUE, VALUE, PATH_SWIGLU,
        )
        if PLANAR:
            packed_offset = ((batch * H + y) * W + x) * P + mode
            tl.store(OUTPUT + packed_offset, out_r, active)
            tl.store(OUTPUT + N * H * W * P + packed_offset, out_i, active)
        else:
            tl.store(OUTPUT + offset, out_r, active)
            tl.store(OUTPUT + offset + P, out_i, active)
    else:
        e0 = tl.sum(tl.sum(r0 * r0 + i0 * i0, axis=0), axis=0)
        e1 = tl.sum(tl.sum(r1 * r1 + i1 * i1, axis=0), axis=0)
        e2 = tl.sum(tl.sum(r2 * r2 + i2 * i2, axis=0), axis=0)
        e3 = tl.sum(tl.sum(r3 * r3 + i3 * i3, axis=0), axis=0)
        if EPILOGUE == "mean_energy":
            energy = e0 + e1 + e2 + e3
            tl.store(OUTPUT + batch * P + modes,
                     libdevice.log1p(energy / (4 * H * W)), modes < P)
        else:
            base = batch * 4 * P + modes
            tl.store(OUTPUT + base, libdevice.log1p(e0 / (H * W)), modes < P)
            tl.store(OUTPUT + base + P, libdevice.log1p(e1 / (H * W)), modes < P)
            tl.store(OUTPUT + base + 2 * P, libdevice.log1p(e2 / (H * W)), modes < P)
            tl.store(OUTPUT + base + 3 * P, libdevice.log1p(e3 / (H * W)), modes < P)


@triton_op("lnet::product_scan_2d_inference", mutates_args={})
def _scan2d_op(
    inputs: list[Tensor], path: list[Tensor], epilogue: str, epsilon: float,
    kind: int, path_swiglu: bool, planar_output: bool, block_modes: int, num_warps: int,
) -> Tensor:
    if not inputs[8].is_cuda:
        packed, descriptor = _packed_reference(
            inputs, path, "full16" if path else epilogue,
            epsilon, kind, path_swiglu, planar_output,
        )
        return packed if path else descriptor
    drive = inputs[8]
    batch, height, width, channels = drive.shape
    modes = channels // 2
    pole_x = cast("Pole", tuple(inputs[:4]))
    pole_y = cast("Pole", tuple(inputs[4:8]))
    auxiliary = static_product_scan_auxiliary(
        pole_x, pole_y, drive[..., :modes], epsilon=epsilon, gain_kind=kind,
    )
    if path:
        output = drive.new_empty((2, batch, height, width, modes) if planar_output else drive.shape)
    else:
        output = drive.new_empty((batch, modes if epilogue == "mean_energy" else 4 * modes),
                                 dtype=torch.float32)
    weights = path or [drive] * 4
    hidden = path[0].shape[1] if path else 1
    value = hidden // 3 if path_swiglu else 1
    wrap_triton(_scan2d_kernel)[(batch, triton.cdiv(modes, block_modes))](
        drive, *pole_x, *pole_y, *auxiliary, *weights, output,
        height, width, modes, batch, kind, epsilon, epilogue,
        planar_output, path_swiglu, hidden,
        max(4, triton.next_power_of_2(height)), max(4, triton.next_power_of_2(width)),
        block_modes,
        max(16, triton.next_power_of_2(hidden)),
        max(16, triton.next_power_of_2(2 * value)), value,
        num_warps=num_warps,
    )
    return output


def run_product_scan_2d_inference(  # noqa: C901, PLR0912
    pole_x: Pole,
    pole_y: Pole,
    drive: Tensor,
    path: Path | None = None,
    *,
    epilogue: Literal["path", "descriptor", "mean_energy"] = "path",
    gain_normalization: ProductGainNormalization = "global",
    epsilon: float = DEFAULT_EPSILON,
    path_swiglu: bool = False,
    planar_output: bool = False,
    block_modes: int = 2,
    num_warps: int = 4,
    experimental_large_grid: bool = False,
) -> Tensor:
    """Evaluate packed small-grid D4 inference without horizontal state storage.

    Default eligibility stops at 16 per axis. Explicit experimental_large_grid
    permits up to 56 only for resource/timing experiments; installers stay at
    7/14. Whole-grid state can spill for larger grids. Block/warp settings are
    explicit so component measurements can select them.
    """
    if drive.ndim != 4 or drive.shape[-1] < 2 or drive.shape[-1] % 2:
        raise ValueError("2D inference requires NHW(2P) drive")
    max_size = 56 if experimental_large_grid else 16
    if min(drive.shape) < 1 or max(drive.shape[1:3]) > max_size:
        raise ValueError(f"2D inference requires positive spatial dimensions <={max_size}")
    if not drive.is_contiguous():
        raise ValueError("2D inference drive must be contiguous")
    if drive.dtype != torch.bfloat16:
        raise TypeError("2D inference requires BF16 drive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("2D inference epsilon must be finite and positive")
    modes = drive.shape[-1] // 2
    for pole in (pole_x, pole_y):
        if len(pole) != 4 or any(value.shape != (1, 1, 1, modes) for value in pole):
            raise ValueError("2D inference requires compact 111P poles")
        if any(value.device != drive.device or not value.is_contiguous() for value in pole):
            raise ValueError("2D inference poles must be contiguous and on the drive device")
        if any(value.dtype != pole[0].dtype for value in pole):
            raise TypeError("2D inference pole coordinates must share a dtype")
        if pole[0].dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("2D inference poles must use FP32 or BF16")
    if block_modes not in (1, 2, 4, 8) or num_warps not in (4, 8):
        raise ValueError("2D inference requires block_modes 1/2/4/8 and num_warps 4/8")
    if epilogue not in ("path", "descriptor", "mean_energy"):
        raise ValueError("unsupported 2D inference epilogue")
    if (epilogue == "path") != (path is not None):
        raise ValueError("2D inference path parameters require the path epilogue")
    if path is None and (path_swiglu or planar_output):
        raise ValueError("path layout and SwiGLU options require path parameters")
    if path is not None:
        w1, b1, w2, b2 = path
        if w1.ndim != 3 or w1.shape[0] != modes or w1.shape[2] != 8:
            raise ValueError("2D inference path input weights must be P-by-hidden-by-8")
        hidden = w1.shape[1]
        if hidden < 1 or (path_swiglu and hidden % 3):
            raise ValueError("2D inference path hidden width is invalid")
        output_hidden = 2 * (hidden // 3) if path_swiglu else hidden
        if b1.shape != (modes, hidden) or w2.shape != (modes, 2, output_hidden):
            raise ValueError("2D inference path parameter shapes are inconsistent")
        if b2.shape != (modes, 2):
            raise ValueError("2D inference path output bias must be P-by-2")
        if any(value.dtype not in (torch.float32, torch.bfloat16) for value in path):
            raise TypeError("2D inference path parameters must use FP32 or BF16")
        if any(value.device != drive.device or not value.is_contiguous() for value in path):
            raise ValueError("2D inference path parameters must be contiguous on the drive device")
    if torch.is_grad_enabled() and any(
        value.requires_grad for value in (*pole_x, *pole_y, drive, *(path or ()))
    ):
        raise ValueError("2D scan is inference-only; use inference_mode or frozen tensors")
    return _scan2d_op(
        [*pole_x, *pole_y, drive], list(path or ()), epilogue, epsilon,
        gain_kind(gain_normalization), path_swiglu, planar_output, block_modes, num_warps,
    )
