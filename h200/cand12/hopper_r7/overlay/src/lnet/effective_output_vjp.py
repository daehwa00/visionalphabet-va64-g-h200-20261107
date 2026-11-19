"""Select verified vertical and associative backward kernels using logical-update receipts."""

from dataclasses import replace
from functools import cache
import hashlib
import json
import os
from pathlib import Path
from types import FunctionType

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .direct_path_scan_kernels import _path_channel_gradient, _path_input, _sum_line
from .kernel_launch_config import (
    _WORKLOAD_RECORDS,
    _read_store,
    device_key,
    geometry_store_signature,
    register_kernel_specializer,
    validated_geometry,
)
from .output_dot_dispatch import dot_revision, select_output_dot
from .output_dot_kernel import _output_weight_bias_vjp
from .output_dot_kernel import _path_backward_kernel as _original_kernel
from .path_backward_layout import BUNDLED_LAYOUTS, LAYOUT_NAME, _LayoutGraphPass, _layout_revision
from .retained_path_dispatch import PROFILE_NAME, VERTICAL_BACKWARD, step_revision
from .segmented_path_scan import _path_backward_kernel as _segmented_backward_kernel
from .segmented_path_scan import _prefix_pair
from .triton_product_scan_kernels import (
    _PACKED_PATH_INPUTS_TL,
    _PACKED_PATH_OUTPUTS_TL,
    _PATH_DOT_INPUTS_TL,
    _compose_complex_pair,
)

VJP_NAME = "effective_output_vjp"
ASSOC_BACKWARD_LAUNCH_NAME = "product_scan_path_collapse_backward"
BUNDLE = Path(__file__).with_name("validated_effective_output_vjps.json")


@triton.jit
def _effective_output_vjp_transpose(
    grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
):
    dy = tl.where(
        channel[None, :, None] == 0,
        grad_real[:, None, :],
        tl.where(channel[None, :, None] == 1, grad_imag[:, None, :], 0.0),
    ).to(tl.bfloat16)
    features = tl.where(feature[None, None, :] == 8, 1.0, hidden).to(tl.bfloat16)
    partial = tl.permute(
        tl.dot(tl.permute(features, (0, 2, 1)), tl.permute(dy, (0, 2, 1))),
        (0, 2, 1),
    )
    row = 2 * mode[:, None] + channel[None, :]
    valid = (mode[:, None] < modes) & (channel[None, :] < 2)
    tl.atomic_add(
        dw + row[:, :, None] * hidden_size + feature[None, None, :],
        partial,
        valid[:, :, None] & (feature[None, None, :] < hidden_size),
        sem="relaxed",
    )
    bias_partial = tl.sum(tl.where(feature[None, None, :] == 8, partial, 0.0), axis=2)
    tl.atomic_add(db + row, bias_partial, valid, sem="relaxed")


@triton.jit
def _effective_output_vjp_rows(
    grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
):
    dy = tl.where(
        channel[None, :, None] == 0,
        grad_real[:, None, :],
        tl.where(channel[None, :, None] == 1, grad_imag[:, None, :], 0.0),
    ).to(tl.bfloat16)
    features = tl.where(feature[None, None, :] == 8, 1.0, hidden).to(tl.bfloat16)
    padded_partial = tl.dot(dy, features)
    real_channel = tl.arange(0, 2)
    row_indices = tl.broadcast_to(
        real_channel[None, :, None], (padded_partial.shape[0], 2, padded_partial.shape[2])
    )
    partial = tl.gather(padded_partial, row_indices, axis=1)
    row = 2 * mode[:, None] + real_channel[None, :]
    valid = mode[:, None] < modes
    tl.atomic_add(
        dw + row[:, :, None] * hidden_size + feature[None, None, :],
        partial,
        valid[:, :, None] & (feature[None, None, :] < hidden_size),
        sem="relaxed",
    )
    bias_partial = tl.sum(tl.where(feature[None, None, :] == 8, partial, 0.0), axis=2)
    tl.atomic_add(db + row, bias_partial, valid, sem="relaxed")


@triton.jit
def _fast_sigmoid(x):
    return libdevice.fast_dividef(1.0, 1.0 + libdevice.fast_expf(-x))


@triton.jit
def _slice8_perm(x, BM: tl.constexpr, TOKENS: tl.constexpr):
    half0, _ = tl.split(tl.permute(tl.reshape(x, (BM, TOKENS, 2, 8)), (0, 1, 3, 2)))
    return half0


@triton.jit
def _pad16_perm(x, BM: tl.constexpr, TOKENS: tl.constexpr):
    return tl.reshape(tl.permute(tl.join(x, tl.zeros_like(x)), (0, 1, 3, 2)), (BM, TOKENS, 16))


@triton.jit
def _path_backward_kernel(
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
    grad_collapsed_real,
    grad_collapsed_imag,
    grad_path_input_weight,
    grad_path_input_bias,
    grad_path_output_weight,
    grad_path_output_bias,
    grad_decay_real,
    grad_decay_imag,
    grad_gamma_real,
    grad_gamma_imag,
    grad_source_real_a,
    height: tl.constexpr,
    width: int,
    line_count: int,
    modes: int,
    packed_path_hidden: int,
    epsilon: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    COARSE: tl.constexpr,
    DOUBLING: tl.constexpr,
):
    batch_size = line_count // width
    scan_width = width // 2 if COARSE else width
    tile = tl.program_id(0)
    mode_group = tile // GROUP
    line_group = tl.program_id(1) * GROUP + tile % GROUP
    batch = line_group // scan_width
    x_group = line_group - batch * scan_width
    grad_decay_real += batch * modes
    grad_decay_imag += batch * modes
    grad_gamma_real += batch * modes
    grad_gamma_imag += batch * modes
    grad_path_input_weight += batch * modes * packed_path_hidden * 8
    grad_path_input_bias += batch * modes * packed_path_hidden
    grad_path_output_weight += batch * modes * 2 * packed_path_hidden
    grad_path_output_bias += batch * modes * 2
    x = x_group + tl.arange(0, 1)[None, :, None]
    y = tl.arange(0, 64)[:, None, None]
    mode_vector = mode_group * BM + tl.arange(0, BM)
    mode = mode_vector[None, None, :]
    valid_line = x < scan_width
    active = valid_line & (y < height) & (mode < modes)
    valid_mode = valid_line & (mode < modes)
    source_x_a = 2 * x + 1 if COARSE else x
    source_x_b = 2 * x if COARSE else x
    offset = ((batch * height + y) * width + source_x_a) * modes + mode
    offset_b = offset - modes if COARSE else offset
    ar = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    sar = tl.load(source_real_a + offset, active, 0).to(tl.float32)
    sai = tl.load(source_imag_a + offset, active, 0).to(tl.float32)
    sbr = tl.load(source_real_b + offset_b, active, 0).to(tl.float32)
    sbi = tl.load(source_imag_b + offset_b, active, 0).to(tl.float32)
    par, pai, pbr, pbi = _prefix_pair(
        ar,
        ai,
        gr * sar - gi * sai,
        gr * sai + gi * sar,
        gr * sbr - gi * sbi,
        gr * sbi + gi * sbr,
        active,
        False,
        DOUBLING,
    )
    nar, nai, nbr, nbi = _prefix_pair(
        ar,
        -ai,
        gr * sar + gi * sai,
        gr * sai - gi * sar,
        gr * sbr + gi * sbi,
        gr * sbi - gi * sbr,
        active,
        True,
        DOUBLING,
    )
    decay_norm = ar * ar + ai * ai
    path_height: tl.constexpr = 64 // 2 if COARSE else 64
    valid_y = (y < height) & (mode < modes)
    x_offset = source_x_a * modes + mode
    x_offset_b = source_x_b * modes + mode
    y_offset = y * modes + mode
    vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
    vx_negative = tl.load(variance_x + width * modes + x_offset_b, mask=valid_mode, other=0.0)
    vy_pos_fwd = tl.load(variance_y + y_offset, mask=valid_y, other=0.0)
    vy_neg_fwd = tl.load(variance_y + height * modes + y_offset, mask=valid_y, other=0.0)
    path_tokens: tl.constexpr = path_height
    pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos_fwd, epsilon))
    pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos_fwd, epsilon))
    na = tl.rsqrt(tl.maximum(vx_positive * vy_neg_fwd, epsilon))
    nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg_fwd, epsilon))
    packed_path = _path_input(
        par, pai, pbr, pbi, nar, nai, nbr, nbi, pa, pb, na, nb, active, COARSE
    )
    path_coordinate = tl.arange(0, 16)
    hidden_coordinate = tl.arange(0, 16)
    h8 = tl.arange(0, 8)
    input_weight_offset = (
        mode_vector[:, None, None] * packed_path_hidden + hidden_coordinate[None, None, :]
    ) * 8 + path_coordinate[None, :, None]
    active_input_weight = tl.load(
        path_input_weight + input_weight_offset,
        mask=(mode_vector[:, None, None] < modes)
        & (hidden_coordinate[None, None, :] < packed_path_hidden)
        & (path_coordinate[None, :, None] < 8),
        other=0.0,
    ).to(tl.bfloat16)
    preactivation = tl.dot(packed_path, active_input_weight)
    active_input_bias = tl.load(
        path_input_bias + mode_vector[:, None] * packed_path_hidden + hidden_coordinate[None, :],
        mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
        other=0.0,
    ).to(tl.bfloat16)
    preactivation = (preactivation + active_input_bias[:, None, :]).to(tl.bfloat16)
    if COARSE:
        cell_y = tl.arange(0, path_height)[:, None, None]
        cell_x = x
        path_offset = ((batch * (height // 2) + cell_y) * (width // 2) + cell_x) * modes + mode
        path_active = (cell_y < height // 2) & (cell_x < width // 2) & (mode < modes)
    else:
        path_offset, path_active = (offset, active)
    physical_offset = tl.reshape(tl.permute(path_offset, (2, 0, 1)), (BM, path_tokens))
    physical_active = tl.reshape(tl.permute(path_active, (2, 0, 1)), (BM, path_tokens))
    active_grad_real = tl.load(
        grad_collapsed_real + physical_offset, mask=physical_active, other=0.0
    ).to(tl.bfloat16)
    active_grad_imag = tl.load(
        grad_collapsed_imag + physical_offset, mask=physical_active, other=0.0
    ).to(tl.bfloat16)
    pre8 = _slice8_perm(preactivation, BM, path_tokens).to(tl.float32)
    sig8 = _fast_sigmoid(pre8)
    hid8 = (pre8 * sig8).to(tl.bfloat16)
    silu8 = sig8 * (1.0 + pre8 * (1.0 - sig8))
    hidden = _pad16_perm(hid8, BM, path_tokens)
    output_offset = mode_vector[:, None] * (2 * packed_path_hidden) + hidden_coordinate[None, :]
    output_mask = (mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden)
    tl.atomic_add(
        grad_path_output_weight + output_offset,
        tl.sum(active_grad_real.to(tl.float32)[:, :, None] * hidden.to(tl.float32), axis=1),
        mask=output_mask,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_path_output_weight + output_offset + packed_path_hidden,
        tl.sum(active_grad_imag.to(tl.float32)[:, :, None] * hidden.to(tl.float32), axis=1),
        mask=output_mask,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_path_output_bias + 2 * mode_vector,
        tl.sum(active_grad_real.to(tl.float32), axis=1),
        mask=mode_vector < modes,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_path_output_bias + 2 * mode_vector + 1,
        tl.sum(active_grad_imag.to(tl.float32), axis=1),
        mask=mode_vector < modes,
        sem="relaxed",
    )
    out8_off = mode_vector[:, None] * (2 * packed_path_hidden) + h8[None, :]
    out8_mask = (mode_vector[:, None] < modes) & (h8[None, :] < packed_path_hidden)
    wr8 = (
        tl.load(path_output_weight + out8_off, mask=out8_mask, other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    wi8 = (
        tl.load(path_output_weight + out8_off + packed_path_hidden, mask=out8_mask, other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    gh8 = (
        active_grad_real.to(tl.float32)[:, :, None] * wr8[:, None, :]
        + active_grad_imag.to(tl.float32)[:, :, None] * wi8[:, None, :]
    )
    gp8 = (gh8 * silu8).to(tl.bfloat16)
    grad_preactivation = _pad16_perm(gp8, BM, path_tokens)
    path_input_weight_gradient = tl.dot(tl.permute(grad_preactivation, (0, 2, 1)), packed_path)
    path_input_weight_gradient_offset = (
        mode_vector[:, None, None] * packed_path_hidden + hidden_coordinate[None, :, None]
    ) * 8 + path_coordinate[None, None, :]
    tl.atomic_add(
        grad_path_input_weight + path_input_weight_gradient_offset,
        path_input_weight_gradient,
        mask=(mode_vector[:, None, None] < modes)
        & (hidden_coordinate[None, :, None] < packed_path_hidden)
        & (path_coordinate[None, None, :] < 8),
        sem="relaxed",
    )
    tl.atomic_add(
        grad_path_input_bias
        + mode_vector[:, None] * packed_path_hidden
        + hidden_coordinate[None, :],
        tl.sum(grad_preactivation.to(tl.float32), axis=1),
        mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
        sem="relaxed",
    )
    grad_packed_path = tl.dot(grad_preactivation, tl.permute(active_input_weight, (0, 2, 1))).to(
        tl.float32
    )
    if COARSE:
        grad_packed_path = grad_packed_path.to(tl.bfloat16).to(tl.float32)
    nr0 = _path_channel_gradient(grad_packed_path, path_coordinate, 2, BM, COARSE).to(tl.bfloat16)
    nr1 = _path_channel_gradient(grad_packed_path, path_coordinate, 3, BM, COARSE).to(tl.bfloat16)
    ni0 = _path_channel_gradient(grad_packed_path, path_coordinate, 6, BM, COARSE).to(tl.bfloat16)
    ni1 = _path_channel_gradient(grad_packed_path, path_coordinate, 7, BM, COARSE).to(tl.bfloat16)
    l0r, l0i, l1r, l1i = _prefix_pair(
        ar,
        -ai,
        _path_channel_gradient(grad_packed_path, path_coordinate, 0, BM, COARSE) * pa,
        _path_channel_gradient(grad_packed_path, path_coordinate, 4, BM, COARSE) * pa,
        _path_channel_gradient(grad_packed_path, path_coordinate, 1, BM, COARSE) * pb,
        _path_channel_gradient(grad_packed_path, path_coordinate, 5, BM, COARSE) * pb,
        active,
        True,
        DOUBLING,
    )
    l2r, l2i, l3r, l3i = _prefix_pair(
        ar,
        ai,
        nr0.to(tl.float32) * na,
        ni0.to(tl.float32) * na,
        nr1.to(tl.float32) * nb,
        ni1.to(tl.float32) * nb,
        active,
        False,
        DOUBLING,
    )
    plane_size = tl.cast(batch_size, tl.int64) * (height * width * modes)
    grad_a_imag_pointer = grad_source_real_a + plane_size
    grad_b_real_pointer = grad_source_real_a + 2 * plane_size
    grad_b_imag_pointer = grad_source_real_a + 3 * plane_size
    tl.store(grad_source_real_a + offset, gr * l0r + gi * l0i + gr * l2r - gi * l2i, mask=active)
    tl.store(grad_a_imag_pointer + offset, -gi * l0r + gr * l0i + gi * l2r + gr * l2i, mask=active)
    tl.store(grad_b_real_pointer + offset_b, gr * l1r + gi * l1i + gr * l3r - gi * l3i, mask=active)
    tl.store(
        grad_b_imag_pointer + offset_b, -gi * l1r + gr * l1i + gi * l3r + gr * l3i, mask=active
    )
    inverse_decay = libdevice.fast_dividef(1.0, decay_norm)
    positive_drive_ar, positive_drive_ai = (gr * sar - gi * sai, gr * sai + gi * sar)
    positive_drive_br, positive_drive_bi = (gr * sbr - gi * sbi, gr * sbi + gi * sbr)
    previous_0r = (ar * (par - positive_drive_ar) + ai * (pai - positive_drive_ai)) * inverse_decay
    previous_0i = (ar * (pai - positive_drive_ai) - ai * (par - positive_drive_ar)) * inverse_decay
    previous_1r = (ar * (pbr - positive_drive_br) + ai * (pbi - positive_drive_bi)) * inverse_decay
    previous_1i = (ar * (pbi - positive_drive_bi) - ai * (pbr - positive_drive_br)) * inverse_decay
    negative_drive_ar, negative_drive_ai = (gr * sar + gi * sai, gr * sai - gi * sar)
    negative_drive_br, negative_drive_bi = (gr * sbr + gi * sbi, gr * sbi - gi * sbr)
    previous_2r = (ar * (nar - negative_drive_ar) - ai * (nai - negative_drive_ai)) * inverse_decay
    previous_2i = (ai * (nar - negative_drive_ar) + ar * (nai - negative_drive_ai)) * inverse_decay
    previous_3r = (ar * (nbr - negative_drive_br) - ai * (nbi - negative_drive_bi)) * inverse_decay
    previous_3i = (ai * (nbr - negative_drive_br) + ar * (nbi - negative_drive_bi)) * inverse_decay
    grad_dr = _sum_line(
        l0r * previous_0r
        + l0i * previous_0i
        + l1r * previous_1r
        + l1i * previous_1i
        + l2r * previous_2r
        + l2i * previous_2i
        + l3r * previous_3r
        + l3i * previous_3i,
        active,
    )
    grad_di = _sum_line(
        -l0r * previous_0i
        + l0i * previous_0r
        - l1r * previous_1i
        + l1i * previous_1r
        + l2r * previous_2i
        - l2i * previous_2r
        + l3r * previous_3i
        - l3i * previous_3r,
        active,
    )
    grad_gr = _sum_line(
        l0r * sar
        + l0i * sai
        + l1r * sbr
        + l1i * sbi
        + l2r * sar
        + l2i * sai
        + l3r * sbr
        + l3i * sbi,
        active,
    )
    grad_gi = _sum_line(
        -l0r * sai
        + l0i * sar
        - l1r * sbi
        + l1i * sbr
        + l2r * sai
        - l2i * sar
        + l3r * sbi
        - l3i * sbr,
        active,
    )
    valid_coefficient_mode = mode_vector < modes
    tl.atomic_add(
        grad_decay_real + mode_vector, grad_dr, mask=valid_coefficient_mode, sem="relaxed"
    )
    tl.atomic_add(
        grad_decay_imag + mode_vector, grad_di, mask=valid_coefficient_mode, sem="relaxed"
    )
    tl.atomic_add(
        grad_gamma_real + mode_vector, grad_gr, mask=valid_coefficient_mode, sem="relaxed"
    )
    tl.atomic_add(
        grad_gamma_imag + mode_vector, grad_gi, mask=valid_coefficient_mode, sem="relaxed"
    )


_coarse_fast_kernel = _path_backward_kernel


@triton.jit
def _path_backward_kernel(
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
    grad_collapsed_real,
    grad_collapsed_imag,
    grad_path_input_weight,
    grad_path_input_bias,
    grad_path_output_weight,
    grad_path_output_bias,
    grad_decay_real,
    grad_decay_imag,
    grad_gamma_real,
    grad_gamma_imag,
    grad_source_real_a,
    height: tl.constexpr,
    width: int,
    line_count: int,
    modes: int,
    packed_path_hidden: int,
    epsilon: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
    COARSE: tl.constexpr,
    DOUBLING: tl.constexpr,
):
    batch_size = line_count // width
    scan_width = width // 2 if COARSE else width
    tile = tl.program_id(0)
    mode_group = tile // GROUP
    line_group = tl.program_id(1) * GROUP + tile % GROUP
    batch = line_group // scan_width
    x_group = line_group - batch * scan_width
    grad_decay_real += batch * modes
    grad_decay_imag += batch * modes
    grad_gamma_real += batch * modes
    grad_gamma_imag += batch * modes
    grad_path_input_weight += batch * modes * packed_path_hidden * 8
    grad_path_input_bias += batch * modes * packed_path_hidden
    grad_path_output_weight += batch * modes * 2 * packed_path_hidden
    grad_path_output_bias += batch * modes * 2
    x = x_group + tl.arange(0, 1)[None, :, None]
    y = tl.arange(0, 64)[:, None, None]
    mode_vector = mode_group * BM + tl.arange(0, BM)
    mode = mode_vector[None, None, :]
    valid_line = x < scan_width
    active = valid_line & (y < height) & (mode < modes)
    valid_mode = valid_line & (mode < modes)
    source_x_a = 2 * x + 1 if COARSE else x
    source_x_b = 2 * x if COARSE else x
    offset = ((batch * height + y) * width + source_x_a) * modes + mode
    offset_b = offset - modes if COARSE else offset
    ar = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    sar = tl.load(source_real_a + offset, active, 0).to(tl.float32)
    sai = tl.load(source_imag_a + offset, active, 0).to(tl.float32)
    sbr = tl.load(source_real_b + offset_b, active, 0).to(tl.float32)
    sbi = tl.load(source_imag_b + offset_b, active, 0).to(tl.float32)
    par, pai, pbr, pbi = _prefix_pair(
        ar,
        ai,
        gr * sar - gi * sai,
        gr * sai + gi * sar,
        gr * sbr - gi * sbi,
        gr * sbi + gi * sbr,
        active,
        False,
        DOUBLING,
    )
    nar, nai, nbr, nbi = _prefix_pair(
        ar,
        -ai,
        gr * sar + gi * sai,
        gr * sai - gi * sar,
        gr * sbr + gi * sbi,
        gr * sbi - gi * sbr,
        active,
        True,
        DOUBLING,
    )
    decay_norm = ar * ar + ai * ai
    path_height: tl.constexpr = 64 // 2 if COARSE else 64
    valid_y = (y < height) & (mode < modes)
    x_offset = source_x_a * modes + mode
    x_offset_b = source_x_b * modes + mode
    y_offset = y * modes + mode
    vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
    vx_negative = tl.load(variance_x + width * modes + x_offset_b, mask=valid_mode, other=0.0)
    vy_pos_fwd = tl.load(variance_y + y_offset, mask=valid_y, other=0.0)
    vy_neg_fwd = tl.load(variance_y + height * modes + y_offset, mask=valid_y, other=0.0)
    path_tokens: tl.constexpr = path_height
    pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos_fwd, epsilon))
    pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos_fwd, epsilon))
    na = tl.rsqrt(tl.maximum(vx_positive * vy_neg_fwd, epsilon))
    nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg_fwd, epsilon))
    packed_path = _path_input(
        par, pai, pbr, pbi, nar, nai, nbr, nbi, pa, pb, na, nb, active, COARSE
    )
    path_coordinate = tl.arange(0, 16)
    hidden_coordinate = tl.arange(0, 16)
    h8 = tl.arange(0, 8)
    input_weight_offset = (
        mode_vector[:, None, None] * packed_path_hidden + hidden_coordinate[None, None, :]
    ) * 8 + path_coordinate[None, :, None]
    active_input_weight = tl.load(
        path_input_weight + input_weight_offset,
        mask=(mode_vector[:, None, None] < modes)
        & (hidden_coordinate[None, None, :] < packed_path_hidden)
        & (path_coordinate[None, :, None] < 8),
        other=0.0,
    ).to(tl.bfloat16)
    preactivation = tl.dot(packed_path, active_input_weight)
    active_input_bias = tl.load(
        path_input_bias + mode_vector[:, None] * packed_path_hidden + hidden_coordinate[None, :],
        mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
        other=0.0,
    ).to(tl.bfloat16)
    preactivation = (preactivation + active_input_bias[:, None, :]).to(tl.bfloat16)
    if COARSE:
        cell_y = tl.arange(0, path_height)[:, None, None]
        cell_x = x
        path_offset = ((batch * (height // 2) + cell_y) * (width // 2) + cell_x) * modes + mode
        path_active = (cell_y < height // 2) & (cell_x < width // 2) & (mode < modes)
    else:
        path_offset, path_active = (offset, active)
    physical_offset = tl.reshape(tl.permute(path_offset, (2, 0, 1)), (BM, path_tokens))
    physical_active = tl.reshape(tl.permute(path_active, (2, 0, 1)), (BM, path_tokens))
    active_grad_real = tl.load(
        grad_collapsed_real + physical_offset, mask=physical_active, other=0.0
    ).to(tl.bfloat16)
    active_grad_imag = tl.load(
        grad_collapsed_imag + physical_offset, mask=physical_active, other=0.0
    ).to(tl.bfloat16)
    pre8 = _slice8_perm(preactivation, BM, path_tokens).to(tl.float32)
    sig8 = _fast_sigmoid(pre8)
    hid8 = (pre8 * sig8).to(tl.bfloat16)
    silu8 = sig8 * (1.0 + pre8 * (1.0 - sig8))
    hidden = _pad16_perm(hid8, BM, path_tokens)
    _output_weight_bias_vjp(
        active_grad_real,
        active_grad_imag,
        hidden,
        mode_vector,
        path_coordinate,
        hidden_coordinate,
        packed_path_hidden,
        modes,
        grad_path_output_weight,
        grad_path_output_bias,
    )
    out8_off = mode_vector[:, None] * (2 * packed_path_hidden) + h8[None, :]
    out8_mask = (mode_vector[:, None] < modes) & (h8[None, :] < packed_path_hidden)
    wr8 = (
        tl.load(path_output_weight + out8_off, mask=out8_mask, other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    wi8 = (
        tl.load(path_output_weight + out8_off + packed_path_hidden, mask=out8_mask, other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    gh8 = (
        active_grad_real.to(tl.float32)[:, :, None] * wr8[:, None, :]
        + active_grad_imag.to(tl.float32)[:, :, None] * wi8[:, None, :]
    )
    gp8 = (gh8 * silu8).to(tl.bfloat16)
    grad_preactivation = _pad16_perm(gp8, BM, path_tokens)
    path_input_weight_gradient = tl.dot(tl.permute(grad_preactivation, (0, 2, 1)), packed_path)
    path_input_weight_gradient_offset = (
        mode_vector[:, None, None] * packed_path_hidden + hidden_coordinate[None, :, None]
    ) * 8 + path_coordinate[None, None, :]
    tl.atomic_add(
        grad_path_input_weight + path_input_weight_gradient_offset,
        path_input_weight_gradient,
        mask=(mode_vector[:, None, None] < modes)
        & (hidden_coordinate[None, :, None] < packed_path_hidden)
        & (path_coordinate[None, None, :] < 8),
        sem="relaxed",
    )
    tl.atomic_add(
        grad_path_input_bias
        + mode_vector[:, None] * packed_path_hidden
        + hidden_coordinate[None, :],
        tl.sum(grad_preactivation.to(tl.float32), axis=1),
        mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
        sem="relaxed",
    )
    grad_packed_path = tl.dot(grad_preactivation, tl.permute(active_input_weight, (0, 2, 1))).to(
        tl.float32
    )
    if COARSE:
        grad_packed_path = grad_packed_path.to(tl.bfloat16).to(tl.float32)
    nr0 = _path_channel_gradient(grad_packed_path, path_coordinate, 2, BM, COARSE).to(tl.bfloat16)
    nr1 = _path_channel_gradient(grad_packed_path, path_coordinate, 3, BM, COARSE).to(tl.bfloat16)
    ni0 = _path_channel_gradient(grad_packed_path, path_coordinate, 6, BM, COARSE).to(tl.bfloat16)
    ni1 = _path_channel_gradient(grad_packed_path, path_coordinate, 7, BM, COARSE).to(tl.bfloat16)
    l0r, l0i, l1r, l1i = _prefix_pair(
        ar,
        -ai,
        _path_channel_gradient(grad_packed_path, path_coordinate, 0, BM, COARSE) * pa,
        _path_channel_gradient(grad_packed_path, path_coordinate, 4, BM, COARSE) * pa,
        _path_channel_gradient(grad_packed_path, path_coordinate, 1, BM, COARSE) * pb,
        _path_channel_gradient(grad_packed_path, path_coordinate, 5, BM, COARSE) * pb,
        active,
        True,
        DOUBLING,
    )
    l2r, l2i, l3r, l3i = _prefix_pair(
        ar,
        ai,
        nr0.to(tl.float32) * na,
        ni0.to(tl.float32) * na,
        nr1.to(tl.float32) * nb,
        ni1.to(tl.float32) * nb,
        active,
        False,
        DOUBLING,
    )
    plane_size = tl.cast(batch_size, tl.int64) * (height * width * modes)
    grad_a_imag_pointer = grad_source_real_a + plane_size
    grad_b_real_pointer = grad_source_real_a + 2 * plane_size
    grad_b_imag_pointer = grad_source_real_a + 3 * plane_size
    tl.store(grad_source_real_a + offset, gr * l0r + gi * l0i + gr * l2r - gi * l2i, mask=active)
    tl.store(grad_a_imag_pointer + offset, -gi * l0r + gr * l0i + gi * l2r + gr * l2i, mask=active)
    tl.store(grad_b_real_pointer + offset_b, gr * l1r + gi * l1i + gr * l3r - gi * l3i, mask=active)
    tl.store(
        grad_b_imag_pointer + offset_b, -gi * l1r + gr * l1i + gi * l3r + gr * l3i, mask=active
    )
    inverse_decay = libdevice.fast_dividef(1.0, decay_norm)
    positive_drive_ar, positive_drive_ai = (gr * sar - gi * sai, gr * sai + gi * sar)
    positive_drive_br, positive_drive_bi = (gr * sbr - gi * sbi, gr * sbi + gi * sbr)
    previous_0r = (ar * (par - positive_drive_ar) + ai * (pai - positive_drive_ai)) * inverse_decay
    previous_0i = (ar * (pai - positive_drive_ai) - ai * (par - positive_drive_ar)) * inverse_decay
    previous_1r = (ar * (pbr - positive_drive_br) + ai * (pbi - positive_drive_bi)) * inverse_decay
    previous_1i = (ar * (pbi - positive_drive_bi) - ai * (pbr - positive_drive_br)) * inverse_decay
    negative_drive_ar, negative_drive_ai = (gr * sar + gi * sai, gr * sai - gi * sar)
    negative_drive_br, negative_drive_bi = (gr * sbr + gi * sbi, gr * sbi - gi * sbr)
    previous_2r = (ar * (nar - negative_drive_ar) - ai * (nai - negative_drive_ai)) * inverse_decay
    previous_2i = (ai * (nar - negative_drive_ar) + ar * (nai - negative_drive_ai)) * inverse_decay
    previous_3r = (ar * (nbr - negative_drive_br) - ai * (nbi - negative_drive_bi)) * inverse_decay
    previous_3i = (ai * (nbr - negative_drive_br) + ar * (nbi - negative_drive_bi)) * inverse_decay
    grad_dr = _sum_line(
        l0r * previous_0r
        + l0i * previous_0i
        + l1r * previous_1r
        + l1i * previous_1i
        + l2r * previous_2r
        + l2i * previous_2i
        + l3r * previous_3r
        + l3i * previous_3i,
        active,
    )
    grad_di = _sum_line(
        -l0r * previous_0i
        + l0i * previous_0r
        - l1r * previous_1i
        + l1i * previous_1r
        + l2r * previous_2i
        - l2i * previous_2r
        + l3r * previous_3i
        - l3i * previous_3r,
        active,
    )
    grad_gr = _sum_line(
        l0r * sar
        + l0i * sai
        + l1r * sbr
        + l1i * sbi
        + l2r * sar
        + l2i * sai
        + l3r * sbr
        + l3i * sbi,
        active,
    )
    grad_gi = _sum_line(
        -l0r * sai
        + l0i * sar
        - l1r * sbi
        + l1i * sbr
        + l2r * sai
        - l2i * sar
        + l3r * sbi
        - l3i * sbr,
        active,
    )
    valid_coefficient_mode = mode_vector < modes
    tl.atomic_add(
        grad_decay_real + mode_vector, grad_dr, mask=valid_coefficient_mode, sem="relaxed"
    )
    tl.atomic_add(
        grad_decay_imag + mode_vector, grad_di, mask=valid_coefficient_mode, sem="relaxed"
    )
    tl.atomic_add(
        grad_gamma_real + mode_vector, grad_gr, mask=valid_coefficient_mode, sem="relaxed"
    )
    tl.atomic_add(
        grad_gamma_imag + mode_vector, grad_gi, mask=valid_coefficient_mode, sem="relaxed"
    )


_rows_template_kernel = _path_backward_kernel


@triton.jit
def _pack_once8(r0, r1, r2, r3, i0, i1, i2, i3, BM: tl.constexpr, TOKENS: tl.constexpr):
    even = tl.join(tl.join(r0, i0), tl.join(r2, i2))
    odd = tl.join(tl.join(r1, i1), tl.join(r3, i3))
    paths = tl.reshape(tl.join(even, odd), (TOKENS, BM, 8))
    return tl.permute(paths, (1, 0, 2))


@triton.jit
def _unpack_once_shape(grad, BM: tl.constexpr, H: tl.constexpr, L: tl.constexpr):
    low = _slice8_perm(grad, BM, H * L)
    tile = tl.reshape(tl.permute(low, (1, 0, 2)), (H, L, BM, 2, 2, 2))
    even, odd = tl.split(tile)
    p0, p2 = tl.split(even)
    p1, p3 = tl.split(odd)
    r0, i0 = tl.split(p0)
    r1, i1 = tl.split(p1)
    r2, i2 = tl.split(p2)
    r3, i3 = tl.split(p3)
    return r0, r1, r2, r3, i0, i1, i2, i3


@triton.jit
def _product_scan_coarse4_associative_backward_kernel(
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
    global_inverse_gain,
    grad_coarse_real,
    grad_coarse_imag,
    grad_collapsed_real,
    grad_collapsed_imag,
    path_input_weight,
    path_input_bias,
    path_output_weight,
    _path_output_bias,
    grad_path_input_weight,
    grad_path_input_bias,
    grad_path_output_weight,
    grad_path_output_bias,
    descriptor_gradient_factor,
    grad_decay_real,
    grad_decay_imag,
    grad_gamma_real,
    grad_gamma_imag,
    grad_source_real_a,
    grad_source_imag_a,
    grad_source_real_b,
    grad_source_imag_b,
    height: tl.constexpr,
    width: int,
    line_count: int,
    modes: int,
    packed_path_hidden: int,
    epsilon: tl.constexpr,
    gain_kind: tl.constexpr,
    HAS_COARSE_GRAD: tl.constexpr,
    FULL_COARSE_GRAD: tl.constexpr,
    COLLAPSE_PATHS_GRAD: tl.constexpr,
    PATH_SWIGLU: tl.constexpr,
    SWIGLU_HIDDEN: tl.constexpr,
    HAS_DESCRIPTOR_GRAD: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_LINES: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_PATH_HIDDEN: tl.constexpr,
    BLOCK_SWIGLU_VALUE: tl.constexpr,
    PACKED_OUTPUT: tl.constexpr = 0,
    COARSE_PATHS: tl.constexpr = 0,
    PLANAR_PATH_OUTPUT: tl.constexpr = 0,
    COMBINED_STATE_GRADS: tl.constexpr = 0,
    MEAN_DESCRIPTOR_GRAD: tl.constexpr = 0,
    BATCH_PARTIAL_GRADS: tl.constexpr = 0,
    ZERO_UNUSED_COARSE_GRADS: tl.constexpr = 1,
    LINES_PER_GROUP: tl.constexpr = 0,
    COMPACT_CHANNELS: tl.constexpr = 0,
):
    batch_size = line_count // width
    scan_width = width
    tile = tl.program_id(0)
    mode_group = tile // LINES_PER_GROUP
    line_group = tl.program_id(1) * LINES_PER_GROUP + tile % LINES_PER_GROUP
    batch = line_group // scan_width
    x_group = line_group - batch * scan_width
    grad_decay_real += batch * modes
    grad_decay_imag += batch * modes
    grad_gamma_real += batch * modes
    grad_gamma_imag += batch * modes
    grad_path_input_weight += batch * modes * packed_path_hidden * _PACKED_PATH_INPUTS_TL
    grad_path_input_bias += batch * modes * packed_path_hidden
    grad_path_output_weight += batch * modes * _PACKED_PATH_OUTPUTS_TL * packed_path_hidden
    grad_path_output_bias += batch * modes * _PACKED_PATH_OUTPUTS_TL

    x = x_group * BLOCK_LINES + tl.arange(0, BLOCK_LINES)[None, :, None]
    y = tl.arange(0, BLOCK_HEIGHT)[:, None, None]
    mode_vector = mode_group * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    mode = mode_vector[None, None, :]
    valid_line = x < scan_width
    active = valid_line & (y < height) & (mode < modes)
    valid_mode = valid_line & (mode < modes)
    offset = ((batch * height + y) * width + x) * modes + mode
    ar = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    scan_ai = tl.where(active, ai, 0.0)
    sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
    sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
    sbr = tl.load(source_real_b + offset, mask=active, other=0.0).to(tl.float32)
    sbi = tl.load(source_imag_b + offset, mask=active, other=0.0).to(tl.float32)
    positive = tl.associative_scan(
        (
            scan_ar,
            scan_ai,
            gr * sar - gi * sai,
            gr * sai + gi * sar,
            gr * sbr - gi * sbi,
            gr * sbi + gi * sbr,
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
    )
    par, pai, pbr, pbi = positive[2], positive[3], positive[4], positive[5]
    negative = tl.associative_scan(
        (
            scan_ar,
            -scan_ai,
            gr * sar + gi * sai,
            gr * sai - gi * sar,
            gr * sbr + gi * sbi,
            gr * sbi - gi * sbr,
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
        reverse=True,
    )
    nar, nai, nbr, nbi = negative[2], negative[3], negative[4], negative[5]
    decay_norm = ar * ar + ai * ai

    path_height: tl.constexpr = BLOCK_HEIGHT
    path_lines: tl.constexpr = BLOCK_LINES
    path_tokens: tl.constexpr = path_height * path_lines
    valid_y = (y < height) & (mode < modes)
    x_offset = x * modes + mode
    y_offset = y * modes + mode
    vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
    vx_negative = tl.load(variance_x + width * modes + x_offset, mask=valid_mode, other=0.0)
    vy_pos_fwd = tl.load(variance_y + y_offset, mask=valid_y, other=0.0)
    vy_neg_fwd = tl.load(variance_y + height * modes + y_offset, mask=valid_y, other=0.0)
    pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos_fwd, epsilon))
    pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos_fwd, epsilon))
    na = tl.rsqrt(tl.maximum(vx_positive * vy_neg_fwd, epsilon))
    nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg_fwd, epsilon))

    paths8 = _pack_once8(
        tl.where(active, par * pa, 0.0).to(tl.bfloat16),
        tl.where(active, pbr * pb, 0.0).to(tl.bfloat16),
        tl.where(active, nar * na, 0.0).to(tl.bfloat16),
        tl.where(active, nbr * nb, 0.0).to(tl.bfloat16),
        tl.where(active, pai * pa, 0.0).to(tl.bfloat16),
        tl.where(active, pbi * pb, 0.0).to(tl.bfloat16),
        tl.where(active, nai * na, 0.0).to(tl.bfloat16),
        tl.where(active, nbi * nb, 0.0).to(tl.bfloat16),
        BLOCK_MODES, path_tokens,
    )
    packed_path = _pad16_perm(paths8, BLOCK_MODES, path_tokens)
    path_coordinate = tl.arange(0, _PATH_DOT_INPUTS_TL)
    hidden_coordinate = tl.arange(0, BLOCK_PATH_HIDDEN)
    h8 = tl.arange(0, 8)
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
        path_input_bias + mode_vector[:, None] * packed_path_hidden + hidden_coordinate[None, :],
        mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
        other=0.0,
    ).to(tl.bfloat16)
    preactivation = (preactivation + active_input_bias[:, None, :]).to(tl.bfloat16)

    physical_offset = tl.reshape(tl.permute(offset, (2, 0, 1)), (BLOCK_MODES, path_tokens))
    physical_active = tl.reshape(tl.permute(active, (2, 0, 1)), (BLOCK_MODES, path_tokens))
    active_grad_real = tl.load(
        grad_collapsed_real + physical_offset, mask=physical_active, other=0.0
    ).to(tl.bfloat16)
    active_grad_imag = tl.load(
        grad_collapsed_imag + physical_offset, mask=physical_active, other=0.0
    ).to(tl.bfloat16)

    pre8 = _slice8_perm(preactivation, BLOCK_MODES, path_tokens).to(tl.float32)
    sig8 = _fast_sigmoid(pre8)
    hid8 = (pre8 * sig8).to(tl.bfloat16)
    silu8 = sig8 * (1.0 + pre8 * (1.0 - sig8))

    output_offset = mode_vector[:, None] * (2 * packed_path_hidden) + h8[None, :]
    output_mask = (mode_vector[:, None] < modes) & (h8[None, :] < packed_path_hidden)
    tl.atomic_add(
        grad_path_output_weight + output_offset,
        tl.sum(active_grad_real.to(tl.float32)[:, :, None] * hid8.to(tl.float32), axis=1),
        mask=output_mask,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_path_output_weight + output_offset + packed_path_hidden,
        tl.sum(active_grad_imag.to(tl.float32)[:, :, None] * hid8.to(tl.float32), axis=1),
        mask=output_mask,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_path_output_bias + 2 * mode_vector,
        tl.sum(active_grad_real.to(tl.float32), axis=1),
        mask=mode_vector < modes,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_path_output_bias + 2 * mode_vector + 1,
        tl.sum(active_grad_imag.to(tl.float32), axis=1),
        mask=mode_vector < modes,
        sem="relaxed",
    )

    out8_off = mode_vector[:, None] * (2 * packed_path_hidden) + h8[None, :]
    out8_mask = (mode_vector[:, None] < modes) & (h8[None, :] < packed_path_hidden)
    wr8 = (
        tl.load(path_output_weight + out8_off, mask=out8_mask, other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    wi8 = (
        tl.load(path_output_weight + out8_off + packed_path_hidden, mask=out8_mask, other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    gh8 = (
        active_grad_real.to(tl.float32)[:, :, None] * wr8[:, None, :]
        + active_grad_imag.to(tl.float32)[:, :, None] * wi8[:, None, :]
    )
    gp8 = (gh8 * silu8).to(tl.bfloat16)
    grad_preactivation = _pad16_perm(gp8, BLOCK_MODES, path_tokens)

    path_input_weight_gradient = tl.dot(tl.permute(grad_preactivation, (0, 2, 1)), packed_path)
    path_input_weight_gradient_offset = (
        mode_vector[:, None, None] * packed_path_hidden + hidden_coordinate[None, :, None]
    ) * _PACKED_PATH_INPUTS_TL + path_coordinate[None, None, :]
    tl.atomic_add(
        grad_path_input_weight + path_input_weight_gradient_offset,
        path_input_weight_gradient,
        mask=(mode_vector[:, None, None] < modes)
        & (hidden_coordinate[None, :, None] < packed_path_hidden)
        & (path_coordinate[None, None, :] < _PACKED_PATH_INPUTS_TL),
        sem="relaxed",
    )
    tl.atomic_add(
        grad_path_input_bias + mode_vector[:, None] * packed_path_hidden + h8[None, :],
        tl.sum(gp8.to(tl.float32), axis=1),
        mask=(mode_vector[:, None] < modes) & (h8[None, :] < packed_path_hidden),
        sem="relaxed",
    )
    grad_packed_path = tl.dot(grad_preactivation, tl.permute(active_input_weight, (0, 2, 1))).to(
        tl.float32
    )

    gp_r0, gp_r1, gp_r2, gp_r3, gp_i0, gp_i1, gp_i2, gp_i3 = _unpack_once_shape(
        grad_packed_path, BLOCK_MODES, BLOCK_HEIGHT, BLOCK_LINES,
    )
    gp_r2_b, gp_r3_b = gp_r2.to(tl.bfloat16), gp_r3.to(tl.bfloat16)
    gp_i2_b, gp_i3_b = gp_i2.to(tl.bfloat16), gp_i3.to(tl.bfloat16)
    positive_adjoint = tl.associative_scan(
        (
            scan_ar,
            -scan_ai,
            tl.where(active, gp_r0 * pa, 0.0),
            tl.where(active, gp_i0 * pa, 0.0),
            tl.where(active, gp_r1 * pb, 0.0),
            tl.where(active, gp_i1 * pb, 0.0),
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
        reverse=True,
    )
    l0r, l0i, l1r, l1i = positive_adjoint[2], positive_adjoint[3], positive_adjoint[4], positive_adjoint[5]

    negative_adjoint = tl.associative_scan(
        (
            scan_ar,
            scan_ai,
            tl.where(active, gp_r2_b.to(tl.float32) * na, 0.0),
            tl.where(active, gp_i2_b.to(tl.float32) * na, 0.0),
            tl.where(active, gp_r3_b.to(tl.float32) * nb, 0.0),
            tl.where(active, gp_i3_b.to(tl.float32) * nb, 0.0),
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
    )
    l2r, l2i, l3r, l3i = negative_adjoint[2], negative_adjoint[3], negative_adjoint[4], negative_adjoint[5]

    plane_size = tl.cast(batch_size, tl.int64) * (height * width * modes)
    grad_a_imag_pointer = grad_source_real_a + plane_size
    grad_b_real_pointer = grad_source_real_a + 2 * plane_size
    grad_b_imag_pointer = grad_source_real_a + 3 * plane_size
    tl.store(grad_source_real_a + offset, gr * l0r + gi * l0i + gr * l2r - gi * l2i, mask=active)
    tl.store(grad_a_imag_pointer + offset, -gi * l0r + gr * l0i + gi * l2r + gr * l2i, mask=active)
    tl.store(grad_b_real_pointer + offset, gr * l1r + gi * l1i + gr * l3r - gi * l3i, mask=active)
    tl.store(grad_b_imag_pointer + offset, -gi * l1r + gr * l1i + gi * l3r + gr * l3i, mask=active)

    inverse_decay = libdevice.fast_dividef(1.0, decay_norm)
    positive_drive_ar, positive_drive_ai = gr * sar - gi * sai, gr * sai + gi * sar
    positive_drive_br, positive_drive_bi = gr * sbr - gi * sbi, gr * sbi + gi * sbr
    previous_0r = (ar * (par - positive_drive_ar) + ai * (pai - positive_drive_ai)) * inverse_decay
    previous_0i = (ar * (pai - positive_drive_ai) - ai * (par - positive_drive_ar)) * inverse_decay
    previous_1r = (ar * (pbr - positive_drive_br) + ai * (pbi - positive_drive_bi)) * inverse_decay
    previous_1i = (ar * (pbi - positive_drive_bi) - ai * (pbr - positive_drive_br)) * inverse_decay
    negative_drive_ar, negative_drive_ai = gr * sar + gi * sai, gr * sai - gi * sar
    negative_drive_br, negative_drive_bi = gr * sbr + gi * sbi, gr * sbi - gi * sbr
    previous_2r = (ar * (nar - negative_drive_ar) - ai * (nai - negative_drive_ai)) * inverse_decay
    previous_2i = (ai * (nar - negative_drive_ar) + ar * (nai - negative_drive_ai)) * inverse_decay
    previous_3r = (ar * (nbr - negative_drive_br) - ai * (nbi - negative_drive_bi)) * inverse_decay
    previous_3i = (ai * (nbr - negative_drive_br) + ar * (nbi - negative_drive_bi)) * inverse_decay
    grad_dr = tl.sum(
        tl.sum(
            tl.where(
                active,
                l0r * previous_0r
                + l0i * previous_0i
                + l1r * previous_1r
                + l1i * previous_1i
                + l2r * previous_2r
                + l2i * previous_2i
                + l3r * previous_3r
                + l3i * previous_3i,
                0.0,
            ),
            axis=0,
        ),
        axis=0,
    )
    grad_di = tl.sum(
        tl.sum(
            tl.where(
                active,
                -l0r * previous_0i
                + l0i * previous_0r
                - l1r * previous_1i
                + l1i * previous_1r
                + l2r * previous_2i
                - l2i * previous_2r
                + l3r * previous_3i
                - l3i * previous_3r,
                0.0,
            ),
            axis=0,
        ),
        axis=0,
    )
    grad_gr = tl.sum(
        tl.sum(
            tl.where(
                active,
                l0r * sar
                + l0i * sai
                + l1r * sbr
                + l1i * sbi
                + l2r * sar
                + l2i * sai
                + l3r * sbr
                + l3i * sbi,
                0.0,
            ),
            axis=0,
        ),
        axis=0,
    )
    grad_gi = tl.sum(
        tl.sum(
            tl.where(
                active,
                -l0r * sai
                + l0i * sar
                - l1r * sbi
                + l1i * sbr
                + l2r * sai
                - l2i * sar
                + l3r * sbi
                - l3i * sbr,
                0.0,
            ),
            axis=0,
        ),
        axis=0,
    )
    valid_coefficient_mode = mode_vector < modes
    tl.atomic_add(grad_decay_real + mode_vector, grad_dr, mask=valid_coefficient_mode, sem="relaxed")
    tl.atomic_add(grad_decay_imag + mode_vector, grad_di, mask=valid_coefficient_mode, sem="relaxed")
    tl.atomic_add(grad_gamma_real + mode_vector, grad_gr, mask=valid_coefficient_mode, sem="relaxed")
    tl.atomic_add(grad_gamma_imag + mode_vector, grad_gi, mask=valid_coefficient_mode, sem="relaxed")


_assoc_fast_kernel = _product_scan_coarse4_associative_backward_kernel


def _bind_vjp(helper, template=_original_kernel):
    # A private namespace changes only the output-VJP dependency. Preserve all
    # caller bytecode, pointer arguments, scan logic, and original global objects.
    fn = template.fn
    impl = FunctionType(
        fn.__code__,
        {**fn.__globals__, "_output_weight_bias_vjp": helper},
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    impl.__annotations__ = fn.__annotations__.copy()
    impl.__kwdefaults__ = fn.__kwdefaults__
    impl.__module__ = __name__
    return triton.jit(impl)


def _bind_transpose():
    # Inductor emits JIT dependency .src without rewriting the caller's global
    # aliases. This shim preserves the canonical call name and emits a uniquely
    # named helper in generated code. Triton inlines the forwarding call.
    @triton.jit
    def _output_weight_bias_vjp(
        grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
    ):
        _effective_output_vjp_transpose(
            grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
        )

    return _bind_vjp(_output_weight_bias_vjp)


def _bind_rows():
    @triton.jit
    def _output_weight_bias_vjp(
        grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
    ):
        _effective_output_vjp_rows(
            grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
        )

    return _bind_vjp(_output_weight_bias_vjp, template=_rows_template_kernel)


_transpose_kernel = _bind_transpose()
_rows_kernel = _bind_rows()
_VARIANTS = {1: _transpose_kernel, 2: _rows_kernel}


@cache
def vjp_revision(parent_revision):
    digest = hashlib.sha256(dot_revision(parent_revision).encode())
    for variant, kernel in sorted(_VARIANTS.items()):
        digest.update(str(variant).encode())
        digest.update(kernel.cache_key.encode())
    digest.update(_coarse_fast_kernel.cache_key.encode())
    digest.update(_assoc_fast_kernel.cache_key.encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _match_vjp_map_for(vjp_map, dtype, compile_profile, target_rev, batch):
    if not isinstance(vjp_map, dict):
        return False, False
    for key, payload in vjp_map.items():
        try:
            fields = json.loads(key)
            shape = fields["shape"]
        except (KeyError, TypeError, ValueError):
            continue
        if (
            fields.get("execution_regime") == "compiled"
            and fields.get("dtype") == dtype
            and fields.get("compile_profile") == compile_profile
            and fields.get("kernel_revision") == target_rev
            and shape.get("batch") == batch
            and (
                shape.get("height"),
                shape.get("width"),
                shape.get("operation"),
                shape.get("coarse"),
            )
            == (56, 56, 1, 0)
        ):
            active = isinstance(payload, dict) and payload.get("blocks", {}).get("VARIANT") == 2
            return True, active
    return False, False


def _rows_vjp_active_for(batch, dtype, compile_profile):
    if dtype != "bfloat16":
        return False
    target_rev = vjp_revision(step_revision(1))
    wl = _WORKLOAD_RECORDS.get()
    if isinstance(wl, dict) and isinstance(wl.get(VJP_NAME), dict):
        found, active = _match_vjp_map_for(wl[VJP_NAME], dtype, compile_profile, target_rev, batch)
        if found:
            return active
    dev = device_key()
    for path in (None, BUNDLE):
        vjp_map = _read_store(path).get(dev, {}).get(VJP_NAME, {})
        found, active = _match_vjp_map_for(vjp_map, dtype, compile_profile, target_rev, batch)
        if found:
            return active
    return False


def _rows_vjp_active(scope):
    if scope.execution_regime != "compiled" or scope.dtype != "bfloat16":
        return False
    return _rows_vjp_active_for(
        dict(scope.shape_items).get("batch"), scope.dtype, scope.compile_profile
    )


def _match_layout_map(layout_map, batch, height, width, modes, dtype, compile_profile, layout_rev):
    if not isinstance(layout_map, dict):
        return False, False
    for key, payload in layout_map.items():
        try:
            fields = json.loads(key)
            shape = fields["shape"]
        except (KeyError, TypeError, ValueError):
            continue
        if (
            fields.get("execution_regime") == "compiled"
            and fields.get("dtype") == dtype
            and fields.get("compile_profile") == compile_profile
            and fields.get("kernel_revision") == layout_rev
            and shape.get("batch") == batch
            and shape.get("height") == height
            and shape.get("width") == width
            and shape.get("modes") == modes
        ):
            blocks = payload.get("blocks", {}) if isinstance(payload, dict) else {}
            active = blocks.get("LINES_PER_GROUP", 0) > 0 and blocks.get("COMPACT_CHANNELS") == 3
            return True, active
    return False, False


def _layout_grouped_compact_active(batch, height, width, modes, dtype, compile_profile):
    layout_rev = _layout_revision()
    wl = _WORKLOAD_RECORDS.get()
    if isinstance(wl, dict) and isinstance(wl.get(LAYOUT_NAME), dict):
        found, active = _match_layout_map(
            wl[LAYOUT_NAME], batch, height, width, modes, dtype, compile_profile, layout_rev
        )
        if found:
            return active
    dev = device_key()
    for path in (None, BUNDLED_LAYOUTS):
        layout_map = _read_store(path).get(dev, {}).get(LAYOUT_NAME, {})
        found, active = _match_layout_map(
            layout_map, batch, height, width, modes, dtype, compile_profile, layout_rev
        )
        if found:
            return active
    return False


def select_effective_vjp(scope, base):
    selected = select_output_dot(scope, base)
    shape = dict(scope.shape_items)
    if (
        scope.execution_regime == "compiled"
        and shape.get("operation") == VERTICAL_BACKWARD
        and shape.get("height") == 56
        and shape.get("width") == 56
        and shape.get("coarse") == 1
        and selected is _segmented_backward_kernel
        and _rows_vjp_active(scope)
    ):
        return _coarse_fast_kernel
    if (
        scope.execution_regime != "compiled"
        or shape.get("operation") != VERTICAL_BACKWARD
        or shape.get("height") != 56
        or shape.get("width") != 56
        or shape.get("coarse") != 0
    ):
        return selected
    exact = replace(scope, kernel_revision=vjp_revision(scope.kernel_revision))
    marker = validated_geometry(VJP_NAME, exact, BUNDLE)
    if selected is not _original_kernel or marker is None:
        return selected
    return _VARIANTS.get(marker.blocks.get("VARIANT"), selected)


def select_effective_layout_backward(scope, base):
    shape = dict(scope.shape_items)
    compiled = (
        scope.execution_regime == "compiled"
        or torch.compiler.is_compiling()
        or get_proxy_mode() is not None
    )
    compile_profile = scope.compile_profile or (
        os.environ.get("LNET_COMPILE_PROFILE") if compiled else None
    )
    if (
        base is not None
        and compiled
        and scope.dtype == "bfloat16"
        and base.blocks.get("BLOCK_LINES") == 1
        and shape.get("height") == shape.get("width")
        and shape.get("height") in (14, 28)
        and shape.get("modes") == 160
        and shape.get("gain_kind") == 0
        and bool(shape.get("full_coarse"))
        and bool(shape.get("path_collapse"))
        and bool(shape.get("batch_partial_gradients"))
        and (
            os.environ.get("LNET_R7_R5_ASSOC") == "1"
            or _rows_vjp_active_for(shape.get("batch"), scope.dtype, compile_profile)
        )
        and _layout_grouped_compact_active(
            shape.get("batch"),
            shape.get("height"),
            shape.get("width"),
            shape.get("modes"),
            scope.dtype,
            compile_profile,
        )
    ):
        return _assoc_fast_kernel
    return None


def vjp_graph_pass(previous):
    register_kernel_specializer(PROFILE_NAME, select_effective_vjp)
    register_kernel_specializer(ASSOC_BACKWARD_LAUNCH_NAME, select_effective_layout_backward)
    signature = geometry_store_signature(VJP_NAME, BUNDLE, omit_empty=True)
    return _LayoutGraphPass(vjp_revision(signature), previous) if signature else previous
