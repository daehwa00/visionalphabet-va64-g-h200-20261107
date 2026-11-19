"""Four-token segmented coarse forward with 8-lane BF16 path packing."""

import triton
import triton.language as tl

from .r6_segment import _segment_pair
from .effective_output_vjp import _pack_once8, _pad16_perm
from .triton_product_scan_kernels import (
    _PACKED_PATH_INPUTS_TL,
    _PACKED_PATH_OUTPUTS_TL,
    _PATH_DOT_INPUTS_TL,
    _PATH_DOT_OUTPUTS_TL,
    _coarse_path_endpoint,
    _compose_complex_pair,
)


@triton.jit
def r10_segmented_coarse_forward(  # noqa: C901, PLR0912
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
    coarse_real,
    coarse_imag,
    path_input_weight,
    path_input_bias,
    path_output_weight,
    path_output_bias,
    collapsed_real,
    collapsed_imag,
    descriptor_energy,
    height: tl.constexpr,
    width: int,
    line_count: int,
    modes: int,
    packed_path_hidden: int,
    epsilon: tl.constexpr,
    gain_kind: tl.constexpr,
    EMIT_COARSE: tl.constexpr,
    FULL_COARSE: tl.constexpr,
    COLLAPSE_PATHS: tl.constexpr,
    PATH_SWIGLU: tl.constexpr,
    SWIGLU_HIDDEN: tl.constexpr,
    EMIT_DESCRIPTOR: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_LINES: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    BLOCK_PATH_HIDDEN: tl.constexpr,
    BLOCK_SWIGLU_VALUE: tl.constexpr,
    SINGLE_LOAD: tl.constexpr = False,
    PACKED_OUTPUT: tl.constexpr = 0,
    COARSE_PATHS: tl.constexpr = 0,
    PLANAR_PATH_OUTPUT: tl.constexpr = 0,
    DETERMINISTIC_DESCRIPTOR: tl.constexpr = 0,
) -> None:
    batch_size = line_count // width
    scan_width = width // 2 if COARSE_PATHS else width
    line_group = tl.program_id(0)
    if BLOCK_LINES == 1:
        batch = line_group // scan_width
        x_group = line_group - batch * scan_width
    else:
        # Grouped programs are interleaved across batches.  Adjacent CTAs then
        # update different descriptor rows instead of contending on one batch.
        x_group = line_group // batch_size
        batch = line_group - x_group * batch_size
    x = x_group * BLOCK_LINES + tl.arange(0, BLOCK_LINES)[None, :, None]
    y = tl.arange(0, BLOCK_HEIGHT)[:, None, None]
    mode_vector = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    mode = mode_vector[None, None, :]
    valid_line = x < scan_width
    active = valid_line & (y < height) & (mode < modes)
    valid_mode = valid_line & (mode < modes)
    # Coarse endpoints need positive-x streams only at odd columns and
    # negative-x streams only at even columns; the other scans are unused.
    source_x_a = 2 * x + 1 if COARSE_PATHS else x
    source_x_b = 2 * x if COARSE_PATHS else x
    offset = ((batch * height + y) * width + source_x_a) * modes + mode
    offset_b = ((batch * height + y) * width + source_x_b) * modes + mode

    ar = tl.load(decay_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
    sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
    sbr = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
    sbi = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
    pos = _segment_pair(
        tl.broadcast_to(ar, (1, BLOCK_LINES, BLOCK_MODES)),
        tl.broadcast_to(ai, (1, BLOCK_LINES, BLOCK_MODES)),
        gr * sar - gi * sai, gr * sai + gi * sar,
        gr * sbr - gi * sbi, gr * sbi + gi * sbr, False, True,
    )
    positive = (ar, ai, pos[0], pos[1], pos[2], pos[3])
    x_offset = source_x_a * modes + mode
    x_offset_b = source_x_b * modes + mode
    y_offset = y * modes + mode
    vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
    vx_negative = tl.load(
        variance_x + width * modes + x_offset_b,
        mask=valid_mode,
        other=0.0,
    )
    vy_positive = tl.load(variance_y + y_offset, mask=active, other=0.0)
    inverse_pa = tl.rsqrt(tl.maximum(vx_positive * vy_positive, epsilon))
    inverse_pb = tl.rsqrt(tl.maximum(vx_negative * vy_positive, epsilon))
    # Zero padded scan lanes once at the normalization boundary.  Keeping the
    # padded recurrence state alive and masking every downstream consumer costs
    # registers, while its zero variance also magnifies the state by rsqrt(eps).
    r0 = tl.where(active, positive[2] * inverse_pa, 0.0)
    i0 = tl.where(active, positive[3] * inverse_pa, 0.0)
    r1 = tl.where(active, positive[4] * inverse_pb, 0.0)
    i1 = tl.where(active, positive[5] * inverse_pb, 0.0)
    if not COARSE_PATHS:
        r0 = r0.to(tl.bfloat16)
        i0 = i0.to(tl.bfloat16)
        r1 = r1.to(tl.bfloat16)
        i1 = i1.to(tl.bfloat16)

    if SINGLE_LOAD:
        sar_n, sai_n, sbr_n, sbi_n = sar, sai, sbr, sbi
    else:
        sar_n = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
        sai_n = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
        sbr_n = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
        sbi_n = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
    neg = _segment_pair(
        tl.broadcast_to(ar, (1, BLOCK_LINES, BLOCK_MODES)),
        tl.broadcast_to(-ai, (1, BLOCK_LINES, BLOCK_MODES)),
        gr * sar_n + gi * sai_n, gr * sai_n - gi * sar_n,
        gr * sbr_n + gi * sbi_n, gr * sbi_n - gi * sbr_n, True, True,
    )
    negative = (ar, ai, neg[0], neg[1], neg[2], neg[3])
    x_offset_n = source_x_a * modes + mode
    x_offset_bn = source_x_b * modes + mode
    y_offset_n = y * modes + mode
    vx_positive_n = tl.load(variance_x + x_offset_n, mask=valid_mode, other=0.0)
    vx_negative_n = tl.load(
        variance_x + width * modes + x_offset_bn,
        mask=valid_mode,
        other=0.0,
    )
    vy_negative = tl.load(
        variance_y + height * modes + y_offset_n,
        mask=active,
        other=0.0,
    )
    inverse_na = tl.rsqrt(tl.maximum(vx_positive_n * vy_negative, epsilon))
    inverse_nb = tl.rsqrt(tl.maximum(vx_negative_n * vy_negative, epsilon))
    r2 = tl.where(active, negative[2] * inverse_na, 0.0)
    i2 = tl.where(active, negative[3] * inverse_na, 0.0)
    r3 = tl.where(active, negative[4] * inverse_nb, 0.0)
    i3 = tl.where(active, negative[5] * inverse_nb, 0.0)

    path_height: tl.constexpr = BLOCK_HEIGHT // 2 if COARSE_PATHS else BLOCK_HEIGHT
    path_lines: tl.constexpr = BLOCK_LINES
    path_tokens: tl.constexpr = path_height * path_lines
    if COARSE_PATHS:
        cell_y = tl.arange(0, path_height)[:, None, None]
        cell_x = x_group * path_lines + tl.arange(0, path_lines)[None, :, None]
        path_offset = ((batch * (height // 2) + cell_y) * (width // 2) + cell_x) * modes + mode
        path_active = (cell_y < height // 2) & (cell_x < width // 2) & (mode < modes)
    else:
        path_offset, path_active = offset, active
    path_coordinate = tl.arange(0, _PATH_DOT_INPUTS_TL)
    paths8 = _pack_once8(
        (_coarse_path_endpoint(r0, 0) if COARSE_PATHS else r0).to(tl.bfloat16),
        (_coarse_path_endpoint(r1, 1) if COARSE_PATHS else r1).to(tl.bfloat16),
        (_coarse_path_endpoint(r2, 2) if COARSE_PATHS else r2).to(tl.bfloat16),
        (_coarse_path_endpoint(r3, 3) if COARSE_PATHS else r3).to(tl.bfloat16),
        (_coarse_path_endpoint(i0, 0) if COARSE_PATHS else i0).to(tl.bfloat16),
        (_coarse_path_endpoint(i1, 1) if COARSE_PATHS else i1).to(tl.bfloat16),
        (_coarse_path_endpoint(i2, 2) if COARSE_PATHS else i2).to(tl.bfloat16),
        (_coarse_path_endpoint(i3, 3) if COARSE_PATHS else i3).to(tl.bfloat16),
        BLOCK_MODES,
        path_tokens,
    )
    packed_path = _pad16_perm(paths8, BLOCK_MODES, path_tokens)
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
        path_input_bias + mode_vector[:, None] * packed_path_hidden + hidden_coordinate[None, :],
        mask=((mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden)),
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
    hidden_float = preactivation.to(tl.float32)
    hidden = (hidden_float * tl.sigmoid(hidden_float)).to(tl.bfloat16)
    output_weight_offset = (
        mode_vector[:, None, None] * _PACKED_PATH_OUTPUTS_TL + output_coordinate[None, None, :]
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
    physical_offset = tl.reshape(
        tl.permute(path_offset, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    physical_active = tl.reshape(
        tl.permute(path_active, (2, 0, 1)),
        (BLOCK_MODES, path_tokens),
    )
    collapsed_real_value = tl.sum(
        tl.where(output_coordinate[None, None, :] == 0, path_output, 0.0),
        axis=2,
    )
    collapsed_imag_value = tl.sum(
        tl.where(output_coordinate[None, None, :] == 1, path_output, 0.0),
        axis=2,
    )

    tl.store(collapsed_real + physical_offset, collapsed_real_value, mask=physical_active)
    tl.store(collapsed_imag + physical_offset, collapsed_imag_value, mask=physical_active)
