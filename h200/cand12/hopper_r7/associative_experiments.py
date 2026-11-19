"""Associative vertical forward and coarse backward r6 candidates."""
import triton
import triton.language as tl

from lnet.hopper_r7_effective_output_vjp import _fast_sigmoid
from lnet.effective_structural_scan import _pack_once8
from lnet.hopper_r7_effective_output_vjp import _pad16_perm, _slice8_perm
from lnet.triton_product_scan_kernels import (
    _PATHS_TL, _PATH_DOT_INPUTS_TL, _PATH_DOT_OUTPUTS_TL,
    _PACKED_PATH_INPUTS_TL, _PACKED_PATH_OUTPUTS_TL,
    _coordinate_offset, _store_complex_coordinates, _coarse_path_endpoint,
    _compose_complex_pair, _collapsed_path_register_mlp,
)
from lnet.r6_segment import _segment_pair


@triton.jit
def associative_forward(  # noqa: C901, PLR0912
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
    PACK2: tl.constexpr = False,
    FAST_SIGMOID: tl.constexpr = False,
    SINGLE_LOAD: tl.constexpr = False,
    SEGMENT: tl.constexpr = False,
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
    if SEGMENT:
        pos = _segment_pair(
            tl.broadcast_to(ar, (1, BLOCK_LINES, BLOCK_MODES)),
            tl.broadcast_to(ai, (1, BLOCK_LINES, BLOCK_MODES)),
            gr * sar - gi * sai, gr * sai + gi * sar,
            gr * sbr - gi * sbi, gr * sbi + gi * sbr, False, True,
        )
        positive = (ar, ai, pos[0], pos[1], pos[2], pos[3])
    else:
        positive = tl.associative_scan(
            (
                tl.where(active, ar, 1.0),
                tl.where(active, ai, 0.0),
                gr * sar - gi * sai,
                gr * sai + gi * sar,
                gr * sbr - gi * sbi,
                gr * sbi + gi * sbr,
            ),
            axis=0,
            combine_fn=_compose_complex_pair,
        )
    if gain_kind == 1:
        inverse_pa = tl.load(global_inverse_gain + mode, mask=valid_mode, other=0.0)
        inverse_pb = inverse_pa
    else:
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
    direct_path_collapse: tl.constexpr = (
        False
    )
    early_bf16_collapse: tl.constexpr = (
        COLLAPSE_PATHS and not COARSE_PATHS and not EMIT_COARSE and not EMIT_DESCRIPTOR
    )
    if direct_path_collapse:
        if COARSE_PATHS:
            r0 = _coarse_path_endpoint(r0, 0).to(tl.bfloat16)
            i0 = _coarse_path_endpoint(i0, 0).to(tl.bfloat16)
            r1 = _coarse_path_endpoint(r1, 1).to(tl.bfloat16)
            i1 = _coarse_path_endpoint(i1, 1).to(tl.bfloat16)
        else:
            r0 = r0.to(tl.bfloat16)
            i0 = i0.to(tl.bfloat16)
            r1 = r1.to(tl.bfloat16)
            i1 = i1.to(tl.bfloat16)
    elif early_bf16_collapse:
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
    if SEGMENT:
        neg = _segment_pair(
            tl.broadcast_to(ar, (1, BLOCK_LINES, BLOCK_MODES)),
            tl.broadcast_to(-ai, (1, BLOCK_LINES, BLOCK_MODES)),
            gr * sar_n + gi * sai_n, gr * sai_n - gi * sar_n,
            gr * sbr_n + gi * sbi_n, gr * sbi_n - gi * sbr_n, True, True,
        )
        negative = (ar, ai, neg[0], neg[1], neg[2], neg[3])
    else:
        negative = tl.associative_scan(
            (
                tl.where(active, ar, 1.0),
                tl.where(active, -ai, 0.0),
                gr * sar_n + gi * sai_n,
                gr * sai_n - gi * sar_n,
                gr * sbr_n + gi * sbi_n,
                gr * sbi_n - gi * sbr_n,
            ),
            axis=0,
            combine_fn=_compose_complex_pair,
            reverse=True,
        )
    if gain_kind == 1:
        inverse_na = tl.load(global_inverse_gain + mode, mask=valid_mode, other=0.0)
        inverse_nb = inverse_na
    else:
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

    if COLLAPSE_PATHS:
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
        if direct_path_collapse:
            if COARSE_PATHS:
                r2 = _coarse_path_endpoint(r2, 2).to(tl.bfloat16)
                i2 = _coarse_path_endpoint(i2, 2).to(tl.bfloat16)
                r3 = _coarse_path_endpoint(r3, 3).to(tl.bfloat16)
                i3 = _coarse_path_endpoint(i3, 3).to(tl.bfloat16)
            collapsed_real_value, collapsed_imag_value = _collapsed_path_register_mlp(
                r0,
                r1,
                r2,
                r3,
                i0,
                i1,
                i2,
                i3,
                path_input_weight,
                path_input_bias,
                path_output_weight,
                path_output_bias,
                mode_vector,
                modes,
                packed_path_hidden,
                path_height,
                path_lines,
                BLOCK_MODES,
            )
            if PLANAR_PATH_OUTPUT:
                plane_size = tl.cast(batch_size, tl.int64) * (
                    (height // 2 if COARSE_PATHS else height)
                    * (width // 2 if COARSE_PATHS else width)
                    * modes
                )
                tl.store(collapsed_real + path_offset, collapsed_real_value, mask=path_active)
                tl.store(
                    collapsed_real + plane_size + path_offset,
                    collapsed_imag_value,
                    mask=path_active,
                )
            else:
                _store_complex_coordinates(
                    collapsed_real,
                    collapsed_imag,
                    _coordinate_offset(path_offset, mode, PACKED_OUTPUT),
                    collapsed_real_value,
                    collapsed_imag_value,
                    path_active,
                    modes,
                    PACKED_OUTPUT,
                )
        else:
            path_coordinate = tl.arange(0, _PATH_DOT_INPUTS_TL)
            if PACK2:
                paths8 = _pack_once8(
                    (_coarse_path_endpoint(r0, 0) if COARSE_PATHS else r0).to(tl.bfloat16),
                    (_coarse_path_endpoint(r1, 1) if COARSE_PATHS else r1).to(tl.bfloat16),
                    (_coarse_path_endpoint(r2, 2) if COARSE_PATHS else r2).to(tl.bfloat16),
                    (_coarse_path_endpoint(r3, 3) if COARSE_PATHS else r3).to(tl.bfloat16),
                    (_coarse_path_endpoint(i0, 0) if COARSE_PATHS else i0).to(tl.bfloat16),
                    (_coarse_path_endpoint(i1, 1) if COARSE_PATHS else i1).to(tl.bfloat16),
                    (_coarse_path_endpoint(i2, 2) if COARSE_PATHS else i2).to(tl.bfloat16),
                    (_coarse_path_endpoint(i3, 3) if COARSE_PATHS else i3).to(tl.bfloat16),
                    BLOCK_MODES, path_tokens,
                )
                packed_path = _pad16_perm(paths8, BLOCK_MODES, path_tokens)
            else:
                r0_flat = tl.reshape(
                    tl.permute(_coarse_path_endpoint(r0, 0) if COARSE_PATHS else r0, (2, 0, 1)),
                    (BLOCK_MODES, path_tokens),
                )
                r1_flat = tl.reshape(
                    tl.permute(_coarse_path_endpoint(r1, 1) if COARSE_PATHS else r1, (2, 0, 1)),
                    (BLOCK_MODES, path_tokens),
                )
                r2_flat = tl.reshape(
                    tl.permute(_coarse_path_endpoint(r2, 2) if COARSE_PATHS else r2, (2, 0, 1)),
                    (BLOCK_MODES, path_tokens),
                )
                r3_flat = tl.reshape(
                    tl.permute(_coarse_path_endpoint(r3, 3) if COARSE_PATHS else r3, (2, 0, 1)),
                    (BLOCK_MODES, path_tokens),
                )
                i0_flat = tl.reshape(
                    tl.permute(_coarse_path_endpoint(i0, 0) if COARSE_PATHS else i0, (2, 0, 1)),
                    (BLOCK_MODES, path_tokens),
                )
                i1_flat = tl.reshape(
                    tl.permute(_coarse_path_endpoint(i1, 1) if COARSE_PATHS else i1, (2, 0, 1)),
                    (BLOCK_MODES, path_tokens),
                )
                i2_flat = tl.reshape(
                    tl.permute(_coarse_path_endpoint(i2, 2) if COARSE_PATHS else i2, (2, 0, 1)),
                    (BLOCK_MODES, path_tokens),
                )
                i3_flat = tl.reshape(
                    tl.permute(_coarse_path_endpoint(i3, 3) if COARSE_PATHS else i3, (2, 0, 1)),
                    (BLOCK_MODES, path_tokens),
                )
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
                if COARSE_PATHS:
                    packed_path = tl.where(
                        path_coordinate[None, None, :] < _PACKED_PATH_INPUTS_TL, packed_path, 0.0,
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
                mask=(
                    (mode_vector[:, None] < modes)
                    & (hidden_coordinate[None, :] < packed_path_hidden)
                ),
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
            elif not PACK2:
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
            else:
                hidden_float = _slice8_perm(preactivation, BLOCK_MODES, path_tokens).to(tl.float32)
                sigmoid = _fast_sigmoid(hidden_float) if FAST_SIGMOID else tl.sigmoid(hidden_float)
                hidden8 = (hidden_float * sigmoid).to(tl.bfloat16).to(tl.float32)
                lane = tl.arange(0, 8)[None, :]
                base = mode_vector[:, None] * (2 * packed_path_hidden) + lane
                w_real = tl.load(path_output_weight + base, mode_vector[:, None] < modes, 0)
                w_imag = tl.load(path_output_weight + base + packed_path_hidden,
                                 mode_vector[:, None] < modes, 0)
                b_real = tl.load(path_output_bias + 2 * mode_vector, mode_vector < modes, 0)
                b_imag = tl.load(path_output_bias + 2 * mode_vector + 1, mode_vector < modes, 0)
                collapsed_real_value = (tl.sum(hidden8 * w_real.to(tl.bfloat16).to(tl.float32)[:, None, :], 2)
                                        + b_real.to(tl.bfloat16).to(tl.float32)[:, None]).to(tl.bfloat16)
                collapsed_imag_value = (tl.sum(hidden8 * w_imag.to(tl.bfloat16).to(tl.float32)[:, None, :], 2)
                                        + b_imag.to(tl.bfloat16).to(tl.float32)[:, None]).to(tl.bfloat16)
            physical_offset = tl.reshape(
                tl.permute(path_offset, (2, 0, 1)), (BLOCK_MODES, path_tokens),
            )
            physical_active = tl.reshape(
                tl.permute(path_active, (2, 0, 1)), (BLOCK_MODES, path_tokens),
            )
            if not PACK2 or PATH_SWIGLU:
                collapsed_real_value = tl.sum(
                    tl.where(output_coordinate[None, None, :] == 0, path_output, 0.0),
                    axis=2,
                )
                collapsed_imag_value = tl.sum(
                    tl.where(output_coordinate[None, None, :] == 1, path_output, 0.0),
                    axis=2,
                )
            if PLANAR_PATH_OUTPUT:
                plane_size = tl.cast(batch_size, tl.int64) * (
                    (height // 2 if COARSE_PATHS else height)
                    * (width // 2 if COARSE_PATHS else width)
                    * modes
                )
                tl.store(
                    collapsed_real + physical_offset, collapsed_real_value, mask=physical_active,
                )
                tl.store(
                    collapsed_real + plane_size + physical_offset,
                    collapsed_imag_value,
                    mask=physical_active,
                )
            else:
                _store_complex_coordinates(
                    collapsed_real, collapsed_imag,
                    _coordinate_offset(physical_offset, mode_vector[:, None], PACKED_OUTPUT),
                    collapsed_real_value, collapsed_imag_value, physical_active,
                    modes, PACKED_OUTPUT,
                )

    if EMIT_COARSE:
        cell = (batch * (height // 2) + y // 2) * (width // 2) + x // 2
        if FULL_COARSE:
            parity_x = x & 1
            parity_y = y & 1
            local0 = 2 * parity_x + parity_y
            local1 = 2 * (1 - parity_x) + parity_y
            local2 = 2 * parity_x + (1 - parity_y)
            local3 = 2 * (1 - parity_x) + (1 - parity_y)
            base0 = ((cell * 4) * 4 + local0) * modes + mode
            base1 = ((cell * 4 + 1) * 4 + local1) * modes + mode
            base2 = ((cell * 4 + 2) * 4 + local2) * modes + mode
            base3 = ((cell * 4 + 3) * 4 + local3) * modes + mode
            _store_complex_coordinates(
                coarse_real,
                coarse_imag,
                _coordinate_offset(base0, mode, PACKED_OUTPUT),
                r0,
                i0,
                active,
                modes,
                PACKED_OUTPUT,
            )
            _store_complex_coordinates(
                coarse_real,
                coarse_imag,
                _coordinate_offset(base1, mode, PACKED_OUTPUT),
                r1,
                i1,
                active,
                modes,
                PACKED_OUTPUT,
            )
            _store_complex_coordinates(
                coarse_real,
                coarse_imag,
                _coordinate_offset(base2, mode, PACKED_OUTPUT),
                r2,
                i2,
                active,
                modes,
                PACKED_OUTPUT,
            )
            _store_complex_coordinates(
                coarse_real,
                coarse_imag,
                _coordinate_offset(base3, mode, PACKED_OUTPUT),
                r3,
                i3,
                active,
                modes,
                PACKED_OUTPUT,
            )
        else:
            coarse_base = (cell * 4) * modes + mode
            positive_endpoint = active & ((y & 1) == 1)
            negative_endpoint = active & ((y & 1) == 0)
            _store_complex_coordinates(
                coarse_real,
                coarse_imag,
                _coordinate_offset(coarse_base, mode, PACKED_OUTPUT),
                r0,
                i0,
                positive_endpoint & ((x & 1) == 1),
                modes,
                PACKED_OUTPUT,
            )
            _store_complex_coordinates(
                coarse_real,
                coarse_imag,
                _coordinate_offset(coarse_base + modes, mode, PACKED_OUTPUT),
                r1,
                i1,
                positive_endpoint & ((x & 1) == 0),
                modes,
                PACKED_OUTPUT,
            )
            _store_complex_coordinates(
                coarse_real,
                coarse_imag,
                _coordinate_offset(coarse_base + 2 * modes, mode, PACKED_OUTPUT),
                r2,
                i2,
                negative_endpoint & ((x & 1) == 1),
                modes,
                PACKED_OUTPUT,
            )
            _store_complex_coordinates(
                coarse_real,
                coarse_imag,
                _coordinate_offset(coarse_base + 3 * modes, mode, PACKED_OUTPUT),
                r3,
                i3,
                negative_endpoint & ((x & 1) == 0),
                modes,
                PACKED_OUTPUT,
            )

    if EMIT_DESCRIPTOR:
        raw0 = tl.sum(tl.sum(r0 * r0 + i0 * i0, axis=0), axis=0)
        raw1 = tl.sum(tl.sum(r1 * r1 + i1 * i1, axis=0), axis=0)
        raw2 = tl.sum(tl.sum(r2 * r2 + i2 * i2, axis=0), axis=0)
        raw3 = tl.sum(tl.sum(r3 * r3 + i3 * i3, axis=0), axis=0)
        valid_descriptor_mode = mode_vector < modes
        if DETERMINISTIC_DESCRIPTOR:
            # One unique row per line group; unused rows stay zero. A separate
            # fixed-order reduction avoids inter-CTA floating-point atomic races.
            descriptor_mode = (batch * width + x_group) * (4 * modes) + mode_vector
            tl.store(
                descriptor_energy + descriptor_mode,
                raw0,
                valid_descriptor_mode,
            )
            tl.store(
                descriptor_energy + descriptor_mode + modes,
                raw1,
                valid_descriptor_mode,
            )
            tl.store(
                descriptor_energy + descriptor_mode + 2 * modes,
                raw2,
                valid_descriptor_mode,
            )
            tl.store(
                descriptor_energy + descriptor_mode + 3 * modes,
                raw3,
                valid_descriptor_mode,
            )
        else:
            descriptor_mode = batch * (4 * modes) + mode_vector
            tl.atomic_add(
                descriptor_energy + descriptor_mode,
                raw0,
                mask=valid_descriptor_mode,
            )
            tl.atomic_add(
                descriptor_energy + descriptor_mode + modes,
                raw1,
                mask=valid_descriptor_mode,
            )
            tl.atomic_add(
                descriptor_energy + descriptor_mode + 2 * modes,
                raw2,
                mask=valid_descriptor_mode,
            )
            tl.atomic_add(
                descriptor_energy + descriptor_mode + 3 * modes,
                raw3,
                mask=valid_descriptor_mode,
            )
