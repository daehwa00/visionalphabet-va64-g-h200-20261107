"""Coarse associative vertical backward r6 kernel."""
import triton
import triton.language as tl

from .effective_output_vjp import _fast_sigmoid, _pack_once8, _unpack_once_shape
from .effective_output_vjp import _pad16_perm, _slice8_perm
from .triton_product_scan_kernels import (
    _PATHS_TL, _PATH_DOT_INPUTS_TL, _PATH_DOT_OUTPUTS_TL,
    _PACKED_PATH_INPUTS_TL, _PACKED_PATH_OUTPUTS_TL,
    _coordinate_offset, _coarse_path_endpoint,
    _compose_complex_pair, _scale_by_gain, _unpack_path_gradient,
)


@triton.jit
def _restore_parity(value, ODD: tl.constexpr):
    zeros = tl.zeros_like(value)
    if ODD:
        packed = tl.join(zeros, value)
    else:
        packed = tl.join(value, zeros)
    return tl.reshape(tl.permute(packed, (0, 3, 1, 2)),
                      (2 * value.shape[0], value.shape[1], value.shape[2]))


@triton.jit
def coarse_associative_backward(  # noqa: C901, PLR0912
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
) -> None:
    batch_size = line_count // width
    scan_width = width // 2 if COARSE_PATHS else width
    if LINES_PER_GROUP:
        # Neighboring CTAs visit all mode tiles within a group of spatial lines.
        tile = tl.program_id(0)
        mode_group = tile // LINES_PER_GROUP
        line_group = tl.program_id(1) * LINES_PER_GROUP + tile % LINES_PER_GROUP
    else:
        mode_group = tl.program_id(1)
        line_group = tl.program_id(0)
    if BLOCK_LINES == 1:
        batch = line_group // scan_width
        x_group = line_group - batch * scan_width
    else:
        # Spread coefficient atomics across batches while each grouped CTA
        # accumulates several neighboring x-lines before issuing its update.
        x_group = line_group // batch_size
        batch = line_group - x_group * batch_size
    if BATCH_PARTIAL_GRADS:
        # Keep parameter atomics local to a batch row before the final reduction.
        grad_decay_real += batch * modes
        grad_decay_imag += batch * modes
        grad_gamma_real += batch * modes
        grad_gamma_imag += batch * modes
        if COLLAPSE_PATHS_GRAD:
            grad_path_input_weight += batch * modes * packed_path_hidden * _PACKED_PATH_INPUTS_TL
            grad_path_input_bias += batch * modes * packed_path_hidden
            output_hidden = 2 * SWIGLU_HIDDEN if PATH_SWIGLU else packed_path_hidden
            grad_path_output_weight += batch * modes * _PACKED_PATH_OUTPUTS_TL * output_hidden
            grad_path_output_bias += batch * modes * _PACKED_PATH_OUTPUTS_TL
    x = x_group * BLOCK_LINES + tl.arange(0, BLOCK_LINES)[None, :, None]
    y = tl.arange(0, BLOCK_HEIGHT)[:, None, None]
    mode_vector = mode_group * BLOCK_MODES + tl.arange(0, BLOCK_MODES)
    mode = mode_vector[None, None, :]
    valid_line = x < scan_width
    active = valid_line & (y < height) & (mode < modes)
    valid_mode = valid_line & (mode < modes)
    # Coarse endpoints need positive-x streams only at odd columns and
    # Negative-x streams are needed only at even columns; odd/even pairs collapse.
    source_x_a = 2 * x + 1 if COARSE_PATHS else x
    source_x_b = 2 * x if COARSE_PATHS else x
    offset = ((batch * height + y) * width + source_x_a) * modes + mode
    offset_b = offset - modes if COARSE_PATHS else offset
    ar = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    scan_ai = tl.where(active, ai, 0.0)
    sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
    sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
    sbr = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
    sbi = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
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

    sar_n = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
    sai_n = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
    sbr_n = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
    sbi_n = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
    negative = tl.associative_scan(
        (
            scan_ar,
            -scan_ai,
            gr * sar_n + gi * sai_n,
            gr * sai_n - gi * sar_n,
            gr * sbr_n + gi * sbi_n,
            gr * sbi_n - gi * sbr_n,
        ),
        axis=0,
        combine_fn=_compose_complex_pair,
        reverse=True,
    )
    nar, nai, nbr, nbi = negative[2], negative[3], negative[4], negative[5]
    decay_norm = ar * ar + ai * ai
    direct_path_collapse_grad: tl.constexpr = (
        COLLAPSE_PATHS_GRAD
        and not PATH_SWIGLU
        and not HAS_DESCRIPTOR_GRAD
        and not HAS_COARSE_GRAD
    )
    if direct_path_collapse_grad:
        path_height: tl.constexpr = BLOCK_HEIGHT // 2 if COARSE_PATHS else BLOCK_HEIGHT
        path_lines: tl.constexpr = BLOCK_LINES
        valid_y = (y < height) & (mode < modes)
        if gain_kind == 1:
            inv_gain = tl.load(global_inverse_gain + mode, mask=mode < modes, other=0.0)
            vx_positive = inv_gain
            vx_negative = inv_gain
            vy_pos_fwd = inv_gain
            vy_neg_fwd = inv_gain
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
            vy_pos_fwd = tl.load(variance_y + y_offset, mask=valid_y, other=0.0)
            vy_neg_fwd = tl.load(
                variance_y + height * modes + y_offset,
                mask=valid_y,
                other=0.0,
            )
        path_tokens: tl.constexpr = path_height * path_lines
        r0_val = tl.where(
            active, _scale_by_gain(par, gain_kind, vx_positive, vx_positive, vy_pos_fwd, epsilon), 0.0
        )
        r1_val = tl.where(
            active, _scale_by_gain(pbr, gain_kind, vx_negative, vx_negative, vy_pos_fwd, epsilon), 0.0
        )
        r2_val = tl.where(
            active, _scale_by_gain(nar, gain_kind, vx_positive, vx_positive, vy_neg_fwd, epsilon), 0.0
        )
        r3_val = tl.where(
            active, _scale_by_gain(nbr, gain_kind, vx_negative, vx_negative, vy_neg_fwd, epsilon), 0.0
        )
        i0_val = tl.where(
            active, _scale_by_gain(pai, gain_kind, vx_positive, vx_positive, vy_pos_fwd, epsilon), 0.0
        )
        i1_val = tl.where(
            active, _scale_by_gain(pbi, gain_kind, vx_negative, vx_negative, vy_pos_fwd, epsilon), 0.0
        )
        i2_val = tl.where(
            active, _scale_by_gain(nai, gain_kind, vx_positive, vx_positive, vy_neg_fwd, epsilon), 0.0
        )
        i3_val = tl.where(
            active, _scale_by_gain(nbi, gain_kind, vx_negative, vx_negative, vy_neg_fwd, epsilon), 0.0
        )
        path_coordinate = tl.arange(0, _PATH_DOT_INPUTS_TL)
        paths8 = _pack_once8(
            (_coarse_path_endpoint(r0_val, 0) if COARSE_PATHS else r0_val).to(tl.bfloat16),
            (_coarse_path_endpoint(r1_val, 1) if COARSE_PATHS else r1_val).to(tl.bfloat16),
            (_coarse_path_endpoint(r2_val, 2) if COARSE_PATHS else r2_val).to(tl.bfloat16),
            (_coarse_path_endpoint(r3_val, 3) if COARSE_PATHS else r3_val).to(tl.bfloat16),
            (_coarse_path_endpoint(i0_val, 0) if COARSE_PATHS else i0_val).to(tl.bfloat16),
            (_coarse_path_endpoint(i1_val, 1) if COARSE_PATHS else i1_val).to(tl.bfloat16),
            (_coarse_path_endpoint(i2_val, 2) if COARSE_PATHS else i2_val).to(tl.bfloat16),
            (_coarse_path_endpoint(i3_val, 3) if COARSE_PATHS else i3_val).to(tl.bfloat16),
            BLOCK_MODES, path_tokens,
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
            path_input_bias
            + mode_vector[:, None] * packed_path_hidden
            + hidden_coordinate[None, :],
            mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
            other=0.0,
        ).to(tl.bfloat16)
        preactivation = (preactivation + active_input_bias[:, None, :]).to(tl.bfloat16)
        if COARSE_PATHS:
            cell_y = tl.arange(0, path_height)[:, None, None]
            cell_x = x_group * path_lines + tl.arange(0, path_lines)[None, :, None]
            path_offset = ((batch * (height // 2) + cell_y) * (width // 2) + cell_x) * modes + mode
            path_active = (cell_y < height // 2) & (cell_x < width // 2) & (mode < modes)
        else:
            path_offset, path_active = offset, active
        physical_offset = tl.reshape(
            tl.permute(path_offset, (2, 0, 1)),
            (BLOCK_MODES, path_tokens),
        )
        physical_active = tl.reshape(
            tl.permute(path_active, (2, 0, 1)),
            (BLOCK_MODES, path_tokens),
        )
        gradient_offset = (
            physical_offset
            if PLANAR_PATH_OUTPUT
            else _coordinate_offset(physical_offset, mode_vector[:, None], PACKED_OUTPUT)
        )
        active_grad_real = tl.load(
            grad_collapsed_real + gradient_offset,
            mask=physical_active,
            other=0.0,
        ).to(tl.bfloat16)
        active_grad_imag = tl.load(
            grad_collapsed_imag + gradient_offset,
            mask=physical_active,
            other=0.0,
        ).to(tl.bfloat16)
        h8 = tl.arange(0, 8)
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
        if path_tokens >= 16:
            path_input_weight_gradient = tl.dot(
                tl.permute(grad_preactivation, (0, 2, 1)),
                packed_path,
            )
        elif path_tokens == 8:
            grad_preactivation_padded = tl.reshape(
                tl.permute(
                    tl.join(grad_preactivation, tl.zeros_like(grad_preactivation)),
                    (0, 3, 1, 2),
                ),
                (BLOCK_MODES, 16, BLOCK_PATH_HIDDEN),
            )
            packed_path_padded = tl.reshape(
                tl.permute(tl.join(packed_path, tl.zeros_like(packed_path)), (0, 3, 1, 2)),
                (BLOCK_MODES, 16, _PATH_DOT_INPUTS_TL),
            )
            path_input_weight_gradient = tl.dot(
                tl.permute(grad_preactivation_padded, (0, 2, 1)),
                packed_path_padded,
            )
        else:
            path_input_weight_gradient = tl.sum(
                grad_preactivation[:, :, :, None] * packed_path[:, :, None, :],
                axis=1,
            )
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
        grad_packed_path = tl.dot(
            grad_preactivation,
            tl.permute(active_input_weight, (0, 2, 1)),
        ).to(tl.float32)
        if COARSE_PATHS:
            grad_packed_path = grad_packed_path.to(tl.bfloat16).to(tl.float32)

        gp_r0, gp_r1, gp_r2, gp_r3, gp_i0, gp_i1, gp_i2, gp_i3 = _unpack_once_shape(
            grad_packed_path, BLOCK_MODES, path_height, BLOCK_LINES,
        )
        gp_r0 = _restore_parity(gp_r0, True)
        gp_r1 = _restore_parity(gp_r1, True)
        gp_i0 = _restore_parity(gp_i0, True)
        gp_i1 = _restore_parity(gp_i1, True)
        gp_r2_b = _restore_parity(gp_r2.to(tl.bfloat16), False)
        gp_r3_b = _restore_parity(gp_r3.to(tl.bfloat16), False)
        gp_i2_b = _restore_parity(gp_i2.to(tl.bfloat16), False)
        gp_i3_b = _restore_parity(gp_i3.to(tl.bfloat16), False)
        vy_pos_b = (
            inv_gain if gain_kind == 1
            else tl.load(variance_y + y * modes + mode, mask=valid_y, other=0.0)
        )
        grad_r0 = _scale_by_gain(
            gp_r0,
            gain_kind, vx_positive, vx_positive, vy_pos_b, epsilon,
        )
        grad_r1 = _scale_by_gain(
            gp_r1,
            gain_kind, vx_negative, vx_negative, vy_pos_b, epsilon,
        )
        grad_i0 = _scale_by_gain(
            gp_i0,
            gain_kind, vx_positive, vx_positive, vy_pos_b, epsilon,
        )
        grad_i1 = _scale_by_gain(
            gp_i1,
            gain_kind, vx_negative, vx_negative, vy_pos_b, epsilon,
        )
        positive_adjoint = tl.associative_scan(
            (
                scan_ar,
                -scan_ai,
                tl.where(active, grad_r0, 0.0),
                tl.where(active, grad_i0, 0.0),
                tl.where(active, grad_r1, 0.0),
                tl.where(active, grad_i1, 0.0),
            ),
            axis=0,
            combine_fn=_compose_complex_pair,
            reverse=True,
        )
        l0r, l0i = positive_adjoint[2], positive_adjoint[3]
        l1r, l1i = positive_adjoint[4], positive_adjoint[5]
        vy_neg_b = (
            inv_gain if gain_kind == 1
            else tl.load(variance_y + height * modes + y * modes + mode, mask=valid_y, other=0.0)
        )
        grad_r2 = _scale_by_gain(gp_r2_b.to(tl.float32), gain_kind, vx_positive, vx_positive, vy_neg_b, epsilon)
        grad_r3 = _scale_by_gain(gp_r3_b.to(tl.float32), gain_kind, vx_negative, vx_negative, vy_neg_b, epsilon)
        grad_i2 = _scale_by_gain(gp_i2_b.to(tl.float32), gain_kind, vx_positive, vx_positive, vy_neg_b, epsilon)
        grad_i3 = _scale_by_gain(gp_i3_b.to(tl.float32), gain_kind, vx_negative, vx_negative, vy_neg_b, epsilon)
        negative_adjoint = tl.associative_scan(
            (
                scan_ar,
                scan_ai,
                tl.where(active, grad_r2, 0.0),
                tl.where(active, grad_i2, 0.0),
                tl.where(active, grad_r3, 0.0),
                tl.where(active, grad_i3, 0.0),
            ),
            axis=0,
            combine_fn=_compose_complex_pair,
        )
        l2r, l2i = negative_adjoint[2], negative_adjoint[3]
        l3r, l3i = negative_adjoint[4], negative_adjoint[5]
    else:
        if gain_kind == 1:
            inverse_pa = tl.load(global_inverse_gain + mode, mask=valid_mode, other=0.0)
            inverse_pb = inverse_pa
            inverse_na = inverse_pa
            inverse_nb = inverse_pa
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
            vy_negative = tl.load(
                variance_y + height * modes + y_offset,
                mask=active,
                other=0.0,
            )
            inverse_pa = tl.rsqrt(tl.maximum(vx_positive * vy_positive, epsilon))
            inverse_pb = tl.rsqrt(tl.maximum(vx_negative * vy_positive, epsilon))
            inverse_na = tl.rsqrt(tl.maximum(vx_positive * vy_negative, epsilon))
            inverse_nb = tl.rsqrt(tl.maximum(vx_negative * vy_negative, epsilon))
        r0, i0 = tl.where(active, par * inverse_pa, 0.0), tl.where(active, pai * inverse_pa, 0.0)
        r1, i1 = tl.where(active, pbr * inverse_pb, 0.0), tl.where(active, pbi * inverse_pb, 0.0)
        r2, i2 = tl.where(active, nar * inverse_na, 0.0), tl.where(active, nai * inverse_na, 0.0)
        r3, i3 = tl.where(active, nbr * inverse_nb, 0.0), tl.where(active, nbi * inverse_nb, 0.0)
        if HAS_DESCRIPTOR_GRAD:
            descriptor_base = batch * (modes if MEAN_DESCRIPTOR_GRAD else 4 * modes) + mode
            raw_factor0 = tl.load(
                descriptor_gradient_factor + descriptor_base,
                mask=valid_mode,
                other=0.0,
            )
            raw_factor1 = tl.load(
                descriptor_gradient_factor
                + descriptor_base
                + (0 if MEAN_DESCRIPTOR_GRAD else 1 * modes),
                mask=valid_mode,
                other=0.0,
            )
            raw_factor2 = tl.load(
                descriptor_gradient_factor
                + descriptor_base
                + (0 if MEAN_DESCRIPTOR_GRAD else 2 * modes),
                mask=valid_mode,
                other=0.0,
            )
            raw_factor3 = tl.load(
                descriptor_gradient_factor
                + descriptor_base
                + (0 if MEAN_DESCRIPTOR_GRAD else 3 * modes),
                mask=valid_mode,
                other=0.0,
            )
            grad_r0 = 2.0 * raw_factor0 * r0 * inverse_pa
            grad_i0 = 2.0 * raw_factor0 * i0 * inverse_pa
            grad_r1 = 2.0 * raw_factor1 * r1 * inverse_pb
            grad_i1 = 2.0 * raw_factor1 * i1 * inverse_pb
            grad_r2 = 2.0 * raw_factor2 * r2 * inverse_na
            grad_i2 = 2.0 * raw_factor2 * i2 * inverse_na
            grad_r3 = 2.0 * raw_factor3 * r3 * inverse_nb
            grad_i3 = 2.0 * raw_factor3 * i3 * inverse_nb

        else:
            # An absent descriptor contributes exactly zero. Avoid retaining
            # zero-times-state expressions through the path MLP backward.
            grad_r0 = tl.full((BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES), 0.0, tl.float32)
            grad_i0 = tl.full((BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES), 0.0, tl.float32)
            grad_r1 = tl.full((BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES), 0.0, tl.float32)
            grad_i1 = tl.full((BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES), 0.0, tl.float32)
            grad_r2 = tl.full((BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES), 0.0, tl.float32)
            grad_i2 = tl.full((BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES), 0.0, tl.float32)
            grad_r3 = tl.full((BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES), 0.0, tl.float32)
            grad_i3 = tl.full((BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES), 0.0, tl.float32)

        if COLLAPSE_PATHS_GRAD:
            path_height: tl.constexpr = BLOCK_HEIGHT // 2 if COARSE_PATHS else BLOCK_HEIGHT
            path_lines: tl.constexpr = BLOCK_LINES
            path_tokens: tl.constexpr = path_height * path_lines
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
            path_coordinate = tl.arange(0, _PATH_DOT_INPUTS_TL)
            path = path_coordinate % _PATHS_TL
            active_path_real = tl.where(
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
                active_path_real,
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
                mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
                other=0.0,
            ).to(tl.bfloat16)
            preactivation = (preactivation + active_input_bias[:, None, :]).to(tl.bfloat16)
            output_coordinate = tl.arange(0, _PATH_DOT_OUTPUTS_TL)
            if COARSE_PATHS:
                cell_y = tl.arange(0, path_height)[:, None, None]
                cell_x = x_group * path_lines + tl.arange(0, path_lines)[None, :, None]
                path_offset = ((batch * (height // 2) + cell_y) * (width // 2) + cell_x) * modes + mode
                path_active = (cell_y < height // 2) & (cell_x < width // 2) & (mode < modes)
            else:
                path_offset, path_active = offset, active
            physical_offset = tl.reshape(
                tl.permute(path_offset, (2, 0, 1)), (BLOCK_MODES, path_tokens),
            )
            physical_active = tl.reshape(
                tl.permute(path_active, (2, 0, 1)), (BLOCK_MODES, path_tokens),
            )
            gradient_offset = (
                physical_offset if PLANAR_PATH_OUTPUT
                else _coordinate_offset(physical_offset, mode_vector[:, None], PACKED_OUTPUT)
            )
            active_grad_real = tl.load(
                grad_collapsed_real + gradient_offset,
                mask=physical_active,
                other=0.0,
            ).to(tl.bfloat16)
            active_grad_imag = tl.load(
                grad_collapsed_imag + gradient_offset,
                mask=physical_active,
                other=0.0,
            ).to(tl.bfloat16)
            if PATH_SWIGLU:
                active_grad_output = tl.where(
                    output_coordinate[None, None, :] == 0,
                    active_grad_real[:, :, None],
                    tl.where(
                        output_coordinate[None, None, :] == 1,
                        active_grad_imag[:, :, None],
                        0.0,
                    ),
                ).to(tl.bfloat16)
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
                gate_sigmoid = tl.sigmoid(gate_logits)
                gate = (gate_logits * gate_sigmoid).to(tl.bfloat16)
                gated_value = tl.where(
                    value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN,
                    value * gate,
                    0.0,
                ).to(tl.bfloat16)
                output_weight_offset = (
                    mode_vector[:, None, None] * _PACKED_PATH_OUTPUTS_TL
                    + output_coordinate[None, :, None]
                ) * (2 * SWIGLU_HIDDEN) + value_coordinate[None, None, :]
                active_output_weight = tl.load(
                    path_output_weight + output_weight_offset,
                    mask=(mode_vector[:, None, None] < modes)
                    & (output_coordinate[None, :, None] < _PACKED_PATH_OUTPUTS_TL)
                    & (value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN),
                    other=0.0,
                ).to(tl.bfloat16)
                grad_gated_value = tl.dot(active_grad_output, active_output_weight).to(tl.float32)
                active_value = value.to(tl.float32)
                active_gate = gate.to(tl.float32)
                grad_value = grad_gated_value * active_gate
                gate_derivative = gate_sigmoid * (1.0 + gate_logits * (1.0 - gate_sigmoid))
                grad_gate_contribution = grad_gated_value * active_value * gate_derivative
                value_match = (
                    hidden_coordinate[None, None, None, :] == value_coordinate[None, None, :, None]
                )
                value_joint_gradient = tl.sum(
                    tl.where(value_match, grad_value[:, :, :, None], 0.0),
                    axis=2,
                )
                gate_match = (
                    hidden_coordinate[None, None, None, :]
                    == (2 * SWIGLU_HIDDEN + value_coordinate % SWIGLU_HIDDEN)[None, None, :, None]
                )
                gate_joint_gradient = tl.sum(
                    tl.where(gate_match, grad_gate_contribution[:, :, :, None], 0.0),
                    axis=2,
                )
                grad_preactivation = (value_joint_gradient + gate_joint_gradient).to(tl.bfloat16)
                if path_tokens >= 16:
                    path_output_weight_gradient = tl.dot(
                        tl.permute(active_grad_output, (0, 2, 1)),
                        gated_value,
                    )
                elif path_tokens == 8:
                    grad_output_padded = tl.reshape(
                        tl.permute(
                            tl.join(active_grad_output, tl.zeros_like(active_grad_output)),
                            (0, 3, 1, 2),
                        ),
                        (BLOCK_MODES, 16, _PATH_DOT_OUTPUTS_TL),
                    )
                    gated_value_padded = tl.reshape(
                        tl.permute(tl.join(gated_value, tl.zeros_like(gated_value)), (0, 3, 1, 2)),
                        (BLOCK_MODES, 16, BLOCK_SWIGLU_VALUE),
                    )
                    path_output_weight_gradient = tl.dot(
                        tl.permute(grad_output_padded, (0, 2, 1)),
                        gated_value_padded,
                    )
                else:
                    path_output_weight_gradient = tl.sum(
                        active_grad_output[:, :, :, None] * gated_value[:, :, None, :],
                        axis=1,
                    )
                output_weight_mask = (
                    (mode_vector[:, None, None] < modes)
                    & (output_coordinate[None, :, None] < _PACKED_PATH_OUTPUTS_TL)
                    & (value_coordinate[None, None, :] < 2 * SWIGLU_HIDDEN)
                )
            else:
                preactivation_float = preactivation.to(tl.float32)
                sigmoid = tl.sigmoid(preactivation_float)
                hidden = (preactivation_float * sigmoid).to(tl.bfloat16)
                # This CFFN has exactly two outputs. Avoid padding its output
                # adjoint and parameter reduction to a 16-column tensor-core tile.
                output_offset = (
                    mode_vector[:, None] * (2 * packed_path_hidden) + hidden_coordinate[None, :]
                )
                output_mask = (mode_vector[:, None] < modes) & (
                    hidden_coordinate[None, :] < packed_path_hidden
                )
                weight_real = (
                    tl.load(path_output_weight + output_offset, mask=output_mask, other=0.0)
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                weight_imag = (
                    tl.load(
                        path_output_weight + output_offset + packed_path_hidden,
                        mask=output_mask,
                        other=0.0,
                    )
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
                grad_hidden = (
                    active_grad_real.to(tl.float32)[:, :, None] * weight_real[:, None, :]
                    + active_grad_imag.to(tl.float32)[:, :, None] * weight_imag[:, None, :]
                )
                grad_preactivation = (
                    grad_hidden * sigmoid * (1.0 + preactivation_float * (1.0 - sigmoid))
                ).to(tl.bfloat16)
                grad_weight_real = tl.sum(
                    active_grad_real.to(tl.float32)[:, :, None] * hidden.to(tl.float32), axis=1
                )
                grad_weight_imag = tl.sum(
                    active_grad_imag.to(tl.float32)[:, :, None] * hidden.to(tl.float32), axis=1
                )
                tl.atomic_add(
                    grad_path_output_weight + output_offset, grad_weight_real, mask=output_mask,
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_path_output_weight + output_offset + packed_path_hidden,
                    grad_weight_imag,
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
            if PATH_SWIGLU:
                tl.atomic_add(
                    grad_path_output_weight + output_weight_offset,
                    path_output_weight_gradient,
                    mask=output_weight_mask,
                    sem="relaxed",
                )
                tl.atomic_add(
                    grad_path_output_bias
                    + mode_vector[:, None] * _PACKED_PATH_OUTPUTS_TL
                    + output_coordinate[None, :],
                    tl.sum(active_grad_output.to(tl.float32), axis=1),
                    mask=(mode_vector[:, None] < modes)
                    & (output_coordinate[None, :] < _PACKED_PATH_OUTPUTS_TL),
                    sem="relaxed",
                )
            if path_tokens >= 16:
                path_input_weight_gradient = tl.dot(
                    tl.permute(grad_preactivation, (0, 2, 1)),
                    packed_path,
                )
            elif path_tokens == 8:
                grad_preactivation_padded = tl.reshape(
                    tl.permute(
                        tl.join(grad_preactivation, tl.zeros_like(grad_preactivation)),
                        (0, 3, 1, 2),
                    ),
                    (BLOCK_MODES, 16, BLOCK_PATH_HIDDEN),
                )
                packed_path_padded = tl.reshape(
                    tl.permute(tl.join(packed_path, tl.zeros_like(packed_path)), (0, 3, 1, 2)),
                    (BLOCK_MODES, 16, _PATH_DOT_INPUTS_TL),
                )
                path_input_weight_gradient = tl.dot(
                    tl.permute(grad_preactivation_padded, (0, 2, 1)),
                    packed_path_padded,
                )
            else:
                path_input_weight_gradient = tl.sum(
                    grad_preactivation[:, :, :, None] * packed_path[:, :, None, :],
                    axis=1,
                )
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
                grad_path_input_bias
                + mode_vector[:, None] * packed_path_hidden
                + hidden_coordinate[None, :],
                tl.sum(grad_preactivation.to(tl.float32), axis=1),
                mask=(mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden),
                sem="relaxed",
            )
            grad_packed_path = tl.dot(
                grad_preactivation,
                tl.permute(active_input_weight, (0, 2, 1)),
            ).to(tl.float32)
            if COARSE_PATHS:
                grad_packed_path = grad_packed_path.to(tl.bfloat16).to(tl.float32)

            grad_r0 += (
                _unpack_path_gradient(
                    grad_packed_path,
                    path_coordinate,
                    0,
                    BLOCK_HEIGHT,
                    BLOCK_LINES,
                    BLOCK_MODES,
                    COARSE_PATHS,
                )
                * inverse_pa
            )
            grad_r1 += (
                _unpack_path_gradient(
                    grad_packed_path,
                    path_coordinate,
                    1,
                    BLOCK_HEIGHT,
                    BLOCK_LINES,
                    BLOCK_MODES,
                    COARSE_PATHS,
                )
                * inverse_pb
            )
            grad_r2 += (
                _unpack_path_gradient(
                    grad_packed_path,
                    path_coordinate,
                    2,
                    BLOCK_HEIGHT,
                    BLOCK_LINES,
                    BLOCK_MODES,
                    COARSE_PATHS,
                )
                * inverse_na
            )
            grad_r3 += (
                _unpack_path_gradient(
                    grad_packed_path,
                    path_coordinate,
                    3,
                    BLOCK_HEIGHT,
                    BLOCK_LINES,
                    BLOCK_MODES,
                    COARSE_PATHS,
                )
                * inverse_nb
            )
            grad_i0 += (
                _unpack_path_gradient(
                    grad_packed_path,
                    path_coordinate,
                    4,
                    BLOCK_HEIGHT,
                    BLOCK_LINES,
                    BLOCK_MODES,
                    COARSE_PATHS,
                )
                * inverse_pa
            )
            grad_i1 += (
                _unpack_path_gradient(
                    grad_packed_path,
                    path_coordinate,
                    5,
                    BLOCK_HEIGHT,
                    BLOCK_LINES,
                    BLOCK_MODES,
                    COARSE_PATHS,
                )
                * inverse_pb
            )
            grad_i2 += (
                _unpack_path_gradient(
                    grad_packed_path,
                    path_coordinate,
                    6,
                    BLOCK_HEIGHT,
                    BLOCK_LINES,
                    BLOCK_MODES,
                    COARSE_PATHS,
                )
                * inverse_na
            )
            grad_i3 += (
                _unpack_path_gradient(
                    grad_packed_path,
                    path_coordinate,
                    7,
                    BLOCK_HEIGHT,
                    BLOCK_LINES,
                    BLOCK_MODES,
                    COARSE_PATHS,
                )
                * inverse_nb
            )

        if HAS_COARSE_GRAD:
            cell = (batch * (height // 2) + y // 2) * (width // 2) + x // 2
            if FULL_COARSE_GRAD:
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
                grad_r0 += (
                    tl.load(
                        grad_coarse_real + _coordinate_offset(base0, mode, PACKED_OUTPUT),
                        mask=active,
                        other=0.0,
                    )
                    * inverse_pa
                )
                grad_i0 += (
                    tl.load(
                        grad_coarse_imag + _coordinate_offset(base0, mode, PACKED_OUTPUT),
                        mask=active,
                        other=0.0,
                    )
                    * inverse_pa
                )
                grad_r1 += (
                    tl.load(
                        grad_coarse_real + _coordinate_offset(base1, mode, PACKED_OUTPUT),
                        mask=active,
                        other=0.0,
                    )
                    * inverse_pb
                )
                grad_i1 += (
                    tl.load(
                        grad_coarse_imag + _coordinate_offset(base1, mode, PACKED_OUTPUT),
                        mask=active,
                        other=0.0,
                    )
                    * inverse_pb
                )
                grad_r2 += (
                    tl.load(
                        grad_coarse_real + _coordinate_offset(base2, mode, PACKED_OUTPUT),
                        mask=active,
                        other=0.0,
                    )
                    * inverse_na
                )
                grad_i2 += (
                    tl.load(
                        grad_coarse_imag + _coordinate_offset(base2, mode, PACKED_OUTPUT),
                        mask=active,
                        other=0.0,
                    )
                    * inverse_na
                )
                grad_r3 += (
                    tl.load(
                        grad_coarse_real + _coordinate_offset(base3, mode, PACKED_OUTPUT),
                        mask=active,
                        other=0.0,
                    )
                    * inverse_nb
                )
                grad_i3 += (
                    tl.load(
                        grad_coarse_imag + _coordinate_offset(base3, mode, PACKED_OUTPUT),
                        mask=active,
                        other=0.0,
                    )
                    * inverse_nb
                )
            else:
                coarse_base = (cell * 4) * modes + mode
                positive_endpoint = active & ((y & 1) == 1)
                negative_endpoint = active & ((y & 1) == 0)
                mask0 = positive_endpoint & ((x & 1) == 1)
                mask1 = positive_endpoint & ((x & 1) == 0)
                mask2 = negative_endpoint & ((x & 1) == 1)
                mask3 = negative_endpoint & ((x & 1) == 0)
                grad_r0 += (
                    tl.load(
                        grad_coarse_real + _coordinate_offset(coarse_base, mode, PACKED_OUTPUT),
                        mask=mask0,
                        other=0.0,
                    )
                    * inverse_pa
                )
                grad_i0 += (
                    tl.load(
                        grad_coarse_imag + _coordinate_offset(coarse_base, mode, PACKED_OUTPUT),
                        mask=mask0,
                        other=0.0,
                    )
                    * inverse_pa
                )
                grad_r1 += (
                    tl.load(
                        grad_coarse_real + _coordinate_offset(coarse_base + modes, mode, PACKED_OUTPUT),
                        mask=mask1,
                        other=0.0,
                    )
                    * inverse_pb
                )
                grad_i1 += (
                    tl.load(
                        grad_coarse_imag + _coordinate_offset(coarse_base + modes, mode, PACKED_OUTPUT),
                        mask=mask1,
                        other=0.0,
                    )
                    * inverse_pb
                )
                grad_r2 += (
                    tl.load(
                        grad_coarse_real
                        + _coordinate_offset(coarse_base + 2 * modes, mode, PACKED_OUTPUT),
                        mask=mask2,
                        other=0.0,
                    )
                    * inverse_na
                )
                grad_i2 += (
                    tl.load(
                        grad_coarse_imag
                        + _coordinate_offset(coarse_base + 2 * modes, mode, PACKED_OUTPUT),
                        mask=mask2,
                        other=0.0,
                    )
                    * inverse_na
                )
                grad_r3 += (
                    tl.load(
                        grad_coarse_real
                        + _coordinate_offset(coarse_base + 3 * modes, mode, PACKED_OUTPUT),
                        mask=mask3,
                        other=0.0,
                    )
                    * inverse_nb
                )
                grad_i3 += (
                    tl.load(
                        grad_coarse_imag
                        + _coordinate_offset(coarse_base + 3 * modes, mode, PACKED_OUTPUT),
                        mask=mask3,
                        other=0.0,
                    )
                    * inverse_nb
                )
        positive_adjoint = tl.associative_scan(
            (
                tl.where(active, ar, 1.0),
                tl.where(active, -ai, 0.0),
                grad_r0,
                grad_i0,
                grad_r1,
                grad_i1,
            ),
            axis=0,
            combine_fn=_compose_complex_pair,
            reverse=True,
        )
        negative_adjoint = tl.associative_scan(
            (
                tl.where(active, ar, 1.0),
                tl.where(active, ai, 0.0),
                grad_r2,
                grad_i2,
                grad_r3,
                grad_i3,
            ),
            axis=0,
            combine_fn=_compose_complex_pair,
        )
        l0r, l0i = positive_adjoint[2], positive_adjoint[3]
        l1r, l1i = positive_adjoint[4], positive_adjoint[5]
        l2r, l2i = negative_adjoint[2], negative_adjoint[3]
        l3r, l3i = negative_adjoint[4], negative_adjoint[5]
    # Reload source streams here rather than keeping four FP32 tiles live across
    # both scans and the MLP backward; done before storing to donated buffers.
    sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
    sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
    sbr = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
    sbi = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
    if COMBINED_STATE_GRADS:
        plane_size = tl.cast(batch_size, tl.int64) * (height * width * modes)
        grad_a_real_pointer = grad_source_real_a
        grad_a_imag_pointer = grad_source_real_a + plane_size
        grad_b_real_pointer = grad_source_real_a + 2 * plane_size
        grad_b_imag_pointer = grad_source_real_a + 3 * plane_size
    else:
        grad_a_real_pointer, grad_a_imag_pointer = grad_source_real_a, grad_source_imag_a
        grad_b_real_pointer, grad_b_imag_pointer = grad_source_real_b, grad_source_imag_b
    tl.store(
        grad_a_real_pointer + offset,
        gr * l0r + gi * l0i + gr * l2r - gi * l2i,
        mask=active,
    )
    tl.store(
        grad_a_imag_pointer + offset,
        -gi * l0r + gr * l0i + gi * l2r + gr * l2i,
        mask=active,
    )
    tl.store(
        grad_b_real_pointer + offset_b,
        gr * l1r + gi * l1i + gr * l3r - gi * l3i,
        mask=active,
    )
    tl.store(
        grad_b_imag_pointer + offset_b,
        -gi * l1r + gr * l1i + gi * l3r + gr * l3i,
        mask=active,
    )

    if COARSE_PATHS and ZERO_UNUSED_COARSE_GRADS:
        tl.store(grad_a_real_pointer + offset_b, 0.0, mask=active)
        tl.store(grad_a_imag_pointer + offset_b, 0.0, mask=active)
        tl.store(grad_b_real_pointer + offset, 0.0, mask=active)
        tl.store(grad_b_imag_pointer + offset, 0.0, mask=active)

    inverse_decay = 1.0 / decay_norm
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
    coefficient_mode = mode_vector
    valid_coefficient_mode = mode_vector < modes
    tl.atomic_add(
        grad_decay_real + coefficient_mode,
        grad_dr,
        mask=valid_coefficient_mode,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_decay_imag + coefficient_mode,
        grad_di,
        mask=valid_coefficient_mode,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_gamma_real + coefficient_mode,
        grad_gr,
        mask=valid_coefficient_mode,
        sem="relaxed",
    )
    tl.atomic_add(
        grad_gamma_imag + coefficient_mode,
        grad_gi,
        mask=valid_coefficient_mode,
        sem="relaxed",
    )


__all__ = [
    "_PACKED_PATH_INPUTS",
    "_PACKED_PATH_OUTPUTS",
    "_finalize_descriptor_kernel",
    "_product_scan_coarse4_associative_backward_kernel",
    "_product_scan_coarse4_associative_forward_kernel",
]
