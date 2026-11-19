"""Compact direct adjoint with multiple columns per CTA."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .effective_output_vjp import (
    _fast_sigmoid,
    _pack_once8,
    _pad16_perm,
    _slice8_perm,
    _unpack_once_shape,
)
from .triton_product_scan_kernels import (
    _PACKED_PATH_INPUTS_TL,
    _PACKED_PATH_OUTPUTS_TL,
    _PATH_DOT_INPUTS_TL,
    _compose_complex_pair,
)


@triton.jit
def r7_direct_backward(
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
    line_tiles = tl.cdiv(scan_width, BLOCK_LINES)
    tile = tl.program_id(0)
    mode_group = tile // LINES_PER_GROUP
    line_group = tl.program_id(1) * LINES_PER_GROUP + tile % LINES_PER_GROUP
    batch = line_group // line_tiles
    x_group = line_group - batch * line_tiles
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
        BLOCK_MODES,
        path_tokens,
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
        grad_packed_path,
        BLOCK_MODES,
        BLOCK_HEIGHT,
        BLOCK_LINES,
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
    l0r, l0i, l1r, l1i = (
        positive_adjoint[2],
        positive_adjoint[3],
        positive_adjoint[4],
        positive_adjoint[5],
    )

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
    l2r, l2i, l3r, l3i = (
        negative_adjoint[2],
        negative_adjoint[3],
        negative_adjoint[4],
        negative_adjoint[5],
    )

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
