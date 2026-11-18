"""Four-row retained adjoint with one tensor-core reduction for the output weight and bias.

The original retained adjoint has independently validated device receipts. This specialized
entry preserves its pointer ABI and uses the same scan, packing and gradient helpers.
"""

import triton
import triton.language as tl

from .direct_path_scan_kernels import _path_channel_gradient, _path_input, _sum_line
from .segmented_path_scan import _prefix_pair


@triton.jit
def _output_weight_bias_vjp(
    grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
):
    # The output projection has two rows and eight hidden features. BF16 dot operands
    # already have sixteen lanes. A column of ones in lane eight also accumulates db.
    dy = tl.where(
        channel[None, :, None] == 0,
        grad_real[:, None, :],
        tl.where(channel[None, :, None] == 1, grad_imag[:, None, :], 0.0),
    ).to(tl.bfloat16)
    features = tl.where(feature[None, None, :] == 8, 1.0, hidden).to(tl.bfloat16)
    partial = tl.dot(dy, features)
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
    sar_n = tl.load(source_real_a + offset, active, 0).to(tl.float32)
    sai_n = tl.load(source_imag_a + offset, active, 0).to(tl.float32)
    sbr_n = tl.load(source_real_b + offset_b, active, 0).to(tl.float32)
    sbi_n = tl.load(source_imag_b + offset_b, active, 0).to(tl.float32)
    nar, nai, nbr, nbi = _prefix_pair(
        ar,
        -ai,
        gr * sar_n + gi * sai_n,
        gr * sai_n - gi * sar_n,
        gr * sbr_n + gi * sbi_n,
        gr * sbi_n - gi * sbr_n,
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
    # The coarse epilogue keeps alternating endpoints; its MLP has half as many tokens.
    pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos_fwd, epsilon))
    pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos_fwd, epsilon))
    na = tl.rsqrt(tl.maximum(vx_positive * vy_neg_fwd, epsilon))
    nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg_fwd, epsilon))
    packed_path = _path_input(
        par, pai, pbr, pbi, nar, nai, nbr, nbi, pa, pb, na, nb, active, COARSE
    )
    path_coordinate = tl.arange(0, 16)
    hidden_coordinate = tl.arange(0, 16)
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
    preactivation_float = preactivation.to(tl.float32)
    sigmoid = tl.sigmoid(preactivation_float)
    hidden = (preactivation_float * sigmoid).to(tl.bfloat16)
    silu_deriv = sigmoid * (1.0 + preactivation_float * (1.0 - sigmoid))
    output_offset = mode_vector[:, None] * (2 * packed_path_hidden) + hidden_coordinate[None, :]
    output_mask = (mode_vector[:, None] < modes) & (hidden_coordinate[None, :] < packed_path_hidden)
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
    weight_real = (
        tl.load(path_output_weight + output_offset, mask=output_mask, other=0.0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    weight_imag = (
        tl.load(
            path_output_weight + output_offset + packed_path_hidden, mask=output_mask, other=0.0
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    grad_hidden = (
        active_grad_real.to(tl.float32)[:, :, None] * weight_real[:, None, :]
        + active_grad_imag.to(tl.float32)[:, :, None] * weight_imag[:, None, :]
    )
    grad_preactivation = (grad_hidden * silu_deriv).to(tl.bfloat16)
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
    # Keep the negative channels in their original BF16 representation across the positive adjoint.
    nr0 = _path_channel_gradient(grad_packed_path, path_coordinate, 2, BM, COARSE).to(tl.bfloat16)
    nr1 = _path_channel_gradient(grad_packed_path, path_coordinate, 3, BM, COARSE).to(tl.bfloat16)
    ni0 = _path_channel_gradient(grad_packed_path, path_coordinate, 6, BM, COARSE).to(tl.bfloat16)
    ni1 = _path_channel_gradient(grad_packed_path, path_coordinate, 7, BM, COARSE).to(tl.bfloat16)
    vy = tl.load(variance_y + y * modes + mode, valid_y, 0.0)
    ga = tl.rsqrt(tl.maximum(vx_positive * vy, epsilon))
    gb = tl.rsqrt(tl.maximum(vx_negative * vy, epsilon))
    l0r, l0i, l1r, l1i = _prefix_pair(
        ar,
        -ai,
        _path_channel_gradient(grad_packed_path, path_coordinate, 0, BM, COARSE) * ga,
        _path_channel_gradient(grad_packed_path, path_coordinate, 4, BM, COARSE) * ga,
        _path_channel_gradient(grad_packed_path, path_coordinate, 1, BM, COARSE) * gb,
        _path_channel_gradient(grad_packed_path, path_coordinate, 5, BM, COARSE) * gb,
        active,
        True,
        DOUBLING,
    )
    vy = tl.load(variance_y + (height + y) * modes + mode, valid_y, 0.0)
    ga = tl.rsqrt(tl.maximum(vx_positive * vy, epsilon))
    gb = tl.rsqrt(tl.maximum(vx_negative * vy, epsilon))
    l2r, l2i, l3r, l3i = _prefix_pair(
        ar,
        ai,
        nr0.to(tl.float32) * ga,
        ni0.to(tl.float32) * ga,
        nr1.to(tl.float32) * gb,
        ni1.to(tl.float32) * gb,
        active,
        False,
        DOUBLING,
    )
    # Reload before writing adjoints; the original FP32 drives are needed for pole gradients.
    sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
    sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
    sbr = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
    sbi = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
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
    inverse_decay = 1.0 / decay_norm
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
