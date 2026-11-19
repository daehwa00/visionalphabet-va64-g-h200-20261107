"""Parity-folded retained coarse adjoint from the r5 standalone screen."""
import triton
import triton.language as tl
from lnet.direct_path_scan_kernels import _sum_line
from lnet.hopper_r8_effective_output_vjp import _fast_sigmoid, _slice8_perm, _pad16_perm
from lnet.effective_structural_scan import _pack_once8
from lnet.triton_product_scan_kernels import _compose_complex_pair

@triton.jit
def _shift_rows(value, H: tl.constexpr, N: tl.constexpr, REVERSE: tl.constexpr):
    row = tl.arange(0, N)[:, None, None]
    before = row + 1 if REVERSE else row - 1
    indices = tl.broadcast_to(before & (N - 1), value.shape)
    return tl.where((before >= 0) & (before < H), tl.gather(value, indices, 0), 0.0)

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
def _split_parity(value):
    bm: tl.constexpr = value.shape[2]
    return tl.split(tl.permute(tl.reshape(value, (32, 2, 1, bm)), (0, 2, 3, 1)))

@triton.jit
def _merge_parity(even, odd):
    return tl.reshape(tl.permute(tl.join(even, odd), (0, 3, 1, 2)), (64, 1, even.shape[2]))

@triton.jit
def _complex_mul(ar, ai, xr, xi):
    return ar * xr - ai * xi, ar * xi + ai * xr

@triton.jit
def _scan_pairs(ar, ai, xr, xi, zr, zi, active, REVERSE: tl.constexpr):
    result = tl.associative_scan(
        (tl.where(active, ar, 1.0), tl.where(active, ai, 0.0),
         tl.where(active, xr, 0.0), tl.where(active, xi, 0.0),
         tl.where(active, zr, 0.0), tl.where(active, zi, 0.0)),
        0, _compose_complex_pair, reverse=REVERSE,
    )
    return result[2], result[3], result[4], result[5]

@triton.jit
def _pair_drive(ar, ai, xr, xi, yr, yi):
    re, im = _complex_mul(ar, ai, xr, xi)
    return re + yr, im + yi

@triton.jit
def c1_parity32(
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
    k = tl.arange(0, 32)[:, None, None]
    pair_active = (k < height // 2) & valid_mode
    a2r, a2i = _complex_mul(ar, ai, ar, ai)
    sapr, sapi = _split_parity(gr * sar - gi * sai), _split_parity(gr * sai + gi * sar)
    sbpr, sbpi = _split_parity(gr * sbr - gi * sbi), _split_parity(gr * sbi + gi * sbr)
    sanr, sani = _split_parity(gr * sar + gi * sai), _split_parity(gr * sai - gi * sar)
    sbnr, sbni = _split_parity(gr * sbr + gi * sbi), _split_parity(gr * sbi - gi * sbr)
    dar, dai = _pair_drive(ar, ai, sapr[0], sapi[0], sapr[1], sapi[1])
    dbr, dbi = _pair_drive(ar, ai, sbpr[0], sbpi[0], sbpr[1], sbpi[1])
    par, pai, pbr, pbi = _scan_pairs(a2r, a2i, dar, dai, dbr, dbi, pair_active, False)
    dar, dai = _pair_drive(ar, -ai, sanr[1], sani[1], sanr[0], sani[0])
    dbr, dbi = _pair_drive(ar, -ai, sbnr[1], sbni[1], sbnr[0], sbni[0])
    nar, nai, nbr, nbi = _scan_pairs(a2r, -a2i, dar, dai, dbr, dbi, pair_active, True)

    vx_positive = tl.load(variance_x + source_x_a * modes + mode, valid_mode, 0.0)
    vx_negative = tl.load(variance_x + width * modes + source_x_b * modes + mode, valid_mode, 0.0)
    vy_pos = tl.load(variance_y + (2 * k + 1) * modes + mode, pair_active, 0.0)
    vy_neg = tl.load(variance_y + (height + 2 * k) * modes + mode, pair_active, 0.0)
    pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos, epsilon))
    pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos, epsilon))
    na = tl.rsqrt(tl.maximum(vx_positive * vy_neg, epsilon))
    nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg, epsilon))
    path_height: tl.constexpr = 32
    path_tokens: tl.constexpr = path_height
    paths8 = _pack_once8(
        tl.where(pair_active, par * pa, 0.0).to(tl.bfloat16),
        tl.where(pair_active, pbr * pb, 0.0).to(tl.bfloat16),
        tl.where(pair_active, nar * na, 0.0).to(tl.bfloat16),
        tl.where(pair_active, nbr * nb, 0.0).to(tl.bfloat16),
        tl.where(pair_active, pai * pa, 0.0).to(tl.bfloat16),
        tl.where(pair_active, pbi * pb, 0.0).to(tl.bfloat16),
        tl.where(pair_active, nai * na, 0.0).to(tl.bfloat16),
        tl.where(pair_active, nbi * nb, 0.0).to(tl.bfloat16),
        BM, path_tokens,
    )
    packed_path = _pad16_perm(paths8, BM, path_tokens)
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
    output_offset = mode_vector[:, None] * (2 * packed_path_hidden) + h8[None, :]
    output_mask = (mode_vector[:, None] < modes) & (h8[None, :] < packed_path_hidden)
    tl.atomic_add(
        grad_path_output_weight + output_offset,
        tl.sum(active_grad_real.to(tl.float32)[:, :, None] * hid8.to(tl.float32), axis=1),
        mask=output_mask, sem="relaxed",
    )
    tl.atomic_add(
        grad_path_output_weight + output_offset + packed_path_hidden,
        tl.sum(active_grad_imag.to(tl.float32)[:, :, None] * hid8.to(tl.float32), axis=1),
        mask=output_mask, sem="relaxed",
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
        grad_path_input_bias + mode_vector[:, None] * packed_path_hidden + h8[None, :],
        tl.sum(gp8.to(tl.float32), axis=1),
        mask=(mode_vector[:, None] < modes) & (h8[None, :] < packed_path_hidden),
        sem="relaxed",
    )
    grad_packed_path = tl.dot(grad_preactivation, tl.permute(active_input_weight, (0, 2, 1))).to(
        tl.float32
    )
    g0, g1, g2, g3, g4, g5, g6, g7 = _unpack_once_shape(
        grad_packed_path.to(tl.bfloat16).to(tl.float32), BM, 32, 1,
    )
    r0, i0, r1, i1 = _scan_pairs(a2r, -a2i, g0 * pa, g4 * pa, g1 * pb, g5 * pb, pair_active, True)
    l0r, l0i = _complex_mul(ar, -ai, r0, i0)
    l1r, l1i = _complex_mul(ar, -ai, r1, i1)
    l0r, l0i = _merge_parity(l0r, r0), _merge_parity(l0i, i0)
    l1r, l1i = _merge_parity(l1r, r1), _merge_parity(l1i, i1)
    half: tl.constexpr = height // 2
    prev_0r, prev_0i = _shift_rows(par, half, 32, False), _shift_rows(pai, half, 32, False)
    prev_1r, prev_1i = _shift_rows(pbr, half, 32, False), _shift_rows(pbi, half, 32, False)
    odd_0r, odd_0i = _pair_drive(ar, ai, prev_0r, prev_0i, sapr[0], sapi[0])
    odd_1r, odd_1i = _pair_drive(ar, ai, prev_1r, prev_1i, sbpr[0], sbpi[0])
    previous_0r, previous_0i = _merge_parity(prev_0r, odd_0r), _merge_parity(prev_0i, odd_0i)
    previous_1r, previous_1i = _merge_parity(prev_1r, odd_1r), _merge_parity(prev_1i, odd_1i)
    grad_dr_pos = _sum_line(l0r * previous_0r + l0i * previous_0i + l1r * previous_1r + l1i * previous_1i, active)
    grad_di_pos = _sum_line(-l0r * previous_0i + l0i * previous_0r - l1r * previous_1i + l1i * previous_1r, active)

    r2, i2, r3, i3 = _scan_pairs(a2r, a2i, g2 * na, g6 * na, g3 * nb, g7 * nb, pair_active, False)
    l2r, l2i = _complex_mul(ar, ai, r2, i2)
    l3r, l3i = _complex_mul(ar, ai, r3, i3)
    l2r, l2i = _merge_parity(r2, l2r), _merge_parity(i2, l2i)
    l3r, l3i = _merge_parity(r3, l3r), _merge_parity(i3, l3i)
    prev_2r, prev_2i = _shift_rows(nar, half, 32, True), _shift_rows(nai, half, 32, True)
    prev_3r, prev_3i = _shift_rows(nbr, half, 32, True), _shift_rows(nbi, half, 32, True)
    even_2r, even_2i = _pair_drive(ar, -ai, prev_2r, prev_2i, sanr[1], sani[1])
    even_3r, even_3i = _pair_drive(ar, -ai, prev_3r, prev_3i, sbnr[1], sbni[1])
    previous_2r, previous_2i = _merge_parity(even_2r, prev_2r), _merge_parity(even_2i, prev_2i)
    previous_3r, previous_3i = _merge_parity(even_3r, prev_3r), _merge_parity(even_3i, prev_3i)
    grad_dr_neg = _sum_line(l2r * previous_2r + l2i * previous_2i + l3r * previous_3r + l3i * previous_3i, active)
    grad_di_neg = _sum_line(l2r * previous_2i - l2i * previous_2r + l3r * previous_3i - l3i * previous_3r, active)
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
    grad_dr = grad_dr_pos + grad_dr_neg
    grad_di = grad_di_pos + grad_di_neg
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
