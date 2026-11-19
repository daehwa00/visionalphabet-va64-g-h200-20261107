"""Device-agnostic structural specializations for the retained 56x56 vertical scan kernels.

Each variant is a selectable candidate over the verified four-row output-dot kernel
(or forward retained kernel) while preserving all existing kernel revisions and receipts:
  - Backward VARIANT 1 (recompute_prev): Hypothesis B prefix intermediate live-range cut.
  - Backward VARIANT 2 (fast_mlp): Single-permute pack/unpack + 8-lane active hidden MLP.
  - Backward VARIANT 3 (half_hidden): Prefix live-range cut (recompute) + fast_mlp.
  - Backward VARIANT 4 (unpack_prev): Prefix live-range cut (packed-path recovery) + fast_mlp.
  - Backward VARIANT 5 (unpack_rows): unpack_prev + Hypothesis C 2-row output VJP gather.
  - Backward VARIANT 6 (col_loop2): Hypothesis A 2-column CTA loop + in-register grad accumulation + unpack_prev.
  - Forward  VARIANT 1 (fast_forward): Single-permute pack + 8-lane active hidden MLP.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
import hashlib
from pathlib import Path

import triton
import triton.language as tl

from .direct_path_scan_kernels import (
    _gains,
    _path_channel_gradient,
    _path_forward_kernel as _original_forward_kernel,
    _scan_states,
    _sum_line,
    _flat,
)
from .effective_output_vjp import (
    _effective_output_vjp_rows,
    _rows_kernel,
    _transpose_kernel,
    select_effective_vjp,
    vjp_revision,
)
from .kernel_launch_config import (
    geometry_store_signature,
    register_kernel_specializer,
    validated_geometry,
)
from .output_dot_kernel import (
    _output_weight_bias_vjp,
    _path_backward_kernel as _original_backward_kernel,
)
from .path_backward_layout import _LayoutGraphPass
from .retained_path_dispatch import PROFILE_NAME, VERTICAL_BACKWARD, VERTICAL_FORWARD
from .segmented_path_scan import _prefix_pair
from .triton_product_scan_kernels import (
    _coarse_path_endpoint,
    _pack_same_resolution_paths,
)

STRUCT_NAME = "effective_structural_scan"
BUNDLE = Path(__file__).with_name("validated_effective_structural_scans.json")


@triton.jit
def _structural_recompute_prev(x):
    return x


@triton.jit
def _structural_fast_mlp(x):
    return x


@triton.jit
def _structural_half_hidden(x):
    return x


@triton.jit
def _structural_unpack_prev(x):
    return x


@triton.jit
def _structural_unpack_rows(x):
    return x


@triton.jit
def _structural_col_loop2(x):
    return x


@triton.jit
def _structural_fast_forward(x):
    return x


@triton.jit
def _take_low8(x, BM: tl.constexpr, N: tl.constexpr):
    lo, _hi = tl.split(tl.permute(tl.reshape(x, (BM, N, 2, 8)), (0, 1, 3, 2)))
    return lo


@triton.jit
def _pad_high8(x, BM: tl.constexpr, N: tl.constexpr):
    padded = tl.permute(tl.join(x, tl.zeros_like(x)), (0, 1, 3, 2))
    return tl.reshape(padded, (BM, N, 16))


@triton.jit
def _prep_chan(value, inv, mask, PATH: tl.constexpr, COARSE: tl.constexpr):
    value = tl.where(mask, value * inv, 0.0)
    if COARSE:
        value = _coarse_path_endpoint(value, PATH)
    return value.to(tl.bfloat16)


@triton.jit
def _pack_once8(r0, r1, r2, r3, i0, i1, i2, i3, BM: tl.constexpr, TOKENS: tl.constexpr):
    even = tl.join(tl.join(r0, i0), tl.join(r2, i2))
    odd = tl.join(tl.join(r1, i1), tl.join(r3, i3))
    paths = tl.reshape(tl.join(even, odd), (TOKENS, BM, 8))
    return tl.permute(paths, (1, 0, 2))


@triton.jit
def _pack_once16(r0, r1, r2, r3, i0, i1, i2, i3, BM: tl.constexpr, TOKENS: tl.constexpr):
    return _pad_high8(_pack_once8(r0, r1, r2, r3, i0, i1, i2, i3, BM, TOKENS), BM, TOKENS)


@triton.jit
def _expand_coarse(v, PATH: tl.constexpr):
    row = tl.arange(0, 2)[None, :, None, None]
    return tl.reshape(
        tl.where(row == (1 if PATH < 2 else 0), v[:, None, :, :], 0.0), (64, 1, v.shape[2])
    )


@triton.jit
def _unpack_8(active8, BM: tl.constexpr, COARSE: tl.constexpr):
    tokens: tl.constexpr = 32 if COARSE else 64
    t = tl.reshape(tl.permute(active8, (1, 0, 2)), (tokens, 1, BM, 2, 2, 2))
    even, odd = tl.split(t)
    p0, p2 = tl.split(even)
    p1, p3 = tl.split(odd)
    g0, g4 = tl.split(p0)
    g2, g6 = tl.split(p2)
    g1, g5 = tl.split(p1)
    g3, g7 = tl.split(p3)
    if COARSE:
        g0 = _expand_coarse(g0, 0)
        g1 = _expand_coarse(g1, 1)
        g2 = _expand_coarse(g2, 2)
        g3 = _expand_coarse(g3, 3)
        g4 = _expand_coarse(g4, 0)
        g5 = _expand_coarse(g5, 1)
        g6 = _expand_coarse(g6, 2)
        g7 = _expand_coarse(g7, 3)
    return g0, g1, g2, g3, g4, g5, g6, g7


@triton.jit
def _unpack_all_once(grad_packed_path, BM: tl.constexpr, COARSE: tl.constexpr):
    tokens: tl.constexpr = 32 if COARSE else 64
    return _unpack_8(_take_low8(grad_packed_path, BM, tokens), BM, COARSE)


@triton.jit
def _pack_fwd_once(r0, r1, r2, r3, i0, i1, i2, i3, g0, g1, g2, g3, mask, BM: tl.constexpr):
    c0 = tl.where(mask, r0 * g0, 0.0).to(tl.bfloat16)
    c1 = tl.where(mask, r1 * g1, 0.0).to(tl.bfloat16)
    c2 = tl.where(mask, r2 * g2, 0.0).to(tl.bfloat16)
    c3 = tl.where(mask, r3 * g3, 0.0).to(tl.bfloat16)
    c4 = tl.where(mask, i0 * g0, 0.0).to(tl.bfloat16)
    c5 = tl.where(mask, i1 * g1, 0.0).to(tl.bfloat16)
    c6 = tl.where(mask, i2 * g2, 0.0).to(tl.bfloat16)
    c7 = tl.where(mask, i3 * g3, 0.0).to(tl.bfloat16)
    return _pack_once16(c0, c1, c2, c3, c4, c5, c6, c7, BM, 64)


def _make_vback_recompute_prev():
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
        batch_size = _structural_recompute_prev(line_count // width)
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
        valid_y = (y < height) & (mode < modes)
        x_offset = source_x_a * modes + mode
        x_offset_b = source_x_b * modes + mode
        y_offset = y * modes + mode
        vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
        vx_negative = tl.load(variance_x + width * modes + x_offset_b, mask=valid_mode, other=0.0)
        vy_pos_fwd = tl.load(variance_y + y_offset, mask=valid_y, other=0.0)
        pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos_fwd, epsilon))
        pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos_fwd, epsilon))
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
        r0 = _flat(par, pa, active, 0, COARSE)
        r1 = _flat(pbr, pb, active, 1, COARSE)
        i0 = _flat(pai, pa, active, 0, COARSE)
        i1 = _flat(pbi, pb, active, 1, COARSE)
        vy_neg_fwd = tl.load(variance_y + height * modes + y_offset, mask=valid_y, other=0.0)
        na = tl.rsqrt(tl.maximum(vx_positive * vy_neg_fwd, epsilon))
        nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg_fwd, epsilon))
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
        r2 = _flat(nar, na, active, 2, COARSE)
        r3 = _flat(nbr, nb, active, 3, COARSE)
        i2 = _flat(nai, na, active, 2, COARSE)
        i3 = _flat(nbi, nb, active, 3, COARSE)
        packed_path = _pack_same_resolution_paths(r0, r1, r2, r3, i0, i1, i2, i3)
        path_height: tl.constexpr = 64 // 2 if COARSE else 64
        path_tokens: tl.constexpr = path_height
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
        grad_packed_path = tl.dot(
            grad_preactivation, tl.permute(active_input_weight, (0, 2, 1))
        ).to(tl.float32)
        if COARSE:
            grad_packed_path = grad_packed_path.to(tl.bfloat16).to(tl.float32)
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
        sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
        sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
        sbr = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
        sbi = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
        plane_size = tl.cast(batch_size, tl.int64) * (height * width * modes)
        grad_a_imag_pointer = grad_source_real_a + plane_size
        grad_b_real_pointer = grad_source_real_a + 2 * plane_size
        grad_b_imag_pointer = grad_source_real_a + 3 * plane_size
        tl.store(grad_source_real_a + offset, gr * l0r + gi * l0i + gr * l2r - gi * l2i, mask=active)
        tl.store(
            grad_a_imag_pointer + offset, -gi * l0r + gr * l0i + gi * l2r + gr * l2i, mask=active
        )
        tl.store(
            grad_b_real_pointer + offset_b, gr * l1r + gi * l1i + gr * l3r - gi * l3i, mask=active
        )
        tl.store(
            grad_b_imag_pointer + offset_b, -gi * l1r + gr * l1i + gi * l3r + gr * l3i, mask=active
        )
        decay_norm = ar * ar + ai * ai
        inverse_decay = 1.0 / decay_norm
        positive_drive_ar, positive_drive_ai = (gr * sar - gi * sai, gr * sai + gi * sar)
        positive_drive_br, positive_drive_bi = (gr * sbr - gi * sbi, gr * sbi + gi * sbr)
        par, pai, pbr, pbi = _prefix_pair(
            ar,
            ai,
            positive_drive_ar,
            positive_drive_ai,
            positive_drive_br,
            positive_drive_bi,
            active,
            False,
            DOUBLING,
        )
        previous_0r = (ar * (par - positive_drive_ar) + ai * (pai - positive_drive_ai)) * inverse_decay
        previous_0i = (ar * (pai - positive_drive_ai) - ai * (par - positive_drive_ar)) * inverse_decay
        previous_1r = (ar * (pbr - positive_drive_br) + ai * (pbi - positive_drive_bi)) * inverse_decay
        previous_1i = (ar * (pbi - positive_drive_bi) - ai * (pbr - positive_drive_br)) * inverse_decay
        grad_dr = _sum_line(
            l0r * previous_0r + l0i * previous_0i + l1r * previous_1r + l1i * previous_1i, active
        )
        grad_di = _sum_line(
            -l0r * previous_0i + l0i * previous_0r - l1r * previous_1i + l1i * previous_1r, active
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
        negative_drive_ar, negative_drive_ai = (gr * sar + gi * sai, gr * sai - gi * sar)
        negative_drive_br, negative_drive_bi = (gr * sbr + gi * sbi, gr * sbi - gi * sbr)
        nar, nai, nbr, nbi = _prefix_pair(
            ar,
            -ai,
            negative_drive_ar,
            negative_drive_ai,
            negative_drive_br,
            negative_drive_bi,
            active,
            True,
            DOUBLING,
        )
        previous_2r = (ar * (nar - negative_drive_ar) - ai * (nai - negative_drive_ai)) * inverse_decay
        previous_2i = (ai * (nar - negative_drive_ar) + ar * (nai - negative_drive_ai)) * inverse_decay
        previous_3r = (ar * (nbr - negative_drive_br) - ai * (nbi - negative_drive_bi)) * inverse_decay
        previous_3i = (ai * (nbr - negative_drive_br) + ar * (nbi - negative_drive_bi)) * inverse_decay
        grad_dr = grad_dr + _sum_line(
            l2r * previous_2r + l2i * previous_2i + l3r * previous_3r + l3i * previous_3i, active
        )
        grad_di = grad_di + _sum_line(
            l2r * previous_2i - l2i * previous_2r + l3r * previous_3i - l3i * previous_3r, active
        )
        pole_mask = (x_group < scan_width) & (mode_vector < modes)
        tl.atomic_add(grad_decay_real + mode_vector, grad_dr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_decay_imag + mode_vector, grad_di, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_real + mode_vector, grad_gr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_imag + mode_vector, grad_gi, mask=pole_mask, sem="relaxed")

    return _path_backward_kernel


def _make_vback_fast_mlp():
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
        batch_size = _structural_fast_mlp(line_count // width)
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
        path_tokens: tl.constexpr = path_height
        valid_y = (y < height) & (mode < modes)
        x_offset = source_x_a * modes + mode
        x_offset_b = source_x_b * modes + mode
        y_offset = y * modes + mode
        vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
        vx_negative = tl.load(variance_x + width * modes + x_offset_b, mask=valid_mode, other=0.0)
        vy_pos_fwd = tl.load(variance_y + y_offset, mask=valid_y, other=0.0)
        vy_neg_fwd = tl.load(variance_y + height * modes + y_offset, mask=valid_y, other=0.0)
        pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos_fwd, epsilon))
        pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos_fwd, epsilon))
        na = tl.rsqrt(tl.maximum(vx_positive * vy_neg_fwd, epsilon))
        nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg_fwd, epsilon))
        packed_path = _pack_once16(
            _prep_chan(par, pa, active, 0, COARSE),
            _prep_chan(pbr, pb, active, 1, COARSE),
            _prep_chan(nar, na, active, 2, COARSE),
            _prep_chan(nbr, nb, active, 3, COARSE),
            _prep_chan(pai, pa, active, 0, COARSE),
            _prep_chan(pbi, pb, active, 1, COARSE),
            _prep_chan(nai, na, active, 2, COARSE),
            _prep_chan(nbi, nb, active, 3, COARSE),
            BM,
            path_tokens,
        )
        path_coordinate = tl.arange(0, 16)
        hidden_coordinate = tl.arange(0, 16)
        h8_coord = tl.arange(0, 8)
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
        pre8 = _take_low8(tl.dot(packed_path, active_input_weight), BM, path_tokens)
        bias8 = tl.load(
            path_input_bias + mode_vector[:, None] * packed_path_hidden + h8_coord[None, :],
            mask=(mode_vector[:, None] < modes) & (h8_coord[None, :] < packed_path_hidden),
            other=0.0,
        ).to(tl.bfloat16)
        pre8 = (pre8 + bias8[:, None, :]).to(tl.bfloat16)
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
        pre8_f = pre8.to(tl.float32)
        sig8 = tl.sigmoid(pre8_f)
        hidden8 = (pre8_f * sig8).to(tl.bfloat16)
        hidden16 = _pad_high8(hidden8, BM, path_tokens)
        output_offset8 = mode_vector[:, None] * (2 * packed_path_hidden) + h8_coord[None, :]
        output_mask8 = (mode_vector[:, None] < modes) & (h8_coord[None, :] < packed_path_hidden)
        _output_weight_bias_vjp(
            active_grad_real,
            active_grad_imag,
            hidden16,
            mode_vector,
            path_coordinate,
            hidden_coordinate,
            packed_path_hidden,
            modes,
            grad_path_output_weight,
            grad_path_output_bias,
        )
        pre8_f2 = pre8.to(tl.float32)
        sig8_2 = tl.sigmoid(pre8_f2)
        silu_deriv8 = sig8_2 * (1.0 + pre8_f2 * (1.0 - sig8_2))
        weight_real8 = (
            tl.load(path_output_weight + output_offset8, mask=output_mask8, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        weight_imag8 = (
            tl.load(
                path_output_weight + output_offset8 + packed_path_hidden,
                mask=output_mask8,
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        grad_hidden8 = (
            active_grad_real.to(tl.float32)[:, :, None] * weight_real8[:, None, :]
            + active_grad_imag.to(tl.float32)[:, :, None] * weight_imag8[:, None, :]
        )
        grad_pre8 = (grad_hidden8 * silu_deriv8).to(tl.bfloat16)
        tl.atomic_add(
            grad_path_input_bias + mode_vector[:, None] * packed_path_hidden + h8_coord[None, :],
            tl.sum(grad_pre8.to(tl.float32), axis=1),
            mask=output_mask8,
            sem="relaxed",
        )
        grad_preactivation = _pad_high8(grad_pre8, BM, path_tokens)
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
        grad_packed_path = tl.dot(
            grad_preactivation, tl.permute(active_input_weight, (0, 2, 1))
        ).to(tl.float32)
        if COARSE:
            grad_packed_path = grad_packed_path.to(tl.bfloat16).to(tl.float32)
        g0, g1, g2, g3, g4, g5, g6, g7 = _unpack_all_once(grad_packed_path, BM, COARSE)
        nr0 = g2.to(tl.bfloat16)
        nr1 = g3.to(tl.bfloat16)
        ni0 = g6.to(tl.bfloat16)
        ni1 = g7.to(tl.bfloat16)
        l0r, l0i, l1r, l1i = _prefix_pair(
            ar, -ai, g0 * pa, g4 * pa, g1 * pb, g5 * pb, active, True, DOUBLING
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
        sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
        sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
        sbr = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
        sbi = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
        plane_size = tl.cast(batch_size, tl.int64) * (height * width * modes)
        grad_a_imag_pointer = grad_source_real_a + plane_size
        grad_b_real_pointer = grad_source_real_a + 2 * plane_size
        grad_b_imag_pointer = grad_source_real_a + 3 * plane_size
        tl.store(grad_source_real_a + offset, gr * l0r + gi * l0i + gr * l2r - gi * l2i, mask=active)
        tl.store(
            grad_a_imag_pointer + offset, -gi * l0r + gr * l0i + gi * l2r + gr * l2i, mask=active
        )
        tl.store(
            grad_b_real_pointer + offset_b, gr * l1r + gi * l1i + gr * l3r - gi * l3i, mask=active
        )
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
        pole_mask = (x_group < scan_width) & (mode_vector < modes)
        tl.atomic_add(grad_decay_real + mode_vector, grad_dr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_decay_imag + mode_vector, grad_di, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_real + mode_vector, grad_gr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_imag + mode_vector, grad_gi, mask=pole_mask, sem="relaxed")

    return _path_backward_kernel


def _make_vback_half_hidden():
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
        batch_size = _structural_half_hidden(line_count // width)
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
        valid_y = (y < height) & (mode < modes)
        x_offset = source_x_a * modes + mode
        x_offset_b = source_x_b * modes + mode
        y_offset = y * modes + mode
        vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
        vx_negative = tl.load(variance_x + width * modes + x_offset_b, mask=valid_mode, other=0.0)
        vy_pos_fwd = tl.load(variance_y + y_offset, mask=valid_y, other=0.0)
        pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos_fwd, epsilon))
        pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos_fwd, epsilon))
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
        r0 = _prep_chan(par, pa, active, 0, COARSE)
        r1 = _prep_chan(pbr, pb, active, 1, COARSE)
        i0 = _prep_chan(pai, pa, active, 0, COARSE)
        i1 = _prep_chan(pbi, pb, active, 1, COARSE)
        vy_neg_fwd = tl.load(variance_y + height * modes + y_offset, mask=valid_y, other=0.0)
        na = tl.rsqrt(tl.maximum(vx_positive * vy_neg_fwd, epsilon))
        nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg_fwd, epsilon))
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
        r2 = _prep_chan(nar, na, active, 2, COARSE)
        r3 = _prep_chan(nbr, nb, active, 3, COARSE)
        i2 = _prep_chan(nai, na, active, 2, COARSE)
        i3 = _prep_chan(nbi, nb, active, 3, COARSE)
        path_height: tl.constexpr = 64 // 2 if COARSE else 64
        path_tokens: tl.constexpr = path_height
        packed_path = _pack_once16(r0, r1, r2, r3, i0, i1, i2, i3, BM, path_tokens)
        path_coordinate = tl.arange(0, 16)
        hidden_coordinate = tl.arange(0, 16)
        h8_coord = tl.arange(0, 8)
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
        pre8 = _take_low8(tl.dot(packed_path, active_input_weight), BM, path_tokens)
        bias8 = tl.load(
            path_input_bias + mode_vector[:, None] * packed_path_hidden + h8_coord[None, :],
            mask=(mode_vector[:, None] < modes) & (h8_coord[None, :] < packed_path_hidden),
            other=0.0,
        ).to(tl.bfloat16)
        pre8 = (pre8 + bias8[:, None, :]).to(tl.bfloat16)
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
        pre8_f = pre8.to(tl.float32)
        sig8 = tl.sigmoid(pre8_f)
        hidden8 = (pre8_f * sig8).to(tl.bfloat16)
        hidden16 = _pad_high8(hidden8, BM, path_tokens)
        _output_weight_bias_vjp(
            active_grad_real,
            active_grad_imag,
            hidden16,
            mode_vector,
            path_coordinate,
            hidden_coordinate,
            packed_path_hidden,
            modes,
            grad_path_output_weight,
            grad_path_output_bias,
        )
        pre8_f2 = pre8.to(tl.float32)
        sig8_2 = tl.sigmoid(pre8_f2)
        silu_deriv8 = sig8_2 * (1.0 + pre8_f2 * (1.0 - sig8_2))
        output_offset8 = mode_vector[:, None] * (2 * packed_path_hidden) + h8_coord[None, :]
        output_mask8 = (mode_vector[:, None] < modes) & (h8_coord[None, :] < packed_path_hidden)
        weight_real8 = (
            tl.load(path_output_weight + output_offset8, mask=output_mask8, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        weight_imag8 = (
            tl.load(
                path_output_weight + output_offset8 + packed_path_hidden,
                mask=output_mask8,
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        grad_hidden8 = (
            active_grad_real.to(tl.float32)[:, :, None] * weight_real8[:, None, :]
            + active_grad_imag.to(tl.float32)[:, :, None] * weight_imag8[:, None, :]
        )
        grad_pre8 = (grad_hidden8 * silu_deriv8).to(tl.bfloat16)
        tl.atomic_add(
            grad_path_input_bias + mode_vector[:, None] * packed_path_hidden + h8_coord[None, :],
            tl.sum(grad_pre8.to(tl.float32), axis=1),
            mask=output_mask8,
            sem="relaxed",
        )
        grad_preactivation = _pad_high8(grad_pre8, BM, path_tokens)
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
        grad_packed_path = tl.dot(
            grad_preactivation, tl.permute(active_input_weight, (0, 2, 1))
        ).to(tl.float32)
        if COARSE:
            grad_packed_path = grad_packed_path.to(tl.bfloat16).to(tl.float32)
        g0, g1, g2, g3, g4, g5, g6, g7 = _unpack_all_once(grad_packed_path, BM, COARSE)
        nr0 = g2.to(tl.bfloat16)
        nr1 = g3.to(tl.bfloat16)
        ni0 = g6.to(tl.bfloat16)
        ni1 = g7.to(tl.bfloat16)
        l0r, l0i, l1r, l1i = _prefix_pair(
            ar, -ai, g0 * pa, g4 * pa, g1 * pb, g5 * pb, active, True, DOUBLING
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
        sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
        sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
        sbr = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
        sbi = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
        plane_size = tl.cast(batch_size, tl.int64) * (height * width * modes)
        grad_a_imag_pointer = grad_source_real_a + plane_size
        grad_b_real_pointer = grad_source_real_a + 2 * plane_size
        grad_b_imag_pointer = grad_source_real_a + 3 * plane_size
        tl.store(grad_source_real_a + offset, gr * l0r + gi * l0i + gr * l2r - gi * l2i, mask=active)
        tl.store(
            grad_a_imag_pointer + offset, -gi * l0r + gr * l0i + gi * l2r + gr * l2i, mask=active
        )
        tl.store(
            grad_b_real_pointer + offset_b, gr * l1r + gi * l1i + gr * l3r - gi * l3i, mask=active
        )
        tl.store(
            grad_b_imag_pointer + offset_b, -gi * l1r + gr * l1i + gi * l3r + gr * l3i, mask=active
        )
        decay_norm = ar * ar + ai * ai
        inverse_decay = 1.0 / decay_norm
        positive_drive_ar, positive_drive_ai = (gr * sar - gi * sai, gr * sai + gi * sar)
        positive_drive_br, positive_drive_bi = (gr * sbr - gi * sbi, gr * sbi + gi * sbr)
        par, pai, pbr, pbi = _prefix_pair(
            ar,
            ai,
            positive_drive_ar,
            positive_drive_ai,
            positive_drive_br,
            positive_drive_bi,
            active,
            False,
            DOUBLING,
        )
        previous_0r = (ar * (par - positive_drive_ar) + ai * (pai - positive_drive_ai)) * inverse_decay
        previous_0i = (ar * (pai - positive_drive_ai) - ai * (par - positive_drive_ar)) * inverse_decay
        previous_1r = (ar * (pbr - positive_drive_br) + ai * (pbi - positive_drive_bi)) * inverse_decay
        previous_1i = (ar * (pbi - positive_drive_bi) - ai * (pbr - positive_drive_br)) * inverse_decay
        grad_dr = _sum_line(
            l0r * previous_0r + l0i * previous_0i + l1r * previous_1r + l1i * previous_1i, active
        )
        grad_di = _sum_line(
            -l0r * previous_0i + l0i * previous_0r - l1r * previous_1i + l1i * previous_1r, active
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
        negative_drive_ar, negative_drive_ai = (gr * sar + gi * sai, gr * sai - gi * sar)
        negative_drive_br, negative_drive_bi = (gr * sbr + gi * sbi, gr * sbi - gi * sbr)
        nar, nai, nbr, nbi = _prefix_pair(
            ar,
            -ai,
            negative_drive_ar,
            negative_drive_ai,
            negative_drive_br,
            negative_drive_bi,
            active,
            True,
            DOUBLING,
        )
        previous_2r = (ar * (nar - negative_drive_ar) - ai * (nai - negative_drive_ai)) * inverse_decay
        previous_2i = (ai * (nar - negative_drive_ar) + ar * (nai - negative_drive_ai)) * inverse_decay
        previous_3r = (ar * (nbr - negative_drive_br) - ai * (nbi - negative_drive_bi)) * inverse_decay
        previous_3i = (ai * (nbr - negative_drive_br) + ar * (nbi - negative_drive_bi)) * inverse_decay
        grad_dr = grad_dr + _sum_line(
            l2r * previous_2r + l2i * previous_2i + l3r * previous_3r + l3i * previous_3i, active
        )
        grad_di = grad_di + _sum_line(
            l2r * previous_2i - l2i * previous_2r + l3r * previous_3i - l3i * previous_3r, active
        )
        pole_mask = (x_group < scan_width) & (mode_vector < modes)
        tl.atomic_add(grad_decay_real + mode_vector, grad_dr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_decay_imag + mode_vector, grad_di, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_real + mode_vector, grad_gr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_imag + mode_vector, grad_gi, mask=pole_mask, sem="relaxed")

    return _path_backward_kernel


def _make_vback_unpack_prev(use_rows_vjp: bool = False):
    vjp_fn = _effective_output_vjp_rows if use_rows_vjp else _output_weight_bias_vjp
    tag_fn = _structural_unpack_rows if use_rows_vjp else _structural_unpack_prev

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
        batch_size = tag_fn(line_count // width)
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
        valid_y = (y < height) & (mode < modes)
        x_offset = source_x_a * modes + mode
        x_offset_b = source_x_b * modes + mode
        y_offset = y * modes + mode
        vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
        vx_negative = tl.load(variance_x + width * modes + x_offset_b, mask=valid_mode, other=0.0)
        vy_pos_fwd = tl.load(variance_y + y_offset, mask=valid_y, other=0.0)
        pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos_fwd, epsilon))
        pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos_fwd, epsilon))
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
        r0 = _prep_chan(par, pa, active, 0, COARSE)
        r1 = _prep_chan(pbr, pb, active, 1, COARSE)
        i0 = _prep_chan(pai, pa, active, 0, COARSE)
        i1 = _prep_chan(pbi, pb, active, 1, COARSE)
        vy_neg_fwd = tl.load(variance_y + height * modes + y_offset, mask=valid_y, other=0.0)
        na = tl.rsqrt(tl.maximum(vx_positive * vy_neg_fwd, epsilon))
        nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg_fwd, epsilon))
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
        r2 = _prep_chan(nar, na, active, 2, COARSE)
        r3 = _prep_chan(nbr, nb, active, 3, COARSE)
        i2 = _prep_chan(nai, na, active, 2, COARSE)
        i3 = _prep_chan(nbi, nb, active, 3, COARSE)
        path_height: tl.constexpr = 64 // 2 if COARSE else 64
        path_tokens: tl.constexpr = path_height
        paths8 = _pack_once8(r0, r1, r2, r3, i0, i1, i2, i3, BM, path_tokens)
        packed_path = _pad_high8(paths8, BM, path_tokens)
        path_coordinate = tl.arange(0, 16)
        hidden_coordinate = tl.arange(0, 16)
        h8_coord = tl.arange(0, 8)
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
        pre8 = _take_low8(tl.dot(packed_path, active_input_weight), BM, path_tokens)
        bias8 = tl.load(
            path_input_bias + mode_vector[:, None] * packed_path_hidden + h8_coord[None, :],
            mask=(mode_vector[:, None] < modes) & (h8_coord[None, :] < packed_path_hidden),
            other=0.0,
        ).to(tl.bfloat16)
        pre8 = (pre8 + bias8[:, None, :]).to(tl.bfloat16)
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
        pre8_f = pre8.to(tl.float32)
        sig8 = tl.sigmoid(pre8_f)
        hidden8 = (pre8_f * sig8).to(tl.bfloat16)
        silu_deriv8 = (sig8 * (1.0 + pre8_f * (1.0 - sig8))).to(tl.bfloat16)
        hidden16 = _pad_high8(hidden8, BM, path_tokens)
        vjp_fn(
            active_grad_real,
            active_grad_imag,
            hidden16,
            mode_vector,
            path_coordinate,
            hidden_coordinate,
            packed_path_hidden,
            modes,
            grad_path_output_weight,
            grad_path_output_bias,
        )
        output_offset8 = mode_vector[:, None] * (2 * packed_path_hidden) + h8_coord[None, :]
        output_mask8 = (mode_vector[:, None] < modes) & (h8_coord[None, :] < packed_path_hidden)
        weight_real8 = (
            tl.load(path_output_weight + output_offset8, mask=output_mask8, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        weight_imag8 = (
            tl.load(
                path_output_weight + output_offset8 + packed_path_hidden,
                mask=output_mask8,
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        grad_hidden8 = (
            active_grad_real.to(tl.float32)[:, :, None] * weight_real8[:, None, :]
            + active_grad_imag.to(tl.float32)[:, :, None] * weight_imag8[:, None, :]
        )
        grad_pre8 = (grad_hidden8 * silu_deriv8.to(tl.float32)).to(tl.bfloat16)
        tl.atomic_add(
            grad_path_input_bias + mode_vector[:, None] * packed_path_hidden + h8_coord[None, :],
            tl.sum(grad_pre8.to(tl.float32), axis=1),
            mask=output_mask8,
            sem="relaxed",
        )
        grad_preactivation = _pad_high8(grad_pre8, BM, path_tokens)
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
        grad_packed_path = tl.dot(
            grad_preactivation, tl.permute(active_input_weight, (0, 2, 1))
        ).to(tl.float32)
        if COARSE:
            grad_packed_path = grad_packed_path.to(tl.bfloat16).to(tl.float32)
        g0, g1, g2, g3, g4, g5, g6, g7 = _unpack_all_once(grad_packed_path, BM, COARSE)
        nr0 = g2.to(tl.bfloat16)
        nr1 = g3.to(tl.bfloat16)
        ni0 = g6.to(tl.bfloat16)
        ni1 = g7.to(tl.bfloat16)
        l0r, l0i, l1r, l1i = _prefix_pair(
            ar, -ai, g0 * pa, g4 * pa, g1 * pb, g5 * pb, active, True, DOUBLING
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
        sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
        sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
        sbr = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
        sbi = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
        plane_size = tl.cast(batch_size, tl.int64) * (height * width * modes)
        grad_a_imag_pointer = grad_source_real_a + plane_size
        grad_b_real_pointer = grad_source_real_a + 2 * plane_size
        grad_b_imag_pointer = grad_source_real_a + 3 * plane_size
        tl.store(grad_source_real_a + offset, gr * l0r + gi * l0i + gr * l2r - gi * l2i, mask=active)
        tl.store(
            grad_a_imag_pointer + offset, -gi * l0r + gr * l0i + gi * l2r + gr * l2i, mask=active
        )
        tl.store(
            grad_b_real_pointer + offset_b, gr * l1r + gi * l1i + gr * l3r - gi * l3i, mask=active
        )
        tl.store(
            grad_b_imag_pointer + offset_b, -gi * l1r + gr * l1i + gi * l3r + gr * l3i, mask=active
        )
        decay_norm = ar * ar + ai * ai
        inverse_decay = 1.0 / decay_norm
        positive_drive_ar, positive_drive_ai = (gr * sar - gi * sai, gr * sai + gi * sar)
        positive_drive_br, positive_drive_bi = (gr * sbr - gi * sbi, gr * sbi + gi * sbr)
        negative_drive_ar, negative_drive_ai = (gr * sar + gi * sai, gr * sai - gi * sar)
        negative_drive_br, negative_drive_bi = (gr * sbr + gi * sbi, gr * sbi - gi * sbr)
        if COARSE:
            par, pai, pbr, pbi = _prefix_pair(
                ar,
                ai,
                positive_drive_ar,
                positive_drive_ai,
                positive_drive_br,
                positive_drive_bi,
                active,
                False,
                DOUBLING,
            )
            nar, nai, nbr, nbi = _prefix_pair(
                ar,
                -ai,
                negative_drive_ar,
                negative_drive_ai,
                negative_drive_br,
                negative_drive_bi,
                active,
                True,
                DOUBLING,
            )
        else:
            u0, u1, u2, u3, u4, u5, u6, u7 = _unpack_8(paths8, BM, COARSE)
            par, pai, pbr, pbi = (
                u0.to(tl.float32) / pa,
                u4.to(tl.float32) / pa,
                u1.to(tl.float32) / pb,
                u5.to(tl.float32) / pb,
            )
            nar, nai, nbr, nbi = (
                u2.to(tl.float32) / na,
                u6.to(tl.float32) / na,
                u3.to(tl.float32) / nb,
                u7.to(tl.float32) / nb,
            )
        previous_0r = (ar * (par - positive_drive_ar) + ai * (pai - positive_drive_ai)) * inverse_decay
        previous_0i = (ar * (pai - positive_drive_ai) - ai * (par - positive_drive_ar)) * inverse_decay
        previous_1r = (ar * (pbr - positive_drive_br) + ai * (pbi - positive_drive_bi)) * inverse_decay
        previous_1i = (ar * (pbi - positive_drive_bi) - ai * (pbr - positive_drive_br)) * inverse_decay
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
        pole_mask = (x_group < scan_width) & (mode_vector < modes)
        tl.atomic_add(grad_decay_real + mode_vector, grad_dr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_decay_imag + mode_vector, grad_di, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_real + mode_vector, grad_gr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_imag + mode_vector, grad_gi, mask=pole_mask, sem="relaxed")

    return _path_backward_kernel


def _make_vback_col_loop2():
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
        STEPS: tl.constexpr = 2
        batch_size = _structural_col_loop2(line_count // width)
        scan_width = width // 2 if COARSE else width
        group_width = scan_width // STEPS
        tile = tl.program_id(0)
        mode_group = tile // GROUP
        line_group = tl.program_id(1) * GROUP + tile % GROUP
        if line_group >= batch_size * group_width:
            return
        batch = line_group // group_width
        x_base = (line_group - batch * group_width) * STEPS
        grad_decay_real += batch * modes
        grad_decay_imag += batch * modes
        grad_gamma_real += batch * modes
        grad_gamma_imag += batch * modes
        grad_path_input_weight += batch * modes * packed_path_hidden * 8
        grad_path_input_bias += batch * modes * packed_path_hidden
        grad_path_output_weight += batch * modes * 2 * packed_path_hidden
        grad_path_output_bias += batch * modes * 2
        y = tl.arange(0, 64)[:, None, None]
        mode_vector = mode_group * BM + tl.arange(0, BM)
        mode = mode_vector[None, None, :]
        valid_mode = mode < modes
        valid_y = (y < height) & valid_mode
        ar = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
        ai = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
        gr = tl.load(gamma_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
        gi = tl.load(gamma_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
        decay_norm = ar * ar + ai * ai
        inverse_decay = 1.0 / decay_norm
        y_offset = y * modes + mode
        vy_pos_fwd = tl.load(variance_y + y_offset, mask=valid_y, other=0.0)
        vy_neg_fwd = tl.load(variance_y + height * modes + y_offset, mask=valid_y, other=0.0)
        path_height: tl.constexpr = 64 // 2 if COARSE else 64
        path_tokens: tl.constexpr = path_height
        path_coordinate = tl.arange(0, 16)
        hidden_coordinate = tl.arange(0, 16)
        h8_coord = tl.arange(0, 8)
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
        bias8 = tl.load(
            path_input_bias + mode_vector[:, None] * packed_path_hidden + h8_coord[None, :],
            mask=(mode_vector[:, None] < modes) & (h8_coord[None, :] < packed_path_hidden),
            other=0.0,
        ).to(tl.bfloat16)
        output_offset8 = mode_vector[:, None] * (2 * packed_path_hidden) + h8_coord[None, :]
        output_mask8 = (mode_vector[:, None] < modes) & (h8_coord[None, :] < packed_path_hidden)
        weight_real8 = (
            tl.load(path_output_weight + output_offset8, mask=output_mask8, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        weight_imag8 = (
            tl.load(
                path_output_weight + output_offset8 + packed_path_hidden,
                mask=output_mask8,
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        acc_dr = tl.zeros((BM,), dtype=tl.float32)
        acc_di = tl.zeros((BM,), dtype=tl.float32)
        acc_gr = tl.zeros((BM,), dtype=tl.float32)
        acc_gi = tl.zeros((BM,), dtype=tl.float32)
        acc_b_in = tl.zeros((BM, 8), dtype=tl.float32)
        plane_size = tl.cast(batch_size, tl.int64) * (height * width * modes)
        grad_a_imag_pointer = grad_source_real_a + plane_size
        grad_b_real_pointer = grad_source_real_a + 2 * plane_size
        grad_b_imag_pointer = grad_source_real_a + 3 * plane_size

        for step in range(STEPS):
            x_group = x_base + step
            x = x_group + tl.arange(0, 1)[None, :, None]
            active = (x < scan_width) & valid_y
            source_x_a = 2 * x + 1 if COARSE else x
            source_x_b = 2 * x if COARSE else x
            offset = ((batch * height + y) * width + source_x_a) * modes + mode
            offset_b = offset - modes if COARSE else offset
            x_offset = source_x_a * modes + mode
            x_offset_b = source_x_b * modes + mode
            vx_positive = tl.load(variance_x + x_offset, mask=valid_mode, other=0.0)
            vx_negative = tl.load(
                variance_x + width * modes + x_offset_b, mask=valid_mode, other=0.0
            )
            pa = tl.rsqrt(tl.maximum(vx_positive * vy_pos_fwd, epsilon))
            pb = tl.rsqrt(tl.maximum(vx_negative * vy_pos_fwd, epsilon))
            na = tl.rsqrt(tl.maximum(vx_positive * vy_neg_fwd, epsilon))
            nb = tl.rsqrt(tl.maximum(vx_negative * vy_neg_fwd, epsilon))
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
            r0 = _prep_chan(par, pa, active, 0, COARSE)
            r1 = _prep_chan(pbr, pb, active, 1, COARSE)
            i0 = _prep_chan(pai, pa, active, 0, COARSE)
            i1 = _prep_chan(pbi, pb, active, 1, COARSE)
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
            r2 = _prep_chan(nar, na, active, 2, COARSE)
            r3 = _prep_chan(nbr, nb, active, 3, COARSE)
            i2 = _prep_chan(nai, na, active, 2, COARSE)
            i3 = _prep_chan(nbi, nb, active, 3, COARSE)
            paths8 = _pack_once8(r0, r1, r2, r3, i0, i1, i2, i3, BM, path_tokens)
            packed_path = _pad_high8(paths8, BM, path_tokens)
            pre8 = (
                _take_low8(tl.dot(packed_path, active_input_weight), BM, path_tokens)
                + bias8[:, None, :]
            ).to(tl.bfloat16)
            if COARSE:
                cell_y = tl.arange(0, path_height)[:, None, None]
                cell_x = x
                path_offset = (
                    (batch * (height // 2) + cell_y) * (width // 2) + cell_x
                ) * modes + mode
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
            pre8_f = pre8.to(tl.float32)
            sig8 = tl.sigmoid(pre8_f)
            hidden8 = (pre8_f * sig8).to(tl.bfloat16)
            silu_deriv8 = (sig8 * (1.0 + pre8_f * (1.0 - sig8))).to(tl.bfloat16)
            hidden16 = _pad_high8(hidden8, BM, path_tokens)
            _output_weight_bias_vjp(
                active_grad_real,
                active_grad_imag,
                hidden16,
                mode_vector,
                path_coordinate,
                hidden_coordinate,
                packed_path_hidden,
                modes,
                grad_path_output_weight,
                grad_path_output_bias,
            )
            grad_hidden8 = (
                active_grad_real.to(tl.float32)[:, :, None] * weight_real8[:, None, :]
                + active_grad_imag.to(tl.float32)[:, :, None] * weight_imag8[:, None, :]
            )
            grad_pre8 = (grad_hidden8 * silu_deriv8.to(tl.float32)).to(tl.bfloat16)
            acc_b_in += tl.sum(grad_pre8.to(tl.float32), axis=1)
            grad_preactivation = _pad_high8(grad_pre8, BM, path_tokens)
            path_input_weight_gradient = tl.dot(
                tl.permute(grad_preactivation, (0, 2, 1)), packed_path
            )
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
            grad_packed_path = tl.dot(
                grad_preactivation, tl.permute(active_input_weight, (0, 2, 1))
            ).to(tl.float32)
            if COARSE:
                grad_packed_path = grad_packed_path.to(tl.bfloat16).to(tl.float32)
            g0, g1, g2, g3, g4, g5, g6, g7 = _unpack_all_once(grad_packed_path, BM, COARSE)
            nr0 = g2.to(tl.bfloat16)
            nr1 = g3.to(tl.bfloat16)
            ni0 = g6.to(tl.bfloat16)
            ni1 = g7.to(tl.bfloat16)
            l0r, l0i, l1r, l1i = _prefix_pair(
                ar, -ai, g0 * pa, g4 * pa, g1 * pb, g5 * pb, active, True, DOUBLING
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
            sar = tl.load(source_real_a + offset, mask=active, other=0.0).to(tl.float32)
            sai = tl.load(source_imag_a + offset, mask=active, other=0.0).to(tl.float32)
            sbr = tl.load(source_real_b + offset_b, mask=active, other=0.0).to(tl.float32)
            sbi = tl.load(source_imag_b + offset_b, mask=active, other=0.0).to(tl.float32)
            tl.store(
                grad_source_real_a + offset, gr * l0r + gi * l0i + gr * l2r - gi * l2i, mask=active
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
            positive_drive_ar, positive_drive_ai = (gr * sar - gi * sai, gr * sai + gi * sar)
            positive_drive_br, positive_drive_bi = (gr * sbr - gi * sbi, gr * sbi + gi * sbr)
            negative_drive_ar, negative_drive_ai = (gr * sar + gi * sai, gr * sai - gi * sar)
            negative_drive_br, negative_drive_bi = (gr * sbr + gi * sbi, gr * sbi - gi * sbr)
            if COARSE:
                par, pai, pbr, pbi = _prefix_pair(
                    ar,
                    ai,
                    positive_drive_ar,
                    positive_drive_ai,
                    positive_drive_br,
                    positive_drive_bi,
                    active,
                    False,
                    DOUBLING,
                )
                nar, nai, nbr, nbi = _prefix_pair(
                    ar,
                    -ai,
                    negative_drive_ar,
                    negative_drive_ai,
                    negative_drive_br,
                    negative_drive_bi,
                    active,
                    True,
                    DOUBLING,
                )
            else:
                u0, u1, u2, u3, u4, u5, u6, u7 = _unpack_8(paths8, BM, COARSE)
                par, pai, pbr, pbi = (
                    u0.to(tl.float32) / pa,
                    u4.to(tl.float32) / pa,
                    u1.to(tl.float32) / pb,
                    u5.to(tl.float32) / pb,
                )
                nar, nai, nbr, nbi = (
                    u2.to(tl.float32) / na,
                    u6.to(tl.float32) / na,
                    u3.to(tl.float32) / nb,
                    u7.to(tl.float32) / nb,
                )
            previous_0r = (
                ar * (par - positive_drive_ar) + ai * (pai - positive_drive_ai)
            ) * inverse_decay
            previous_0i = (
                ar * (pai - positive_drive_ai) - ai * (par - positive_drive_ar)
            ) * inverse_decay
            previous_1r = (
                ar * (pbr - positive_drive_br) + ai * (pbi - positive_drive_bi)
            ) * inverse_decay
            previous_1i = (
                ar * (pbi - positive_drive_bi) - ai * (pbr - positive_drive_br)
            ) * inverse_decay
            previous_2r = (
                ar * (nar - negative_drive_ar) - ai * (nai - negative_drive_ai)
            ) * inverse_decay
            previous_2i = (
                ai * (nar - negative_drive_ar) + ar * (nai - negative_drive_ai)
            ) * inverse_decay
            previous_3r = (
                ar * (nbr - negative_drive_br) - ai * (nbi - negative_drive_bi)
            ) * inverse_decay
            previous_3i = (
                ai * (nbr - negative_drive_br) + ar * (nbi - negative_drive_bi)
            ) * inverse_decay
            acc_dr += _sum_line(
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
            acc_di += _sum_line(
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
            acc_gr += _sum_line(
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
            acc_gi += _sum_line(
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

        tl.atomic_add(
            grad_path_input_bias + mode_vector[:, None] * packed_path_hidden + h8_coord[None, :],
            acc_b_in,
            mask=output_mask8,
            sem="relaxed",
        )
        pole_mask = mode_vector < modes
        tl.atomic_add(grad_decay_real + mode_vector, acc_dr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_decay_imag + mode_vector, acc_di, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_real + mode_vector, acc_gr, mask=pole_mask, sem="relaxed")
        tl.atomic_add(grad_gamma_imag + mode_vector, acc_gi, mask=pole_mask, sem="relaxed")

    return _path_backward_kernel


def _make_vfwd_fast_pack():
    @triton.jit
    def _path_forward_kernel(
        AR,
        AI,
        GR,
        GI,
        SA,
        IA,
        SB,
        IB,
        VX,
        VY,
        WI,
        BI,
        WO,
        BO,
        OR,
        OI,
        H: tl.constexpr,
        W: tl.constexpr,
        M: tl.constexpr,
        EPS: tl.constexpr,
        BM: tl.constexpr,
        GROUP: tl.constexpr,
        DOUBLING: tl.constexpr,
    ):
        mode = _structural_fast_forward(tl.program_id(0) // GROUP * BM + tl.arange(0, BM))
        line = tl.program_id(1) * GROUP + tl.program_id(0) % GROUP
        x = line % W
        batch = line // W
        y = tl.arange(0, 64)[:, None, None]
        m = mode[None, None, :]
        mask = y < H
        off = ((batch * H + y) * W + x) * M + m
        ar = tl.load(AR + m)
        ai = tl.load(AI + m)
        gr = tl.load(GR + m)
        gi = tl.load(GI + m)
        sa = tl.load(SA + off, mask, 0).to(tl.float32)
        ia = tl.load(IA + off, mask, 0).to(tl.float32)
        sb = tl.load(SB + off, mask, 0).to(tl.float32)
        ib = tl.load(IB + off, mask, 0).to(tl.float32)
        r0, i0, r1, i1, r2, i2, r3, i3 = _scan_states(
            ar, ai, gr, gi, sa, ia, sb, ib, mask, DOUBLING
        )
        g0, g1, g2, g3 = _gains(VX, VY, x, x, y, m, H, W, M, EPS)
        packed = _pack_fwd_once(r0, r1, r2, r3, i0, i1, i2, i3, g0, g1, g2, g3, mask, BM)
        coord = tl.arange(0, 16)
        h8 = tl.arange(0, 8)
        offset = (mode[:, None, None] * 8 + coord[None, None, :]) * 8 + coord[None, :, None]
        weight = tl.load(
            WI + offset, (coord[None, None, :] < 8) & (coord[None, :, None] < 8), 0
        ).to(tl.bfloat16)
        bias8 = tl.load(BI + mode[:, None] * 8 + h8[None, :], True, 0).to(tl.bfloat16)
        pre8 = (
            (_take_low8(tl.dot(packed, weight), BM, 64) + bias8[:, None, :])
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        hidden = _pad_high8((pre8 * tl.sigmoid(pre8)).to(tl.bfloat16), BM, 64)
        wo = tl.load(
            WO + (mode[:, None, None] * 2 + coord[None, None, :]) * 8 + coord[None, :, None],
            (coord[None, :, None] < 8) & (coord[None, None, :] < 2),
            0,
        ).to(tl.bfloat16)
        bo = tl.load(BO + mode[:, None] * 2 + coord[None, :], coord[None, :] < 2, 0).to(tl.bfloat16)
        result = (tl.dot(hidden, wo) + bo[:, None, :]).to(tl.bfloat16)
        flat = tl.reshape(tl.permute(off, (2, 0, 1)), (BM, 64))
        tl.store(
            OR + flat,
            tl.sum(tl.where(coord[None, None, :] == 0, result, 0.0), 2),
            tl.arange(0, 64)[None, :] < H,
        )
        tl.store(
            OI + flat,
            tl.sum(tl.where(coord[None, None, :] == 1, result, 0.0), 2),
            tl.arange(0, 64)[None, :] < H,
        )

    return _path_forward_kernel


_recompute_prev_kernel = _make_vback_recompute_prev()
_fast_mlp_kernel = _make_vback_fast_mlp()
_half_hidden_kernel = _make_vback_half_hidden()
_unpack_prev_kernel = _make_vback_unpack_prev(use_rows_vjp=False)
_unpack_rows_kernel = _make_vback_unpack_prev(use_rows_vjp=True)
_col_loop2_kernel = _make_vback_col_loop2()
_fast_forward_kernel = _make_vfwd_fast_pack()

_BACKWARD_VARIANTS = {
    1: _recompute_prev_kernel,
    2: _fast_mlp_kernel,
    3: _half_hidden_kernel,
    4: _unpack_prev_kernel,
    5: _unpack_rows_kernel,
    6: _col_loop2_kernel,
}
_FORWARD_VARIANTS = {
    1: _fast_forward_kernel,
}


@cache
def structural_revision(parent_revision: str) -> str:
    digest = hashlib.sha256(vjp_revision(parent_revision).encode())
    for variant, kernel in sorted(_BACKWARD_VARIANTS.items()):
        digest.update(f"b{variant}".encode())
        digest.update(kernel.cache_key.encode())
    for variant, kernel in sorted(_FORWARD_VARIANTS.items()):
        digest.update(f"f{variant}".encode())
        digest.update(kernel.cache_key.encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def select_structural_scan(scope, base):
    selected = select_effective_vjp(scope, base)
    shape = dict(scope.shape_items)
    if (
        scope.execution_regime != "compiled"
        or shape.get("height") != 56
        or shape.get("width") != 56
    ):
        return selected
    op = shape.get("operation")
    if op == VERTICAL_BACKWARD:
        exact = replace(scope, kernel_revision=structural_revision(scope.kernel_revision))
        marker = validated_geometry(STRUCT_NAME, exact, BUNDLE)
        if (
            selected not in (_original_backward_kernel, _transpose_kernel, _rows_kernel)
            or marker is None
            or not marker.blocks
        ):
            return selected
        return _BACKWARD_VARIANTS.get(marker.blocks.get("VARIANT"), selected)
    if op == VERTICAL_FORWARD and shape.get("coarse") == 0:
        exact = replace(scope, kernel_revision=structural_revision(scope.kernel_revision))
        marker = validated_geometry(STRUCT_NAME, exact, BUNDLE)
        if base is None or not base.blocks or marker is None or not marker.blocks:
            return selected
        return _FORWARD_VARIANTS.get(marker.blocks.get("VARIANT"), selected)
    return selected


def structural_graph_pass(previous):
    register_kernel_specializer(PROFILE_NAME, select_structural_scan)
    signature = geometry_store_signature(STRUCT_NAME, BUNDLE, omit_empty=True)
    return _LayoutGraphPass(structural_revision(signature), previous) if signature else previous
