"""Horizontal BF16 scans with adjacent mode tiles scheduled together."""
import triton
import triton.language as tl

from .triton_bidirectional_product_scan import _compose_complex


@triton.jit
def r8_horizontal_forward(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    source_real,
    source_imag,
    positive_real,
    positive_imag,
    negative_real,
    negative_imag,
    height: int,
    width: int,
    line_count: int,
    modes: int,
    SOURCE_REAL_STRIDE: tl.constexpr,
    SOURCE_IMAG_STRIDE: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    COMBINED_STATES: tl.constexpr,
    BLOCK_LINES: tl.constexpr,
) -> None:
    # Adjacent CTAs cover neighboring BF16 channels on the same line.
    lane = tl.arange(0, BLOCK_LINES * BLOCK_MODES)[None, :]
    line = tl.program_id(1) * BLOCK_LINES + lane // BLOCK_MODES
    mode = tl.program_id(0) * BLOCK_MODES + lane % BLOCK_MODES
    x = tl.arange(0, BLOCK_WIDTH)[:, None]
    valid_mode = (line < line_count) & (mode < modes)
    active = valid_mode & (x < width)
    offset = (line * width + x) * modes + mode
    ar = tl.load(decay_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    source_real_offset = (line * width + x) * SOURCE_REAL_STRIDE + mode
    source_imag_offset = (line * width + x) * SOURCE_IMAG_STRIDE + mode
    sr = tl.load(source_real + source_real_offset, mask=active, other=0.0).to(tl.float32)
    si = tl.load(source_imag + source_imag_offset, mask=active, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    scan_ai = tl.where(active, ai, 0.0)
    positive = tl.associative_scan(
        (scan_ar, scan_ai, gr * sr - gi * si, gr * si + gi * sr),
        axis=0,
        combine_fn=_compose_complex,
    )
    pr, pi = positive[2], positive[3]
    tl.store(positive_real + offset, pr, mask=active)
    if COMBINED_STATES:
        plane_size = tl.cast(line_count, tl.int64) * (width * modes)
        tl.store(positive_real + plane_size + offset, pi, mask=active)
    else:
        tl.store(positive_imag + offset, pi, mask=active)
    # The reverse scan reuses the two source loads after the positive stores.
    negative = tl.associative_scan(
        (scan_ar, -scan_ai, gr * sr + gi * si, gr * si - gi * sr),
        axis=0,
        combine_fn=_compose_complex,
        reverse=True,
    )
    nr, ni = negative[2], negative[3]
    if COMBINED_STATES:
        tl.store(positive_real + 2 * plane_size + offset, nr, mask=active)
        tl.store(positive_real + 3 * plane_size + offset, ni, mask=active)
    else:
        tl.store(negative_real + offset, nr, mask=active)
        tl.store(negative_imag + offset, ni, mask=active)


@triton.jit
def r8_horizontal_backward(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    source_real,
    source_imag,
    positive_real,
    positive_imag,
    negative_real,
    negative_imag,
    grad_positive_real,
    grad_positive_imag,
    grad_negative_real,
    grad_negative_imag,
    grad_decay_real,
    grad_decay_imag,
    grad_gamma_real,
    grad_gamma_imag,
    grad_source_real,
    grad_source_imag,
    height: int,
    width: int,
    line_count: int,
    modes: int,
    SOURCE_REAL_STRIDE: tl.constexpr,
    SOURCE_IMAG_STRIDE: tl.constexpr,
    GRAD_SOURCE_STRIDE: tl.constexpr,
    PACKED_SOURCE_GRADIENT: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    RECOMPUTE_STATES: tl.constexpr,
    STORAGE_KIND: tl.constexpr,
    BLOCK_LINES: tl.constexpr,
    COARSE_EVEN_ODD: tl.constexpr = 0,
) -> None:
    # Adjacent CTAs cover neighboring BF16 channels on the same line.
    lane = tl.arange(0, BLOCK_LINES * BLOCK_MODES)[None, :]
    line = tl.program_id(1) * BLOCK_LINES + lane // BLOCK_MODES
    mode = tl.program_id(0) * BLOCK_MODES + lane % BLOCK_MODES
    x = tl.arange(0, BLOCK_WIDTH)[:, None]
    valid_mode = (line < line_count) & (mode < modes)
    active = valid_mode & (x < width)
    offset = (line * width + x) * modes + mode
    ar = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    pos_mask = active & ((x & 1) == 1) if COARSE_EVEN_ODD else active
    neg_mask = active & ((x & 1) == 0) if COARSE_EVEN_ODD else active
    gpr = tl.load(grad_positive_real + offset, mask=pos_mask, other=0.0).to(tl.float32)
    gpi = tl.load(grad_positive_imag + offset, mask=pos_mask, other=0.0).to(tl.float32)
    gnr = tl.load(grad_negative_real + offset, mask=neg_mask, other=0.0).to(tl.float32)
    gni = tl.load(grad_negative_imag + offset, mask=neg_mask, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    positive_adjoint = tl.associative_scan(
        (scan_ar, tl.where(active, -ai, 0.0), gpr, gpi),
        axis=0,
        combine_fn=_compose_complex,
        reverse=True,
    )
    negative_adjoint = tl.associative_scan(
        (scan_ar, tl.where(active, ai, 0.0), gnr, gni),
        axis=0,
        combine_fn=_compose_complex,
    )
    plr, pli = positive_adjoint[2], positive_adjoint[3]
    nlr, nli = negative_adjoint[2], negative_adjoint[3]
    source_real_offset = (line * width + x) * SOURCE_REAL_STRIDE + mode
    source_imag_offset = (line * width + x) * SOURCE_IMAG_STRIDE + mode
    sr = tl.load(source_real + source_real_offset, mask=active, other=0.0).to(tl.float32)
    si = tl.load(source_imag + source_imag_offset, mask=active, other=0.0).to(tl.float32)
    positive_previous_offset = offset - modes
    negative_previous_offset = offset + modes
    ppr = tl.load(
        positive_real + positive_previous_offset,
        mask=active & (x > 0),
        other=0.0,
    ).to(tl.float32)
    ppi = tl.load(
        positive_imag + positive_previous_offset,
        mask=active & (x > 0),
        other=0.0,
    ).to(tl.float32)
    npr = tl.load(
        negative_real + negative_previous_offset,
        mask=active & (x < width - 1),
        other=0.0,
    ).to(tl.float32)
    npi = tl.load(
        negative_imag + negative_previous_offset,
        mask=active & (x < width - 1),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        grad_source_real + (line * width + x) * GRAD_SOURCE_STRIDE + mode,
        plr * gr + pli * gi + nlr * gr - nli * gi,
        mask=active,
    )
    if PACKED_SOURCE_GRADIENT:
        tl.store(
            grad_source_real + (line * width + x) * GRAD_SOURCE_STRIDE + mode + modes,
            -plr * gi + pli * gr + nlr * gi + nli * gr,
            mask=active,
        )
    else:
        tl.store(
            grad_source_imag + (line * width + x) * GRAD_SOURCE_STRIDE + mode,
            -plr * gi + pli * gr + nlr * gi + nli * gr,
            mask=active,
        )
    cdr = tl.sum(
        tl.where(active, plr * ppr + pli * ppi + nlr * npr + nli * npi, 0.0),
        axis=0,
    )
    cdi = tl.sum(
        tl.where(active, -plr * ppi + pli * ppr + nlr * npi - nli * npr, 0.0),
        axis=0,
    )
    cgr = tl.sum(
        tl.where(active, plr * sr + pli * si + nlr * sr + nli * si, 0.0),
        axis=0,
    )
    cgi = tl.sum(
        tl.where(active, -plr * si + pli * sr + nlr * si - nli * sr, 0.0),
        axis=0,
    )
    tl.atomic_add(grad_decay_real + mode, cdr[None, :], mask=valid_mode, sem="relaxed")
    tl.atomic_add(grad_decay_imag + mode, cdi[None, :], mask=valid_mode, sem="relaxed")
    tl.atomic_add(grad_gamma_real + mode, cgr[None, :], mask=valid_mode, sem="relaxed")
    tl.atomic_add(grad_gamma_imag + mode, cgi[None, :], mask=valid_mode, sem="relaxed")
