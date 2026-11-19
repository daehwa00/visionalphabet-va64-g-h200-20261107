"""Four-token segmented forward scan for packed horizontal BF16 states."""

import triton
import triton.language as tl
from .r6_segment import _segment


@triton.jit
def r10_hfwd56_seg4(
    decay_real, decay_imag, gamma_real, gamma_imag, source_real, source_imag,
    positive_real, positive_imag, negative_real, negative_imag,
    height, width, line_count, modes,
    SOURCE_REAL_STRIDE: tl.constexpr, SOURCE_IMAG_STRIDE: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr, BLOCK_MODES: tl.constexpr, COMBINED_STATES: tl.constexpr,
):
    line = tl.program_id(0)
    x = tl.arange(0, BLOCK_WIDTH)[:, None]
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)[None, :]
    valid_lane = mode < modes
    active = (x < width) & valid_lane
    ar = tl.load(decay_real + mode, valid_lane, 0).to(tl.float32)
    ai = tl.load(decay_imag + mode, valid_lane, 0).to(tl.float32)
    gr = tl.load(gamma_real + mode, valid_lane, 0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, valid_lane, 0).to(tl.float32)
    sr = tl.load(source_real + (line * width + x) * SOURCE_REAL_STRIDE + mode, active, 0).to(tl.float32)
    si = tl.load(source_imag + (line * width + x) * SOURCE_IMAG_STRIDE + mode, active, 0).to(tl.float32)
    offset = (line * width + x) * modes + mode
    plane_size = tl.cast(line_count, tl.int64) * width * modes
    pr, pi = _segment(ar, ai, gr * sr - gi * si, gr * si + gi * sr, False)
    tl.store(positive_real + offset, pr, active)
    tl.store(positive_real + plane_size + offset, pi, active)
    nr, ni = _segment(ar, -ai, gr * sr + gi * si, gr * si - gi * sr, True)
    tl.store(positive_real + 2 * plane_size + offset, nr, active)
    tl.store(positive_real + 3 * plane_size + offset, ni, active)
