import triton
import triton.language as tl
from lnet.r6_segment import _segment, _segment_pair


@triton.jit
def bifwd_seg4(
    decay_real, decay_imag, gamma_real, gamma_imag, source_real, source_imag,
    positive_real, positive_imag, negative_real, negative_imag,
    height, width, line_count, modes,
    SOURCE_REAL_STRIDE: tl.constexpr, SOURCE_IMAG_STRIDE: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr, BLOCK_MODES: tl.constexpr, COMBINED_STATES: tl.constexpr,
    DUAL: tl.constexpr,
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
    if DUAL:
        rx = BLOCK_WIDTH - 1 - x
        reverse_active = (rx < width) & valid_lane
        reverse_offset = (line * width + rx) * modes + mode
        rr = tl.load(source_real + (line * width + rx) * SOURCE_REAL_STRIDE + mode,
                     reverse_active, 0).to(tl.float32)
        ri = tl.load(source_imag + (line * width + rx) * SOURCE_IMAG_STRIDE + mode,
                     reverse_active, 0).to(tl.float32)
        pr, pi, nr, ni = _segment_pair(
            ar, ai, gr * sr - gi * si, gr * si + gi * sr,
            gr * rr + gi * ri, gi * rr - gr * ri, False, False,
        )
        tl.store(positive_real + offset, pr, active)
        tl.store(positive_real + plane_size + offset, pi, active)
        tl.store(positive_real + 2 * plane_size + reverse_offset, nr, reverse_active)
        tl.store(positive_real + 3 * plane_size + reverse_offset, -ni, reverse_active)
    else:
        pr, pi = _segment(ar, ai, gr * sr - gi * si, gr * si + gi * sr, False)
        tl.store(positive_real + offset, pr, active)
        tl.store(positive_real + plane_size + offset, pi, active)
        nr, ni = _segment(ar, -ai, gr * sr + gi * si, gr * si - gi * sr, True)
        tl.store(positive_real + 2 * plane_size + offset, nr, active)
        tl.store(positive_real + 3 * plane_size + offset, ni, active)
