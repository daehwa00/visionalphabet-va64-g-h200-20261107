"""Four-token complex prefixes for r6."""
import triton
import triton.language as tl

from .triton_bidirectional_product_scan import _compose_complex
from .triton_product_scan_kernels import _compose_complex_pair


@triton.jit
def _split4(value, REVERSE: tl.constexpr):
    groups: tl.constexpr = value.shape[0] // 4
    lanes: tl.constexpr = value.numel // value.shape[0]
    pairs = tl.permute(tl.reshape(value, (groups, 2, 2, lanes)), (0, 3, 1, 2))
    even, odd = tl.split(pairs)
    row0, row2 = tl.split(even)
    row1, row3 = tl.split(odd)
    if REVERSE:
        return row3, row2, row1, row0
    return row0, row1, row2, row3


@triton.jit
def _join4(r0, r1, r2, r3, REVERSE: tl.constexpr):
    if REVERSE:
        tile = tl.join(tl.join(r3, r1), tl.join(r2, r0))
    else:
        tile = tl.join(tl.join(r0, r2), tl.join(r1, r3))
    return tl.reshape(tl.permute(tile, (0, 2, 3, 1)), (r0.shape[0] * 4, r0.shape[1]))


@triton.jit
def _powers(ar, ai):
    a2r, a2i = ar * ar - ai * ai, ar * ai + ai * ar
    a3r, a3i = ar * a2r - ai * a2i, ar * a2i + ai * a2r
    a4r, a4i = a2r * a2r - a2i * a2i, a2r * a2i + a2i * a2r
    return ar, ai, a2r, a2i, a3r, a3i, a4r, a4i


@triton.jit
def _local(real, imag, ar, ai, REVERSE: tl.constexpr):
    r0, r1, r2, r3 = _split4(real, REVERSE)
    i0, i1, i2, i3 = _split4(imag, REVERSE)
    r1, i1 = ar * r0 - ai * i0 + r1, ar * i0 + ai * r0 + i1
    r2, i2 = ar * r1 - ai * i1 + r2, ar * i1 + ai * r1 + i2
    r3, i3 = ar * r2 - ai * i2 + r3, ar * i2 + ai * r2 + i3
    return r0, i0, r1, i1, r2, i2, r3, i3


@triton.jit
def _complete(local, terminal_r, terminal_i, power, REVERSE: tl.constexpr, ENDPOINT: tl.constexpr):
    groups: tl.constexpr = terminal_r.shape[0]
    index = tl.arange(0, groups)[:, None]
    neighbor = index + 1 if REVERSE else index - 1
    gather = tl.broadcast_to(neighbor & (groups - 1), terminal_r.shape)
    valid = (neighbor >= 0) & (neighbor < groups)
    cr = tl.where(valid, tl.gather(terminal_r, gather, axis=0), 0.0)
    ci = tl.where(valid, tl.gather(terminal_i, gather, axis=0), 0.0)
    r1 = power[2] * cr - power[3] * ci + local[2]
    i1 = power[2] * ci + power[3] * cr + local[3]
    if ENDPOINT:
        r0, i0 = tl.zeros_like(r1), tl.zeros_like(i1)
        r2, i2 = r0, i0
    else:
        r0 = power[0] * cr - power[1] * ci + local[0]
        i0 = power[0] * ci + power[1] * cr + local[1]
        r2 = power[4] * cr - power[5] * ci + local[4]
        i2 = power[4] * ci + power[5] * cr + local[5]
    return _join4(r0, r1, r2, terminal_r, REVERSE), _join4(i0, i1, i2, terminal_i, REVERSE)


@triton.jit
def _segment(ar, ai, r, i, REVERSE: tl.constexpr):
    power = _powers(ar, ai)
    local = _local(r, i, ar, ai, REVERSE)
    last_r, last_i = local[6], local[7]
    scan = tl.associative_scan(
        (tl.broadcast_to(power[6], last_r.shape), tl.broadcast_to(power[7], last_r.shape), last_r, last_i),
        0, _compose_complex, reverse=REVERSE,
    )
    return _complete(local, scan[2], scan[3], power, REVERSE, False)


@triton.jit
def _segment_pair(ar, ai, r, i, s, t, REVERSE: tl.constexpr, ENDPOINT: tl.constexpr):
    time: tl.constexpr = r.shape[0]
    lanes: tl.constexpr = r.numel // time
    ar = tl.reshape(ar, (1, lanes))
    ai = tl.reshape(ai, (1, lanes))
    power = _powers(ar, ai)
    a = _local(r, i, ar, ai, REVERSE)
    b = _local(s, t, ar, ai, REVERSE)
    scan = tl.associative_scan(
        (tl.broadcast_to(power[6], a[6].shape), tl.broadcast_to(power[7], a[6].shape),
         a[6], a[7], b[6], b[7]),
        0, _compose_complex_pair, reverse=REVERSE,
    )
    out_r, out_i = _complete(a, scan[2], scan[3], power, REVERSE, ENDPOINT)
    out_s, out_t = _complete(b, scan[4], scan[5], power, REVERSE, ENDPOINT)
    return (tl.reshape(out_r, r.shape), tl.reshape(out_i, r.shape),
            tl.reshape(out_s, r.shape), tl.reshape(out_t, r.shape))
