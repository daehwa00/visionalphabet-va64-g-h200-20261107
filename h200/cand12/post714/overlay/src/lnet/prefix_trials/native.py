import triton
import triton.language as tl
from ..segmented_path_scan import _local_prefix, _complete_prefix
from ..triton_product_scan_kernels import _compose_complex_pair


@triton.jit
def _segment_four(ar, ai, xr, xi, zr, zi, REVERSE: tl.constexpr):
    # Each four-row block contributes its terminal state and the same pole to the fourth power.
    modes: tl.constexpr = xr.shape[2]
    ar = tl.reshape(ar, (1, modes))
    ai = tl.reshape(ai, (1, modes))
    first = _local_prefix(xr, xi, ar, ai, REVERSE)
    second = _local_prefix(zr, zi, ar, ai, REVERSE)
    xr, xi, zr, zi = first[6], first[7], second[6], second[7]
    pr, pi = ar * ar - ai * ai, ar * ai + ai * ar
    pr, pi = pr * pr - pi * pi, pr * pi + pi * pr
    group = tl.arange(0, 16)[:, None]
    pr=tl.broadcast_to(pr,(16,modes));pi=tl.broadcast_to(pi,(16,modes))
    scan=tl.associative_scan((pr,pi,xr,xi,zr,zi),0,_compose_complex_pair,reverse=REVERSE)
    xr,xi,zr,zi=scan[2],scan[3],scan[4],scan[5]
    neighbor = group + 1 if REVERSE else group - 1
    idx = tl.broadcast_to(neighbor & 15, (16, modes))
    valid = neighbor < 16 if REVERSE else neighbor >= 0
    xr, xi = _complete_prefix(ar, ai, first, xr, xi, idx, valid, REVERSE)
    zr, zi = _complete_prefix(ar, ai, second, zr, zi, idx, valid, REVERSE)
    return xr, xi, zr, zi
