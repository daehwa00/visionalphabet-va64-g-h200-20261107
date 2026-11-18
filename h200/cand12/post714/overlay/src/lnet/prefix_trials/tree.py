import triton
import triton.language as tl
from ..segmented_path_scan import _split_rows


@triton.jit
def _local_prefix(real, imag, ar, ai, REVERSE: tl.constexpr):
    r0, r1, r2, r3 = _split_rows(real, REVERSE)
    i0, i1, i2, i3 = _split_rows(imag, REVERSE)
    r1, i1 = ar * r0 - ai * i0 + r1, ar * i0 + ai * r0 + i1
    tail_r,tail_i=ar*r2-ai*i2+r3, ar*i2+ai*r2+i3
    r2,i2=ar*r1-ai*i1+r2, ar*i1+ai*r1+i2
    a2r,a2i=ar*ar-ai*ai,ar*ai+ai*ar
    r3,i3=a2r*r1-a2i*i1+tail_r,a2r*i1+a2i*r1+tail_i
    return r0, i0, r1, i1, r2, i2, r3, i3
