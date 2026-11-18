import triton
import triton.language as tl
from .rows import _row_prefix


@triton.jit
def _prefix_pair(ar,ai,xr,xi,zr,zi,active,REVERSE:tl.constexpr,DOUBLING:tl.constexpr):
    return _row_prefix(ar,ai,tl.where(active,xr,0.0),tl.where(active,xi,0.0),
                         tl.where(active,zr,0.0),tl.where(active,zi,0.0),8, REVERSE)
