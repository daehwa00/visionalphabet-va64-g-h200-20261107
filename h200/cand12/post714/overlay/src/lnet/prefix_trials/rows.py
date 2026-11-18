"""Short complex recurrences within rows, then a scan over segment endpoints."""
import triton
import triton.language as tl



@triton.jit
def _split(values, ROWS: tl.constexpr, REVERSE: tl.constexpr):
    m: tl.constexpr = values.shape[2]
    if ROWS == 2:
        even, odd = tl.split(tl.permute(tl.reshape(values, (32, 2, m)), (0, 2, 1)))
        if REVERSE:
            return odd, even
        else:
            return even, odd
    else:
        tile = tl.permute(tl.reshape(values, (8, 2, 2, 2, m)), (0, 4, 1, 2, 3))
        even, odd = tl.split(tile)
        e0, e2 = tl.split(even)
        o1, o3 = tl.split(odd)
        a, e = tl.split(e0)
        b, f = tl.split(o1)
        c, g = tl.split(e2)
        d, h = tl.split(o3)
        if REVERSE:
            return h, g, f, e, d, c, b, a
        else:
            return a, b, c, d, e, f, g, h


@triton.jit
def _join(values, ROWS: tl.constexpr, REVERSE: tl.constexpr):
    m: tl.constexpr = values[0].shape[1]
    if ROWS == 2:
        if REVERSE:
            pair = tl.join(values[1], values[0])
        else:
            pair = tl.join(values[0], values[1])
        return tl.reshape(tl.permute(pair, (0, 2, 1)), (64, 1, m))
    else:
        if REVERSE:
            a,b,c,d,e,f,g,h = values[7],values[6],values[5],values[4],values[3],values[2],values[1],values[0]
        else:
            a,b,c,d,e,f,g,h = values
        even = tl.join(tl.join(a,e),tl.join(c,g))
        odd = tl.join(tl.join(b,f),tl.join(d,h))
        return tl.reshape(tl.permute(tl.join(even,odd), (0,2,3,4,1)), (64,1,m))


@triton.jit
def _local(values_r, values_i, ar, ai, ROWS: tl.constexpr, REVERSE: tl.constexpr):
    rs = _split(values_r, ROWS, REVERSE)
    ims = _split(values_i, ROWS, REVERSE)
    real = (rs[0],)
    imag = (ims[0],)
    for j in tl.static_range(1, ROWS):
        rr = ar * real[j-1] - ai * imag[j-1] + rs[j]
        ii = ar * imag[j-1] + ai * real[j-1] + ims[j]
        real += (rr,)
        imag += (ii,)
    return real, imag


@triton.jit
def _restore(real, imag, end_r, end_i, ar, ai, idx, valid, ROWS: tl.constexpr, REVERSE: tl.constexpr):
    cr = tl.where(valid, tl.gather(end_r, idx, 0), 0.0)
    ci = tl.where(valid, tl.gather(end_i, idx, 0), 0.0)
    pr, pi = ar, ai
    out_r = (pr * cr - pi * ci + real[0],)
    out_i = (pr * ci + pi * cr + imag[0],)
    for j in tl.static_range(1, ROWS-1):
        pr, pi = ar * pr - ai * pi, ai * pr + ar * pi
        out_r += (pr * cr - pi * ci + real[j],)
        out_i += (pr * ci + pi * cr + imag[j],)
    return _join(out_r + (end_r,), ROWS, REVERSE), _join(out_i + (end_i,), ROWS, REVERSE)


@triton.jit
def _row_prefix(ar, ai, xr, xi, zr, zi, ROWS: tl.constexpr, REVERSE: tl.constexpr):
    m: tl.constexpr = xr.shape[2]
    groups: tl.constexpr = 64 // ROWS
    ar, ai = tl.reshape(ar,(1,m)), tl.reshape(ai,(1,m))
    ax, bx = _local(xr, xi, ar, ai, ROWS, REVERSE)
    az, bz = _local(zr, zi, ar, ai, ROWS, REVERSE)
    xr, xi, zr, zi = ax[ROWS-1], bx[ROWS-1], az[ROWS-1], bz[ROWS-1]
    pr, pi = ar * ar - ai * ai, ar * ai + ai * ar
    if ROWS == 8:
        pr, pi = pr * pr - pi * pi, pr * pi + pi * pr
        pr, pi = pr * pr - pi * pi, pr * pi + pi * pr
    group = tl.arange(0, groups)[:,None]
    for k in tl.static_range(5 if ROWS == 2 else 3):
        neighbor = group + (1 << k) if REVERSE else group - (1 << k)
        idx = tl.broadcast_to(neighbor & (groups-1), (groups,m))
        valid = neighbor < groups if REVERSE else neighbor >= 0
        ur,ui = tl.gather(xr,idx,0),tl.gather(xi,idx,0)
        uz,uj = tl.gather(zr,idx,0),tl.gather(zi,idx,0)
        xr,xi = tl.where(valid,pr*ur-pi*ui+xr,xr),tl.where(valid,pr*ui+pi*ur+xi,xi)
        zr,zi = tl.where(valid,pr*uz-pi*uj+zr,zr),tl.where(valid,pr*uj+pi*uz+zi,zi)
        pr,pi = pr*pr-pi*pi,pr*pi+pi*pr
    neighbor = group+1 if REVERSE else group-1
    idx=tl.broadcast_to(neighbor & (groups-1),(groups,m))
    valid=neighbor<groups if REVERSE else neighbor>=0
    xr,xi=_restore(ax,bx,xr,xi,ar,ai,idx,valid,ROWS,REVERSE)
    zr,zi=_restore(az,bz,zr,zi,ar,ai,idx,valid,ROWS,REVERSE)
    return xr,xi,zr,zi
