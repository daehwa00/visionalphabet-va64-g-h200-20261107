"""Four-row complex prefix for the verified 64-row retained vertical adjoint.

The straight doubling kernel remains available at its original code/graph revision.
"""

from __future__ import annotations

from types import FunctionType
from pathlib import Path
from dataclasses import replace
from functools import cache
import hashlib

import triton
import triton.language as tl

from .direct_path_scan_kernels import _path_backward_kernel as _original_kernel
from .kernel_launch_config import (
    geometry_store_signature,
    register_kernel_specializer,
    validated_geometry,
)
from .path_backward_layout import _LayoutGraphPass
from .retained_path_dispatch import PROFILE_NAME, VERTICAL_BACKWARD


@triton.jit
def _split_rows(value, REVERSE: tl.constexpr):
    modes: tl.constexpr = value.shape[2]
    pairs = tl.permute(tl.reshape(value, (16, 2, 2, modes)), (0, 3, 1, 2))
    even, odd = tl.split(pairs)
    row0, row2 = tl.split(even)
    row1, row3 = tl.split(odd)
    if REVERSE:
        return row3, row2, row1, row0
    else:
        return row0, row1, row2, row3


@triton.jit
def _join_rows(row0, row1, row2, row3, REVERSE: tl.constexpr):
    modes: tl.constexpr = row0.shape[1]
    if REVERSE:
        tile = tl.join(tl.join(row3, row1), tl.join(row2, row0))
    else:
        tile = tl.join(tl.join(row0, row2), tl.join(row1, row3))
    return tl.reshape(tl.permute(tile, (0, 2, 3, 1)), (64, 1, modes))


@triton.jit
def _local_prefix(real, imag, ar, ai, REVERSE: tl.constexpr):
    r0, r1, r2, r3 = _split_rows(real, REVERSE)
    i0, i1, i2, i3 = _split_rows(imag, REVERSE)
    r1, i1 = ar * r0 - ai * i0 + r1, ar * i0 + ai * r0 + i1
    r2, i2 = ar * r1 - ai * i1 + r2, ar * i1 + ai * r1 + i2
    r3, i3 = ar * r2 - ai * i2 + r3, ar * i2 + ai * r2 + i3
    return r0, i0, r1, i1, r2, i2, r3, i3


@triton.jit
def _complete_prefix(ar, ai, local, last_r, last_i, idx, valid, REVERSE: tl.constexpr):
    cr = tl.where(valid, tl.gather(last_r, idx, 0), 0.0)
    ci = tl.where(valid, tl.gather(last_i, idx, 0), 0.0)
    r0 = ar * cr - ai * ci + local[0]
    i0 = ar * ci + ai * cr + local[1]
    a2r, a2i = ar * ar - ai * ai, ar * ai + ai * ar
    r1 = a2r * cr - a2i * ci + local[2]
    i1 = a2r * ci + a2i * cr + local[3]
    a3r, a3i = ar * a2r - ai * a2i, ar * a2i + ai * a2r
    r2 = a3r * cr - a3i * ci + local[4]
    i2 = a3r * ci + a3i * cr + local[5]
    return _join_rows(r0, r1, r2, last_r, REVERSE), _join_rows(i0, i1, i2, last_i, REVERSE)


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
    for k in tl.static_range(4):
        offset = 1 << k
        neighbor = group + offset if REVERSE else group - offset
        idx = tl.broadcast_to(neighbor & 15, (16, modes))
        valid = neighbor < 16 if REVERSE else neighbor >= 0
        ur, ui = tl.gather(xr, idx, 0), tl.gather(xi, idx, 0)
        uz, uj = tl.gather(zr, idx, 0), tl.gather(zi, idx, 0)
        xr, xi = (
            tl.where(valid, pr * ur - pi * ui + xr, xr),
            tl.where(valid, pr * ui + pi * ur + xi, xi),
        )
        zr, zi = (
            tl.where(valid, pr * uz - pi * uj + zr, zr),
            tl.where(valid, pr * uj + pi * uz + zi, zi),
        )
        pr, pi = pr * pr - pi * pi, pr * pi + pi * pr
    neighbor = group + 1 if REVERSE else group - 1
    idx = tl.broadcast_to(neighbor & 15, (16, modes))
    valid = neighbor < 16 if REVERSE else neighbor >= 0
    xr, xi = _complete_prefix(ar, ai, first, xr, xi, idx, valid, REVERSE)
    zr, zi = _complete_prefix(ar, ai, second, zr, zi, idx, valid, REVERSE)
    return xr, xi, zr, zi


@triton.jit
def _prefix_pair(ar, ai, xr, xi, zr, zi, active, REVERSE: tl.constexpr, DOUBLING: tl.constexpr):
    # The selector accepts only doubling profiles. AOT's mutation-analysis trace
    # may present placeholder constexpr values before the real kernel is lowered.
    return _segment_four(
        ar,
        ai,
        tl.where(active, xr, 0.0),
        tl.where(active, xi, 0.0),
        tl.where(active, zr, 0.0),
        tl.where(active, zi, 0.0),
        REVERSE,
    )


def _bind_prefix():
    # Use one canonical adjoint body and independent immutable JIT namespaces.
    # Keep the original function and its transitive cache key intact on uncalibrated devices.
    fn = _original_kernel.fn
    # Bind the single scan dependency in a private globals dictionary. The code,
    # pointer ABI and all other dependencies come from the canonical Triton body.
    impl = FunctionType(
        fn.__code__,
        {**fn.__globals__, "_prefix_pair": _prefix_pair},
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    impl.__annotations__ = fn.__annotations__.copy()
    impl.__kwdefaults__ = fn.__kwdefaults__
    impl.__module__ = __name__
    return triton.jit(impl)


_path_backward_kernel = _bind_prefix()

SEGMENT_NAME = "retained_four_row_prefix"
BUNDLE = Path(__file__).with_name("validated_segmented_scans.json")


@cache
def segment_revision(parent_revision):
    h = hashlib.sha256(parent_revision.encode())
    h.update(_path_backward_kernel.cache_key.encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def select_segment(scope, base):
    shape = dict(scope.shape_items)
    if shape.get("operation") != VERTICAL_BACKWARD or scope.execution_regime != "compiled":
        return None
    # Keep direct and coarse, compilation options, precision, GPU and batch exact.
    scoped = replace(scope, kernel_revision=segment_revision(scope.kernel_revision))
    verified = validated_geometry(SEGMENT_NAME, scoped, BUNDLE)
    if (
        base is None
        or verified is None
        or verified != base
        or base.blocks.get("DOUBLING") != 1
        or shape.get("height") != 56
    ):
        return None
    return _path_backward_kernel


def segment_graph_pass(previous):
    register_kernel_specializer(PROFILE_NAME, select_segment)
    signature = geometry_store_signature(SEGMENT_NAME, BUNDLE, omit_empty=True)
    if not signature:
        return previous
    return _LayoutGraphPass(segment_revision(signature), previous)
