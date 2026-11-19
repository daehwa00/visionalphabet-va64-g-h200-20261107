"""Opt-in geometry and compile-cache identity for qualified r6 scan kernels."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

import triton

from .kernel_launch_config import LaunchGeometry, autotuned
from .path_backward_layout import _LayoutGraphPass

R6_SOURCES = (
    "r6_dispatch.py",
    "r6_associative.py",
    "r6_segment.py",
    "r6_horizontal.py",
    "r7_associative.py",
    "r7_direct.py",
    "r7_dispatch.py",
)


def active_r6_flags() -> dict[str, str]:
    flags = {
        name: os.environ.get(name, "")
        for name in ("LNET_R6_HBACK", "LNET_R6_CABWD", "LNET_R7_AFWD", "LNET_R7_DIRECT")
    }
    return {name: value for name, value in flags.items() if value == "1"}


def legacy_dispatch_source(path: Path) -> bytes:
    """Hash the original dispatch independently of opt-in sections."""

    def baseline(match: re.Match[bytes]) -> bytes:
        return b"".join(
            re.findall(rb"^ *# R6 LEGACY: (.*\n)", match[0], flags=re.MULTILINE)
        )

    return re.sub(
        rb"^ *# BEGIN R6 DISPATCH\n.*?^ *# END R6 DISPATCH\n",
        baseline,
        path.read_bytes(),
        flags=re.MULTILINE | re.DOTALL,
    )


def r6_graph_pass(previous):
    flags = active_r6_flags()
    if not flags:
        return previous
    digest = hashlib.sha256(repr(flags).encode())
    for name in (
        *R6_SOURCES,
        "product_scan_ops.py",
        "retained_path_dispatch.py",
        "path_backward_layout.py",
    ):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return _LayoutGraphPass(digest.hexdigest(), previous)


def associative_launch(height, width, batch, modes, *, coarse, backward=True):
    """Return a fixed launch after the caller checks the scan ABI."""
    if not backward or not coarse or modes != 160 or height != width:
        return None
    if height not in (14, 28):
        return None
    from .r6_associative import coarse_associative_backward

    scan_width = width // 2
    bh, group = (16, 4) if height == 14 else (32, 1)
    if batch * scan_width % group:
        return None
    geometry = LaunchGeometry.build(
        num_warps=2, blocks={"BLOCK_MODES": 8, "BLOCK_LINES": 1}
    )
    kernel = autotuned(
        coarse_associative_backward, "r6_cabwd", key=(), geometry=geometry
    )
    grid = (triton.cdiv(modes, 8) * group, batch * scan_width // group)
    return kernel, grid, bh, group, {}


def horizontal_launch(*, coarse):
    from .r6_horizontal import hback56

    geometry = LaunchGeometry.build(
        num_warps=4, blocks={"BM": 16 if coarse else 32, "GROUP": 1}
    )
    return autotuned(hback56, "r6_hback", key=(), geometry=geometry)
