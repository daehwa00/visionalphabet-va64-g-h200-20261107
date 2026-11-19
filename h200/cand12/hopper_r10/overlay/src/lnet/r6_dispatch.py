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
    "r8_dispatch.py",
    "r8_horizontal.py",
    "r9_rms.py",
    "r10_horizontal.py",
    "r10_rms.py",
)


def active_r6_flags() -> dict[str, str]:
    flags = {
        name: os.environ.get(name, "")
        for name in (
            "LNET_R6_HBACK", "LNET_R6_CABWD", "LNET_R7_AFWD", "LNET_R7_DIRECT",
            "LNET_R8_HFWD", "LNET_R8_HBACK", "LNET_R9_RMS_DW",
            "LNET_R10_HFWD56_SEG4", "LNET_R10_RMS_DW56", "LNET_R10_TUNED_LAUNCH",
            "LNET_R10_SEG4PACK2",
        )
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
    if "LNET_R8_HFWD" in flags or "LNET_R8_HBACK" in flags:
        digest.update(Path(__file__).with_name("triton_bidirectional_product_scan.py").read_bytes())
    if "LNET_R9_RMS_DW" in flags:
        from .r9_rms import RMSScaleGradientPass

        previous = RMSScaleGradientPass(previous)
    if "LNET_R10_RMS_DW56" in flags:
        from .r10_rms import RMS56ScaleGradientPass

        previous = RMS56ScaleGradientPass(previous)
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
    if (
        batch * scan_width % group
        or batch * scan_width // group > 65535
        or not 0 < 4 * batch * height * width * modes < 2**31
    ):
        return None
    tuned = height == 28 and os.environ.get("LNET_R10_TUNED_LAUNCH") == "1"
    geometry = LaunchGeometry.build(
        num_warps=4 if tuned else 2, num_stages=2 if tuned else 1,
        blocks={"BLOCK_MODES": 8, "BLOCK_LINES": 1},
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
