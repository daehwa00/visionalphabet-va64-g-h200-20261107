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
)


# BEGIN R10 DISPATCH
R10_SOURCES = (
    "r10_dispatch.py",
    "r10_vertical.py",
    "r10_horizontal.py",
    "r10_rms.py",
    "r10_segmented_coarse.py",
)


# END R10 DISPATCH
# BEGIN R11 DISPATCH
R11_SOURCES = (
    "r11_dispatch.py",
    "r11_cuda_vscan.py",
    "r11_cuda_hscan.py",
)


# END R11 DISPATCH
# BEGIN R12 DISPATCH
R12_SOURCES = (
    "r12_cuda_scans.py",
    "r11_dispatch.py",
    "r11_cuda_vscan.py",
    "r11_cuda_hscan.py",
    "r6_dispatch.py",
    "product_scan_pipeline.py",
    "retained_path_dispatch.py",
)


def _pre_r12_source(path: Path) -> bytes:
    def baseline(match: re.Match[bytes]) -> bytes:
        return b"".join(re.findall(rb"^ *# R12 LEGACY: (.*\n)", match[0], flags=re.MULTILINE))

    return re.sub(
        rb"^ *# BEGIN R12 DISPATCH\n.*?^ *# END R12 DISPATCH\n",
        baseline,
        path.read_bytes(),
        flags=re.MULTILINE | re.DOTALL,
    )


# END R12 DISPATCH
def active_r6_flags() -> dict[str, str]:
    flags = {
        name: os.environ.get(name, "")
        for name in (
            "LNET_R6_HBACK", "LNET_R6_CABWD", "LNET_R7_AFWD", "LNET_R7_DIRECT",
            "LNET_R8_HFWD", "LNET_R8_HBACK", "LNET_R9_RMS_DW",
            # BEGIN R10 DISPATCH
            "LNET_R10_RMS_DW",
            "LNET_R10_VFWD",
            "LNET_R10_HFWD56_SEG4",
            "LNET_R10_RMS_DW56",
            "LNET_R10_TUNED_LAUNCH",
            "LNET_R10_SEG4PACK2",
            # END R10 DISPATCH
            # BEGIN R11 DISPATCH
            "LNET_R11_CUDA_VFWD",
            "LNET_R11_CUDA_VBACK",
            "LNET_R11_CUDA_HFWD",
            "LNET_R11_CUDA_HBACK",
            # END R11 DISPATCH
            # BEGIN R12 DISPATCH
            "LNET_R12_PACKED",
            "LNET_R12_PRECISE",
            "LNET_R12_INDEX32",
            # END R12 DISPATCH
        )
    }
    return {name: value for name, value in flags.items() if value == "1"}
# BEGIN R11 DISPATCH


def _pre_r11_source(path: Path) -> bytes:
    def baseline(match: re.Match[bytes]) -> bytes:
        return b"".join(re.findall(rb"^ *# R11 LEGACY: (.*\n)", match[0], flags=re.MULTILINE))

    return re.sub(
        rb"^ *# BEGIN R11 DISPATCH\n.*?^ *# END R11 DISPATCH\n",
        baseline,
        # BEGIN R12 DISPATCH
        # R12 LEGACY:         path.read_bytes(),
        _pre_r12_source(path),
        # END R12 DISPATCH
        flags=re.MULTILINE | re.DOTALL,
    )
# END R11 DISPATCH
# BEGIN R10 DISPATCH


def _pre_r10_source(path: Path) -> bytes:
    def baseline(match: re.Match[bytes]) -> bytes:
        return b"".join(re.findall(rb"^ *# R10 LEGACY: (.*\n)", match[0], flags=re.MULTILINE))

    return re.sub(
        rb"^ *# BEGIN R10 DISPATCH\n.*?^ *# END R10 DISPATCH\n",
        baseline,
        # BEGIN R11 DISPATCH
        # R11 LEGACY:         path.read_bytes(),
        _pre_r11_source(path),
        # END R11 DISPATCH
        flags=re.MULTILINE | re.DOTALL,
    )
# END R10 DISPATCH


def legacy_dispatch_source(path: Path) -> bytes:
    """Hash the original dispatch independently of opt-in sections."""

    def baseline(match: re.Match[bytes]) -> bytes:
        return b"".join(
            re.findall(rb"^ *# R6 LEGACY: (.*\n)", match[0], flags=re.MULTILINE)
        )

    return re.sub(
        rb"^ *# BEGIN R6 DISPATCH\n.*?^ *# END R6 DISPATCH\n",
        baseline,
        # BEGIN R10 DISPATCH
        # R10 LEGACY:         path.read_bytes(),
        _pre_r10_source(path),
        # END R10 DISPATCH
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
        # BEGIN R10 DISPATCH
        # R10 LEGACY:         digest.update(Path(__file__).with_name(name).read_bytes())
        digest.update(_pre_r10_source(Path(__file__).with_name(name)))
        # END R10 DISPATCH
    if "LNET_R8_HFWD" in flags or "LNET_R8_HBACK" in flags:
        # BEGIN R10 DISPATCH
        # R10 LEGACY:         digest.update(Path(__file__).with_name("triton_bidirectional_product_scan.py").read_bytes())
        digest.update(_pre_r10_source(Path(__file__).with_name("triton_bidirectional_product_scan.py")))
        # END R10 DISPATCH
    if "LNET_R9_RMS_DW" in flags:
        from .r9_rms import RMSScaleGradientPass

        previous = RMSScaleGradientPass(previous)
    # BEGIN R10 DISPATCH
    if any(name.startswith("LNET_R10_") for name in flags):
        for name in (
            *R10_SOURCES,
            "r6_dispatch.py",
            "r7_dispatch.py",
            "r8_dispatch.py",
            "r9_rms.py",
            "retained_path_dispatch.py",
            "triton_bidirectional_product_scan.py",
        ):
            # BEGIN R11 DISPATCH
            # R11 LEGACY:             digest.update(Path(__file__).with_name(name).read_bytes())
            digest.update(_pre_r11_source(Path(__file__).with_name(name)))
            # END R11 DISPATCH
    if "LNET_R10_RMS_DW" in flags:
        from .r10_rms import RMSYScaleGradientPass

        previous = RMSYScaleGradientPass(previous)
    if "LNET_R10_RMS_DW56" in flags:
        from .r10_rms import RMS56ScaleGradientPass

        previous = RMS56ScaleGradientPass(previous)
    # END R10 DISPATCH
    # BEGIN R11 DISPATCH
    if any(name.startswith("LNET_R11_") for name in flags):
        for name in (
            *R11_SOURCES,
            "r6_dispatch.py",
            "product_scan_ops.py",
            "retained_path_dispatch.py",
            "triton_bidirectional_product_scan.py",
        ):
            # BEGIN R12 DISPATCH
            # R12 LEGACY:             digest.update(Path(__file__).with_name(name).read_bytes())
            digest.update(_pre_r12_source(Path(__file__).with_name(name)))
            # END R12 DISPATCH
    # END R11 DISPATCH
    # BEGIN R12 DISPATCH
    if any(name.startswith("LNET_R12_") for name in flags):
        for name in R12_SOURCES:
            digest.update(Path(__file__).with_name(name).read_bytes())
    # END R12 DISPATCH
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
    # BEGIN R10 DISPATCH
    # R10 LEGACY:     if batch * scan_width % group:
    # R10 LEGACY:         return None
    # R10 LEGACY:     geometry = LaunchGeometry.build(
    # R10 LEGACY:         num_warps=2, blocks={"BLOCK_MODES": 8, "BLOCK_LINES": 1}
    # R10 LEGACY:     )
    if (
        batch * scan_width % group
        or batch * scan_width // group > 65535
        or not 0 < 4 * batch * height * width * modes < 2**31
    ):
        return None
    tuned = height == 28 and os.environ.get("LNET_R10_TUNED_LAUNCH") == "1"
    geometry = LaunchGeometry.build(
        num_warps=4 if tuned else 2,
        num_stages=2 if tuned else 1,
        blocks={"BLOCK_MODES": 8, "BLOCK_LINES": 1},
    )
    # END R10 DISPATCH
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
