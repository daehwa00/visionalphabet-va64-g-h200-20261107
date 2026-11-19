"""Opt-in geometry and compile-cache identity for qualified r6 scan kernels."""

from __future__ import annotations

from functools import cache
import hashlib
import json
import math
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
)


def active_r6_flags() -> dict[str, str]:
    flags = {
        name: os.environ.get(name, "")
        for name in ("LNET_R6_HBACK", "LNET_R6_CABWD", "LNET_R7_R5_ASSOC")
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


@cache
def _winner_data(path):
    return json.loads(Path(path).read_text()) if path else {}


def _winner(tag):
    return _winner_data(os.environ.get("LNET_R7_WINNERS", "")).get(tag, {})


def r6_graph_pass(previous):
    flags = active_r6_flags()
    if not flags:
        return previous
    digest = hashlib.sha256(repr(flags).encode())
    digest.update(json.dumps(_winner_data(os.environ.get("LNET_R7_WINNERS", "")), sort_keys=True).encode())
    for name in (
        *R6_SOURCES,
        "product_scan_ops.py",
        "retained_path_dispatch.py",
        "path_backward_layout.py",
        "effective_output_vjp.py",
    ):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return _LayoutGraphPass(digest.hexdigest(), previous)


def associative_launch(height, width, batch, modes, *, coarse, backward=True):
    """Return a fixed launch after the caller checks the scan ABI."""
    if not backward or not coarse or batch < 1 or modes != 160 or height != width:
        return None
    if height not in (14, 28):
        return None
    from .r6_associative import coarse_associative_backward

    scan_width = width // 2
    bh = 16 if height == 14 else 32
    group = math.gcd(4, batch * scan_width) if height == 14 else 1
    winner = _winner(f"H2-{height}-1")
    tile = winner.get("bm", 8)
    group = winner.get("group", group) if (batch * scan_width) % winner.get("group", group) == 0 else group
    geometry = LaunchGeometry.build(
        num_warps=winner.get("warps", 2), blocks={"BLOCK_MODES": tile, "BLOCK_LINES": 1}
    )
    kernel = autotuned(
        coarse_associative_backward, "r6_cabwd", key=(), geometry=geometry
    )
    grid = (triton.cdiv(modes, tile) * group, batch * scan_width // group)
    return kernel, grid, bh, group, {}


def horizontal_launch(*, coarse):
    from .r6_horizontal import hback56

    winner = _winner(f"H5-56-{int(coarse)}-backward")
    geometry = LaunchGeometry.build(
        num_warps=winner.get("warps", 4), blocks={"BM": winner.get("bm", 16 if coarse else 32), "GROUP": 1}
    )
    return autotuned(hback56, "r6_hback", key=(), geometry=geometry)
