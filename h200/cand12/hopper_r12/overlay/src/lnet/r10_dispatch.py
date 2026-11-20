"""Opt-in launch policy for segmented vertical forward."""

from dataclasses import replace
import os

from .kernel_launch_config import LaunchGeometry


def vertical_forward_geometry(batch, height, width, modes):
    if os.environ.get("LNET_R10_VFWD") != "1" or (height, width, modes) != (56, 56, 160):
        return None
    lines = batch * width
    if not 0 < 4 * lines * height * modes < 2**31 or lines % 4 or lines // 4 > 65535:
        return None
    from .r10_vertical import r10_vfwd56

    geometry = LaunchGeometry.build(
        num_warps=2, blocks={"BM": 8, "GROUP": 4, "DOUBLING": 0},
    )
    return replace(geometry, implementation=r10_vfwd56)
