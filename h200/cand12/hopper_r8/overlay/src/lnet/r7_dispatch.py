"""Opt-in launch geometry for qualified r7 scans."""

import triton

from .kernel_launch_config import LaunchGeometry, autotuned

FORWARD_GEOMETRY = {
    (56, True): (64, 16, 1, 4, False),
    (28, False): (32, 8, 1, 4, True),
    (28, True): (32, 8, 1, 4, False),
    (14, False): (16, 16, 1, 4, False),
    (14, True): (16, 8, 4, 2, False),
}


def forward_launch(height, width, batch, modes, *, coarse):
    """Use the fixed geometry for a supported dense BF16 collapse."""
    if height != width or modes != 160 or (height, coarse) not in FORWARD_GEOMETRY:
        return None
    if not 0 < batch * height * width * modes < 2**31:
        return None
    from .r7_associative import r7_associative_forward

    block_height, block_modes, block_lines, warps, single_load = FORWARD_GEOMETRY[height, coarse]
    geometry = LaunchGeometry.build(
        num_warps=warps,
        blocks={"BLOCK_MODES": block_modes, "BLOCK_LINES": block_lines},
    )
    kernel = autotuned(r7_associative_forward, "r7_afwd", key=(), geometry=geometry)
    return kernel, block_height, {"SINGLE_LOAD": single_load}


def direct_backward_launch(height, width, batch, modes):
    """Use two neighboring columns per CTA when the line groups are complete."""
    if height != width or height not in (14, 28) or modes != 160:
        return None
    block_modes, block_lines, group = 4, 2, 4
    line_tiles = batch * triton.cdiv(width, block_lines)
    if line_tiles % group or not 0 < batch * height * width * modes * 4 < 2**31:
        return None
    from .r7_direct import r7_direct_backward

    geometry = LaunchGeometry.build(
        num_warps=2,
        blocks={"BLOCK_MODES": block_modes, "BLOCK_LINES": block_lines},
    )
    kernel = autotuned(r7_direct_backward, "r7_direct_backward", key=(), geometry=geometry)
    return kernel, triton.next_power_of_2(height), group
