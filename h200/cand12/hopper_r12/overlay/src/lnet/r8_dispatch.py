"""Opt-in launch policy for r8 BF16 scans."""

import os

import torch
# BEGIN R10 DISPATCH
import triton
# END R10 DISPATCH

from .kernel_launch_config import LaunchGeometry, autotuned


def horizontal_shape_supported(batch, height, width, modes):
    return (
        height == width
        and height in (14, 28, 56)
        and modes == 160
        and 0 < batch * height <= 65535
        and 0 < 4 * batch * height * width * modes < 2**31
    )


def _horizontal_launch(source, kernel, warps):
    batch, height, _, modes = source.shape
    geometry = LaunchGeometry.build(num_warps=warps, blocks={"BLOCK_MODES": 32, "BLOCK_LINES": 1})
    tuner = autotuned(kernel, kernel.__name__, key=(), geometry=geometry)
    return tuner, (modes // 32, batch * height)


def horizontal_forward_launch(source):
    # BEGIN R10 DISPATCH
    batch, height, _, modes = source.shape
    if (
        os.environ.get("LNET_R10_HFWD56_SEG4") == "1"
        and source.is_cuda
        and source.dtype == torch.bfloat16
        and height == 56
        and horizontal_shape_supported(*source.shape)
    ):
        from .r10_horizontal import r10_hfwd56_seg4

        geometry = LaunchGeometry.build(num_warps=4, num_stages=3, blocks={"BLOCK_MODES": 32})
        kernel = autotuned(r10_hfwd56_seg4, "r10_hfwd56_seg4", key=(), geometry=geometry)
        return kernel, (batch * height, triton.cdiv(modes, 32))
    # END R10 DISPATCH
    if (
        os.environ.get("LNET_R8_HFWD") != "1"
        or not source.is_cuda
        or source.dtype != torch.bfloat16
        or not horizontal_shape_supported(*source.shape)
        or source.shape[1] == 14
    ):
        return None
    from .r8_horizontal import r8_horizontal_forward

    warps = 4 if source.shape[1] == 56 else 2
    return _horizontal_launch(source, r8_horizontal_forward, warps)


def horizontal_backward_launch(source, *, saved_states, packed_gradient, coarse):
    if (
        os.environ.get("LNET_R8_HBACK") != "1"
        or not source.is_cuda
        or source.dtype != torch.bfloat16
        or not saved_states
        or not packed_gradient
        or not horizontal_shape_supported(*source.shape)
        or source.shape[1] == 56
        or (source.shape[1] == 14 and coarse)
    ):
        return None
    from .r8_horizontal import r8_horizontal_backward

    warps = 4 if source.shape[1] == 28 else 2
    return _horizontal_launch(source, r8_horizontal_backward, warps)
