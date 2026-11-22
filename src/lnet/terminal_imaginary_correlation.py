"""Terminal correlations from updated vertical states and their writes."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from .general_cuda_scans import (
    general_scans_enabled,
    horizontal_forward,
    try_general_horizontal_forward,
)
from .product_scan_reference import bidirectional_product_scan_reference
from .triton_bidirectional_product_scan import triton_bidirectional_product_scan

PoleTensors = tuple[Tensor, Tensor, Tensor, Tensor]
SourcePair = tuple[Tensor, Tensor]
StateQuad = tuple[Tensor, Tensor, Tensor, Tensor]


def _scan(pole: PoleTensors, source: SourcePair) -> StateQuad:
    if source[0].is_cuda and general_scans_enabled(forward=True):
        if torch.compiler.is_compiling():
            # Let the custom op plan its launch with real CUDA tensors.
            states = horizontal_forward(list(pole), *source)
        else:
            states = try_general_horizontal_forward(pole, source)
        if states is not None:
            return cast("StateQuad", states.unbind(0))
    return (
        triton_bidirectional_product_scan(pole, source)
        if source[0].is_cuda
        else bidirectional_product_scan_reference(pole, source)
    )


def _vertical_correlations(
    pole: PoleTensors,
    source: SourcePair,
    *,
    normalize: bool = False,
    forward_vertical_only: bool = False,
) -> list[tuple[Tensor, Tensor]]:
    real, imag = (v.float() for v in source)
    states = _scan(pole, (real.transpose(1, 2).contiguous(), imag.transpose(1, 2).contiguous()))
    gr, gi = (v.float().reshape(1, 1, 1, -1) for v in pole[2:])
    pairs: list[tuple[Tensor, Tensor]] = []
    for reverse in ((False,) if forward_vertical_only else (False, True)):
        gain_imag = -gi if reverse else gi
        ur, ui = gr * real - gain_imag * imag, gr * imag + gain_imag * real
        sr, si = (v.transpose(1, 2) for v in states[2 * int(reverse) : 2 * int(reverse) + 2])
        re = (2 * (sr * ur + si * ui) - ur.square() - ui.square()).mean((1, 2))
        im = (2 * (si * ur - sr * ui)).mean((1, 2))
        if normalize:
            denominator = (ur.square() + ui.square()).mean((1, 2)).clamp_min(1e-6).detach()
            re, im = re / denominator, im / denominator
        pairs.append((re, im))
    return pairs


def complex_correlation_descriptor(
    pole_x: PoleTensors,
    pole_y: PoleTensors,
    drive: Tensor,
    *,
    gain_normalization: str | None = None,
    normalize: bool = False,
    forward_vertical_only: bool = False,
) -> Tensor:
    """Return the selected directions as real parts followed by imaginary parts."""
    del gain_normalization
    real_drive, imag_drive = drive.chunk(2, dim=-1)
    horizontal = _scan(pole_x, (real_drive, imag_drive))
    pairs = [
        _vertical_correlations(
            pole_y,
            (horizontal[h], horizontal[h + 1]),
            normalize=normalize,
            forward_vertical_only=forward_vertical_only,
        )
        for h in (0, 2)
    ]
    return torch.cat(
        [
            pairs[h][rev][component]
            for component in (0, 1)
            for rev in range(len(pairs[0]))
            for h in (0, 1)
        ],
        dim=-1,
    )
