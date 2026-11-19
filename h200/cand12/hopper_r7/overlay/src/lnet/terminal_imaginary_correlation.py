"""Only terminal readout: mean(2*s*conj(u)-abs(u)**2), without normalization.

Use updated linear states and actual vertical writes gamma_y*horizontal_state.
Reuse the bidirectional scan on transposed fields for the vertical axis.
Output all four directional real parts followed by all imaginary parts.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .product_scan_reference import bidirectional_product_scan_reference
from .triton_bidirectional_product_scan import triton_bidirectional_product_scan

PoleTensors = tuple[Tensor, Tensor, Tensor, Tensor]
SourcePair = tuple[Tensor, Tensor]
StateQuad = tuple[Tensor, Tensor, Tensor, Tensor]


def _scan(pole: PoleTensors, source: SourcePair) -> StateQuad:
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
) -> list[tuple[Tensor, Tensor]]:
    real, imag = (v.float() for v in source)
    states = _scan(pole, (real.transpose(1, 2).contiguous(), imag.transpose(1, 2).contiguous()))
    gr, gi = (v.float().reshape(1, 1, 1, -1) for v in pole[2:])
    pairs: list[tuple[Tensor, Tensor]] = []
    for reverse in (False, True):
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


def complex_vertical_correlation(
    pole: PoleTensors,
    source: SourcePair,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    return _vertical_correlations(pole, source)[int(reverse)]


def complex_correlation_descriptor(
    pole_x: PoleTensors,
    pole_y: PoleTensors,
    drive: Tensor,
    *,
    gain_normalization: str | None = None,
    normalize: bool = False,
) -> Tensor:
    # Compatibility keyword does not apply an energy normalization to raw G.
    del gain_normalization
    real_drive, imag_drive = drive.chunk(2, dim=-1)
    horizontal = _scan(pole_x, (real_drive, imag_drive))
    pairs = [
        _vertical_correlations(pole_y, (horizontal[0], horizontal[1]), normalize=normalize),
        _vertical_correlations(pole_y, (horizontal[2], horizontal[3]), normalize=normalize),
    ]
    return torch.cat(
        [pairs[h][rev][component] for component in (0, 1) for rev in (0, 1) for h in (0, 1)],
        dim=-1,
    )
