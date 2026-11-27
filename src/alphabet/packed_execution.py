"""Shared packed-real space-to-depth reduction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def weighted_s2d_packed(source: Tensor, weight: Tensor) -> Tensor:
    """Reduce physical 2x2 cells directly, without an expanded S2D copy.

    The reshape is a view. Inductor fuses multiplication and the four-cell
    reduction; there is no permute/contiguous materialization of 4C channels.
    """
    batch, height, width, channels = source.shape
    if height % 2 == 0 and width % 2 == 0 and weight.shape == (channels, 4):
        cells = source.reshape(batch, height // 2, 2, width // 2, 2, channels)
        coefficients = weight.transpose(0, 1).reshape(1, 1, 2, 1, 2, channels)
        return (cells * coefficients).sum(dim=(2, 4))
    modes = channels // 2
    if height % 2 or width % 2 or channels % 2 or weight.shape != (modes, 4):
        raise ValueError("packed S2D carry requires even spatial sizes and C-by-4 weights")
    cells = source.reshape(batch, height // 2, 2, width // 2, 2, 2, modes)
    coefficients = weight.transpose(0, 1).reshape(1, 1, 2, 1, 2, 1, modes)
    return (cells * coefficients).sum(dim=(2, 4)).flatten(-2)
