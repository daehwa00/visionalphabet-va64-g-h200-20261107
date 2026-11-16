"""Only terminal readout: mean(2*s*conj(u)-abs(u)**2), without normalization.

Use updated linear states and actual vertical writes gamma_y*horizontal_state.
Reuse the bidirectional scan on transposed fields for the vertical axis.
Output all four directional real parts followed by all imaginary parts.
"""
import torch
from .pac_product_scan_reference import bidirectional_product_scan_reference
from .pac_triton_bidirectional_product_scan import pac_triton_bidirectional_product_scan


def _scan(pole, source):
    return (pac_triton_bidirectional_product_scan(pole, source) if source[0].is_cuda
            else bidirectional_product_scan_reference(pole, source))


def _vertical_correlations(pole, source):
    real, imag = (v.float() for v in source)
    states = _scan(pole, (real.transpose(1, 2).contiguous(), imag.transpose(1, 2).contiguous()))
    gr, gi = (v.float().reshape(1, 1, 1, -1) for v in pole[2:])
    pairs = []
    for reverse in (False, True):
        gain_imag = -gi if reverse else gi
        ur, ui = gr * real - gain_imag * imag, gr * imag + gain_imag * real
        sr, si = (v.transpose(1, 2) for v in states[2 * int(reverse):2 * int(reverse) + 2])
        re = (2 * (sr * ur + si * ui) - ur.square() - ui.square()).mean((1, 2))
        im = (2 * (si * ur - sr * ui)).mean((1, 2))
        pairs.append((re, im))
    return pairs


def complex_vertical_correlation(pole, source, *, reverse=False):
    return _vertical_correlations(pole, source)[int(reverse)]


def complex_correlation_descriptor(pole_x, pole_y, drive, *, gain_normalization=None):
    # Compatibility keyword does not apply an energy normalization to raw G.
    horizontal = _scan(pole_x, drive.chunk(2, dim=-1))
    pairs = [_vertical_correlations(pole_y, h) for h in (horizontal[:2], horizontal[2:])]
    return torch.cat([pairs[h][rev][component] for component in (0, 1) for rev in (0, 1) for h in (0, 1)], dim=-1)
