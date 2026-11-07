"""GAP of signed, gain-normalized terminal D4 complex features."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .pac_product_scan_contracts import DEFAULT_EPSILON, ComplexField, ProductGainNormalization
from .pac_product_scan_reference import static_variance_tables_reference
from .pac_triton_bidirectional_product_scan import pac_triton_bidirectional_product_scan

Pole = tuple[Tensor, Tensor, Tensor, Tensor]


def signed_feature_pool(real: Tensor, imag: Tensor) -> Tensor:
    """B,H,W,D,P -> B,(D,real/imag,P), without energy or log transforms."""
    if real.shape != imag.shape or real.ndim != 5 or real.shape[-2] != 4:
        raise ValueError("signed GAP requires matching BHW4P tensors")
    return torch.stack((real.float().mean((1, 2)), imag.float().mean((1, 2))), 2).flatten(1)


def feature_pool_product_scan(
    pole_x: Pole,
    pole_y: Pole,
    source: ComplexField,
    *,
    gain_normalization: ProductGainNormalization = "pointwise",
    memory_policy: str = "retain",
    epsilon: float = DEFAULT_EPSILON,
) -> Tensor:
    """Reuse differentiable axis scans; retain FP32 vertical states before GAP.

    Direction order is (+y,+x), (+y,-x), (-y,+x), (-y,-x).
    Reference-gain gradients are stopped, as in the energy readout.
    """
    if gain_normalization not in ("pointwise", "global"):
        raise ValueError("invalid gain normalization")
    if memory_policy not in ("retain", "recompute"):
        raise ValueError("invalid scan memory policy")

    def run(*inputs: Tensor) -> Tensor:
        px, py, drive = tuple(inputs[:4]), tuple(inputs[4:8]), tuple(inputs[8:])
        batch, height, width, modes = drive[0].shape
        ar, ai, br, bi = pac_triton_bidirectional_product_scan(px, drive)
        vertical_source = tuple(
            torch.cat(pair, 0).float().transpose(1, 2).contiguous()
            for pair in ((ar, br), (ai, bi))
        )
        yr, yi, rr, ri = pac_triton_bidirectional_product_scan(py, vertical_source)
        real = torch.stack((yr[:batch], yr[batch:], rr[:batch], rr[batch:]), -2).transpose(1, 2)
        imag = torch.stack((yi[:batch], yi[batch:], ri[:batch], ri[batch:]), -2).transpose(1, 2)
        vx, vy = static_variance_tables_reference(px, py, width, height)
        variances = torch.stack(
            tuple(vy[y, :, None, :] * vx[x, None, :, :] for y, x in ((0, 0), (0, 1), (1, 0), (1, 1))),
            -2,
        ).unsqueeze(0)
        if gain_normalization == "global":
            variances = variances.mean((1, 2), keepdim=True)
        inverse = variances.detach().clamp_min(epsilon).rsqrt()
        return signed_feature_pool(real * inverse, imag * inverse)

    inputs = (*pole_x, *pole_y, *source)
    if memory_policy == "recompute" and torch.is_grad_enabled() and any(t.requires_grad for t in inputs):
        return checkpoint(run, *inputs, use_reentrant=False, preserve_rng_state=False)
    return run(*inputs)
