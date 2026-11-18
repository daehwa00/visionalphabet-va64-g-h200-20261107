"""Terminal mean-direction energy readout, using the unchanged D4 scan kernels."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor
from torch.library import triton_op
from torch.utils.checkpoint import checkpoint

from .pac_product_scan_contracts import (
    DEFAULT_EPSILON,
    ComplexField,
    ProductGainNormalization,
    gain_kind,
    gain_normalization,
)
from .pac_product_scan_normalization import static_product_scan_auxiliary
from .pac_product_scan_ops import (
    DESCRIPTOR_FORWARD_LAUNCH_NAME,
    _launch_product_scan4_forward,
    _product_scan_descriptor4_backward_op,
)
from .pac_product_scan_reference import _product_scans_from_tables_reference
from .pac_triton_bidirectional_product_scan import pac_triton_bidirectional_product_scan

Pole = tuple[Tensor, Tensor, Tensor, Tensor]


class _Context(Protocol):
    saved_tensors: tuple[Tensor, ...]
    epsilon: float
    kind: int

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def mean_direction_energy(real: Tensor, imag: Tensor) -> Tensor:
    """Reduce B,H,W,D,P states; cast before squaring and keep the gradient."""
    if real.shape != imag.shape or real.ndim != 5:
        raise ValueError("mean energy requires matching BH W D P tensors")
    return torch.log1p((real.float().square() + imag.float().square()).mean((1, 2, 3)))


def _reference(*tensors: Tensor, epsilon: float, kind: int) -> Tensor:
    pole = tuple(tensors[:4])
    source_a, source_b = tuple(tensors[4:6]), tuple(tensors[6:8])
    variance_x = tensors[8]
    parts = []
    for real, imag, variance in _product_scans_from_tables_reference(
        pole, source_a, source_b, variance_x
    ):
        if gain_normalization(kind) == "global":
            active_variance = variance.mean((1, 2), keepdim=True)
        else:
            active_variance = variance
        energy = (
            real.float().square() + imag.float().square()
        ) / active_variance.detach().clamp_min(epsilon)
        parts.append(energy.mean((1, 2)))
    return torch.log1p(torch.stack(parts, dim=1).mean(1))


@triton_op("lnet::pac_mean_energy_descriptor", mutates_args={})
def _mean_op(
    dr: Tensor,
    di: Tensor,
    gr: Tensor,
    gi: Tensor,
    ar: Tensor,
    ai: Tensor,
    br: Tensor,
    bi: Tensor,
    vx: Tensor,
    vy: Tensor,
    inverse: Tensor,
    epsilon: float,
    kind: int,
) -> Tensor:
    if not ar.is_cuda:
        return _reference(
            dr, di, gr, gi, ar, ai, br, bi, vx, vy, inverse, epsilon=epsilon, kind=kind
        )
    return _launch_product_scan4_forward(
        dr,
        di,
        gr,
        gi,
        ar,
        ai,
        br,
        bi,
        vx,
        vy,
        inverse,
        epsilon,
        kind,
        emit_coarse=False,
        mean_direction_energy=True,
        launch_name=DESCRIPTOR_FORWARD_LAUNCH_NAME,
    )[2]


def _setup(ctx: _Context, inputs: tuple, output: Tensor) -> None:
    *tensors, epsilon, kind = inputs
    ctx.epsilon, ctx.kind = epsilon, kind
    ctx.save_for_backward(*tensors, output)


def _backward(ctx: _Context, gradient: Tensor | None) -> tuple[Tensor | None, ...]:
    *tensors, descriptor = ctx.saved_tensors
    if gradient is None:
        return (None,) * 13
    if not tensors[4].is_cuda:
        active = tuple(t.detach().requires_grad_() for t in tensors[:8])
        with torch.enable_grad():
            out = _reference(*active, *tensors[8:], epsilon=ctx.epsilon, kind=ctx.kind)
            gradients = torch.autograd.grad(out, active, gradient)
    else:
        height, width = tensors[4].shape[1:3]
        factor = gradient * torch.exp(-descriptor) / (4 * height * width)
        factor = factor[:, None, :].expand(-1, 4, -1).reshape(gradient.shape[0], -1).contiguous()
        gradients = _product_scan_descriptor4_backward_op(
            *tensors,
            factor,
            ctx.epsilon,
            ctx.kind,
        )
    return *gradients, None, None, None, None, None


torch.library.register_autograd("lnet::pac_mean_energy_descriptor", _backward, setup_context=_setup)


def mean_energy_product_scan(
    pole_x: Pole,
    pole_y: Pole,
    source: ComplexField,
    *,
    gain_normalization: ProductGainNormalization = "pointwise",
    memory_policy: str = "retain",
    epsilon: float = DEFAULT_EPSILON,
) -> Tensor:
    """Compute all four scans and average normalized energies before log1p."""
    if memory_policy not in ("retain", "recompute"):
        raise ValueError("invalid scan memory policy")

    def run(*inputs: Tensor) -> Tensor:
        px, py, drive = tuple(inputs[:4]), tuple(inputs[4:8]), tuple(inputs[8:])
        ar, ai, br, bi = pac_triton_bidirectional_product_scan(px, drive)
        kind = gain_kind(gain_normalization)
        vx, vy, inverse = static_product_scan_auxiliary(px, py, ar, epsilon=epsilon, gain_kind=kind)
        return _mean_op(*py, ar, ai, br, bi, vx, vy, inverse, epsilon, kind)

    inputs = (*pole_x, *pole_y, *source)
    if (
        memory_policy == "recompute"
        and torch.is_grad_enabled()
        and any(t.requires_grad for t in inputs)
    ):
        return checkpoint(run, *inputs, use_reentrant=False, preserve_rng_state=False)
    return run(*inputs)
