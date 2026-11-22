from __future__ import annotations

import torch
from torch import Tensor


def pole_transition(
    damping: Tensor,
    frequency: Tensor,
    dt: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the real 2D representation of exp((-damping + i frequency) dt)."""
    step = dt.to(dtype=damping.dtype, device=damping.device) if isinstance(dt, Tensor) else dt
    scaled_decay = torch.exp(-damping * step)
    phase = frequency * step
    return scaled_decay * torch.cos(phase), scaled_decay * torch.sin(phase)


def discrete_pole(
    damping: Tensor,
    frequency: Tensor,
    dt: float | Tensor,
    threshold: float = 1.0e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    step = dt.to(dtype=damping.dtype, device=damping.device) if isinstance(dt, Tensor) else dt
    decay_real, decay_imag = pole_transition(damping, frequency, step)
    shifted_real = decay_real - 1.0
    shifted_imag = decay_imag
    pole_real = -damping
    pole_imag = frequency
    denominator = pole_real.square() + pole_imag.square()
    gamma_real = (shifted_real * pole_real + shifted_imag * pole_imag) / denominator
    gamma_imag = (shifted_imag * pole_real - shifted_real * pole_imag) / denominator
    small = torch.sqrt((pole_real * step).square() + (pole_imag * step).square()) < threshold
    return (
        decay_real,
        decay_imag,
        torch.where(small, torch.ones_like(gamma_real) * step, gamma_real),
        torch.where(small, torch.zeros_like(gamma_imag), gamma_imag),
    )
