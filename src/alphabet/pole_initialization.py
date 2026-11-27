"""Reusable pole damping initialization transforms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch import nn

    from .complex_scan_stage import ComplexScanStage


DAMPING_QUANTILE_POWER = 0.40
NONTERMINAL_GEOMETRIC_DAMPING_RANGE = (0.06, 0.57)
TERMINAL_GEOMETRIC_DAMPING_RANGE = (0.18, 0.85)
TERMINAL_DAMPING_MAX = 2.0
POLE_ORIENTATIONS = 8
POLE_MAXIMUM_PHASES = (
    math.pi * 0.75,
    math.pi * 0.70,
    math.pi * 0.60,
    math.pi * 0.65,
)
POLE_FREQUENCY_SCALES = (0.96, 0.86, 0.88, 0.90)
POLE_DAMPING_SCALES = (1.15, 1.20, 1.20, 1.15)
POLE_DAMPING_OFFSETS = (0, 3, 7, 10, 10, 7, 3, 0)
POLE_LOG_ANISOTROPY = (-0.20, -0.14, -0.08, -0.03, 0.03, 0.08, 0.14, 0.20)
POLE_DAMPING_INDEX_MULTIPLIER = 5


def damping(stage: ComplexScanStage, axis: str) -> Tensor:
    logits = getattr(stage, f"damping_logits_{axis}")
    return stage.damping_min + (stage.damping_max - stage.damping_min) * logits.sigmoid()


def damping_logits(stage: ComplexScanStage, values: Tensor) -> Tensor:
    ratio = ((values - stage.damping_min) / (stage.damping_max - stage.damping_min)).clamp(
        1.0e-5,
        1.0 - 1.0e-5,
    )
    return torch.logit(ratio)


def target_geometric_damping(
    geometric: Tensor,
    *,
    lower: float,
    upper: float,
) -> Tensor:
    log_geometric = geometric.log()
    span = log_geometric.max() - log_geometric.min()
    if not torch.isfinite(span) or span <= 0:
        raise RuntimeError("short-memory initialization requires a non-degenerate damping atlas")
    quantile = (log_geometric - log_geometric.min()) / span
    shaped_quantile = quantile.pow(DAMPING_QUANTILE_POWER)
    return torch.exp(math.log(lower) + (math.log(upper) - math.log(lower)) * shaped_quantile)


def install_short_damping(stage: ComplexScanStage, *, terminal: bool) -> None:
    damping_x = damping(stage, "x").detach()
    damping_y = damping(stage, "y").detach()
    geometric = torch.sqrt(damping_x * damping_y)
    log_anisotropy = 0.5 * torch.log(damping_x / damping_y)
    lower, upper = (
        TERMINAL_GEOMETRIC_DAMPING_RANGE if terminal else NONTERMINAL_GEOMETRIC_DAMPING_RANGE
    )
    target_geometric = target_geometric_damping(
        geometric,
        lower=lower,
        upper=upper,
    )
    target_x = target_geometric * log_anisotropy.exp()
    target_y = target_geometric * (-log_anisotropy).exp()
    if terminal:
        stage.damping_max = TERMINAL_DAMPING_MAX
    if target_x.min() <= stage.damping_min or target_y.min() <= stage.damping_min:
        raise RuntimeError("short-memory damping crossed the configured minimum")
    if target_x.max() >= stage.damping_max or target_y.max() >= stage.damping_max:
        raise RuntimeError("short-memory damping crossed the configured maximum")
    with torch.no_grad():
        stage.damping_logits_x.copy_(damping_logits(stage, target_x))
        stage.damping_logits_y.copy_(damping_logits(stage, target_y))


def _nearest_coprime_multiplier(radial_levels: int, preferred: int) -> int:
    if radial_levels <= 0 or preferred <= 0:
        raise ValueError("pole atlas dimensions must be positive")
    for distance in range(radial_levels + preferred):
        candidates = (preferred,) if distance == 0 else (preferred - distance, preferred + distance)
        for candidate in candidates:
            if candidate > 0 and math.gcd(candidate, radial_levels) == 1:
                return candidate
    raise RuntimeError("failed to find a coprime damping permutation")


def _canonical_phase_atlas(
    modes: int,
    maximum_phase: float,
    *,
    like: Tensor,
) -> tuple[Tensor, Tensor]:
    if modes <= 0 or modes % POLE_ORIENTATIONS:
        raise ValueError("Pole poles require complete canonical-8 radial groups")
    radial_levels = modes // POLE_ORIENTATIONS
    radial = torch.logspace(
        math.log10(maximum_phase / 8.0),
        math.log10(maximum_phase),
        radial_levels,
        dtype=like.dtype,
        device=like.device,
    ).repeat_interleave(POLE_ORIENTATIONS)
    orientation = torch.linspace(
        0.0,
        math.pi / 2.0,
        POLE_ORIENTATIONS,
        dtype=like.dtype,
        device=like.device,
    ).repeat(radial_levels)
    return radial * torch.cos(orientation), radial * torch.sin(orientation)


def install_calibrated_initialization(
    stage: ComplexScanStage,
    maximum_phase: float,
    frequency_scale: float,
    damping_scale: float,
) -> None:
    phase_x, phase_y = _canonical_phase_atlas(
        stage.modes,
        maximum_phase * frequency_scale,
        like=stage.phase_x,
    )
    radial_levels = stage.modes // POLE_ORIENTATIONS
    values = torch.logspace(
        math.log10(0.04 * damping_scale),
        math.log10(0.35 * damping_scale),
        radial_levels,
        dtype=stage.damping_logits_x.dtype,
        device=stage.damping_logits_x.device,
    ).repeat_interleave(POLE_ORIENTATIONS)
    ratio = ((values - stage.damping_min) / (stage.damping_max - stage.damping_min)).clamp(
        1.0e-4,
        1.0 - 1.0e-4,
    )
    logits = torch.logit(ratio)
    with torch.no_grad():
        stage.phase_x.copy_(phase_x)
        stage.phase_y.copy_(phase_y)
        stage.damping_logits_x.copy_(logits)
        stage.damping_logits_y.copy_(logits)


def install_decoupled_initialization(stage: ComplexScanStage) -> None:
    if stage.modes % POLE_ORIENTATIONS:
        raise ValueError("Pole poles require complete canonical orientation groups")
    radial_levels = stage.modes // POLE_ORIENTATIONS
    multiplier = _nearest_coprime_multiplier(
        radial_levels,
        POLE_DAMPING_INDEX_MULTIPLIER,
    )
    original_x = damping(stage, "x").detach().reshape(radial_levels, POLE_ORIENTATIONS)
    original_y = damping(stage, "y").detach().reshape_as(original_x)
    if not torch.equal(original_x, original_y) or not torch.allclose(
        original_x,
        original_x[:, :1].expand_as(original_x),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise RuntimeError("Pole decoupling requires an isotropic radial atlas")

    radial = torch.arange(radial_levels, device=original_x.device).view(-1, 1)
    offsets = torch.tensor(POLE_DAMPING_OFFSETS, device=original_x.device).view(1, -1)
    indices = (multiplier * radial + offsets) % radial_levels
    base_damping = original_x[:, 0][indices]
    anisotropy = torch.tensor(
        POLE_LOG_ANISOTROPY,
        dtype=base_damping.dtype,
        device=base_damping.device,
    ).view(1, -1)
    damping_x = base_damping * anisotropy.exp()
    damping_y = base_damping * (-anisotropy).exp()
    if damping_x.min() <= stage.damping_min or damping_y.min() <= stage.damping_min:
        raise RuntimeError("Pole decoupling crossed the minimum damping bound")
    if damping_x.max() >= stage.damping_max or damping_y.max() >= stage.damping_max:
        raise RuntimeError("Pole decoupling crossed the maximum damping bound")
    ratio_x = ((damping_x.flatten() - stage.damping_min) / (stage.damping_max - stage.damping_min))
    ratio_y = ((damping_y.flatten() - stage.damping_min) / (stage.damping_max - stage.damping_min))
    with torch.no_grad():
        stage.damping_logits_x.copy_(torch.logit(ratio_x.clamp(1.0e-4, 1.0 - 1.0e-4)))
        stage.damping_logits_y.copy_(torch.logit(ratio_y.clamp(1.0e-4, 1.0 - 1.0e-4)))


def install_pole_initialization(
    model: nn.Module,
    stage_names: tuple[str, str, str, str],
) -> None:
    stages = tuple(getattr(model, name) for name in stage_names)
    for stage, maximum_phase, frequency_scale, damping_scale in zip(
        stages,
        POLE_MAXIMUM_PHASES,
        POLE_FREQUENCY_SCALES,
        POLE_DAMPING_SCALES,
        strict=True,
    ):
        install_calibrated_initialization(
            stage,
            maximum_phase,
            frequency_scale,
            damping_scale,
        )
        install_decoupled_initialization(stage)
    for name, stage in zip(stage_names, stages, strict=True):
        install_short_damping(stage, terminal=name == "terminal")


__all__ = [
    "DAMPING_QUANTILE_POWER",
    "NONTERMINAL_GEOMETRIC_DAMPING_RANGE",
    "POLE_DAMPING_INDEX_MULTIPLIER",
    "POLE_DAMPING_OFFSETS",
    "POLE_DAMPING_SCALES",
    "POLE_FREQUENCY_SCALES",
    "POLE_LOG_ANISOTROPY",
    "POLE_MAXIMUM_PHASES",
    "POLE_ORIENTATIONS",
    "TERMINAL_DAMPING_MAX",
    "TERMINAL_GEOMETRIC_DAMPING_RANGE",
    "damping",
    "damping_logits",
    "install_calibrated_initialization",
    "install_decoupled_initialization",
    "install_pole_initialization",
    "install_short_damping",
    "target_geometric_damping",
]
