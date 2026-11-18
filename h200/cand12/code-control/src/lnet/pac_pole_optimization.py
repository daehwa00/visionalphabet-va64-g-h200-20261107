"""Log-damping parameterization and snapshots for PAC pole banks."""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from torch import Tensor, nn

from .complex_scan_stage import ComplexScanStage

if TYPE_CHECKING:
    from collections.abc import Mapping


class _BankInitialization(TypedDict):
    grid_size: int
    short_indices: list[int]
    long_indices: list[int]
    long_index_mask: str
    orientation_count: int
    radial_levels: int
    long_memory_length_start: float
    long_memory_length_end: float
    long_memory_length_distribution: str
    long_damping_formula: str
    axis_pairing: str


class _InitializationSnapshot(TypedDict):
    name: str
    multiscale: bool
    long_index_mask: str
    orientation_count: int
    phase_policy: str
    banks: dict[str, _BankInitialization]


class _PoleBankSnapshot(TypedDict):
    name: str
    bounds: dict[str, float]
    parameterization: str
    effective_damping_x: list[float]
    effective_damping_y: list[float]
    raw_phase_x: list[float]
    raw_phase_y: list[float]
    grid_size: int


class PoleSnapshot(TypedDict):
    schema: str
    initialization: _InitializationSnapshot
    banks: list[_PoleBankSnapshot]


EXPECTED_K128_BANKS = 12
_ROOT_GRID_SIZES = {
    "stage1": 56,
    "stage2": 28,
    "stage3": 14,
    "terminal": 7,
}
_INITIALIZATION_ATTRIBUTE = "_pac_pole_optimization_initialization"


def _scan_banks(model: nn.Module) -> list[tuple[str, ComplexScanStage]]:
    banks = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, ComplexScanStage)
    ]
    if len(banks) != EXPECTED_K128_BANKS:
        raise ValueError(f"K128 pole optimization expects 12 banks; found {len(banks)}")
    return banks


def _grid_size(
    name: str,
    grid_sizes: Mapping[str, int] | None,
) -> int | None:
    if grid_sizes is not None and name in grid_sizes:
        size = int(grid_sizes[name])
        if size <= 0:
            raise ValueError(f"grid size for {name!r} must be positive")
        return size

    parts = name.split(".")
    for index, part in enumerate(parts):
        if part in _ROOT_GRID_SIZES:
            return _ROOT_GRID_SIZES[part]
        if (
            part in {"same_resolution_blocks", "extra_same_resolution_blocks"}
            and index + 1 < len(parts)
            and parts[index + 1].isdigit()
        ):
            size = int(parts[index + 1])
            if size > 0:
                return size
    return None


def _parameterization(stage: ComplexScanStage) -> str:
    has_log = stage.log_damping_x is not None and stage.log_damping_y is not None
    has_logits = stage.damping_logits_x is not None and stage.damping_logits_y is not None
    if has_log and not has_logits:
        return "log_damping"
    if has_logits and not has_log:
        return "bounded_sigmoid"
    raise RuntimeError("pole bank has an incomplete damping parameterization")


def _interior_damping(values: Tensor, stage: ComplexScanStage) -> Tensor:
    lower = torch.as_tensor(stage.damping_min, dtype=values.dtype, device=values.device)
    upper = torch.as_tensor(stage.damping_max, dtype=values.dtype, device=values.device)
    lower_inside = torch.nextafter(lower, upper)
    upper_inside = torch.nextafter(upper, lower)
    return torch.minimum(torch.maximum(values, lower_inside), upper_inside)


def _float_list(values: Tensor) -> list[float]:
    return cast("list[float]", values.detach().reshape(-1).cpu().tolist())


def _multiscale_values(
    stage: ComplexScanStage,
    damping_x: Tensor,
    damping_y: Tensor,
    *,
    grid_size: int,
) -> tuple[Tensor, Tensor, _BankInitialization]:
    minimum = float(stage.damping_min)
    maximum_memory = min(4.0 * grid_size, 0.9 / minimum)
    if maximum_memory < grid_size:
        message = f"grid size {grid_size} and damping minimum {minimum:g}"
        raise ValueError(f"{message} do not admit the requested long-memory interval")

    modes = stage.modes
    if modes % 16 != 0:
        message = "balanced multiscale initialization requires an 8-direction atlas"
        raise ValueError(f"{message} with an even number of radial levels (modes divisible by 16)")
    indices = torch.arange(modes, device=damping_x.device)
    long_mask = ((indices // 8 + indices % 8) % 2) == 1
    long_indices = indices[long_mask]
    short_indices = indices[~long_mask]
    memory_lengths = torch.logspace(
        math.log10(float(grid_size)),
        math.log10(maximum_memory),
        steps=long_indices.numel(),
        dtype=damping_x.dtype,
        device=damping_x.device,
    )
    long_damping = _interior_damping(memory_lengths.reciprocal(), stage)

    initialized_x = damping_x.detach().reshape(-1).clone()
    initialized_y = damping_y.detach().reshape(-1).clone()
    initialized_x[long_indices] = long_damping
    initialized_y[long_indices] = long_damping.flip(0)
    # The original bank values are already strictly inside their bounds.  Keep
    # the short-memory half untouched; only the newly generated values need
    # interior projection.
    if short_indices.numel() + long_indices.numel() != modes:
        raise RuntimeError("alternating pole indices did not cover the bank")
    metadata: _BankInitialization = {
        "grid_size": grid_size,
        "short_indices": cast("list[int]", short_indices.tolist()),
        "long_indices": cast("list[int]", long_indices.tolist()),
        "long_index_mask": "((index // 8 + index % 8) % 2 == 1)",
        "orientation_count": 8,
        "radial_levels": modes // 8,
        "long_memory_length_start": float(grid_size),
        "long_memory_length_end": float(maximum_memory),
        "long_memory_length_distribution": "geometric",
        "long_damping_formula": "d = 1 / L",
        "axis_pairing": "x ascending L, y reversed L",
    }
    return initialized_x, initialized_y, metadata


def _initial_damping_values(
    name: str,
    stage: ComplexScanStage,
    *,
    multiscale_initialization: bool,
    grid_sizes: Mapping[str, int] | None,
) -> tuple[Tensor, Tensor, _BankInitialization | None]:
    if _parameterization(stage) != "bounded_sigmoid":
        raise RuntimeError(f"pole bank {name!r} is already log-parameterized")
    grid_size = _grid_size(name, grid_sizes)
    if multiscale_initialization and grid_size is None:
        raise ValueError(
            f"multiscale initialization needs an explicit grid size for pole bank {name!r}"
        )

    damping_x, damping_y = stage._damping_fields()
    damping_x = damping_x.detach().reshape(-1)
    damping_y = damping_y.detach().reshape(-1)
    initialization: _BankInitialization | None = None
    if multiscale_initialization:
        if grid_size is None:
            raise RuntimeError(f"pole bank {name!r} lost its resolved grid size")
        damping_x, damping_y, initialization = _multiscale_values(
            stage,
            damping_x,
            damping_y,
            grid_size=grid_size,
        )
    if (
        not torch.isfinite(damping_x).all()
        or not torch.isfinite(damping_y).all()
        or (damping_x <= 0).any()
        or (damping_y <= 0).any()
    ):
        raise RuntimeError(f"pole bank {name!r} has invalid effective damping")
    return damping_x, damping_y, initialization


def install_log_damping_(
    model: nn.Module,
    *,
    multiscale_initialization: bool = False,
    grid_sizes: Mapping[str, int] | None = None,
) -> nn.Module:
    """Replace all K128 bounded-logit damping parameters with log damping.

    Existing effective damping is retained unless ``multiscale_initialization``
    is true.  In that mode the 8-direction atlas uses a radial/orientation
    checkerboard: ``((index // 8 + index % 8) % 2 == 1)`` selects long-memory
    poles, and the other poles retain their damping.  K128 banks have modes
    divisible by 16, giving an even number of radial levels and equal long/short
    counts per orientation.  Custom module paths use ``grid_sizes`` keyed by
    full bank path and must follow the same atlas rule.
    """
    banks = _scan_banks(model)
    if grid_sizes is not None:
        bank_names = {name for name, _ in banks}
        unknown = set(grid_sizes) - bank_names
        if unknown:
            raise ValueError(f"grid_sizes contains unknown pole banks: {sorted(unknown)}")

    pending: list[
        tuple[str, ComplexScanStage, Tensor, Tensor, _BankInitialization | None]
    ] = []
    for name, stage in banks:
        damping_x, damping_y, initialization = _initial_damping_values(
            name,
            stage,
            multiscale_initialization=multiscale_initialization,
            grid_sizes=grid_sizes,
        )
        pending.append((name, stage, damping_x, damping_y, initialization))

    per_bank_initialization: dict[str, _BankInitialization] = {}
    with torch.no_grad():
        for name, stage, damping_x, damping_y, initialization in pending:
            stage.register_parameter("log_damping_x", nn.Parameter(damping_x.log().clone()))
            stage.register_parameter("log_damping_y", nn.Parameter(damping_y.log().clone()))
            stage.register_parameter("damping_logits_x", None)
            stage.register_parameter("damping_logits_y", None)
            if initialization is not None:
                per_bank_initialization[name] = initialization

    model.__dict__[_INITIALIZATION_ATTRIBUTE] = {
        "name": "balanced_short_long_v1"
        if multiscale_initialization
        else "preserve_effective_damping",
        "multiscale": bool(multiscale_initialization),
        "long_index_mask": "((index // 8 + index % 8) % 2 == 1)",
        "orientation_count": 8,
        "phase_policy": "unchanged",
        "banks": per_bank_initialization,
    }
    return model


@torch.no_grad()
def project_log_damping_(model: nn.Module) -> nn.Module:
    """Project log-damping parameters to each bank's configured bounds."""
    banks = _scan_banks(model)
    for name, stage in banks:
        if _parameterization(stage) != "log_damping":
            raise RuntimeError(f"pole bank {name!r} has not been log-parameterized")
        lower = math.log(float(stage.damping_min))
        upper = math.log(float(stage.damping_max))
        log_damping_x = stage.log_damping_x
        log_damping_y = stage.log_damping_y
        if log_damping_x is None or log_damping_y is None:
            raise RuntimeError(f"pole bank {name!r} is missing log-damping parameters")
        log_damping_x.clamp_(min=lower, max=upper)
        log_damping_y.clamp_(min=lower, max=upper)
    return model


def _initialization_snapshot(model: nn.Module) -> _InitializationSnapshot:
    value = model.__dict__.get(_INITIALIZATION_ATTRIBUTE)
    if isinstance(value, dict):
        return cast("_InitializationSnapshot", cast("object", value))
    return {
        "name": "preexisting",
        "multiscale": False,
        "long_index_mask": "((index // 8 + index % 8) % 2 == 1)",
        "orientation_count": 8,
        "phase_policy": "unchanged",
        "banks": {},
    }


@torch.no_grad()
def pole_snapshot(
    model: nn.Module,
    *,
    grid_sizes: Mapping[str, int] | None = None,
) -> PoleSnapshot:
    """Return a JSON-serializable snapshot of every K128 pole bank."""
    banks = _scan_banks(model)
    payload_banks: list[_PoleBankSnapshot] = []
    for name, stage in banks:
        grid_size = _grid_size(name, grid_sizes)
        if grid_size is None:
            raise ValueError(f"pole snapshot needs an explicit grid size for bank {name!r}")
        damping_x, damping_y = stage._damping_fields()
        payload_banks.append(
            {
                "name": name,
                "bounds": {
                    "min": float(stage.damping_min),
                    "max": float(stage.damping_max),
                },
                "parameterization": _parameterization(stage),
                "effective_damping_x": _float_list(damping_x),
                "effective_damping_y": _float_list(damping_y),
                "raw_phase_x": _float_list(stage.phase_x),
                "raw_phase_y": _float_list(stage.phase_y),
                "grid_size": grid_size,
            }
        )
    return {
        "schema": "lnet.pac_pole_snapshot.v1",
        "initialization": _initialization_snapshot(model),
        "banks": payload_banks,
    }


__all__ = [
    "EXPECTED_K128_BANKS",
    "install_log_damping_",
    "pole_snapshot",
    "project_log_damping_",
]
