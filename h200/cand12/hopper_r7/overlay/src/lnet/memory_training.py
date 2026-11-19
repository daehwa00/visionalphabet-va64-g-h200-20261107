"""Opt-in real-memory training execution with unchanged parameters and resume keys."""

from __future__ import annotations

# Execution flags are the internal Reader dispatch contract.
# ruff: noqa: SLF001
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

from .memory_backbone import FactorizedMemoryReader


def enable_memory_training_(
    model: nn.Module,
    *,
    spatial: bool = True,
    vector_weight_gradient: bool = True,
) -> dict[str, int]:
    """Set execution flags before compilation and return the enabled Reader count.

    Parameters, optimizer references, state_dict keys and RNG are untouched.
    Weight normalization remains differentiable and is recomputed every forward.
    Autotune caches only launch choices, never weight-derived tensors. The vector
    gradient is the measured default; disabling it supports a geometry-only ablation.
    """
    readers = [m for m in model.modules() if isinstance(m, FactorizedMemoryReader)]
    if not readers:
        raise ValueError("Expected a real-memory model")
    if any(reader.execution_mode != "factorized_triton" for reader in readers):
        raise ValueError("Training tuning requires factorized_triton Readers")
    for reader in readers:
        reader._real_training_autotune = spatial
        reader._real_training_vector_gradient = spatial and vector_weight_gradient
    return {"readers": len(readers) if spatial else 0}
