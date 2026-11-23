"""CUDA-Graph-safe replacement for PyTorch's matrix-exp orthogonal map.

PyTorch's orthogonal parametrization uses ``torch.matrix_exp``.  That is a
good general-purpose implementation, but its CUDA implementation is not a
stable capture boundary for a whole VisionAlphabet training step.  This
module keeps PyTorch's dynamic-trivialization representation exactly as-is
and replaces only the exponential evaluation with a fixed computation graph.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Final

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx

DEFAULT_TAYLOR_DEGREE: Final = 14
DEFAULT_SCALING_STEPS: Final = 2


class _AutocastIsolatedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, left: Tensor, right: Tensor) -> Tensor:
        ctx.save_for_backward(left, right)
        return left @ right

    @staticmethod
    def backward(ctx: FunctionCtx, gradient: Tensor) -> tuple[Tensor, Tensor]:
        left, right = ctx.saved_tensors
        # AOTAutograd may trace this backward under the ambient forward
        # autocast policy. Keep the polynomial VJP at its forward dtype.
        with torch.amp.autocast(device_type=left.device.type, enabled=False):
            return gradient @ right.mT, left.mT @ gradient


def fixed_taylor_matrix_exp(
    matrix: Tensor,
    *,
    taylor_degree: int = DEFAULT_TAYLOR_DEGREE,
    scaling_steps: int = DEFAULT_SCALING_STEPS,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """Approximate ``exp(matrix)`` with a fixed capture-safe FP32 graph.

    The default degree-14 polynomial uses Paterson-Stockmeyer blocks.
    Other degrees retain the configurable Horner evaluation.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        message = f"expected one square matrix, got shape={tuple(matrix.shape)}"
        raise ValueError(message)
    if not matrix.is_floating_point():
        message = f"expected floating-point matrix, got dtype={matrix.dtype}"
        raise TypeError(message)
    if taylor_degree < 1:
        message = f"taylor_degree must be positive, got {taylor_degree}"
        raise ValueError(message)
    if scaling_steps < 0:
        message = f"scaling_steps must be non-negative, got {scaling_steps}"
        raise ValueError(message)
    if compute_dtype not in (None, torch.float32, torch.float64):
        message = f"compute_dtype must be float32 or float64, got {compute_dtype}"
        raise ValueError(message)

    if compute_dtype is None and matrix.dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    with torch.amp.autocast(device_type=matrix.device.type, enabled=False):
        active = matrix if compute_dtype is None else matrix.to(dtype=compute_dtype)
        identity = torch.eye(active.shape[0], dtype=active.dtype, device=active.device)
        scaled = active * (2.0**-scaling_steps)
        if taylor_degree == DEFAULT_TAYLOR_DEGREE:
            return _paterson_stockmeyer_14(scaled, identity, scaling_steps).to(matrix.dtype)
        result = identity
        for denominator in range(taylor_degree, 0, -1):
            result = identity + _AutocastIsolatedMatmul.apply(scaled, result) / denominator
        for _ in range(scaling_steps):
            result = _AutocastIsolatedMatmul.apply(result, result)
        return result.to(dtype=matrix.dtype)


def _paterson_stockmeyer_14(scaled: Tensor, identity: Tensor, scaling_steps: int) -> Tensor:
    squared = _AutocastIsolatedMatmul.apply(scaled, scaled)
    cubed = _AutocastIsolatedMatmul.apply(squared, scaled)
    fourth = _AutocastIsolatedMatmul.apply(squared, squared)
    powers = (identity, scaled, squared, cubed)

    def block(start: int, count: int) -> Tensor:
        value = powers[0] / math.factorial(start)
        for i in range(1, count):
            value = value + powers[i] / math.factorial(start + i)
        return value

    result = block(12, 3)
    for start in (8, 4):
        result = _AutocastIsolatedMatmul.apply(result, fourth) + block(start, 4)
    residual = _AutocastIsolatedMatmul.apply(result, fourth) + (scaled + squared / 2 + cubed / 6)
    # Square I + R while keeping unit diagonal entries out of the matmuls.
    for _ in range(scaling_steps):
        residual = 2 * residual + _AutocastIsolatedMatmul.apply(residual, residual)
    return identity + residual


class CaptureSafeMatrixExpOrthogonal(nn.Module):
    """Drop-in forward map for PyTorch matrix-exp dynamic trivialization."""

    base: Tensor

    def __init__(
        self,
        shape: torch.Size | tuple[int, ...],
        base: Tensor,
        *,
        taylor_degree: int = DEFAULT_TAYLOR_DEGREE,
        scaling_steps: int = DEFAULT_SCALING_STEPS,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if len(shape) != 2:
            message = f"expected a matrix shape, got {tuple(shape)}"
            raise ValueError(message)
        if taylor_degree < 1:
            message = f"taylor_degree must be positive, got {taylor_degree}"
            raise ValueError(message)
        if scaling_steps < 0:
            message = f"scaling_steps must be non-negative, got {scaling_steps}"
            raise ValueError(message)

        rows, columns = int(shape[-2]), int(shape[-1])
        expected_base_shape = (max(rows, columns), max(rows, columns))
        if tuple(base.shape) != expected_base_shape:
            message = (
                f"expected dynamic-trivialization base shape {expected_base_shape}, "
                f"got {tuple(base.shape)}"
            )
            raise ValueError(message)
        self.shape = torch.Size((rows, columns))
        self.taylor_degree = taylor_degree
        self.scaling_steps = scaling_steps
        self.compute_dtype = compute_dtype
        self.register_buffer("base", base)
        self.base = base

    def forward(self, coordinates: Tensor) -> Tensor:
        rows, columns = coordinates.shape[-2:]
        transposed = rows < columns
        working = coordinates.mT if transposed else coordinates
        rows, columns = working.shape[-2:]

        lower = working.tril()
        if rows != columns:
            padding = lower.new_zeros((rows, rows - columns))
            lower = torch.cat((lower, padding), dim=-1)
        skew = lower - lower.mT
        orthogonal = fixed_taylor_matrix_exp(
            skew,
            taylor_degree=self.taylor_degree,
            scaling_steps=self.scaling_steps,
            compute_dtype=self.compute_dtype,
        )
        if rows != columns:
            orthogonal = orthogonal[:, :columns]
        orthogonal = torch.matmul(self.base, orthogonal)
        return orthogonal.mT if transposed else orthogonal


def replace_matrix_exp_orthogonal_(
    module: nn.Module,
    name: str = "weight",
    *,
    taylor_degree: int = DEFAULT_TAYLOR_DEGREE,
    scaling_steps: int = DEFAULT_SCALING_STEPS,
    compute_dtype: torch.dtype | None = None,
) -> bool:
    """Replace one existing PyTorch matrix-exp parametrization in place.

    The containing ``ParametrizationList`` and its ``original`` parameter are
    retained.  The existing ``base`` tensor is registered on the replacement,
    preserving parameter identity, optimizer references, parameter count, and
    state-dict keys in both directions.
    """
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or name not in parametrizations:
        return False
    parametrization_list = parametrizations[name]
    replacements: list[tuple[int, nn.Module]] = []
    for index, candidate in enumerate(parametrization_list):
        orthogonal_map = getattr(candidate, "orthogonal_map", None)
        if getattr(orthogonal_map, "name", None) != "matrix_exp":
            continue
        base = candidate._buffers.get("base")  # noqa: SLF001
        shape = getattr(candidate, "shape", None)
        if not isinstance(base, Tensor) or shape is None:
            message = "matrix-exp orthogonal parametrization must use dynamic trivialization"
            raise ValueError(message)
        replacement = CaptureSafeMatrixExpOrthogonal(
            shape,
            base,
            taylor_degree=taylor_degree,
            scaling_steps=scaling_steps,
            compute_dtype=compute_dtype,
        )
        replacement.train(candidate.training)
        replacements.append((index, replacement))

    for index, replacement in replacements:
        parametrization_list[index] = replacement
    return bool(replacements)


def prepare_capture_safe_orthogonal_(
    model: nn.Module,
    *,
    taylor_degree: int = DEFAULT_TAYLOR_DEGREE,
    scaling_steps: int = DEFAULT_SCALING_STEPS,
    compute_dtype: torch.dtype | None = None,
    auto: bool = True,
) -> tuple[str, ...]:
    """Replace every existing matrix-exp orthogonal parametrization in ``model``.

    Returns the qualified tensor paths that were changed.  Calling the
    function again is idempotent and returns an empty tuple.

    The environment can force preparation on or off. In auto mode, callers
    opt in via ``auto``, which defaults to True for explicit preparation.
    """
    policy = os.environ.get("LNET_CAPTURE_SAFE_ORTHOGONAL", "auto")
    if policy not in {"auto", "on", "off"}:
        raise ValueError(f"invalid LNET_CAPTURE_SAFE_ORTHOGONAL: {policy!r}")
    if policy == "off" or (policy == "auto" and not auto):
        return ()

    replaced: list[str] = []
    for module_path, module in tuple(model.named_modules()):
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        replaced.extend(
            f"{module_path}.{name}" if module_path else name
            for name in tuple(parametrizations.keys())
            if replace_matrix_exp_orthogonal_(
                module,
                name,
                taylor_degree=taylor_degree,
                scaling_steps=scaling_steps,
                compute_dtype=compute_dtype,
            )
        )
    return tuple(replaced)


__all__ = [
    "DEFAULT_SCALING_STEPS",
    "DEFAULT_TAYLOR_DEGREE",
    "CaptureSafeMatrixExpOrthogonal",
    "fixed_taylor_matrix_exp",
    "prepare_capture_safe_orthogonal_",
    "replace_matrix_exp_orthogonal_",
]
