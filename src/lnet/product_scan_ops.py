"""Static four-path vertical product scan with selectable output epilogues.

The forward kernel deliberately does not materialize the four full-resolution
vertical product states.  Non-terminal stages write direction-aligned coarse
endpoints, while terminal stages skip those stores entirely.  Both epilogues
accumulate exact full-grid directional energies into the descriptor buffer.
"""

from __future__ import annotations

# ruff: noqa: FBT001
# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
# pyright: reportGeneralTypeIssues=false, reportMissingParameterType=false
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import os
from typing import Protocol, cast

import torch
import triton
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .general_cuda_scans import (
    general_scans_enabled,
    try_general_vertical_backward,
    try_general_vertical_forward,
)
from .product_scan_contracts import (
    DEFAULT_EPSILON,
    ComplexField,
    FusedOutputs,
    ProductGainNormalization,
    gain_kind,
    gain_normalization,
    validate_global_inverse_gain,
    validate_product_scan,
)
from .product_scan_normalization import static_product_scan_auxiliary
from .product_scan_reference import (
    _product_scan_coarse4_from_tables_reference,
    _product_scan_descriptor4_from_tables_reference,
    _product_scan_full16_from_tables_reference,
    d4_grouped_cell_path_collapse_reference,
    d4_grouped_cell_path_swiglu_reference,
)
from .triton_product_scan_kernels import (
    _PACKED_PATH_INPUTS,
    _PACKED_PATH_OUTPUTS,
    _finalize_descriptor_kernel,
    _product_scan_coarse4_associative_backward_kernel,
    _product_scan_coarse4_associative_forward_kernel,
)

FORWARD_LAUNCH_NAME = "product_scan_coarse4_forward"
FULL16_FORWARD_LAUNCH_NAME = "product_scan_full16_forward"
PATH_COLLAPSE_FORWARD_LAUNCH_NAME = "product_scan_path_collapse_forward"
DESCRIPTOR_FORWARD_LAUNCH_NAME = "product_scan_descriptor4_forward"
FINALIZE_LAUNCH_NAME = "product_scan_coarse4_descriptor_finalize"
BACKWARD_LAUNCH_NAME = "product_scan_coarse4_backward"
FULL16_BACKWARD_LAUNCH_NAME = "product_scan_full16_backward"
PATH_COLLAPSE_BACKWARD_LAUNCH_NAME = "product_scan_path_collapse_backward"
PathCollapseGradients = tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]
DESCRIPTOR_BACKWARD_LAUNCH_NAME = "product_scan_descriptor4_backward"

COARSE_PATH_FORWARD_LAUNCH_NAME = "product_scan_coarse_path_forward"
COARSE_PATH_BACKWARD_LAUNCH_NAME = "product_scan_coarse_path_backward"
_SCAN_BLOCK_MODES = 8
_PATH_BLOCK_MODES = 4
_FINALIZE_BLOCK_MODES = 32
_SCAN_NUM_WARPS = 4


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    epsilon: float
    gain_kind: int
    path_swiglu: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _launch_product_scan4_backward(  # noqa: C901, PLR0912, PLR0915
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    grad_coarse_real: Tensor,
    grad_coarse_imag: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
    *,
    emit_coarse: bool,
    full_coarse: bool = False,
    path_collapse: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    path_swiglu: bool = False,
    has_descriptor_grad: bool = True,
    launch_name: str,
    source_gradient_buffers: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    packed_output: bool = False,
    coarse_path: bool = False,
    planar_output: bool = False,
    combined_state_gradients: bool = False,
    mean_direction_energy: bool = False,
    zero_unused_coarse_grads: bool = True,
) -> tuple[Tensor, ...]:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    validate_product_scan(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        epsilon,
        gain_kind,
        emit_coarse=emit_coarse,
    )
    validate_global_inverse_gain(
        global_inverse_gain,
        modes=source_real_a.shape[-1],
        gain_kind=gain_kind,
        reference=source_real_a,
    )
    if not decay_real.is_cuda:
        raise RuntimeError("the fused product adjoint kernel requires CUDA")
    batch, height, width, modes = source_real_a.shape
    _bp_env = os.environ.get("VA_REAL_MEMORY_SCAN_BATCH_PARTIALS")
    if _bp_env is None:
        _opt_env = os.environ.get("VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS")
        if _opt_env is not None:
            batch_partial_gradients = _opt_env == "1"
        else:
            batch_partial_gradients = path_collapse is not None and height >= 14
    else:
        batch_partial_gradients = _bp_env == "1"
    if coarse_path and (path_collapse is None or height % 2 or width % 2 or emit_coarse):
        raise ValueError("coarse path collapse requires even dimensions and collapsed-only output")
    if coarse_path and has_descriptor_grad:
        raise ValueError("coarse path collapse has no descriptor gradient")
    output_height = height // 2 if coarse_path else height
    output_width = width // 2 if coarse_path else width
    if emit_coarse:
        expected_shape = (
            (batch, height // 2, width // 2, 4, 4, modes)
            if full_coarse
            else (batch, height // 2, width // 2, 4, modes)
        )
        if grad_coarse_real.shape != expected_shape:
            raise ValueError("fused product adjoint received an invalid coarse gradient")
        if grad_coarse_imag.shape != grad_coarse_real.shape:
            raise ValueError("fused product adjoint coarse gradients must match")
    empty = source_real_a.new_empty((0,))
    if path_collapse is None:
        grad_collapsed_real = empty
        grad_collapsed_imag = empty
        path_tensors = (empty, empty, empty, empty)
        path_gradients = tuple(torch.empty_like(value) for value in path_tensors)
        packed_path_hidden = 1
        swiglu_hidden = 0
    else:
        expected_collapsed_shape = (
            (batch, output_height, output_width, modes)
            if not packed_output and grad_coarse_real.ndim == 4
            else (batch, output_height, output_width, 1, modes)
        )
        if grad_coarse_real.shape != expected_collapsed_shape:
            raise ValueError("fused scan-path adjoint received an invalid output gradient")
        if grad_coarse_imag.shape != grad_coarse_real.shape:
            raise ValueError("fused scan-path output gradients must match")
        grad_collapsed_real = grad_coarse_real if packed_output else grad_coarse_real.contiguous()
        grad_collapsed_imag = grad_coarse_imag if packed_output else grad_coarse_imag.contiguous()
        path_tensors = tuple(value.contiguous() for value in path_collapse)
        path_gradients = tuple(
            value.new_zeros((batch, *value.shape))
            if batch_partial_gradients
            else torch.zeros_like(value, memory_format=torch.contiguous_format)
            for value in path_tensors
        )
        packed_path_hidden = path_tensors[0].shape[1]
        swiglu_hidden = packed_path_hidden // 3 if path_swiglu else 0
    if has_descriptor_grad:
        descriptor_modes = modes if mean_direction_energy else 4 * modes
        if descriptor_gradient_factor.shape != (batch, descriptor_modes):
            raise ValueError("fused product adjoint received an invalid descriptor-gradient factor")
    elif descriptor_gradient_factor.numel() != 0:
        raise ValueError("descriptor-free product adjoint requires an empty gradient factor")
    coefficients = tuple(value.contiguous() for value in pole)
    sources = tuple(value.contiguous() for value in (*source_a, *source_b))
    coefficient_gradients = tuple(
        value.new_zeros((batch, *value.shape))
        if batch_partial_gradients
        else torch.zeros_like(value, memory_format=torch.contiguous_format)
        for value in coefficients
    )
    if combined_state_gradients:
        gradient_storage = source_real_a.new_empty((4, *source_real_a.shape))
        source_gradients = tuple(gradient_storage.unbind(0))
    elif source_gradient_buffers is None:
        source_gradients = tuple(
            torch.empty_like(value, memory_format=torch.contiguous_format) for value in sources
        )
    else:
        if any(
            gradient.shape != source.shape
            or gradient.dtype != source.dtype
            or gradient.device != source.device
            or not gradient.is_contiguous()
            for gradient, source in zip(source_gradient_buffers, sources, strict=True)
        ):
            raise ValueError("fused product adjoint reuse buffers must match contiguous sources")
        source_gradients = source_gradient_buffers
    if (
        general_scans_enabled()
        and not (emit_coarse or has_descriptor_grad or mean_direction_energy)
        and source_gradient_buffers is None
    ):
        result = try_general_vertical_backward(
            pole, (*source_a, *source_b), (variance_x, variance_y), path_collapse or (),
            grad_collapsed_real, grad_collapsed_imag,
            "coarse_path" if coarse_path else "full16", epsilon, gain_kind, path_swiglu,
        )
        if result is not None:
            return (*result[:4], *result[4].unbind(0), *result[5:])
    line_count = batch * width
    block_height = (
        max(32, triton.next_power_of_2(height))
        if coarse_path else triton.next_power_of_2(height)
    )
    coarse_grad_inputs = (
        (
            grad_coarse_real if packed_output else grad_coarse_real.contiguous(),
            grad_coarse_imag if packed_output else grad_coarse_imag.contiguous(),
        )
        if emit_coarse
        else (empty, empty)
    )

    block_modes = (
        _PATH_BLOCK_MODES
        if launch_name == PATH_COLLAPSE_BACKWARD_LAUNCH_NAME else _SCAN_BLOCK_MODES
    )
    wrap_triton(_product_scan_coarse4_associative_backward_kernel)[
        (batch * output_width, triton.cdiv(modes, block_modes))
    ](
        *coefficients,
        *sources,
        variance_x.contiguous(),
        variance_y.contiguous(),
        global_inverse_gain.contiguous(),
        *coarse_grad_inputs,
        grad_collapsed_real,
        grad_collapsed_imag,
        *path_tensors,
        *path_gradients,
        descriptor_gradient_factor.contiguous(),
        *coefficient_gradients,
        *(
            (gradient_storage, source_real_a, source_real_a, source_real_a)
            if combined_state_gradients else source_gradients
        ),
        height,
        width,
        line_count,
        modes,
        packed_path_hidden,
        epsilon=epsilon,
        gain_kind=gain_kind,
        HAS_COARSE_GRAD=emit_coarse,
        FULL_COARSE_GRAD=full_coarse,
        COLLAPSE_PATHS_GRAD=path_collapse is not None,
        PATH_SWIGLU=path_swiglu,
        SWIGLU_HIDDEN=swiglu_hidden,
        HAS_DESCRIPTOR_GRAD=has_descriptor_grad,
        PACKED_OUTPUT=packed_output,
        COARSE_PATHS=coarse_path,
        PLANAR_PATH_OUTPUT=planar_output,
        COMBINED_STATE_GRADS=combined_state_gradients,
        MEAN_DESCRIPTOR_GRAD=mean_direction_energy,
        BATCH_PARTIAL_GRADS=batch_partial_gradients,
        ZERO_UNUSED_COARSE_GRADS=zero_unused_coarse_grads,
        LINES_PER_GROUP=0,
        COMPACT_CHANNELS=0,
        BLOCK_LINES=1,
        BLOCK_MODES=block_modes,
        num_warps=_SCAN_NUM_WARPS,
        BLOCK_HEIGHT=block_height,
        BLOCK_PATH_HIDDEN=max(16, triton.next_power_of_2(packed_path_hidden)),
        BLOCK_SWIGLU_VALUE=(
            max(16, triton.next_power_of_2(2 * swiglu_hidden)) if path_swiglu else 16
        ),
    )
    if batch_partial_gradients:
        # Each CTA accumulates into its batch row; reduce rows only once.
        coefficient_gradients = tuple(value.sum(dim=0) for value in coefficient_gradients)
        if path_collapse is not None:
            path_gradients = tuple(value.sum(dim=0) for value in path_gradients)
    base_gradients = (*coefficient_gradients, *source_gradients)
    if path_collapse is None:
        return base_gradients
    return *base_gradients, *path_gradients


@triton_op("lnet::product_scan_coarse4_backward", mutates_args={})
def _product_scan_coarse4_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    grad_coarse_real: Tensor,
    grad_coarse_imag: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return cast(
        "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
        _launch_product_scan4_backward(
            decay_real,
            decay_imag,
            gamma_real,
            gamma_imag,
            source_real_a,
            source_imag_a,
            source_real_b,
            source_imag_b,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_coarse_real,
            grad_coarse_imag,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=True,
            full_coarse=False,
            launch_name=BACKWARD_LAUNCH_NAME,
        ),
    )


@triton_op("lnet::product_scan_full16_backward", mutates_args={})
def _product_scan_full16_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    grad_full_real: Tensor,
    grad_full_imag: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return cast(
        "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
        _launch_product_scan4_backward(
            decay_real,
            decay_imag,
            gamma_real,
            gamma_imag,
            source_real_a,
            source_imag_a,
            source_real_b,
            source_imag_b,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_full_real,
            grad_full_imag,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=True,
            full_coarse=True,
            launch_name=FULL16_BACKWARD_LAUNCH_NAME,
        ),
    )


@triton_op("lnet::product_scan_path_collapse_backward", mutates_args={})
def _product_scan_path_collapse_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    grad_collapsed_real: Tensor,
    grad_collapsed_imag: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
    path_swiglu: bool,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    return cast(
        "PathCollapseGradients",
        _launch_product_scan4_backward(
            decay_real,
            decay_imag,
            gamma_real,
            gamma_imag,
            source_real_a,
            source_imag_a,
            source_real_b,
            source_imag_b,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_collapsed_real,
            grad_collapsed_imag,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=False,
            path_collapse=(
                path_input_weight,
                path_input_bias,
                path_output_weight,
                path_output_bias,
            ),
            path_swiglu=path_swiglu,
            has_descriptor_grad=False,
            launch_name=PATH_COLLAPSE_BACKWARD_LAUNCH_NAME,
        ),
    )


@triton_op("lnet::product_scan_descriptor4_backward", mutates_args={})
def _product_scan_descriptor4_backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    descriptor_gradient_factor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    empty = source_real_a.new_empty((0,))
    return cast(
        "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
        _launch_product_scan4_backward(
            decay_real,
            decay_imag,
            gamma_real,
            gamma_imag,
            source_real_a,
            source_imag_a,
            source_real_b,
            source_imag_b,
            variance_x,
            variance_y,
            global_inverse_gain,
            empty,
            empty,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=False,
            full_coarse=False,
            launch_name=DESCRIPTOR_BACKWARD_LAUNCH_NAME,
        ),
    )


def _launch_product_scan4_forward(  # noqa: C901, PLR0912, PLR0915
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    epsilon: float,
    gain_kind: int,
    *,
    emit_coarse: bool,
    full_coarse: bool = False,
    path_collapse: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    path_swiglu: bool = False,
    emit_descriptor: bool = True,
    mean_direction_energy: bool = False,
    packed_output: bool = False,
    coarse_path: bool = False,
    planar_output: bool = False,
    squeeze_collapsed: bool = False,
    launch_name: str,
) -> FusedOutputs:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    validate_product_scan(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        epsilon,
        gain_kind,
        emit_coarse=emit_coarse,
    )
    validate_global_inverse_gain(
        global_inverse_gain,
        modes=source_real_a.shape[-1],
        gain_kind=gain_kind,
        reference=source_real_a,
    )
    sources = tuple(value.contiguous() for value in (*source_a, *source_b))
    coefficients = tuple(value.contiguous() for value in pole)
    batch, height, width, modes = source_real_a.shape
    if coarse_path and (path_collapse is None or height % 2 or width % 2 or emit_coarse):
        raise ValueError("coarse path collapse requires even dimensions and collapsed-only output")
    if coarse_path and emit_descriptor:
        raise ValueError("coarse path collapse omits the full-grid descriptor")
    output_height = height // 2 if coarse_path else height
    output_width = width // 2 if coarse_path else width
    coarse_shape = (
        (
            (batch, height // 2, width // 2, 4, 4, modes)
            if full_coarse
            else (batch, height // 2, width // 2, 4, modes)
        )
        if emit_coarse
        else (0,)
    )
    packed_states = source_real_a.new_empty((0,))
    if packed_output and emit_coarse:
        packed_states = source_real_a.new_empty((*coarse_shape[:-1], 2 * modes))
        coarse_real = packed_states
        coarse_imag = source_real_a.new_empty((0,))
    else:
        coarse_real = source_real_a.new_empty(coarse_shape)
        coarse_imag = torch.empty_like(coarse_real)
    if path_collapse is None:
        empty = torch.empty((0,), dtype=torch.float32, device=source_real_a.device)
        path_tensors = (empty, empty, empty, empty)
        collapsed_real = torch.empty((0,), dtype=source_real_a.dtype, device=source_real_a.device)
        collapsed_imag = torch.empty_like(collapsed_real)
        packed_path_hidden = 1
        swiglu_hidden = 0
    else:
        path_input_weight, path_input_bias, path_output_weight, path_output_bias = (
            value.contiguous() for value in path_collapse
        )
        swiglu_hidden = path_input_weight.shape[1] // 3 if path_swiglu else 0
        expected_output_hidden = 2 * swiglu_hidden if path_swiglu else path_input_weight.shape[1]
        if (
            path_input_weight.ndim != 3
            or path_input_weight.shape[0] != modes
            or path_input_weight.shape[2] != _PACKED_PATH_INPUTS
            or path_input_bias.shape != path_input_weight.shape[:2]
            or (path_swiglu and (path_input_weight.shape[1] < 3 or path_input_weight.shape[1] % 3))
            or path_output_weight.shape != (modes, _PACKED_PATH_OUTPUTS, expected_output_hidden)
            or path_output_bias.shape != (modes, _PACKED_PATH_OUTPUTS)
        ):
            raise ValueError("scan-path collapse parameters have incompatible shapes")
        path_tensors = (
            path_input_weight,
            path_input_bias,
            path_output_weight,
            path_output_bias,
        )
        if any(
            value.device != source_real_a.device
            or value.dtype != torch.float32
            or not value.is_contiguous()
            for value in path_tensors
        ):
            raise TypeError("scan-path collapse parameters must be contiguous FP32 CUDA tensors")
        collapsed_shape = (
            (batch, output_height, output_width, modes)
            if squeeze_collapsed
            else (batch, output_height, output_width, 1, modes)
        )
        if packed_output:
            packed_states = source_real_a.new_empty(
                (2, batch, output_height, output_width, modes)
                if planar_output else (batch, output_height, output_width, 2 * modes)
            )
            collapsed_real = packed_states
            collapsed_imag = source_real_a.new_empty((0,))
        else:
            collapsed_real = source_real_a.new_empty(collapsed_shape)
            collapsed_imag = torch.empty_like(collapsed_real)
        packed_path_hidden = path_input_weight.shape[1]
    deterministic_descriptor = emit_descriptor and torch.are_deterministic_algorithms_enabled()
    if emit_descriptor:
        descriptor_energy = torch.zeros(
            (batch, width, 4 * modes) if deterministic_descriptor else (batch, 4 * modes),
            dtype=torch.float32,
            device=source_real_a.device,
        )
        descriptor = torch.empty(
            (batch, modes if mean_direction_energy else 4 * modes),
            dtype=torch.float32,
            device=source_real_a.device,
        )
    else:
        descriptor_energy = torch.empty((0,), dtype=torch.float32, device=source_real_a.device)
        descriptor = torch.empty_like(descriptor_energy)
    if general_scans_enabled(forward=True) and not (
        emit_coarse or emit_descriptor or mean_direction_energy
    ):
        result = try_general_vertical_forward(
            pole, (*source_a, *source_b), (variance_x, variance_y), path_collapse or (),
            "coarse_path" if coarse_path else "full16", epsilon, gain_kind,
            packed_output, planar_output, path_swiglu, squeeze_collapsed,
        )
        if result is not None:
            return result
    line_count = batch * width
    block_modes = (
        _PATH_BLOCK_MODES if launch_name == PATH_COLLAPSE_FORWARD_LAUNCH_NAME else _SCAN_BLOCK_MODES
    )
    wrap_triton(_product_scan_coarse4_associative_forward_kernel)[
        (batch * output_width, triton.cdiv(modes, block_modes))
    ](
        *coefficients,
        *sources,
        variance_x.contiguous(),
        variance_y.contiguous(),
        global_inverse_gain.contiguous(),
        coarse_real,
        coarse_imag,
        *path_tensors,
        collapsed_real,
        collapsed_imag,
        descriptor_energy,
        height,
        width,
        line_count,
        modes,
        packed_path_hidden,
        epsilon=epsilon,
        gain_kind=gain_kind,
        EMIT_COARSE=emit_coarse,
        FULL_COARSE=full_coarse,
        COLLAPSE_PATHS=path_collapse is not None,
        PATH_SWIGLU=path_swiglu,
        SWIGLU_HIDDEN=swiglu_hidden,
        EMIT_DESCRIPTOR=emit_descriptor,
        DETERMINISTIC_DESCRIPTOR=deterministic_descriptor,
        PACKED_OUTPUT=packed_output,
        COARSE_PATHS=coarse_path,
        PLANAR_PATH_OUTPUT=planar_output,
        BLOCK_HEIGHT=(
            max(32, triton.next_power_of_2(height))
            if coarse_path else triton.next_power_of_2(height)
        ),
        BLOCK_LINES=1,
        BLOCK_MODES=block_modes,
        num_warps=_SCAN_NUM_WARPS,
        BLOCK_PATH_HIDDEN=max(16, triton.next_power_of_2(packed_path_hidden)),
        BLOCK_SWIGLU_VALUE=(
            max(16, triton.next_power_of_2(2 * swiglu_hidden)) if path_swiglu else 16
        ),
    )
    if emit_descriptor:
        if deterministic_descriptor:
            descriptor_energy = descriptor_energy.sum(dim=1)
        wrap_triton(_finalize_descriptor_kernel)[
            (batch, triton.cdiv(modes, _FINALIZE_BLOCK_MODES))
        ](
            descriptor_energy,
            descriptor,
            height * width,
            modes,
            MEAN_DIRECTION_ENERGY=mean_direction_energy,
            BLOCK_MODES=_FINALIZE_BLOCK_MODES,
            num_warps=_SCAN_NUM_WARPS,
        )
    if packed_output:
        return packed_states, source_real_a.new_empty((0,)), descriptor
    if path_collapse is not None:
        return collapsed_real, collapsed_imag, descriptor
    return coarse_real, coarse_imag, descriptor


@triton_op("lnet::product_scan_coarse4", mutates_args={})
def _product_scan_coarse4_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    epsilon: float,
    gain_kind: int,
) -> FusedOutputs:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    if not source_real_a.is_cuda:
        return _product_scan_coarse4_from_tables_reference(
            pole,
            source_a,
            source_b,
            variance_x,
            variance_y,
            epsilon=epsilon,
            gain_normalization=gain_normalization(gain_kind),
        )
    return _launch_product_scan4_forward(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
        emit_coarse=True,
        full_coarse=False,
        launch_name=FORWARD_LAUNCH_NAME,
    )


@triton_op("lnet::product_scan_full16", mutates_args={})
def _product_scan_full16_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    epsilon: float,
    gain_kind: int,
) -> FusedOutputs:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    if not source_real_a.is_cuda:
        return _product_scan_full16_from_tables_reference(
            pole,
            source_a,
            source_b,
            variance_x,
            variance_y,
            epsilon=epsilon,
            gain_normalization=gain_normalization(gain_kind),
        )
    return _launch_product_scan4_forward(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
        emit_coarse=True,
        full_coarse=True,
        launch_name=FULL16_FORWARD_LAUNCH_NAME,
    )


@triton_op("lnet::product_scan_path_collapse", mutates_args={})
def _product_scan_path_collapse_op(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    epsilon: float,
    gain_kind: int,
    path_swiglu: bool,
) -> FusedOutputs:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    path = (path_input_weight, path_input_bias, path_output_weight, path_output_bias)
    if not source_real_a.is_cuda:
        full_real, full_imag, descriptor = _product_scan_full16_from_tables_reference(
            pole,
            source_a,
            source_b,
            variance_x,
            variance_y,
            epsilon=epsilon,
            gain_normalization=gain_normalization(gain_kind),
        )
        collapse_reference = (
            d4_grouped_cell_path_swiglu_reference
            if path_swiglu
            else d4_grouped_cell_path_collapse_reference
        )
        collapsed = collapse_reference(full_real, full_imag, *path)
        return collapsed[0], collapsed[1], descriptor
    return _launch_product_scan4_forward(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
        emit_coarse=False,
        path_collapse=path,
        path_swiglu=path_swiglu,
        emit_descriptor=False,
        launch_name=PATH_COLLAPSE_FORWARD_LAUNCH_NAME,
    )


@triton_op("lnet::product_scan_descriptor4", mutates_args={})
def _product_scan_descriptor4_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real_a: Tensor,
    source_imag_a: Tensor,
    source_real_b: Tensor,
    source_imag_b: Tensor,
    variance_x: Tensor,
    variance_y: Tensor,
    global_inverse_gain: Tensor,
    epsilon: float,
    gain_kind: int,
) -> Tensor:
    pole = (decay_real, decay_imag, gamma_real, gamma_imag)
    source_a = (source_real_a, source_imag_a)
    source_b = (source_real_b, source_imag_b)
    if not source_real_a.is_cuda:
        return _product_scan_descriptor4_from_tables_reference(
            pole,
            source_a,
            source_b,
            variance_x,
            variance_y,
            epsilon=epsilon,
            gain_normalization=gain_normalization(gain_kind),
        )
    return _launch_product_scan4_forward(
        *pole,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
        emit_coarse=False,
        full_coarse=False,
        launch_name=DESCRIPTOR_FORWARD_LAUNCH_NAME,
    )[2]


def _setup_coarse_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, ...],
    output: FusedOutputs,
) -> None:
    *tensors, epsilon, gain_kind = inputs
    ctx.epsilon = float(epsilon)
    ctx.gain_kind = int(gain_kind)
    ctx.save_for_backward(*tensors, output[2])


def _coarse_backward_impl(
    ctx: _AutogradContext,
    grad_coarse_real: Tensor | None,
    grad_coarse_imag: Tensor | None,
    grad_descriptor: Tensor | None,
    *,
    full_coarse: bool,
) -> tuple[Tensor | None, ...]:
    (
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        descriptor,
    ) = ctx.saved_tensors
    batch, height, width, modes = source_real_a.shape
    coarse_shape = (
        (batch, height // 2, width // 2, 4, 4, modes)
        if full_coarse
        else (batch, height // 2, width // 2, 4, modes)
    )
    if grad_coarse_real is None:
        grad_coarse_real = source_real_a.new_zeros(coarse_shape)
    if grad_coarse_imag is None:
        grad_coarse_imag = source_real_a.new_zeros(coarse_shape)
    if grad_descriptor is None:
        grad_descriptor = torch.zeros_like(descriptor)
    if not source_real_a.is_cuda:
        differentiable = tuple(
            value.detach().requires_grad_()
            for value in (
                decay_real,
                decay_imag,
                gamma_real,
                gamma_imag,
                source_real_a,
                source_imag_a,
                source_real_b,
                source_imag_b,
            )
        )
        with torch.enable_grad():
            reference = (
                _product_scan_full16_from_tables_reference
                if full_coarse
                else _product_scan_coarse4_from_tables_reference
            )
            outputs = reference(
                cast("tuple[Tensor, Tensor, Tensor, Tensor]", differentiable[:4]),
                cast("ComplexField", differentiable[4:6]),
                cast("ComplexField", differentiable[6:8]),
                variance_x,
                variance_y,
                epsilon=ctx.epsilon,
                gain_normalization=gain_normalization(ctx.gain_kind),
            )
            gradients = torch.autograd.grad(
                outputs,
                differentiable,
                (grad_coarse_real, grad_coarse_imag, grad_descriptor),
            )
        return *gradients, None, None, None, None, None
    # The factor is constant across every x-line.  Materialize it once per
    # batch/mode so the scan kernel does not repeat four exponentials for each
    # of the ``width`` programs; AOTInductor can fuse this expression into the
    # descriptor-gradient producer.
    descriptor_gradient_factor = (
        grad_descriptor * torch.exp(-descriptor) * (1.0 / (height * width))
    ).contiguous()
    backward = (
        _product_scan_full16_backward_op if full_coarse else _product_scan_coarse4_backward_op
    )
    gradients = backward(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        grad_coarse_real.contiguous(),
        grad_coarse_imag.contiguous(),
        descriptor_gradient_factor,
        ctx.epsilon,
        ctx.gain_kind,
    )
    return *gradients, None, None, None, None, None


def _coarse_backward(
    ctx: _AutogradContext,
    grad_coarse_real: Tensor | None,
    grad_coarse_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    return _coarse_backward_impl(
        ctx,
        grad_coarse_real,
        grad_coarse_imag,
        grad_descriptor,
        full_coarse=False,
    )


def _full16_backward(
    ctx: _AutogradContext,
    grad_full_real: Tensor | None,
    grad_full_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    return _coarse_backward_impl(
        ctx,
        grad_full_real,
        grad_full_imag,
        grad_descriptor,
        full_coarse=True,
    )


def _setup_path_collapse_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor | float | int | bool, ...],
    output: FusedOutputs,
) -> None:
    del output
    *tensors, epsilon, gain_kind, path_swiglu = inputs
    ctx.epsilon = float(epsilon)
    ctx.gain_kind = int(gain_kind)
    ctx.path_swiglu = bool(path_swiglu)
    ctx.save_for_backward(*cast("tuple[Tensor, ...]", tuple(tensors)))


def _path_collapse_backward(
    ctx: _AutogradContext,
    grad_collapsed_real: Tensor | None,
    grad_collapsed_imag: Tensor | None,
    _grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    (
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        path_input_weight,
        path_input_bias,
        path_output_weight,
        path_output_bias,
    ) = ctx.saved_tensors
    if not source_real_a.is_cuda:
        raise RuntimeError("retained scan-path autograd requires CUDA")
    batch, height, width, modes = source_real_a.shape
    collapsed_shape = (batch, height, width, 1, modes)
    if grad_collapsed_real is None:
        grad_collapsed_real = source_real_a.new_zeros(collapsed_shape)
    if grad_collapsed_imag is None:
        grad_collapsed_imag = source_real_a.new_zeros(collapsed_shape)
    gradients = _product_scan_path_collapse_backward_op(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        path_input_weight,
        path_input_bias,
        path_output_weight,
        path_output_bias,
        grad_collapsed_real.contiguous(),
        grad_collapsed_imag.contiguous(),
        source_real_a.new_empty((0,)),
        ctx.epsilon,
        ctx.gain_kind,
        ctx.path_swiglu,
    )
    return (
        *gradients[:8],
        None,
        None,
        None,
        *gradients[8:],
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "lnet::product_scan_coarse4",
    _coarse_backward,
    setup_context=_setup_coarse_context,
)
torch.library.register_autograd(
    "lnet::product_scan_full16",
    _full16_backward,
    setup_context=_setup_coarse_context,
)
torch.library.register_autograd(
    "lnet::product_scan_path_collapse",
    _path_collapse_backward,
    setup_context=_setup_path_collapse_context,
)


def _setup_descriptor_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, ...],
    output: Tensor,
) -> None:
    *tensors, epsilon, gain_kind = inputs
    ctx.epsilon = float(epsilon)
    ctx.gain_kind = int(gain_kind)
    ctx.save_for_backward(*tensors, output)


def _descriptor_backward(
    ctx: _AutogradContext,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    (
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        descriptor,
    ) = ctx.saved_tensors
    if grad_descriptor is None:
        grad_descriptor = torch.zeros_like(descriptor)
    if not source_real_a.is_cuda:
        differentiable = tuple(
            value.detach().requires_grad_()
            for value in (
                decay_real,
                decay_imag,
                gamma_real,
                gamma_imag,
                source_real_a,
                source_imag_a,
                source_real_b,
                source_imag_b,
            )
        )
        with torch.enable_grad():
            output = _product_scan_descriptor4_from_tables_reference(
                cast("tuple[Tensor, Tensor, Tensor, Tensor]", differentiable[:4]),
                cast("ComplexField", differentiable[4:6]),
                cast("ComplexField", differentiable[6:8]),
                variance_x,
                variance_y,
                epsilon=ctx.epsilon,
                gain_normalization=gain_normalization(ctx.gain_kind),
            )
            gradients = torch.autograd.grad(output, differentiable, grad_descriptor)
        return *gradients, None, None, None, None, None
    height, width = source_real_a.shape[1:3]
    descriptor_gradient_factor = (
        grad_descriptor * torch.exp(-descriptor) * (1.0 / (height * width))
    ).contiguous()
    gradients = _product_scan_descriptor4_backward_op(
        decay_real,
        decay_imag,
        gamma_real,
        gamma_imag,
        source_real_a,
        source_imag_a,
        source_real_b,
        source_imag_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        descriptor_gradient_factor,
        ctx.epsilon,
        ctx.gain_kind,
    )
    return *gradients, None, None, None, None, None


torch.library.register_autograd(
    "lnet::product_scan_descriptor4",
    _descriptor_backward,
    setup_context=_setup_descriptor_context,
)


def triton_product_scan_coarse4(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> FusedOutputs:
    """Return coarse product paths and an exact full-grid product-Q descriptor."""
    gain_kind_value = gain_kind(gain_normalization)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source_a[0],
        epsilon=epsilon,
        gain_kind=gain_kind_value,
    )
    return _product_scan_coarse4_op(
        *pole_y,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind_value,
    )


def triton_product_scan_full16(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> FusedOutputs:
    """Return all direction-relative 2x2 product states and the descriptor."""
    gain_kind_value = gain_kind(gain_normalization)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source_a[0],
        epsilon=epsilon,
        gain_kind=gain_kind_value,
    )
    return _product_scan_full16_op(
        *pole_y,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind_value,
    )


def triton_product_scan_descriptor4(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_a: ComplexField,
    source_b: ComplexField,
    *,
    epsilon: float = DEFAULT_EPSILON,
    gain_normalization: ProductGainNormalization = "pointwise",
) -> Tensor:
    """Return an exact full-grid product-Q descriptor without coarse states."""
    gain_kind_value = gain_kind(gain_normalization)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source_a[0],
        epsilon=epsilon,
        gain_kind=gain_kind_value,
    )
    return _product_scan_descriptor4_op(
        *pole_y,
        *source_a,
        *source_b,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind_value,
    )


__all__ = [
    "triton_product_scan_coarse4",
    "triton_product_scan_descriptor4",
    "triton_product_scan_full16",
]
