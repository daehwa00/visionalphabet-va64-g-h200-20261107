"""Horizontal-to-vertical D4 product scan composition and memory policy."""

from __future__ import annotations

# ruff: noqa: FBT001, FBT003
# The composition boundary intentionally uses package-private primitive ops so
# their intermediate autograd contexts do not retain horizontal scan states.
# pyright: reportPrivateUsage=false
from typing import TYPE_CHECKING, Literal, Protocol, cast

import torch
from torch import Tensor
from torch.library import triton_op

from .general_cuda_scans import (
    _general_vertical_inputs,
    _should_save_vscan,
    _vertical_forward_saved,
    can_use_general_hscan,
    general_scans_enabled,
    horizontal_backward,
    horizontal_forward,
    try_general_vertical_forward,
    vertical_backward,
    vertical_forward_packed,
)
from .product_scan_contracts import (
    DEFAULT_EPSILON,
    ProductGainNormalization,
    gain_kind,
    gain_normalization,
)
from .product_scan_normalization import static_product_scan_auxiliary
from .product_scan_ops import (
    BACKWARD_LAUNCH_NAME,
    COARSE_PATH_BACKWARD_LAUNCH_NAME,
    COARSE_PATH_FORWARD_LAUNCH_NAME,
    DESCRIPTOR_BACKWARD_LAUNCH_NAME,
    DESCRIPTOR_FORWARD_LAUNCH_NAME,
    FORWARD_LAUNCH_NAME,
    FULL16_BACKWARD_LAUNCH_NAME,
    FULL16_FORWARD_LAUNCH_NAME,
    PATH_COLLAPSE_BACKWARD_LAUNCH_NAME,
    PATH_COLLAPSE_FORWARD_LAUNCH_NAME,
    _launch_product_scan4_backward,
    _launch_product_scan4_forward,
    _product_scan_coarse4_op,
    _product_scan_descriptor4_op,
    _product_scan_full16_op,
    _product_scan_path_collapse_backward_op,
    _product_scan_path_collapse_op,
    triton_product_scan_coarse4,
    triton_product_scan_descriptor4,
    triton_product_scan_full16,
)
from .product_scan_reference import (
    d4_grouped_cell_path_collapse_reference,
    d4_grouped_cell_path_swiglu_reference,
    d4_grouped_path_collapse_reference,
    d4_grouped_path_swiglu_reference,
    product_scan_full4_reference,
)
from .triton_bidirectional_product_scan import (
    BACKWARD_LAUNCH_NAME as HORIZONTAL_BACKWARD_LAUNCH_NAME,
)
from .triton_bidirectional_product_scan import (
    RECOMPUTE_BACKWARD_LAUNCH_NAME as HORIZONTAL_RECOMPUTE_BACKWARD_LAUNCH_NAME,
)
from .triton_bidirectional_product_scan import (
    _forward_op as _horizontal_forward_op,
)
from .triton_bidirectional_product_scan import (
    _forward_planes_op as _horizontal_forward_planes_op,
)
from .triton_bidirectional_product_scan import (
    _launch_backward as _launch_horizontal_backward,
)
from .triton_bidirectional_product_scan import (
    _recomputed_backward_op as _horizontal_recomputed_backward_op,
)
from .triton_bidirectional_product_scan import triton_bidirectional_product_scan

if TYPE_CHECKING:
    from collections.abc import Callable

ProductScanEpilogue = Literal["coarse", "full16", "descriptor"]
ScanMemoryPolicy = Literal["retain", "recompute"]
ProductScanOutput = tuple[Tensor, Tensor, Tensor] | Tensor
PathCollapseParameters = tuple[Tensor, Tensor, Tensor, Tensor]
PathCollapsePipelineGradients = tuple[
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
    Tensor,
    Tensor,
]
ProductScanInputGradients = tuple[
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


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    epsilon: float
    gain_kind: int
    path_swiglu: bool

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _product_scan_pipeline(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    *,
    epilogue: ProductScanEpilogue,
    gain_normalization: ProductGainNormalization,
    epsilon: float,
) -> ProductScanOutput:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    positive_real, positive_imag, negative_real, negative_imag = (
        triton_bidirectional_product_scan(
            pole_x,
            (excitation_real, excitation_imag),
        )
    )
    positive_horizontal = positive_real, positive_imag
    negative_horizontal = negative_real, negative_imag
    if epilogue == "coarse":
        return triton_product_scan_coarse4(
            pole_x,
            pole_y,
            positive_horizontal,
            negative_horizontal,
            gain_normalization=gain_normalization,
            epsilon=epsilon,
        )
    if epilogue == "full16":
        return triton_product_scan_full16(
            pole_x,
            pole_y,
            positive_horizontal,
            negative_horizontal,
            gain_normalization=gain_normalization,
            epsilon=epsilon,
        )
    return triton_product_scan_descriptor4(
        pole_x,
        pole_y,
        positive_horizontal,
        negative_horizontal,
        gain_normalization=gain_normalization,
        epsilon=epsilon,
    )


def _recomputed_forward_inputs(
    inputs: tuple[Tensor, ...],
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor],
]:
    return (
        cast("tuple[Tensor, Tensor, Tensor, Tensor]", inputs[:4]),
        cast("tuple[Tensor, Tensor, Tensor, Tensor]", inputs[4:8]),
        cast("tuple[Tensor, Tensor]", inputs[8:10]),
    )


@triton_op("lnet::product_scan_pipeline_coarse_recompute", mutates_args={})
def _coarse_recompute_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor]:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    horizontal = _horizontal_forward_op(*pole_x, excitation_real, excitation_imag)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    return _product_scan_coarse4_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
    )


@triton_op("lnet::product_scan_pipeline_full16_recompute", mutates_args={})
def _full16_recompute_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    epsilon: float,
    gain_kind: int,
) -> tuple[Tensor, Tensor, Tensor]:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    horizontal = _horizontal_forward_op(*pole_x, excitation_real, excitation_imag)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    return _product_scan_full16_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
    )


@triton_op("lnet::product_scan_pipeline_path_collapse_recompute", mutates_args={})
def _path_collapse_recompute_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    epsilon: float,
    gain_kind: int,
    path_swiglu: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    horizontal = _horizontal_forward_op(*pole_x, excitation_real, excitation_imag)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    return _product_scan_path_collapse_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        path_input_weight,
        path_input_bias,
        path_output_weight,
        path_output_bias,
        epsilon,
        gain_kind,
        path_swiglu,
    )


@triton_op("lnet::product_scan_pipeline_descriptor_recompute", mutates_args={})
def _descriptor_recompute_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    epsilon: float,
    gain_kind: int,
) -> Tensor:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    horizontal = _horizontal_forward_op(*pole_x, excitation_real, excitation_imag)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    return _product_scan_descriptor4_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        epsilon,
        gain_kind,
    )


def _setup_recompute_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor | float | int, ...],
    descriptor: Tensor,
) -> None:
    *tensors, epsilon, gain_kind = inputs
    ctx.epsilon = float(epsilon)
    ctx.gain_kind = int(gain_kind)
    ctx.save_for_backward(*cast("tuple[Tensor, ...]", tuple(tensors)), descriptor)


def _setup_coarse_recompute_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor | float | int, ...],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    _setup_recompute_context(ctx, inputs, output[2])


def _setup_descriptor_recompute_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor | float | int, ...],
    output: Tensor,
) -> None:
    _setup_recompute_context(ctx, inputs, output)


def _setup_path_collapse_recompute_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor | float | int | bool, ...],
    output: tuple[Tensor, Tensor, Tensor],
) -> None:
    *tensors, epsilon, gain_kind, path_swiglu = inputs
    ctx.epsilon = float(epsilon)
    ctx.gain_kind = int(gain_kind)
    ctx.path_swiglu = bool(path_swiglu)
    ctx.save_for_backward(*cast("tuple[Tensor, ...]", tuple(tensors)), output[2])


def _reference_recompute_backward(
    ctx: _AutogradContext,
    grad_outputs: tuple[Tensor, ...],
    *,
    epilogue: ProductScanEpilogue,
) -> tuple[Tensor | None, ...]:
    *inputs, _ = ctx.saved_tensors
    differentiable = tuple(value.detach().requires_grad_() for value in inputs)
    pole_x, pole_y, source = _recomputed_forward_inputs(differentiable)
    with torch.enable_grad():
        output = _product_scan_pipeline(
            *pole_x,
            *pole_y,
            *source,
            epilogue=epilogue,
            gain_normalization=gain_normalization(ctx.gain_kind),
            epsilon=ctx.epsilon,
        )
        outputs = output if isinstance(output, tuple) else (output,)
        gradients = torch.autograd.grad(outputs, differentiable, grad_outputs)
    return *gradients, None, None


def _cuda_recompute_backward_impl(
    inputs: tuple[Tensor, ...],
    descriptor: Tensor,
    grad_coarse_real: Tensor,
    grad_coarse_imag: Tensor,
    grad_descriptor: Tensor,
    *,
    emit_coarse: bool,
    full_coarse: bool,
    epsilon: float,
    gain_kind: int,
) -> ProductScanInputGradients:
    pole_x, pole_y, source = _recomputed_forward_inputs(inputs)
    horizontal = _horizontal_forward_op(*pole_x, *source)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source[0],
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    height, width = source[0].shape[1:3]
    descriptor_gradient_factor = (
        grad_descriptor * torch.exp(-descriptor) * (1.0 / (height * width))
    ).contiguous()
    if emit_coarse:
        launch_name = FULL16_BACKWARD_LAUNCH_NAME if full_coarse else BACKWARD_LAUNCH_NAME
        vertical_gradients = _launch_product_scan4_backward(
            *pole_y,
            *horizontal,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_coarse_real.contiguous(),
            grad_coarse_imag.contiguous(),
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=True,
            full_coarse=full_coarse,
            launch_name=launch_name,
            source_gradient_buffers=horizontal,
        )
    else:
        vertical_gradients = _launch_product_scan4_backward(
            *pole_y,
            *horizontal,
            variance_x,
            variance_y,
            global_inverse_gain,
            grad_coarse_real,
            grad_coarse_imag,
            descriptor_gradient_factor,
            epsilon,
            gain_kind,
            emit_coarse=False,
            full_coarse=False,
            launch_name=DESCRIPTOR_BACKWARD_LAUNCH_NAME,
            source_gradient_buffers=horizontal,
        )
    horizontal_gradients = _horizontal_recomputed_backward_op(
        *pole_x,
        *source,
        *horizontal,
    )
    return cast(
        "ProductScanInputGradients",
        (
            *horizontal_gradients[:4],
            *vertical_gradients[:4],
            *horizontal_gradients[4:],
        ),
    )


@triton_op("lnet::product_scan_pipeline_coarse_recompute_backward", mutates_args={})
def _coarse_recompute_backward_cuda_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    descriptor: Tensor,
    grad_coarse_real: Tensor,
    grad_coarse_imag: Tensor,
    grad_descriptor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> ProductScanInputGradients:
    inputs = (
        decay_x_real,
        decay_x_imag,
        gamma_x_real,
        gamma_x_imag,
        decay_y_real,
        decay_y_imag,
        gamma_y_real,
        gamma_y_imag,
        excitation_real,
        excitation_imag,
    )
    return _cuda_recompute_backward_impl(
        inputs,
        descriptor,
        grad_coarse_real,
        grad_coarse_imag,
        grad_descriptor,
        emit_coarse=True,
        full_coarse=False,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )


@triton_op("lnet::product_scan_pipeline_full16_recompute_backward", mutates_args={})
def _full16_recompute_backward_cuda_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    descriptor: Tensor,
    grad_full_real: Tensor,
    grad_full_imag: Tensor,
    grad_descriptor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> ProductScanInputGradients:
    inputs = (
        decay_x_real,
        decay_x_imag,
        gamma_x_real,
        gamma_x_imag,
        decay_y_real,
        decay_y_imag,
        gamma_y_real,
        gamma_y_imag,
        excitation_real,
        excitation_imag,
    )
    return _cuda_recompute_backward_impl(
        inputs,
        descriptor,
        grad_full_real,
        grad_full_imag,
        grad_descriptor,
        emit_coarse=True,
        full_coarse=True,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )


@triton_op("lnet::product_scan_pipeline_path_collapse_recompute_backward", mutates_args={})
def _path_collapse_recompute_backward_cuda_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    path_input_weight: Tensor,
    path_input_bias: Tensor,
    path_output_weight: Tensor,
    path_output_bias: Tensor,
    grad_collapsed_real: Tensor,
    grad_collapsed_imag: Tensor,
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
    Tensor,
    Tensor,
]:
    pole_x = decay_x_real, decay_x_imag, gamma_x_real, gamma_x_imag
    pole_y = decay_y_real, decay_y_imag, gamma_y_real, gamma_y_imag
    source = excitation_real, excitation_imag
    path = path_input_weight, path_input_bias, path_output_weight, path_output_bias
    horizontal = _horizontal_forward_op(*pole_x, *source)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        excitation_real,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )
    empty_descriptor_gradient = excitation_real.new_empty((0,))
    vertical_gradients = _product_scan_path_collapse_backward_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        *path,
        grad_collapsed_real,
        grad_collapsed_imag,
        empty_descriptor_gradient,
        epsilon,
        gain_kind,
        path_swiglu,
    )
    horizontal_gradients = _horizontal_recomputed_backward_op(
        *pole_x,
        *source,
        *vertical_gradients[4:8],
    )
    return (
        *horizontal_gradients[:4],
        *vertical_gradients[:4],
        *horizontal_gradients[4:],
        *vertical_gradients[8:12],
    )


@triton_op("lnet::product_scan_pipeline_descriptor_recompute_backward", mutates_args={})
def _descriptor_recompute_backward_cuda_op(
    decay_x_real: Tensor,
    decay_x_imag: Tensor,
    gamma_x_real: Tensor,
    gamma_x_imag: Tensor,
    decay_y_real: Tensor,
    decay_y_imag: Tensor,
    gamma_y_real: Tensor,
    gamma_y_imag: Tensor,
    excitation_real: Tensor,
    excitation_imag: Tensor,
    descriptor: Tensor,
    grad_descriptor: Tensor,
    epsilon: float,
    gain_kind: int,
) -> ProductScanInputGradients:
    inputs = (
        decay_x_real,
        decay_x_imag,
        gamma_x_real,
        gamma_x_imag,
        decay_y_real,
        decay_y_imag,
        gamma_y_real,
        gamma_y_imag,
        excitation_real,
        excitation_imag,
    )
    empty = excitation_real.new_empty((0,))
    return _cuda_recompute_backward_impl(
        inputs,
        descriptor,
        empty,
        empty,
        grad_descriptor,
        emit_coarse=False,
        full_coarse=False,
        epsilon=epsilon,
        gain_kind=gain_kind,
    )


def _coarse_recompute_backward_impl(
    ctx: _AutogradContext,
    grad_coarse_real: Tensor | None,
    grad_coarse_imag: Tensor | None,
    grad_descriptor: Tensor | None,
    *,
    full_coarse: bool,
) -> tuple[Tensor | None, ...]:
    source_real = ctx.saved_tensors[8]
    batch, height, width, modes = source_real.shape
    coarse_shape = (
        (batch, height // 2, width // 2, 4, 4, modes)
        if full_coarse
        else (batch, height // 2, width // 2, 4, modes)
    )
    coarse_real = (
        source_real.new_zeros(coarse_shape) if grad_coarse_real is None else grad_coarse_real
    )
    coarse_imag = (
        source_real.new_zeros(coarse_shape) if grad_coarse_imag is None else grad_coarse_imag
    )
    descriptor = ctx.saved_tensors[-1]
    descriptor_grad = torch.zeros_like(descriptor) if grad_descriptor is None else grad_descriptor
    if not source_real.is_cuda:
        return _reference_recompute_backward(
            ctx,
            (coarse_real, coarse_imag, descriptor_grad),
            epilogue="full16" if full_coarse else "coarse",
        )
    inputs = ctx.saved_tensors
    backward = (
        _full16_recompute_backward_cuda_op if full_coarse else _coarse_recompute_backward_cuda_op
    )
    gradients = backward(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        inputs[9],
        descriptor,
        coarse_real,
        coarse_imag,
        descriptor_grad,
        ctx.epsilon,
        ctx.gain_kind,
    )
    return *gradients, None, None


def _coarse_recompute_backward(
    ctx: _AutogradContext,
    grad_coarse_real: Tensor | None,
    grad_coarse_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    return _coarse_recompute_backward_impl(
        ctx,
        grad_coarse_real,
        grad_coarse_imag,
        grad_descriptor,
        full_coarse=False,
    )


def _full16_recompute_backward(
    ctx: _AutogradContext,
    grad_full_real: Tensor | None,
    grad_full_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    return _coarse_recompute_backward_impl(
        ctx,
        grad_full_real,
        grad_full_imag,
        grad_descriptor,
        full_coarse=True,
    )


def _path_collapse_recompute_backward(
    ctx: _AutogradContext,
    grad_collapsed_real: Tensor | None,
    grad_collapsed_imag: Tensor | None,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    del grad_descriptor
    source_real = ctx.saved_tensors[8]
    batch, height, width, modes = source_real.shape
    output_shape = (batch, height, width, 1, modes)
    collapsed_real = (
        source_real.new_zeros(output_shape)
        if grad_collapsed_real is None
        else grad_collapsed_real.contiguous()
    )
    collapsed_imag = (
        source_real.new_zeros(output_shape)
        if grad_collapsed_imag is None
        else grad_collapsed_imag.contiguous()
    )
    if not source_real.is_cuda:
        raise RuntimeError("the fused scan-path recompute boundary requires CUDA")
    inputs = ctx.saved_tensors[:-1]
    backward = cast(
        "Callable[..., PathCollapsePipelineGradients]",
        _path_collapse_recompute_backward_cuda_op,
    )
    gradients = backward(
        *inputs,
        collapsed_real,
        collapsed_imag,
        ctx.epsilon,
        ctx.gain_kind,
        ctx.path_swiglu,
    )
    return *gradients, None, None, None


def _descriptor_recompute_backward(
    ctx: _AutogradContext,
    grad_descriptor: Tensor | None,
) -> tuple[Tensor | None, ...]:
    source_real = ctx.saved_tensors[8]
    descriptor = ctx.saved_tensors[-1]
    descriptor_grad = torch.zeros_like(descriptor) if grad_descriptor is None else grad_descriptor
    if not source_real.is_cuda:
        return _reference_recompute_backward(
            ctx,
            (descriptor_grad,),
            epilogue="descriptor",
        )
    inputs = ctx.saved_tensors
    gradients = _descriptor_recompute_backward_cuda_op(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        inputs[9],
        descriptor,
        descriptor_grad,
        ctx.epsilon,
        ctx.gain_kind,
    )
    return *gradients, None, None


torch.library.register_autograd(
    "lnet::product_scan_pipeline_coarse_recompute",
    _coarse_recompute_backward,
    setup_context=_setup_coarse_recompute_context,
)
torch.library.register_autograd(
    "lnet::product_scan_pipeline_full16_recompute",
    _full16_recompute_backward,
    setup_context=_setup_coarse_recompute_context,
)
torch.library.register_autograd(
    "lnet::product_scan_pipeline_path_collapse_recompute",
    _path_collapse_recompute_backward,
    setup_context=_setup_path_collapse_recompute_context,
)
torch.library.register_autograd(
    "lnet::product_scan_pipeline_descriptor_recompute",
    _descriptor_recompute_backward,
    setup_context=_setup_descriptor_recompute_context,
)


@torch.compiler.disable
def _run_memory_efficient_pipeline(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    *,
    epilogue: ProductScanEpilogue,
    epsilon: float,
    gain_kind: int,
) -> ProductScanOutput:
    """Keep donation inside the scan boundary when an outer model is compiled."""
    semantic_args = epsilon, gain_kind
    if epilogue == "coarse":
        return _coarse_recompute_op(*pole_x, *pole_y, *source, *semantic_args)
    if epilogue == "full16":
        return _full16_recompute_op(*pole_x, *pole_y, *source, *semantic_args)
    return _descriptor_recompute_op(*pole_x, *pole_y, *source, *semantic_args)


@torch.compiler.disable
def _run_memory_efficient_path_collapse(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    path_collapse: PathCollapseParameters,
    *,
    epsilon: float,
    gain_kind: int,
    path_swiglu: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep the fused scan-path op opaque to outer Inductor graphs."""
    return _path_collapse_recompute_op(
        *pole_x,
        *pole_y,
        *source,
        *path_collapse,
        epsilon,
        gain_kind,
        path_swiglu,
    )


def run_product_scan_pipeline(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    *,
    epilogue: ProductScanEpilogue,
    gain_normalization: ProductGainNormalization,
    memory_policy: ScanMemoryPolicy = "retain",
    epsilon: float = DEFAULT_EPSILON,
) -> ProductScanOutput:
    """Run the optimized D4 scan, optionally replaying only its scan segment."""
    if epilogue not in {"coarse", "full16", "descriptor"}:
        message = f"unsupported product scan epilogue: {epilogue}"
        raise ValueError(message)
    if memory_policy not in {"retain", "recompute"}:
        message = f"unsupported scan memory policy: {memory_policy}"
        raise ValueError(message)
    should_recompute = (
        memory_policy == "recompute"
        and torch.is_grad_enabled()
        and any(value.requires_grad for value in (*pole_x, *pole_y, *source))
    )
    if not should_recompute:
        return _product_scan_pipeline(
            *pole_x,
            *pole_y,
            *source,
            epilogue=epilogue,
            gain_normalization=gain_normalization,
            epsilon=epsilon,
        )
    memory_efficient_pipeline = cast(
        "Callable[..., ProductScanOutput]",
        _run_memory_efficient_pipeline,
    )
    return memory_efficient_pipeline(
        pole_x,
        pole_y,
        source,
        epilogue=epilogue,
        epsilon=epsilon,
        gain_kind=gain_kind(gain_normalization),
    )


def run_product_scan_path_collapse_pipeline(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    path_collapse: PathCollapseParameters,
    *,
    gain_normalization: ProductGainNormalization,
    memory_policy: ScanMemoryPolicy = "recompute",
    path_swiglu: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor]:
    """Fuse full-resolution D4 scan and per-mode path collapse."""
    if memory_policy not in {"retain", "recompute"}:
        message = f"unsupported scan memory policy: {memory_policy}"
        raise ValueError(message)
    if not source[0].is_cuda:
        if source[0].shape[1] % 2 == 0 and source[0].shape[2] % 2 == 0:
            full = run_product_scan_pipeline(
                pole_x,
                pole_y,
                source,
                epilogue="full16",
                gain_normalization=gain_normalization,
                memory_policy="retain",
                epsilon=epsilon,
            )
            if not isinstance(full, tuple):
                raise TypeError("full16 reference did not return directional cells")
            collapse_reference = (
                d4_grouped_cell_path_swiglu_reference
                if path_swiglu
                else d4_grouped_cell_path_collapse_reference
            )
            return collapse_reference(full[0], full[1], *path_collapse)
        full = product_scan_full4_reference(
            pole_x,
            pole_y,
            source,
            gain_normalization=gain_normalization,
            epsilon=epsilon,
        )
        collapse_reference = (
            d4_grouped_path_swiglu_reference if path_swiglu else d4_grouped_path_collapse_reference
        )
        return collapse_reference(full[0], full[1], *path_collapse)

    semantic_args = epsilon, gain_kind(gain_normalization)
    should_recompute = (
        memory_policy == "recompute"
        and torch.is_grad_enabled()
        and any(value.requires_grad for value in (*pole_x, *pole_y, *source, *path_collapse))
    )
    if should_recompute:
        memory_efficient_path_collapse = cast(
            "Callable[..., tuple[Tensor, Tensor, Tensor]]",
            _run_memory_efficient_path_collapse,
        )
        output = memory_efficient_path_collapse(
            pole_x,
            pole_y,
            source,
            path_collapse,
            epsilon=semantic_args[0],
            gain_kind=semantic_args[1],
            path_swiglu=path_swiglu,
        )
        return output[0], output[1]

    horizontal = _horizontal_forward_op(*pole_x, *source)
    variance_x, variance_y, global_inverse_gain = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source[0],
        epsilon=epsilon,
        gain_kind=semantic_args[1],
    )
    output = _product_scan_path_collapse_op(
        *pole_y,
        *horizontal,
        variance_x,
        variance_y,
        global_inverse_gain,
        *path_collapse,
        *semantic_args,
        path_swiglu,
    )
    return output[0], output[1]


__all__ = [
    "PathCollapseParameters",
    "ProductScanEpilogue",
    "ProductScanOutput",
    "ScanMemoryPolicy",
    "run_product_scan_coarse_path_collapse_packed",
    "run_product_scan_coarse_path_collapse_planes_packed",
    "run_product_scan_path_collapse_pipeline",
    "run_product_scan_path_collapse_pipeline_packed",
    "run_product_scan_path_collapse_planes_packed",
    "run_product_scan_pipeline",
    "run_product_scan_pipeline_packed",
]


class _PackedAutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    epilogue: str
    epsilon: float
    kind: int
    recompute: bool
    path_swiglu: bool
    path_count: int
    planar_output: bool

    def set_materialize_grads(self, value: bool) -> None: ...

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...


def _packed_source_coordinates(source: Tensor) -> tuple[Tensor, Tensor]:
    if source.ndim != 4 or source.shape[-1] == 0 or source.shape[-1] % 2:
        raise ValueError("packed scan source must have shape NHW(2M)")
    if not source.is_contiguous():
        raise ValueError("packed scan source must be contiguous NHW(2M)")
    modes = source.shape[-1] // 2
    return source[..., :modes], source[..., modes:]


def _packed_reference(
    inputs: list[Tensor],
    path: list[Tensor],
    epilogue: str,
    epsilon: float,
    kind: int,
    path_swiglu: bool,
    planar_output: bool,
) -> tuple[Tensor, Tensor]:
    pole_x = cast("tuple[Tensor, Tensor, Tensor, Tensor]", tuple(inputs[:4]))
    pole_y = cast("tuple[Tensor, Tensor, Tensor, Tensor]", tuple(inputs[4:8]))
    source = _packed_source_coordinates(inputs[8])
    if epilogue == "mean_energy":
        raise ValueError("mean_energy epilogue is removed")
    if path and epilogue == "coarse_path":
        coarse_real, coarse_imag, _ = cast(
            "tuple[Tensor, Tensor, Tensor]",
            run_product_scan_pipeline(
                pole_x, pole_y, source, epilogue="coarse",
                gain_normalization=gain_normalization(kind),
                memory_policy="retain", epsilon=epsilon,
            ),
        )
        real, imag = d4_grouped_path_collapse_reference(coarse_real, coarse_imag, *path)
        return (
            torch.stack((real.squeeze(-2), imag.squeeze(-2)), dim=0)
            if planar_output else torch.cat((real.squeeze(-2), imag.squeeze(-2)), dim=-1)
        ), inputs[8].new_empty((0,))
    if path:
        real, imag = run_product_scan_path_collapse_pipeline(
            pole_x,
            pole_y,
            source,
            cast("PathCollapseParameters", tuple(path)),
            gain_normalization=gain_normalization(kind),
            memory_policy="retain",
            path_swiglu=path_swiglu,
            epsilon=epsilon,
        )
        return (
            torch.stack((real.squeeze(-2), imag.squeeze(-2)), dim=0)
            if planar_output else torch.cat((real.squeeze(-2), imag.squeeze(-2)), dim=-1)
        ), inputs[8].new_empty((0,))
    output = run_product_scan_pipeline(
        pole_x,
        pole_y,
        source,
        epilogue=cast(
            "ProductScanEpilogue", "coarse" if epilogue == "coarse_no_descriptor" else epilogue
        ),
        gain_normalization=gain_normalization(kind),
        memory_policy="retain",
        epsilon=epsilon,
    )
    if isinstance(output, Tensor):
        return inputs[8].new_empty((0,)), output
    return torch.cat(output[:2], dim=-1), (
        inputs[8].new_empty((0,)) if epilogue == "coarse_no_descriptor" else output[2]
    )


def _pipeline_launch_name(epilogue: str, has_path: bool, *, backward: bool) -> str:
    if epilogue == "coarse_path":
        return COARSE_PATH_BACKWARD_LAUNCH_NAME if backward else COARSE_PATH_FORWARD_LAUNCH_NAME
    if has_path:
        return PATH_COLLAPSE_BACKWARD_LAUNCH_NAME if backward else PATH_COLLAPSE_FORWARD_LAUNCH_NAME
    if epilogue == "full16":
        return FULL16_BACKWARD_LAUNCH_NAME if backward else FULL16_FORWARD_LAUNCH_NAME
    if epilogue in ("descriptor", "mean_energy"):
        return DESCRIPTOR_BACKWARD_LAUNCH_NAME if backward else DESCRIPTOR_FORWARD_LAUNCH_NAME
    return BACKWARD_LAUNCH_NAME if backward else FORWARD_LAUNCH_NAME


def _packed_horizontal_forward(
    pole: tuple[Tensor, Tensor, Tensor, Tensor], source: tuple[Tensor, Tensor],
) -> Tensor:
    if general_scans_enabled(forward=True):
        coefficients = list(pole)
        if can_use_general_hscan(*source, coefficients):
            return horizontal_forward(coefficients, *source)
    return _horizontal_forward_planes_op(*pole, *source)


def _run_packed_cuda_forward(
    inputs: list[Tensor],
    path: list[Tensor],
    epilogue: str,
    epsilon: float,
    kind: int,
    recompute: bool,
    path_swiglu: bool,
    *,
    packed_output: bool,
    planar_output: bool,
    squeeze_collapsed: bool,
) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
    pole_x = cast("tuple[Tensor, Tensor, Tensor, Tensor]", tuple(inputs[:4]))
    pole_y = cast("tuple[Tensor, Tensor, Tensor, Tensor]", tuple(inputs[4:8]))
    source = _packed_source_coordinates(inputs[8])
    horizontal_storage = _packed_horizontal_forward(pole_x, source)
    auxiliary = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source[0],
        epsilon=epsilon,
        gain_kind=kind,
    )

    if general_scans_enabled(forward=True):
        operands = _general_vertical_inputs(
            pole_y, horizontal_storage, auxiliary, path, epilogue, kind, path_swiglu,
        )
        saved = [] if recompute else [horizontal_storage]
        if operands is not None and _should_save_vscan(horizontal_storage, operands[-1]):
            output, secondary, features = _vertical_forward_saved(
                *operands, float(epsilon), planar=planar_output,
                precise=True, separate=not packed_output,
            )
            descriptor = output.new_empty((0,), dtype=torch.float32)
            return output, secondary, descriptor, [*saved, features]
        if packed_output and operands is not None:
            output = vertical_forward_packed(*operands, float(epsilon), planar_output)
            return (
                output,
                output.new_empty((0,)),
                output.new_empty((0,), dtype=torch.float32),
                saved,
            )
        if not packed_output:
            result = try_general_vertical_forward(
                pole_y, horizontal_storage, auxiliary, path, epilogue, epsilon, kind,
                packed_output, planar_output, path_swiglu, squeeze_collapsed,
            )
            if result is not None:
                return (*result, saved)
    horizontal = horizontal_storage.unbind(0)
    primary, secondary, descriptor = _launch_product_scan4_forward(
        *pole_y,
        *horizontal,
        *auxiliary,
        epsilon,
        kind,
        emit_coarse=not path and epilogue not in ("descriptor", "mean_energy"),
        full_coarse=epilogue == "full16",
        path_collapse=cast("PathCollapseParameters", tuple(path)) if path else None,
        path_swiglu=path_swiglu,
        emit_descriptor=not path and epilogue != "coarse_no_descriptor",
        packed_output=packed_output,
        planar_output=planar_output,
        squeeze_collapsed=squeeze_collapsed,
        mean_direction_energy=epilogue == "mean_energy",
        coarse_path=epilogue == "coarse_path",
        launch_name=_pipeline_launch_name(epilogue, bool(path), backward=False),
    )
    return primary, secondary, descriptor, ([] if recompute else [horizontal_storage])


def _general_packed_backward(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    horizontal: Tensor,
    auxiliary: tuple[Tensor, Tensor, Tensor],
    path: list[Tensor],
    grad_real: Tensor,
    grad_imag: Tensor,
    epilogue: str,
    epsilon: float,
    kind: int,
    path_swiglu: bool,
    saved_features: Tensor | None,
) -> list[Tensor] | None:
    x_coefficients = list(pole_x)
    operands = _general_vertical_inputs(
        pole_y, horizontal, auxiliary, path, epilogue, kind, path_swiglu,
    )
    if operands is None or not can_use_general_hscan(*source, x_coefficients):
        return None
    if grad_real.ndim == 5:
        grad_real, grad_imag = grad_real.squeeze(-2), grad_imag.squeeze(-2)
    py, states, tables, weights, coarse = operands
    vertical = vertical_backward(
        py, states, tables, weights, grad_real, grad_imag, coarse, float(epsilon),
        compact=coarse, saved_features=saved_features,
    )
    horizontal_grads = horizontal_backward(
        x_coefficients, list(source), horizontal, vertical[4], compact=coarse,
    )
    return [*horizontal_grads[:4], *vertical[:4], horizontal_grads[4], *vertical[5:]]


def _run_packed_cuda_backward(
    inputs: list[Tensor],
    path: list[Tensor],
    horizontal_states: list[Tensor],
    descriptor: Tensor,
    grad_real: Tensor,
    grad_imag: Tensor,
    grad_descriptor: Tensor,
    epilogue: str,
    epsilon: float,
    kind: int,
    path_swiglu: bool,
    *,
    packed_output: bool,
    planar_output: bool,
) -> list[Tensor]:
    pole_x = cast("tuple[Tensor, Tensor, Tensor, Tensor]", tuple(inputs[:4]))
    pole_y = cast("tuple[Tensor, Tensor, Tensor, Tensor]", tuple(inputs[4:8]))
    source = _packed_source_coordinates(inputs[8])
    saved_features = None
    if horizontal_states and horizontal_states[-1].ndim == 3:
        *horizontal_states, saved_features = horizontal_states
    horizontal_storage = (
        horizontal_states[0] if horizontal_states
        else _packed_horizontal_forward(pole_x, source)
    )
    auxiliary = static_product_scan_auxiliary(
        pole_x,
        pole_y,
        source[0],
        epsilon=epsilon,
        gain_kind=kind,
    )
    factor = (
        grad_descriptor * torch.exp(-descriptor) / (
            (4 if epilogue == "mean_energy" else 1) * source[0].shape[1] * source[0].shape[2]
        )
        if not path and epilogue != "coarse_no_descriptor"
        else descriptor
    )
    if general_scans_enabled():
        result = _general_packed_backward(
            pole_x, pole_y, source, horizontal_storage, auxiliary, path,
            grad_real, grad_imag, epilogue, epsilon, kind, path_swiglu, saved_features,
        )
        if result is not None:
            return result
    if not packed_output:
        grad_real, grad_imag = grad_real.contiguous(), grad_imag.contiguous()
    horizontal = cast("tuple[Tensor, Tensor, Tensor, Tensor]", horizontal_storage.unbind(0))
    vertical = _launch_product_scan4_backward(
        *pole_y,
        *horizontal,
        *auxiliary,
        grad_real,
        grad_imag,
        factor,
        epsilon,
        kind,
        emit_coarse=not path and epilogue not in ("descriptor", "mean_energy"),
        full_coarse=epilogue == "full16",
        path_collapse=cast("PathCollapseParameters", tuple(path)) if path else None,
        path_swiglu=path_swiglu,
        has_descriptor_grad=not path and epilogue != "coarse_no_descriptor",
        packed_output=packed_output,
        planar_output=planar_output,
        combined_state_gradients=True,
        mean_direction_energy=epilogue == "mean_energy",
        coarse_path=epilogue == "coarse_path",
        zero_unused_coarse_grads=epilogue != "coarse_path",
        launch_name=_pipeline_launch_name(epilogue, bool(path), backward=True),
        # AOT functionalization cannot preserve donation through separate read
        # and write pointer arguments. Keep recomputed states and adjoints in
        # independent allocations, just as the retained path does.
        source_gradient_buffers=None,
    )
    packed_gradient = torch.empty_like(inputs[8])
    horizontal_gradients = _launch_horizontal_backward(
        pole_x,
        source,
        horizontal if horizontal_states else None,
        cast("tuple[Tensor, Tensor, Tensor, Tensor]", tuple(vertical[4:8])),
        launch_name=(
            HORIZONTAL_BACKWARD_LAUNCH_NAME
            if horizontal_states
            else HORIZONTAL_RECOMPUTE_BACKWARD_LAUNCH_NAME
        ),
        source_gradient_buffer=packed_gradient,
        coarse_even_odd=epilogue == "coarse_path",
    )
    return [*horizontal_gradients[:4], *vertical[:4], packed_gradient, *vertical[8:]]


@triton_op("lnet::product_scan_pipeline_storage_v2", mutates_args={})
def _packed_pipeline_op(
    inputs: list[Tensor],
    path: list[Tensor],
    epilogue: str,
    epsilon: float,
    kind: int,
    recompute: bool,
    path_swiglu: bool,
    planar_output: bool,
) -> list[Tensor]:
    """Produce one packed output allocation; hidden outputs support retain policy."""
    if not inputs[8].is_cuda:
        return list(_packed_reference(
            inputs, path, epilogue, epsilon, kind, path_swiglu, planar_output,
        ))
    packed, _, descriptor, saved = _run_packed_cuda_forward(
        inputs,
        path,
        epilogue,
        epsilon,
        kind,
        recompute,
        path_swiglu,
        packed_output=True,
        planar_output=planar_output,
        squeeze_collapsed=False,
    )
    return [packed, descriptor, *saved]


def _setup_packed_context(
    ctx: _PackedAutogradContext,
    inputs: tuple[list[Tensor], list[Tensor], str, float, int, bool, bool, bool],
    output: list[Tensor],
) -> None:
    ctx.set_materialize_grads(False)
    (
        tensors, path, ctx.epilogue, ctx.epsilon, ctx.kind,
        ctx.recompute, ctx.path_swiglu, ctx.planar_output,
    ) = inputs
    ctx.path_count = len(path)
    ctx.mark_non_differentiable(*output[2:])
    ctx.save_for_backward(*tensors, *path, output[1], *output[2:])


@triton_op("lnet::product_scan_pipeline_storage_backward_v2", mutates_args={})
def _packed_pipeline_backward_op(
    inputs: list[Tensor],
    path: list[Tensor],
    horizontal_states: list[Tensor],
    descriptor: Tensor,
    grad_packed: Tensor,
    grad_descriptor: Tensor,
    epilogue: str,
    epsilon: float,
    kind: int,
    path_swiglu: bool,
    planar_output: bool,
) -> list[Tensor]:
    modes = inputs[8].shape[-1] // 2
    # Incoming gradients may have zero strides (e.g. sum); materialize only those.
    grad_packed = grad_packed.contiguous()
    if planar_output:
        grad_real, grad_imag = grad_packed[0], grad_packed[1]
    else:
        grad_real, grad_imag = grad_packed[..., :modes], grad_packed[..., modes:]
    if path:
        grad_real, grad_imag = grad_real.unsqueeze(-2), grad_imag.unsqueeze(-2)
    return _run_packed_cuda_backward(
        inputs,
        path,
        horizontal_states,
        descriptor,
        grad_real,
        grad_imag,
        grad_descriptor,
        epilogue,
        epsilon,
        kind,
        path_swiglu,
        packed_output=True,
        planar_output=planar_output,
    )


def _packed_pipeline_backward(
    ctx: _PackedAutogradContext,
    output_gradients: list[Tensor | None],
) -> tuple[list[Tensor], list[Tensor], None, None, None, None, None, None]:
    saved = ctx.saved_tensors
    inputs = list(saved[:9])
    path = list(saved[9 : 9 + ctx.path_count])
    descriptor = saved[9 + ctx.path_count]
    horizontal = list(saved[10 + ctx.path_count :])
    grad_packed, grad_descriptor = output_gradients[:2]
    source = inputs[8]
    modes = source.shape[-1] // 2
    if grad_packed is None:
        if path:
            shape = (
                (source.shape[0], source.shape[1] // 2, source.shape[2] // 2, 2 * modes)
                if ctx.epilogue == "coarse_path" else source.shape
            )
            if ctx.planar_output:
                shape = (2, *shape[:-1], modes)
        elif ctx.epilogue in ("descriptor", "mean_energy"):
            shape = (0,)
        else:
            shape = (
                source.shape[0],
                source.shape[1] // 2,
                source.shape[2] // 2,
                *((4, 4) if ctx.epilogue == "full16" else (4,)),
                2 * modes,
            )
        grad_packed = source.new_zeros(shape)
    if grad_descriptor is None:
        grad_descriptor = torch.zeros_like(descriptor)
    if not source.is_cuda:
        differentiable = [value.detach().requires_grad_() for value in (*inputs, *path)]
        with torch.enable_grad():
            packed, desc = _packed_reference(
                differentiable[:9],
                differentiable[9:],
                ctx.epilogue,
                ctx.epsilon,
                ctx.kind,
                ctx.path_swiglu,
                ctx.planar_output,
            )
            outputs, gradients = (
                ((packed,), (grad_packed,))
                if path or ctx.epilogue == "coarse_no_descriptor"
                else (
                    ((desc,), (grad_descriptor,))
                    if ctx.epilogue in ("descriptor", "mean_energy")
                    else ((packed, desc), (grad_packed, grad_descriptor))
                )
            )
            computed = torch.autograd.grad(outputs, differentiable, gradients, allow_unused=True)
        result = [
            torch.zeros_like(value) if gradient is None else gradient
            for value, gradient in zip(differentiable, computed, strict=True)
        ]
    else:
        result = _packed_pipeline_backward_op(
            inputs,
            path,
            horizontal,
            descriptor,
            grad_packed,
            grad_descriptor,
            ctx.epilogue,
            ctx.epsilon,
            ctx.kind,
            ctx.path_swiglu,
            ctx.planar_output,
        )
    return result[:9], result[9:], None, None, None, None, None, None


torch.library.register_autograd(
    "lnet::product_scan_pipeline_storage_v2",
    _packed_pipeline_backward,
    setup_context=_setup_packed_context,
)


@triton_op("lnet::product_scan_pipeline_planes_v1", mutates_args={})
def _packed_pipeline_planes_op(
    inputs: list[Tensor],
    path: list[Tensor],
    epilogue: str,
    epsilon: float,
    kind: int,
    recompute: bool,
    path_swiglu: bool,
) -> list[Tensor]:
    """Produce separate contiguous real and imaginary planes directly."""
    if not inputs[8].is_cuda:
        planar, descriptor = _packed_reference(
            inputs, path, epilogue, epsilon, kind, path_swiglu, True,
        )
        return [planar[0].clone(), planar[1].clone(), descriptor]
    real, imag, descriptor, saved = _run_packed_cuda_forward(
        inputs,
        path,
        epilogue,
        epsilon,
        kind,
        recompute,
        path_swiglu,
        packed_output=False,
        planar_output=False,
        squeeze_collapsed=True,
    )
    return [real, imag, descriptor, *saved]


def _setup_packed_planes_context(
    ctx: _PackedAutogradContext,
    inputs: tuple[list[Tensor], list[Tensor], str, float, int, bool, bool],
    output: list[Tensor],
) -> None:
    ctx.set_materialize_grads(False)
    (
        tensors, path, ctx.epilogue, ctx.epsilon, ctx.kind,
        ctx.recompute, ctx.path_swiglu,
    ) = inputs
    ctx.planar_output = False
    ctx.path_count = len(path)
    ctx.mark_non_differentiable(*output[3:])
    ctx.save_for_backward(*tensors, *path, output[2], *output[3:])


@triton_op("lnet::product_scan_pipeline_planes_backward_v1", mutates_args={})
def _packed_pipeline_planes_backward_op(
    inputs: list[Tensor],
    path: list[Tensor],
    horizontal_states: list[Tensor],
    descriptor: Tensor,
    grad_real: Tensor,
    grad_imag: Tensor,
    grad_descriptor: Tensor,
    epilogue: str,
    epsilon: float,
    kind: int,
    path_swiglu: bool,
) -> list[Tensor]:
    return _run_packed_cuda_backward(
        inputs,
        path,
        horizontal_states,
        descriptor,
        grad_real,
        grad_imag,
        grad_descriptor,
        epilogue,
        epsilon,
        kind,
        path_swiglu,
        packed_output=False,
        planar_output=False,
    )


def _packed_pipeline_planes_backward(
    ctx: _PackedAutogradContext,
    output_gradients: list[Tensor | None],
) -> tuple[list[Tensor], list[Tensor], None, None, None, None, None]:
    saved = ctx.saved_tensors
    inputs = list(saved[:9])
    path = list(saved[9 : 9 + ctx.path_count])
    descriptor = saved[9 + ctx.path_count]
    horizontal = list(saved[10 + ctx.path_count :])
    grad_real, grad_imag, grad_descriptor = output_gradients[:3]
    source = inputs[8]
    modes = source.shape[-1] // 2
    plane_shape = (
        (source.shape[0], source.shape[1] // 2, source.shape[2] // 2, modes)
        if ctx.epilogue == "coarse_path"
        else (*source.shape[:-1], modes)
    )
    if grad_real is None:
        grad_real = source.new_zeros(plane_shape)
    if grad_imag is None:
        grad_imag = source.new_zeros(plane_shape)
    if grad_descriptor is None:
        grad_descriptor = torch.zeros_like(descriptor)
    if not source.is_cuda:
        differentiable = [value.detach().requires_grad_() for value in (*inputs, *path)]
        with torch.enable_grad():
            planar, _ = _packed_reference(
                differentiable[:9],
                differentiable[9:],
                ctx.epilogue,
                ctx.epsilon,
                ctx.kind,
                ctx.path_swiglu,
                True,
            )
            computed = torch.autograd.grad(
                (planar[0], planar[1]),
                differentiable,
                (grad_real, grad_imag),
                allow_unused=True,
            )
        result = [
            torch.zeros_like(value) if gradient is None else gradient
            for value, gradient in zip(differentiable, computed, strict=True)
        ]
    else:
        result = _packed_pipeline_planes_backward_op(
            inputs,
            path,
            horizontal,
            descriptor,
            grad_real,
            grad_imag,
            grad_descriptor,
            ctx.epilogue,
            ctx.epsilon,
            ctx.kind,
            ctx.path_swiglu,
        )
    return result[:9], result[9:], None, None, None, None, None


torch.library.register_autograd(
    "lnet::product_scan_pipeline_planes_v1",
    _packed_pipeline_planes_backward,
    setup_context=_setup_packed_planes_context,
)


def run_product_scan_pipeline_packed(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_packed: Tensor,
    *,
    epilogue: ProductScanEpilogue | Literal["mean_energy"],
    gain_normalization: ProductGainNormalization,
    memory_policy: ScanMemoryPolicy = "retain",
    epsilon: float = DEFAULT_EPSILON,
    emit_descriptor: bool = True,
) -> tuple[Tensor, Tensor] | Tensor:
    """Scan NHW(2M) storage directly into packed directional coarse states."""
    _packed_source_coordinates(source_packed)
    if epilogue not in {"coarse", "full16", "descriptor", "mean_energy"}:
        raise ValueError(f"unsupported product scan epilogue: {epilogue}")
    if memory_policy not in {"retain", "recompute"}:
        raise ValueError(f"unsupported scan memory policy: {memory_policy}")
    if not emit_descriptor and epilogue != "coarse":
        raise ValueError("descriptor omission requires the coarse epilogue")
    output = _packed_pipeline_op(
        [*pole_x, *pole_y, source_packed],
        [],
        epilogue if emit_descriptor else "coarse_no_descriptor",
        epsilon,
        gain_kind(gain_normalization),
        memory_policy == "recompute",
        False,
        False,
    )
    return output[1] if epilogue in ("descriptor", "mean_energy") else (output[0], output[1])


def run_product_scan_path_collapse_pipeline_packed(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_packed: Tensor,
    parameters: PathCollapseParameters,
    *,
    gain_normalization: ProductGainNormalization,
    memory_policy: ScanMemoryPolicy = "recompute",
    path_swiglu: bool = False,
    epsilon: float = DEFAULT_EPSILON,
    planar_output: bool = False,
) -> Tensor:
    """Fuse scan and path collapse, writing a single NHW(2M) allocation."""
    _packed_source_coordinates(source_packed)
    if memory_policy not in {"retain", "recompute"}:
        raise ValueError(f"unsupported scan memory policy: {memory_policy}")
    return _packed_pipeline_op(
        [*pole_x, *pole_y, source_packed],
        list(parameters),
        "full16",
        epsilon,
        gain_kind(gain_normalization),
        memory_policy == "recompute",
        path_swiglu,
        planar_output,
    )[0]


def run_product_scan_path_collapse_planes_packed(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_packed: Tensor,
    parameters: PathCollapseParameters,
    *,
    gain_normalization: ProductGainNormalization,
    memory_policy: ScanMemoryPolicy = "recompute",
    path_swiglu: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[Tensor, Tensor]:
    """Fuse scan and path collapse, writing separate contiguous real/imag planes."""
    _packed_source_coordinates(source_packed)
    if memory_policy not in {"retain", "recompute"}:
        raise ValueError(f"unsupported scan memory policy: {memory_policy}")
    outputs = _packed_pipeline_planes_op(
        [*pole_x, *pole_y, source_packed],
        list(parameters),
        "full16",
        epsilon,
        gain_kind(gain_normalization),
        memory_policy == "recompute",
        path_swiglu,
    )
    return outputs[0], outputs[1]


def _validate_coarse_cuda_drive(source: Tensor, normalization: ProductGainNormalization) -> None:
    if not source.is_cuda or source.dtype is torch.bfloat16:
        return
    if (
        source.dtype in (torch.float16, torch.float32)
        and normalization == "pointwise"
        and general_scans_enabled(forward=True)
    ):
        return
    raise TypeError("fused coarse path collapse requires a BF16 CUDA drive")


def run_product_scan_coarse_path_collapse_packed(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_packed: Tensor,
    parameters: PathCollapseParameters,
    *,
    gain_normalization: ProductGainNormalization,
    epsilon: float = DEFAULT_EPSILON,
    planar_output: bool = False,
    memory_policy: ScanMemoryPolicy = "retain",
) -> Tensor:
    """Fuse the four directional 2x2 endpoints and Path CFFN without a coarse buffer.

    The canonical retained-state path keeps horizontal states for backward.
    The general CUDA path also supports FP16 and FP32 pointwise drives.
    """
    _packed_source_coordinates(source_packed)
    if memory_policy not in {"retain", "recompute"}:
        raise ValueError(f"unsupported scan memory policy: {memory_policy}")
    _validate_coarse_cuda_drive(source_packed, gain_normalization)
    if source_packed.shape[1] % 2 or source_packed.shape[2] % 2:
        raise ValueError("coarse path collapse requires even spatial dimensions")
    return _packed_pipeline_op(
        [*pole_x, *pole_y, source_packed], list(parameters), "coarse_path",
        epsilon, gain_kind(gain_normalization), memory_policy == "recompute", False, planar_output,
    )[0]


def run_product_scan_coarse_path_collapse_planes_packed(
    pole_x: tuple[Tensor, Tensor, Tensor, Tensor],
    pole_y: tuple[Tensor, Tensor, Tensor, Tensor],
    source_packed: Tensor,
    parameters: PathCollapseParameters,
    *,
    gain_normalization: ProductGainNormalization,
    epsilon: float = DEFAULT_EPSILON,
    memory_policy: ScanMemoryPolicy = "retain",
) -> tuple[Tensor, Tensor]:
    """Fuse coarse path collapse, writing separate contiguous real/imag planes."""
    _packed_source_coordinates(source_packed)
    if memory_policy not in {"retain", "recompute"}:
        raise ValueError(f"unsupported scan memory policy: {memory_policy}")
    _validate_coarse_cuda_drive(source_packed, gain_normalization)
    if source_packed.shape[1] % 2 or source_packed.shape[2] % 2:
        raise ValueError("coarse path collapse requires even spatial dimensions")
    outputs = _packed_pipeline_planes_op(
        [*pole_x, *pole_y, source_packed],
        list(parameters),
        "coarse_path",
        epsilon,
        gain_kind(gain_normalization),
        memory_policy == "recompute",
        False,
    )
    return outputs[0], outputs[1]
