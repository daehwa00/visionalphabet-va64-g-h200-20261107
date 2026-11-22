"""Measured bidirectional static product scan for the horizontal D4 path."""

from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportMissingParameterType=false
from typing import Protocol, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from .general_cuda_scans import (
    general_scans_enabled,
    try_general_horizontal_backward,
    try_general_horizontal_forward,
)
from .product_scan_reference import bidirectional_product_scan_reference

ComplexField = tuple[Tensor, Tensor]
BidirectionalField = tuple[Tensor, Tensor, Tensor, Tensor]

FORWARD_LAUNCH_NAME = "bidirectional_product_scan_forward"
BACKWARD_LAUNCH_NAME = "bidirectional_product_scan_backward"
RECOMPUTE_BACKWARD_LAUNCH_NAME = "bidirectional_product_scan_backward_recompute"
_BLOCK_MODES = 16
_NUM_WARPS = 4


class _AutogradContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


def _validate(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
) -> None:
    coefficients = decay_real, decay_imag, gamma_real, gamma_imag
    if source_real.shape != source_imag.shape or source_real.ndim != 4:
        raise ValueError("bidirectional product scan requires matching NHWM sources")
    for value in (source_real, source_imag):
        if value.stride(-1) != 1 or value.stride(-2) not in (
            source_real.shape[-1],
            2 * source_real.shape[-1],
        ):
            raise ValueError("bidirectional sources require dense coordinate rows")
        if not value.is_contiguous() and value.stride() != (
            value.shape[1] * value.shape[2] * value.stride(-2),
            value.shape[2] * value.stride(-2),
            value.stride(-2),
            1,
        ):
            raise ValueError("bidirectional sources require NHWM row-dense strides")
    if any(value.shape != (1, 1, 1, source_real.shape[-1]) for value in coefficients):
        raise ValueError("bidirectional product scan requires compact 111M poles")
    if any(value.device != source_real.device for value in (*coefficients, source_imag)):
        raise ValueError("bidirectional product scan tensors must share one device")
    if any(value.dtype != decay_real.dtype for value in coefficients[1:]):
        raise TypeError("bidirectional product scan poles must share one dtype")
    if source_imag.dtype != source_real.dtype:
        raise TypeError("bidirectional product scan sources must share one dtype")


@triton.jit
def _compose_complex(
    left_ar,
    left_ai,
    left_r,
    left_i,
    right_ar,
    right_ai,
    right_r,
    right_i,
):
    return (
        right_ar * left_ar - right_ai * left_ai,
        right_ai * left_ar + right_ar * left_ai,
        right_ar * left_r - right_ai * left_i + right_r,
        right_ai * left_r + right_ar * left_i + right_i,
    )


@triton.jit
def _bidirectional_forward_kernel(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    source_real,
    source_imag,
    positive_real,
    positive_imag,
    negative_real,
    negative_imag,
    height: int,
    width: int,
    line_count: int,
    modes: int,
    SOURCE_REAL_STRIDE: tl.constexpr,
    SOURCE_IMAG_STRIDE: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    COMBINED_STATES: tl.constexpr,
) -> None:
    line = tl.program_id(0)
    batch = line // height
    y = line - batch * height
    valid_line = line < line_count
    x = tl.arange(0, BLOCK_WIDTH)[:, None]
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)[None, :]
    active = valid_line & (x < width) & (mode < modes)
    offset = ((batch * height + y) * width + x) * modes + mode
    ar = tl.load(decay_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=mode < modes, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=mode < modes, other=0.0).to(tl.float32)
    source_real_offset = (line * width + x) * SOURCE_REAL_STRIDE + mode
    source_imag_offset = (line * width + x) * SOURCE_IMAG_STRIDE + mode
    sr = tl.load(source_real + source_real_offset, mask=active, other=0.0).to(tl.float32)
    si = tl.load(source_imag + source_imag_offset, mask=active, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    scan_ai = tl.where(active, ai, 0.0)
    positive = tl.associative_scan(
        (scan_ar, scan_ai, gr * sr - gi * si, gr * si + gi * sr),
        axis=0,
        combine_fn=_compose_complex,
    )
    negative = tl.associative_scan(
        (scan_ar, -scan_ai, gr * sr + gi * si, gr * si - gi * sr),
        axis=0,
        combine_fn=_compose_complex,
        reverse=True,
    )
    tl.store(positive_real + offset, positive[2], mask=active)
    if COMBINED_STATES:
        plane_size = tl.cast(line_count, tl.int64) * (width * modes)
        tl.store(positive_real + plane_size + offset, positive[3], mask=active)
        tl.store(positive_real + 2 * plane_size + offset, negative[2], mask=active)
        tl.store(positive_real + 3 * plane_size + offset, negative[3], mask=active)
    else:
        tl.store(positive_imag + offset, positive[3], mask=active)
        tl.store(negative_real + offset, negative[2], mask=active)
        tl.store(negative_imag + offset, negative[3], mask=active)


@triton.jit
def _bidirectional_backward_kernel(
    decay_real,
    decay_imag,
    gamma_real,
    gamma_imag,
    source_real,
    source_imag,
    positive_real,
    positive_imag,
    negative_real,
    negative_imag,
    grad_positive_real,
    grad_positive_imag,
    grad_negative_real,
    grad_negative_imag,
    grad_decay_real,
    grad_decay_imag,
    grad_gamma_real,
    grad_gamma_imag,
    grad_source_real,
    grad_source_imag,
    height: int,
    width: int,
    line_count: int,
    modes: int,
    SOURCE_REAL_STRIDE: tl.constexpr,
    SOURCE_IMAG_STRIDE: tl.constexpr,
    GRAD_SOURCE_STRIDE: tl.constexpr,
    PACKED_SOURCE_GRADIENT: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_MODES: tl.constexpr,
    RECOMPUTE_STATES: tl.constexpr,
    STORAGE_KIND: tl.constexpr,
    COARSE_EVEN_ODD: tl.constexpr = 0,
) -> None:
    line = tl.program_id(0)
    batch = line // height
    y = line - batch * height
    valid_line = line < line_count
    x = tl.arange(0, BLOCK_WIDTH)[:, None]
    mode = tl.program_id(1) * BLOCK_MODES + tl.arange(0, BLOCK_MODES)[None, :]
    active = valid_line & (x < width) & (mode < modes)
    valid_mode = valid_line & (mode < modes)
    offset = ((batch * height + y) * width + x) * modes + mode
    ar = tl.load(decay_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    ai = tl.load(decay_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gr = tl.load(gamma_real + mode, mask=valid_mode, other=0.0).to(tl.float32)
    gi = tl.load(gamma_imag + mode, mask=valid_mode, other=0.0).to(tl.float32)
    pos_mask = active & ((x & 1) == 1) if COARSE_EVEN_ODD else active
    neg_mask = active & ((x & 1) == 0) if COARSE_EVEN_ODD else active
    gpr = tl.load(grad_positive_real + offset, mask=pos_mask, other=0.0).to(tl.float32)
    gpi = tl.load(grad_positive_imag + offset, mask=pos_mask, other=0.0).to(tl.float32)
    gnr = tl.load(grad_negative_real + offset, mask=neg_mask, other=0.0).to(tl.float32)
    gni = tl.load(grad_negative_imag + offset, mask=neg_mask, other=0.0).to(tl.float32)
    scan_ar = tl.where(active, ar, 1.0)
    positive_adjoint = tl.associative_scan(
        (scan_ar, tl.where(active, -ai, 0.0), gpr, gpi),
        axis=0,
        combine_fn=_compose_complex,
        reverse=True,
    )
    negative_adjoint = tl.associative_scan(
        (scan_ar, tl.where(active, ai, 0.0), gnr, gni),
        axis=0,
        combine_fn=_compose_complex,
    )
    plr, pli = positive_adjoint[2], positive_adjoint[3]
    nlr, nli = negative_adjoint[2], negative_adjoint[3]
    source_real_offset = (line * width + x) * SOURCE_REAL_STRIDE + mode
    source_imag_offset = (line * width + x) * SOURCE_IMAG_STRIDE + mode
    sr = tl.load(source_real + source_real_offset, mask=active, other=0.0).to(tl.float32)
    si = tl.load(source_imag + source_imag_offset, mask=active, other=0.0).to(tl.float32)
    if RECOMPUTE_STATES:
        positive_previous_offset = offset - modes
        negative_previous_offset = offset + modes
        psr = tl.load(
            source_real + source_real_offset - SOURCE_REAL_STRIDE,
            mask=active & (x > 0),
            other=0.0,
        ).to(tl.float32)
        psi = tl.load(
            source_imag + source_imag_offset - SOURCE_IMAG_STRIDE,
            mask=active & (x > 0),
            other=0.0,
        ).to(tl.float32)
        nsr = tl.load(
            source_real + source_real_offset + SOURCE_REAL_STRIDE,
            mask=active & (x < width - 1),
            other=0.0,
        ).to(tl.float32)
        nsi = tl.load(
            source_imag + source_imag_offset + SOURCE_IMAG_STRIDE,
            mask=active & (x < width - 1),
            other=0.0,
        ).to(tl.float32)
        positive_previous = tl.associative_scan(
            (scan_ar, tl.where(active, ai, 0.0), gr * psr - gi * psi, gr * psi + gi * psr),
            axis=0,
            combine_fn=_compose_complex,
        )
        negative_previous = tl.associative_scan(
            (scan_ar, tl.where(active, -ai, 0.0), gr * nsr + gi * nsi, gr * nsi - gi * nsr),
            axis=0,
            combine_fn=_compose_complex,
            reverse=True,
        )
        ppr, ppi = positive_previous[2], positive_previous[3]
        npr, npi = negative_previous[2], negative_previous[3]
        # Match the retained path, whose horizontal states cross a storage
        # boundary before the vertical adjoint consumes them.
        if STORAGE_KIND == 1:
            ppr, ppi = ppr.to(tl.bfloat16).to(tl.float32), ppi.to(tl.bfloat16).to(tl.float32)
            npr, npi = npr.to(tl.bfloat16).to(tl.float32), npi.to(tl.bfloat16).to(tl.float32)
        elif STORAGE_KIND == 2:
            ppr, ppi = ppr.to(tl.float16).to(tl.float32), ppi.to(tl.float16).to(tl.float32)
            npr, npi = npr.to(tl.float16).to(tl.float32), npi.to(tl.float16).to(tl.float32)
    else:
        positive_previous_offset = offset - modes
        negative_previous_offset = offset + modes
        ppr = tl.load(
            positive_real + positive_previous_offset,
            mask=active & (x > 0),
            other=0.0,
        ).to(tl.float32)
        ppi = tl.load(
            positive_imag + positive_previous_offset,
            mask=active & (x > 0),
            other=0.0,
        ).to(tl.float32)
        npr = tl.load(
            negative_real + negative_previous_offset,
            mask=active & (x < width - 1),
            other=0.0,
        ).to(tl.float32)
        npi = tl.load(
            negative_imag + negative_previous_offset,
            mask=active & (x < width - 1),
            other=0.0,
        ).to(tl.float32)
    tl.store(
        grad_source_real + (line * width + x) * GRAD_SOURCE_STRIDE + mode,
        plr * gr + pli * gi + nlr * gr - nli * gi,
        mask=active,
    )
    if PACKED_SOURCE_GRADIENT:
        tl.store(
            grad_source_real + (line * width + x) * GRAD_SOURCE_STRIDE + mode + modes,
            -plr * gi + pli * gr + nlr * gi + nli * gr,
            mask=active,
        )
    else:
        tl.store(
            grad_source_imag + (line * width + x) * GRAD_SOURCE_STRIDE + mode,
            -plr * gi + pli * gr + nlr * gi + nli * gr,
            mask=active,
        )
    cdr = tl.sum(
        tl.where(active, plr * ppr + pli * ppi + nlr * npr + nli * npi, 0.0),
        axis=0,
    )
    cdi = tl.sum(
        tl.where(active, -plr * ppi + pli * ppr + nlr * npi - nli * npr, 0.0),
        axis=0,
    )
    cgr = tl.sum(
        tl.where(active, plr * sr + pli * si + nlr * sr + nli * si, 0.0),
        axis=0,
    )
    cgi = tl.sum(
        tl.where(active, -plr * si + pli * sr + nlr * si - nli * sr, 0.0),
        axis=0,
    )
    tl.atomic_add(grad_decay_real + mode, cdr[None, :], mask=valid_mode, sem="relaxed")
    tl.atomic_add(grad_decay_imag + mode, cdi[None, :], mask=valid_mode, sem="relaxed")
    tl.atomic_add(grad_gamma_real + mode, cgr[None, :], mask=valid_mode, sem="relaxed")
    tl.atomic_add(grad_gamma_imag + mode, cgi[None, :], mask=valid_mode, sem="relaxed")


@triton_op("lnet::bidirectional_product_scan", mutates_args={})
def _forward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
) -> BidirectionalField:
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    source = source_real, source_imag
    if not source_real.is_cuda:
        source = source_real.contiguous(), source_imag.contiguous()
    _validate(*pole, *source)
    if not source_real.is_cuda:
        return bidirectional_product_scan_reference(pole, source)
    if general_scans_enabled(forward=True):
        general_states = try_general_horizontal_forward(pole, source)
        if general_states is not None:
            # This custom op returns independent tensors. The planes op can
            # hand the packed allocation directly to callers that support it.
            return cast(
                "BidirectionalField", tuple(plane.clone() for plane in general_states.unbind(0))
            )
    outputs = cast(
        "BidirectionalField",
        tuple(
            torch.empty_like(source_real, memory_format=torch.contiguous_format) for _ in range(4)
        ),
    )
    batch, height, width, modes = source_real.shape
    line_count = batch * height
    wrap_triton(_bidirectional_forward_kernel)[(line_count, triton.cdiv(modes, _BLOCK_MODES))](
        *(value.contiguous() for value in pole),
        *source,
        *outputs,
        height,
        width,
        line_count,
        modes,
        SOURCE_REAL_STRIDE=source_real.stride(-2),
        SOURCE_IMAG_STRIDE=source_imag.stride(-2),
        BLOCK_WIDTH=triton.next_power_of_2(width),
        BLOCK_MODES=_BLOCK_MODES,
        num_warps=_NUM_WARPS,
        COMBINED_STATES=False,
    )
    return outputs


@triton_op("lnet::bidirectional_product_scan_planes", mutates_args={})
def _forward_planes_op(
    decay_real: Tensor, decay_imag: Tensor, gamma_real: Tensor, gamma_imag: Tensor,
    source_real: Tensor, source_imag: Tensor,
) -> Tensor:
    """Internal scan storage: four contiguous coordinate planes, one allocation."""
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    source = source_real, source_imag
    if not source_real.is_cuda:
        source = source_real.contiguous(), source_imag.contiguous()
    _validate(*pole, *source)
    if not source_real.is_cuda:
        return torch.stack(bidirectional_product_scan_reference(pole, source), dim=0)
    if general_scans_enabled(forward=True):
        general_states = try_general_horizontal_forward(pole, source)
        if general_states is not None:
            return general_states
    output = source_real.new_empty((4, *source_real.shape))
    dummy = source_real.new_empty((0,))
    batch, height, width, modes = source_real.shape
    line_count = batch * height
    wrap_triton(_bidirectional_forward_kernel)[(line_count, triton.cdiv(modes, _BLOCK_MODES))](
        *(value.contiguous() for value in pole), *source, output, dummy, dummy, dummy,
        height, width, line_count, modes,
        SOURCE_REAL_STRIDE=source_real.stride(-2), SOURCE_IMAG_STRIDE=source_imag.stride(-2),
        BLOCK_WIDTH=triton.next_power_of_2(width),
        BLOCK_MODES=_BLOCK_MODES,
        num_warps=_NUM_WARPS, COMBINED_STATES=True,
    )
    return output


def _storage_kind(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float16:
        return 2
    return 0


def _launch_backward(
    pole: tuple[Tensor, Tensor, Tensor, Tensor],
    source: ComplexField,
    states: BidirectionalField | None,
    output_gradients: BidirectionalField,
    *,
    launch_name: str,  # noqa: ARG001
    source_gradient_buffer: Tensor | None = None,
    coarse_even_odd: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    source_real, source_imag = source
    batch, height, width, modes = source_real.shape
    line_count = batch * height
    if states is not None and general_scans_enabled():
        general_grads = try_general_horizontal_backward(pole, source, states, output_gradients)
        if general_grads is not None:
            *pole_grads, packed_grad = general_grads
            if source_gradient_buffer is None:
                real_grad, imag_grad = packed_grad.split(modes, dim=-1)
                return (*pole_grads, real_grad.clone(), imag_grad.clone())
            source_gradient_buffer.copy_(packed_grad)
            return (*pole_grads, *source_gradient_buffer.split(modes, dim=-1))
    coefficient_gradients = cast(
        "tuple[Tensor, Tensor, Tensor, Tensor]",
        tuple(torch.zeros_like(value, memory_format=torch.contiguous_format) for value in pole),
    )
    source_gradients = (
        (
            torch.empty_like(source_real, memory_format=torch.contiguous_format),
            torch.empty_like(source_imag, memory_format=torch.contiguous_format),
        )
        if source_gradient_buffer is None
        else (source_gradient_buffer[..., :modes], source_gradient_buffer[..., modes:])
    )
    wrap_triton(_bidirectional_backward_kernel)[(line_count, triton.cdiv(modes, _BLOCK_MODES))](
        *(value.contiguous() for value in pole),
        *source,
        *(tuple(value.contiguous() for value in states) if states else (source_real,) * 4),
        *(value.contiguous() for value in output_gradients),
        *coefficient_gradients,
        *(
            source_gradients
            if source_gradient_buffer is None
            else (
                source_gradient_buffer,
                source_real.new_empty((0,)),
            )
        ),
        height,
        width,
        line_count,
        modes,
        SOURCE_REAL_STRIDE=source_real.stride(-2),
        SOURCE_IMAG_STRIDE=source_imag.stride(-2),
        GRAD_SOURCE_STRIDE=source_gradients[0].stride(-2),
        PACKED_SOURCE_GRADIENT=source_gradient_buffer is not None,
        BLOCK_WIDTH=triton.next_power_of_2(width),
        BLOCK_MODES=_BLOCK_MODES,
        num_warps=_NUM_WARPS,
        RECOMPUTE_STATES=states is None,
        STORAGE_KIND=_storage_kind(source_real.dtype),
        COARSE_EVEN_ODD=coarse_even_odd,
    )
    return *coefficient_gradients, *source_gradients


@triton_op("lnet::bidirectional_product_scan_backward", mutates_args={})
def _backward_op(
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
    positive_real: Tensor,
    positive_imag: Tensor,
    negative_real: Tensor,
    negative_imag: Tensor,
    grad_positive_real: Tensor,
    grad_positive_imag: Tensor,
    grad_negative_real: Tensor,
    grad_negative_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    source = source_real, source_imag
    _validate(*pole, *source)
    return _launch_backward(
        pole,
        source,
        (positive_real, positive_imag, negative_real, negative_imag),
        (grad_positive_real, grad_positive_imag, grad_negative_real, grad_negative_imag),
        launch_name=BACKWARD_LAUNCH_NAME,
    )


@triton_op("lnet::bidirectional_product_scan_backward_recompute", mutates_args={})
def _recomputed_backward_op(  # pyright: ignore[reportUnusedFunction]
    decay_real: Tensor,
    decay_imag: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    source_real: Tensor,
    source_imag: Tensor,
    grad_positive_real: Tensor,
    grad_positive_imag: Tensor,
    grad_negative_real: Tensor,
    grad_negative_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    pole = decay_real, decay_imag, gamma_real, gamma_imag
    source = source_real, source_imag
    _validate(*pole, *source)
    return _launch_backward(
        pole,
        source,
        None,
        (grad_positive_real, grad_positive_imag, grad_negative_real, grad_negative_imag),
        launch_name=RECOMPUTE_BACKWARD_LAUNCH_NAME,
    )


def _setup_context(
    ctx: _AutogradContext,
    inputs: tuple[Tensor, ...],
    output: BidirectionalField,
) -> None:
    ctx.save_for_backward(*inputs, *output)


def _backward(
    ctx: _AutogradContext,
    grad_positive_real: Tensor | None,
    grad_positive_imag: Tensor | None,
    grad_negative_real: Tensor | None,
    grad_negative_imag: Tensor | None,
) -> tuple[Tensor | None, ...]:
    *inputs, positive_real, positive_imag, negative_real, negative_imag = ctx.saved_tensors
    source_real = inputs[4]
    gradients = [
        torch.zeros_like(source_real) if value is None else value.contiguous()
        for value in (
            grad_positive_real,
            grad_positive_imag,
            grad_negative_real,
            grad_negative_imag,
        )
    ]
    if not source_real.is_cuda:
        differentiable = tuple(value.detach().requires_grad_() for value in inputs)
        with torch.enable_grad():
            outputs = bidirectional_product_scan_reference(
                (differentiable[0], differentiable[1], differentiable[2], differentiable[3]),
                (differentiable[4], differentiable[5]),
            )
            return torch.autograd.grad(outputs, differentiable, gradients)
    return _backward_op(
        *inputs,
        positive_real,
        positive_imag,
        negative_real,
        negative_imag,
        *gradients,
    )


torch.library.register_autograd(
    "lnet::bidirectional_product_scan",
    _backward,
    setup_context=_setup_context,
)


def triton_bidirectional_product_scan(
    pole: tuple[Tensor, Tensor, Tensor, Tensor],
    source: ComplexField,
) -> BidirectionalField:
    """Return positive and negative horizontal states without duplicate dispatch."""
    return _forward_op(*pole, *source)


__all__ = ["triton_bidirectional_product_scan"]
