"""Opt-in packed scan launches and CUDA source variants."""

from __future__ import annotations

# The low-level launchers are shared with the R11 custom ops.
# pyright: reportPrivateUsage=false
# ruff: noqa: FBT001, PLC0415, SLF001
import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor import config as inductor_config

from . import r11_cuda_hscan as hscan
from . import r11_cuda_vscan as vscan

if TYPE_CHECKING:
    from collections.abc import Sequence


def vertical_source(*, index32: bool, bias_last: bool) -> str:
    source = vscan._CUDA_SOURCE
    if index32:
        source = source.replace("long long", "int")
    if bias_last:
        # Accumulate the eight output terms in two FP32 sums, then add the bias.
        # Unroll by two so each iteration selects a fixed accumulator register.
        source = source.replace(
            "float acc_r = bor;\n                float acc_i = boi;\n"
            "                #pragma unroll 1",
            "float acc_r[2] = {};\n                float acc_i[2] = {};\n"
            "                #pragma unroll 2",
        )
        source = source.replace("acc_r += hid", "acc_r[h & 1] += hid")
        source = source.replace("acc_i += hid", "acc_i[h & 1] += hid")
        source = source.replace("f32_to_bf16(acc_r);", "f32_to_bf16((acc_r[0] + acc_r[1]) + bor);")
        source = source.replace("f32_to_bf16(acc_i);", "f32_to_bf16((acc_i[0] + acc_i[1]) + boi);")
        source = "\n".join(
            line for line in source.split("\n") if "__global__ void vscan_bwd_" not in line
        )
    return source


def _use_index32(plane_elements: int) -> bool:
    # The vertical backward writes up to the fourth BF16 adjoint plane.
    return os.environ.get("LNET_R12_INDEX32") == "1" and 4 * plane_elements <= 2**31


@lru_cache(maxsize=16)
def _vertical_module(device: int, *, index32: bool, bias_last: bool) -> vscan._VScanCudaModule:
    source = vertical_source(index32=index32, bias_last=bias_last)
    return vscan._VScanCudaModule(device, source, forward_only=bias_last)


def vscan_module(
    device: int, plane_elements: int, *, forward: bool = False
) -> vscan._VScanCudaModule:
    index32 = _use_index32(plane_elements)
    bias_last = forward and os.environ.get("LNET_R12_PRECISE") == "1"
    if index32 or bias_last:
        return _vertical_module(device, index32=index32, bias_last=bias_last)
    return vscan._get_module(device)


@triton.jit
def _reduce_parameter_grads(  # noqa: ANN202
    partials,  # noqa: ANN001
    gradients,  # noqa: ANN001
    batch,  # noqa: ANN001
    sizes: tl.constexpr,
    starts: tl.constexpr,
    batch_block: tl.constexpr,
    block: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tl.arange(0, batch_block)
    for i in tl.static_range(len(sizes)):
        if (tile >= starts[i]) & (tile < starts[i + 1]):
            columns = (tile - starts[i]) * block + tl.arange(0, block)
            values = tl.load(
                partials[i] + rows[:, None] * sizes[i] + columns[None, :],
                (rows[:, None] < batch) & (columns[None, :] < sizes[i]),
                other=0.0,
            )
            tl.store(gradients[i] + columns, tl.sum(values, axis=0), columns < sizes[i])


def _sum_parameter_grads(partials: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    batch = partials[0].shape[0]
    # Keep a bound on the number of input rows reduced by one CTA.
    if batch > 2048:
        return tuple(partial.sum(0) for partial in partials)
    gradients = tuple(partial.new_empty(partial.shape[1:]) for partial in partials)
    sizes = tuple(gradient.numel() for gradient in gradients)
    block = 32
    starts = [0]
    for size in sizes:
        starts.append(starts[-1] + triton.cdiv(size, block))
    _reduce_parameter_grads[(starts[-1],)](
        partials,
        gradients,
        batch,
        sizes,
        tuple(starts),
        triton.next_power_of_2(batch),
        block,
    )
    return gradients


@torch.library.custom_op("lnet::r12_vscan_forward_packed", mutates_args=())
def vertical_forward(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool,
    squeeze: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    return vscan._launch_vscan_forward(
        *pole,
        *states.unbind(0),
        *auxiliary,
        *path,
        coarse,
        squeeze,
        epsilon,
    )


@vertical_forward.register_fake
def _fake_vertical_forward(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    coarse: bool,
    squeeze: bool,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    return vscan._fake_vscan_forward(
        *pole,
        *states.unbind(0),
        *auxiliary,
        *path,
        coarse,
        squeeze,
        epsilon,
    )


@torch.library.custom_op("lnet::r12_vscan_backward_packed", mutates_args=())
def vertical_backward(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    grad_real: Tensor,
    grad_imag: Tensor,
    coarse: bool,
    epsilon: float,
) -> list[Tensor]:
    return list(
        vscan._launch_vscan_backward(
            *pole,
            *states.unbind(0),
            *auxiliary,
            *path,
            grad_real,
            grad_imag,
            coarse,
            epsilon,
            reduce_gradients=_sum_parameter_grads,
        )
    )


@vertical_backward.register_fake
def _fake_vertical_backward(
    pole: list[Tensor],
    states: Tensor,
    auxiliary: list[Tensor],
    path: list[Tensor],
    grad_real: Tensor,
    grad_imag: Tensor,
    coarse: bool,
    epsilon: float,
) -> list[Tensor]:
    return list(
        vscan._fake_vscan_backward(
            *pole,
            *states.unbind(0),
            *auxiliary,
            *path,
            grad_real,
            grad_imag,
            coarse,
            epsilon,
        )
    )


@torch.library.custom_op("lnet::r12_hscan_backward_packed", mutates_args=())
def horizontal_backward(
    pole: list[Tensor],
    source: list[Tensor],
    states: Tensor,
    adjoints: Tensor,
    coarse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    real, imag = source
    packed = real.new_empty((*real.shape[:-1], 2 * real.shape[-1]))
    gradients = hscan._launch_hscan_backward(
        *pole,
        real,
        imag,
        *states.unbind(0),
        *adjoints.unbind(0),
        packed,
        coarse,
        reduce_gradients=_sum_parameter_grads,
    )
    return (*gradients, packed)


@horizontal_backward.register_fake
def _fake_horizontal_backward(
    pole: list[Tensor],
    source: list[Tensor],
    states: Tensor,  # noqa: ARG001
    adjoints: Tensor,  # noqa: ARG001
    coarse: bool,  # noqa: ARG001
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    real = source[0]
    packed = real.new_empty((*real.shape[:-1], 2 * real.shape[-1]))
    return (*(torch.empty_like(p) for p in pole), packed)


def packed_pipeline_enabled() -> bool:
    from .retained_path_dispatch import _LEGACY_TRIAL

    return os.environ.get("LNET_R12_PACKED") == "1" and not _LEGACY_TRIAL.get()


def try_packed_horizontal_backward(
    pole: Sequence[Tensor],
    source: tuple[Tensor, Tensor],
    states: Tensor,
    adjoints: Tensor,
    coarse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None:
    from .r11_dispatch import _poles_match

    if (
        os.environ.get("LNET_R11_CUDA_HBACK") != "1"
        or torch.are_deterministic_algorithms_enabled()
        or inductor_config.emulate_precision_casts
        or not _poles_match(pole, source[0])
        or not hscan.can_use_cuda_hscan(*source)
    ):
        return None
    return horizontal_backward(list(pole), list(source), states, adjoints, coarse)
