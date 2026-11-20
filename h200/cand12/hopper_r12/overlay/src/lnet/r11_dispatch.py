"""Opt-in dispatch for R11 NVRTC CUDA C++ vertical and horizontal scan kernels."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch._inductor import config as inductor_config

from .r11_cuda_hscan import (
    can_use_cuda_hscan,
    r11_cuda_hscan_backward_into_op,
    r11_cuda_hscan_forward_op,
)
from .r11_cuda_vscan import (
    can_use_cuda_vscan,
    r11_cuda_vscan_backward,
    r11_cuda_vscan_forward_op,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _poles_match(pole: Sequence[Tensor], reference: Tensor) -> bool:
    modes = reference.shape[-1]
    return len(pole) == 4 and all(
        p.shape == (1, 1, 1, modes)
        and p.dtype == torch.float32
        and p.device == reference.device
        for p in pole
    )


def _path_match(path: Sequence[Tensor], reference: Tensor) -> bool:
    modes = reference.shape[-1]
    return (
        len(path) == 4
        and tuple(p.shape for p in path) == ((modes, 8, 8), (modes, 8), (modes, 2, 8), (modes, 2))
        and all(p.dtype == torch.float32 and p.device == reference.device for p in path)
    )


def try_r11_vertical_forward(
    pole: Sequence[Tensor],
    # BEGIN R12 DISPATCH
    # R12 LEGACY:     horizontal: Sequence[Tensor],
    horizontal: Sequence[Tensor] | Tensor,
    # END R12 DISPATCH
    auxiliary: Sequence[Tensor],
    path: Sequence[Tensor],
    epilogue: str,
    epsilon: float,
    kind: int,
    packed: bool,  # noqa: FBT001
    planar: bool,  # noqa: FBT001
    swiglu: bool,  # noqa: FBT001
    squeeze: bool,  # noqa: FBT001
) -> tuple[Tensor, Tensor, Tensor] | None:
    if (
        os.environ.get("LNET_R11_CUDA_VFWD") != "1"
        or epilogue not in ("full16", "coarse_path")
        or kind != 0
        or packed
        or planar
        or swiglu
        or torch.are_deterministic_algorithms_enabled()
        or inductor_config.emulate_precision_casts
    ):
        return None
    # BEGIN R12 DISPATCH
    # A packed input stays intact at the opaque CUDA op boundary.
    packed_states = horizontal if isinstance(horizontal, Tensor) else None
    if packed_states is not None and (packed_states.ndim != 5 or not packed_states.is_contiguous()):
        return None
    # END R12 DISPATCH
    if len(horizontal) != 4 or len(auxiliary) < 2:
        return None
    source = horizontal[0]
    coarse = epilogue == "coarse_path"
    if not (
        _poles_match(pole, source)
        and _path_match(path, source)
        and can_use_cuda_vscan(source, path[0], coarse=coarse)
        # BEGIN R12 DISPATCH
        # R12 LEGACY:         and all(
        and (packed_states is not None or all(
        # END R12 DISPATCH
            h.shape == source.shape
            and h.dtype == torch.bfloat16
            and h.device == source.device
            for h in horizontal[1:]
        # BEGIN R12 DISPATCH
        # R12 LEGACY:         )
        ))
        # END R12 DISPATCH
        and all(
            v.dtype == torch.float32 and v.device == source.device
            for v in auxiliary[:2]
        )
    ):
        return None
    # BEGIN R12 DISPATCH
    if packed_states is not None:
        from .r12_cuda_scans import vertical_forward

        out_r, out_i = vertical_forward(
            list(pole), packed_states, list(auxiliary[:2]), list(path),
            coarse, squeeze, float(epsilon),
        )
        return out_r, out_i, torch.empty(0, device=source.device, dtype=torch.float32)
    # END R12 DISPATCH
    out_r, out_i = r11_cuda_vscan_forward_op(
        pole[0],
        pole[1],
        pole[2],
        pole[3],
        horizontal[0],
        horizontal[1],
        horizontal[2],
        horizontal[3],
        auxiliary[0],
        auxiliary[1],
        path[0],
        path[1],
        path[2],
        path[3],
        coarse,
        squeeze,
        float(epsilon),
    )
    desc = torch.empty(0, device=source.device, dtype=torch.float32)
    return out_r, out_i, desc


def try_r11_vertical_backward(
    pole: Sequence[Tensor],
    # BEGIN R12 DISPATCH
    # R12 LEGACY:     horizontal: Sequence[Tensor],
    horizontal: Sequence[Tensor] | Tensor,
    # END R12 DISPATCH
    auxiliary: Sequence[Tensor],
    path: Sequence[Tensor],
    grad_real: Tensor,
    grad_imag: Tensor,
    epilogue: str,
    epsilon: float,
    kind: int,
    packed: bool,  # noqa: FBT001
    planar: bool,  # noqa: FBT001
    swiglu: bool,  # noqa: FBT001
    *,
    zero_unused_coarse_grads: bool = False,
) -> tuple[Tensor, ...] | None:
    if (
        os.environ.get("LNET_R11_CUDA_VBACK") != "1"
        or epilogue not in ("full16", "coarse_path")
        or kind != 0
        or packed
        or planar
        or swiglu
        or torch.are_deterministic_algorithms_enabled()
        or inductor_config.emulate_precision_casts
    ):
        return None
    coarse = epilogue == "coarse_path"
    if coarse and zero_unused_coarse_grads:
        return None
    # BEGIN R12 DISPATCH
    # A packed input stays intact at the opaque CUDA op boundary.
    packed_states = horizontal if isinstance(horizontal, Tensor) else None
    if packed_states is not None and (packed_states.ndim != 5 or not packed_states.is_contiguous()):
        return None
    # END R12 DISPATCH
    if len(horizontal) != 4 or len(auxiliary) < 2:
        return None
    source = horizontal[0]
    if not (
        _poles_match(pole, source)
        and _path_match(path, source)
        and can_use_cuda_vscan(source, path[0], coarse=coarse)
        # BEGIN R12 DISPATCH
        # R12 LEGACY:         and all(
        and (packed_states is not None or all(
        # END R12 DISPATCH
            h.shape == source.shape
            and h.dtype == torch.bfloat16
            and h.device == source.device
            for h in horizontal[1:]
        # BEGIN R12 DISPATCH
        # R12 LEGACY:         )
        ))
        # END R12 DISPATCH
        and all(
            v.dtype == torch.float32 and v.device == source.device
            for v in auxiliary[:2]
        )
    ):
        return None
    b, h, w, m = source.shape
    oh = h // 2 if coarse else h
    ow = w // 2 if coarse else w
    if (
        grad_real.shape != grad_imag.shape
        or grad_real.shape not in ((b, oh, ow, m), (b, oh, ow, 1, m))
        or grad_real.dtype != torch.bfloat16
        or grad_imag.dtype != torch.bfloat16
        or grad_real.device != source.device
        or grad_imag.device != source.device
    ):
        return None
    # BEGIN R12 DISPATCH
    if packed_states is not None:
        from .r12_cuda_scans import vertical_backward

        return tuple(vertical_backward(
            list(pole), packed_states, list(auxiliary[:2]), list(path),
            grad_real, grad_imag, coarse, float(epsilon),
        ))
    # END R12 DISPATCH
    return r11_cuda_vscan_backward(
        pole[0],
        pole[1],
        pole[2],
        pole[3],
        horizontal[0],
        horizontal[1],
        horizontal[2],
        horizontal[3],
        auxiliary[0],
        auxiliary[1],
        path[0],
        path[1],
        path[2],
        path[3],
        grad_real,
        grad_imag,
        coarse=coarse,
        epsilon=float(epsilon),
    )


def try_r11_horizontal_forward(
    pole: Sequence[Tensor],
    source: tuple[Tensor, Tensor],
) -> Tensor | None:
    if (
        os.environ.get("LNET_R11_CUDA_HFWD") != "1"
        or torch.are_deterministic_algorithms_enabled()
        or inductor_config.emulate_precision_casts
    ):
        return None
    source_real, source_imag = source
    if not (_poles_match(pole, source_real) and can_use_cuda_hscan(source_real, source_imag)):
        return None
    return r11_cuda_hscan_forward_op(
        pole[0],
        pole[1],
        pole[2],
        pole[3],
        source_real,
        source_imag,
    )


def try_r11_horizontal_backward(
    pole: Sequence[Tensor],
    source: tuple[Tensor, Tensor],
    states: Tensor | Sequence[Tensor] | None,
    adjoints: Sequence[Tensor],
    packed_result: Tensor | None,
    *,
    coarse: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    if (
        os.environ.get("LNET_R11_CUDA_HBACK") != "1"
        or states is None
        or packed_result is None
        or torch.are_deterministic_algorithms_enabled()
        or inductor_config.emulate_precision_casts
    ):
        return None
    source_real, source_imag = source
    if not (
        _poles_match(pole, source_real)
        and can_use_cuda_hscan(source_real, source_imag)
        and packed_result.is_contiguous()
        and packed_result.dtype == torch.bfloat16
        and packed_result.device == source_real.device
        and packed_result.shape == (*source_real.shape[:-1], 2 * source_real.shape[-1])
        and len(adjoints) == 4
        and all(
            a.shape == source_real.shape
            and a.dtype == torch.bfloat16
            and a.device == source_real.device
            for a in adjoints
        )
    ):
        return None
    if isinstance(states, Tensor):
        if (
            states.shape != (4, *source_real.shape)
            or states.dtype != torch.bfloat16
            or states.device != source_real.device
        ):
            return None
        state_planes = states.unbind(0)
    else:
        if len(states) != 4 or any(
            s.shape != source_real.shape
            or s.dtype != torch.bfloat16
            or s.device != source_real.device
            for s in states
        ):
            return None
        state_planes = tuple(states)
    return r11_cuda_hscan_backward_into_op(
        pole[0],
        pole[1],
        pole[2],
        pole[3],
        source_real,
        source_imag,
        state_planes[0],
        state_planes[1],
        state_planes[2],
        state_planes[3],
        adjoints[0],
        adjoints[1],
        adjoints[2],
        adjoints[3],
        packed_result,
        coarse,
    )
