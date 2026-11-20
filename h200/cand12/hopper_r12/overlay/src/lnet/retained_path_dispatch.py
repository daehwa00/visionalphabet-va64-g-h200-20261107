"""Exact-scope dispatch for the retained BF16 path MLP and its horizontal adjoint."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import cache
from pathlib import Path

import torch
from torch import Tensor
from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.library import wrap_triton

from .direct_path_scan_kernels import (
    _horizontal_path_backward_kernel,
    _path_backward_kernel,
    _path_forward_kernel,
)
from .kernel_launch_config import (
    LaunchGeometry,
    autotuned,
    geometry_store_signature,
    make_launch_scope,
    validated_geometry,
)
from .path_backward_layout import _LayoutGraphPass, _driver_api_version
from .product_scan_contracts import DEFAULT_EPSILON

PROFILE_NAME = "retained_path_step"
BUNDLE = Path(__file__).with_name("validated_retained_steps.json")
VERTICAL_FORWARD, VERTICAL_BACKWARD, HORIZONTAL_BACKWARD = range(3)
_KERNELS = (_path_forward_kernel, _path_backward_kernel, _horizontal_path_backward_kernel)


_LEGACY_TRIAL = ContextVar("legacy_path_layout_trial", default=False)


@contextmanager
def legacy_layout_trial():
    """Let the original layout calibrator exercise its own kernels on a previously tuned GPU."""
    token = _LEGACY_TRIAL.set(True)
    try:
        yield
    finally:
        _LEGACY_TRIAL.reset(token)


def step_geometry(
    operation: int, group: int, modes: int, warps: int, doubling: int
) -> LaunchGeometry:
    if group not in (1, 2, 4, 8) or modes not in (2, 4, 8, 16, 32) or warps not in (1, 2, 4, 8):
        raise ValueError("Unsupported retained scan geometry")
    if doubling not in (0, 1) or (operation == HORIZONTAL_BACKWARD and (group != 1 or doubling)):
        raise ValueError("Unsupported retained scan algorithm")
    blocks = {"BM": modes, "GROUP": group}
    if operation != HORIZONTAL_BACKWARD:
        blocks["DOUBLING"] = doubling
    return LaunchGeometry.build(num_warps=warps, blocks=blocks)


@cache
def step_revision(operation: int) -> str:
    digest = hashlib.sha256(_KERNELS[operation].cache_key.encode())
    for path in (Path(__file__), Path(__file__).with_name("product_scan_pipeline.py")):
        # BEGIN R6 DISPATCH
        if path == Path(__file__):
            from .r6_dispatch import legacy_dispatch_source

            digest.update(legacy_dispatch_source(path))
            continue
        # END R6 DISPATCH
        # BEGIN R12 DISPATCH
        # R12 LEGACY:         digest.update(path.read_bytes())
        from .r6_dispatch import _pre_r12_source

        digest.update(_pre_r12_source(path))
        # END R12 DISPATCH
    return digest.hexdigest()


def step_graph_pass(previous: CustomGraphPass) -> CustomGraphPass:
    if _LEGACY_TRIAL.get():
        return previous
    signature = geometry_store_signature(PROFILE_NAME, BUNDLE, omit_empty=True)
    # No records for this device: retain the existing compiled-graph identity.
    if not signature:
        return previous
    key = "|".join([signature, *(step_revision(i) for i in range(len(_KERNELS)))])
    return _LayoutGraphPass(hashlib.sha256(key.encode()).hexdigest(), previous)


def _selection(
    source: Tensor,
    path: list[Tensor],
    epilogue: str,
    epsilon: float,
    kind: int,
    packed: bool,
    planar: bool,
    swiglu: bool,
    operation: int,
) -> LaunchGeometry | None:
    batch, height, width, modes = source.shape
    if not (
        not _LEGACY_TRIAL.get()
        and height == width == 56
        and not torch.is_inference_mode_enabled()
        and source.dtype == torch.bfloat16
        and epilogue in ("full16", "coarse_path")
        and epsilon == DEFAULT_EPSILON
        and kind == 0
        and not (packed or planar or swiglu)
        and source.is_cuda
        and len(path) == 4
        and tuple(x.shape for x in path) == ((modes, 8, 8), (modes, 8), (modes, 2, 8), (modes, 2))
        and all(
            x.is_contiguous() and x.dtype == torch.float32 and x.device == source.device
            for x in path
        )
    ):
        return None
    # The specialized vertical adjoint accumulates independent parameter rows per image.
    partials = os.environ.get(
        "VA_REAL_MEMORY_SCAN_BATCH_PARTIALS",
        os.environ.get("VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS"),
    )
    if operation == VERTICAL_BACKWARD and partials not in (None, "1"):
        return None

    # BEGIN R10 DISPATCH
    # BEGIN R6 DISPATCH
    if (
        operation == VERTICAL_FORWARD
        and epilogue == "full16"
        and not torch.are_deterministic_algorithms_enabled()
        and not inductor_config.emulate_precision_casts
    ):
        from .r10_dispatch import vertical_forward_geometry

        segmented = vertical_forward_geometry(batch, height, width, modes)
        if segmented is not None:
            return segmented
    # END R6 DISPATCH
    # END R10 DISPATCH
    shape = dict(
        batch=batch,
        height=height,
        width=width,
        modes=modes,
        operation=operation,
        coarse=int(epilogue == "coarse_path"),
        grad_enabled=int(torch.is_grad_enabled()),
        source_requires_grad=int(source.requires_grad),
        weight_requires_grad=int(path[0].requires_grad),
        source_row_stride=source.stride(-2),
        deterministic=int(torch.are_deterministic_algorithms_enabled()),
        emulate_precision_casts=int(inductor_config.emulate_precision_casts),
        cudnn_benchmark=int(torch.backends.cudnn.benchmark),
        cudnn_tf32=int(torch.backends.cudnn.allow_tf32),
    )
    with torch.cuda.device(source.device):
        scope = make_launch_scope(
            _KERNELS[operation],
            source,
            shape=shape,
            execution_regime="compiled"
            if torch.compiler.is_compiling() or get_proxy_mode() is not None
            else "eager",
        )
        scope = replace(
            scope,
            kernel_revision=step_revision(operation),
            cuda_driver_version=scope.cuda_driver_version or _driver_api_version(),
        )
        plan = validated_geometry(PROFILE_NAME, scope, BUNDLE)
    if plan is None:
        return None
    blocks = plan.blocks
    tile, group = blocks.get("BM", 0), blocks.get("GROUP", 0)
    scan_width = (
        width // 2 if epilogue == "coarse_path" and operation != HORIZONTAL_BACKWARD else width
    )
    try:
        expected = step_geometry(operation, group, tile, plan.num_warps, blocks.get("DOUBLING", 0))
    except ValueError:
        return None
    if (
        plan != expected
        or modes % tile
        or batch * scan_width % group
        or batch * height * width * modes * 4 >= 2**31
    ):
        return None
    return plan


def _kernel(operation: int, plan: LaunchGeometry):
    # Use a single-config Autotuner: Inductor preserves the verified warp count across lowering.
    return autotuned(_KERNELS[operation], f"{PROFILE_NAME}_{operation}", key=(), geometry=plan)


def _matching_poles(pole: tuple[Tensor, ...], source: Tensor) -> bool:
    return all(
        p.shape == (1, 1, 1, source.shape[-1])
        and p.dtype == torch.float32
        and p.device == source.device
        for p in pole
    )


def try_forward(
    pole, horizontal, auxiliary, path, epilogue, epsilon, kind, packed, planar, swiglu, squeeze
):
    # BEGIN R11 DISPATCH
    if not _LEGACY_TRIAL.get() and os.environ.get("LNET_R11_CUDA_VFWD") == "1":
        from .r11_dispatch import try_r11_vertical_forward

        r11_out = try_r11_vertical_forward(
            pole, horizontal, auxiliary, path, epilogue, epsilon, kind, packed, planar, swiglu, squeeze
        )
        if r11_out is not None:
            return r11_out
    # END R11 DISPATCH
    if epilogue != "full16" or not _matching_poles(pole, horizontal[0]):
        return None
    source = horizontal[0]
    plan = _selection(
        source, path, epilogue, epsilon, kind, packed, planar, swiglu, VERTICAL_FORWARD
    )
    if plan is None:
        return None
    b, h, w, m = source.shape
    shape = (b, h, w, m) if squeeze else (b, h, w, 1, m)
    real, imag = source.new_empty(shape), source.new_empty(shape)
    desc = torch.empty(0, device=source.device, dtype=torch.float32)

    def grid(meta):
        return meta["GROUP"] * (m // meta["BM"]), b * w // meta["GROUP"]

    wrap_triton(_kernel(VERTICAL_FORWARD, plan))[grid](
        *(v.contiguous() for v in (*pole, *horizontal, *auxiliary[:2])),
        *path,
        real,
        imag,
        H=h,
        W=w,
        M=m,
        EPS=epsilon,
    )
    return real, imag, desc


def try_backward(
    pole,
    horizontal,
    auxiliary,
    path,
    grad_real,
    grad_imag,
    epilogue,
    epsilon,
    kind,
    packed,
    planar,
    swiglu,
):
    # BEGIN R11 DISPATCH
    if not _LEGACY_TRIAL.get() and os.environ.get("LNET_R11_CUDA_VBACK") == "1":
        from .r11_dispatch import try_r11_vertical_backward

        r11_grads = try_r11_vertical_backward(
            pole,
            horizontal,
            auxiliary,
            path,
            grad_real,
            grad_imag,
            epilogue,
            epsilon,
            kind,
            packed,
            planar,
            swiglu,
        )
        if r11_grads is not None:
            return r11_grads
    # END R11 DISPATCH
    source = horizontal[0]
    if not _matching_poles(pole, source):
        return None
    plan = _selection(
        source, path, epilogue, epsilon, kind, packed, planar, swiglu, VERTICAL_BACKWARD
    )
    if plan is None:
        return None
    b, h, w, m = source.shape
    coarse = epilogue == "coarse_path"
    width = w // 2 if coarse else w
    output_h = h // 2 if coarse else h
    if (
        grad_real.shape != grad_imag.shape
        or grad_real.shape not in ((b, output_h, width, m), (b, output_h, width, 1, m))
        or grad_real.dtype != source.dtype
        or grad_imag.dtype != source.dtype
    ):
        return None
    param_grads = [v.new_zeros((b, *v.shape)) for v in (*pole, *path)]
    state_grads = source.new_empty((4, b, h, w, m))

    def grid(meta):
        return meta["GROUP"] * (m // meta["BM"]), b * width // meta["GROUP"]

    wrap_triton(_kernel(VERTICAL_BACKWARD, plan))[grid](
        *(v.contiguous() for v in (*pole, *horizontal, *auxiliary[:2])),
        *path[:3],
        grad_real.contiguous(),
        grad_imag.contiguous(),
        *param_grads[4:],
        *param_grads[:4],
        state_grads,
        h,
        w,
        b * w,
        m,
        path[0].shape[1],
        epsilon=epsilon,
        COARSE=coarse,
    )
    grads = [v.sum(0) for v in param_grads]
    return (*grads[:4], *state_grads.unbind(0), *grads[4:])


def try_horizontal(
    pole, sources, states, adjoints, result, path, epilogue, epsilon, kind, packed, planar, swiglu
):
    # BEGIN R11 DISPATCH
    if not _LEGACY_TRIAL.get() and os.environ.get("LNET_R11_CUDA_HBACK") == "1":
        from .r11_dispatch import try_r11_horizontal_backward

        r11_hgrads = try_r11_horizontal_backward(
            pole,
            sources,
            states,
            adjoints,
            result,
            coarse=epilogue == "coarse_path",
        )
        if r11_hgrads is not None:
            return r11_hgrads
    # END R11 DISPATCH
    source = sources[0]
    if (
        states is None
        or not _matching_poles(pole, source)
        or not states.is_contiguous()
        or states.shape != (4, *source.shape)
        or not result.is_contiguous()
        or result.shape != (*source.shape[:-1], 2 * source.shape[-1])
        or source.stride(-2) != sources[1].stride(-2)
        or any(
            v.stride()
            != (v.shape[1] * v.shape[2] * v.stride(-2), v.shape[2] * v.stride(-2), v.stride(-2), 1)
            for v in sources
        )
    ):
        return None
    plan = _selection(
        source, path, epilogue, epsilon, kind, packed, planar, swiglu, HORIZONTAL_BACKWARD
    )
    if plan is None:
        return None
    b, h, w, m = source.shape
    grads = tuple(torch.zeros_like(v, memory_format=torch.contiguous_format) for v in pole)
    # BEGIN R6 DISPATCH
    if (
        os.environ.get("LNET_R6_HBACK") == "1"
        and (h, w, m) == (56, 56, 160)
        and source.stride(-2) == 320
        and states.dtype == result.dtype == sources[1].dtype == torch.bfloat16
        and states.device == result.device == sources[1].device == source.device
        and len(adjoints) == 4
        and all(
            v.shape == source.shape and v.is_contiguous()
            and v.dtype == source.dtype and v.device == source.device
            for v in adjoints
        )
    ):
        from .r6_dispatch import horizontal_launch

        kernel = horizontal_launch(coarse=epilogue == "coarse_path")
        wrap_triton(kernel)[lambda meta: (m // meta["BM"], b * h)](
            *(v.contiguous() for v in pole),
            *sources,
            states,
            *adjoints,
            *grads,
            result,
            grads[0],
            H=h,
            W=w,
            M=m,
            B=b,
            STRIDE=source.stride(-2),
            COARSE=epilogue == "coarse_path",
            SEGMENT=True,
            PARTIAL=False,
        )
        return grads
    # END R6 DISPATCH
    wrap_triton(_kernel(HORIZONTAL_BACKWARD, plan))[lambda meta: (m // meta["BM"], b * h)](
        *(v.contiguous() for v in pole),
        *sources,
        states,
        *(v.contiguous() for v in adjoints),
        *grads,
        result,
        H=h,
        W=w,
        M=m,
        B=b,
        STRIDE=source.stride(-2),
        COARSE=epilogue == "coarse_path",
    )
    return grads
