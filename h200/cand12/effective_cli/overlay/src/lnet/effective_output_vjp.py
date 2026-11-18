"""Select a four-row output gradient reduction using a verified logical-update receipt."""

from dataclasses import replace
from functools import cache
import hashlib
from pathlib import Path
from types import FunctionType

import triton
import triton.language as tl

from .kernel_launch_config import (
    geometry_store_signature,
    register_kernel_specializer,
    validated_geometry,
)
from .output_dot_dispatch import dot_revision, select_output_dot
from .output_dot_kernel import _path_backward_kernel as _original_kernel
from .path_backward_layout import _LayoutGraphPass
from .retained_path_dispatch import PROFILE_NAME, VERTICAL_BACKWARD

VJP_NAME = "effective_output_vjp"
BUNDLE = Path(__file__).with_name("validated_effective_output_vjps.json")


@triton.jit
def _effective_output_vjp_transpose(
    grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
):
    dy = tl.where(
        channel[None, :, None] == 0,
        grad_real[:, None, :],
        tl.where(channel[None, :, None] == 1, grad_imag[:, None, :], 0.0),
    ).to(tl.bfloat16)
    features = tl.where(feature[None, None, :] == 8, 1.0, hidden).to(tl.bfloat16)
    partial = tl.permute(
        tl.dot(tl.permute(features, (0, 2, 1)), tl.permute(dy, (0, 2, 1))),
        (0, 2, 1),
    )
    row = 2 * mode[:, None] + channel[None, :]
    valid = (mode[:, None] < modes) & (channel[None, :] < 2)
    tl.atomic_add(
        dw + row[:, :, None] * hidden_size + feature[None, None, :],
        partial,
        valid[:, :, None] & (feature[None, None, :] < hidden_size),
        sem="relaxed",
    )
    bias_partial = tl.sum(tl.where(feature[None, None, :] == 8, partial, 0.0), axis=2)
    tl.atomic_add(db + row, bias_partial, valid, sem="relaxed")


@triton.jit
def _effective_output_vjp_rows(
    grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
):
    dy = tl.where(
        channel[None, :, None] == 0,
        grad_real[:, None, :],
        tl.where(channel[None, :, None] == 1, grad_imag[:, None, :], 0.0),
    ).to(tl.bfloat16)
    features = tl.where(feature[None, None, :] == 8, 1.0, hidden).to(tl.bfloat16)
    padded_partial = tl.dot(dy, features)
    real_channel = tl.arange(0, 2)
    row_indices = tl.broadcast_to(
        real_channel[None, :, None], (padded_partial.shape[0], 2, padded_partial.shape[2])
    )
    partial = tl.gather(padded_partial, row_indices, axis=1)
    row = 2 * mode[:, None] + real_channel[None, :]
    valid = mode[:, None] < modes
    tl.atomic_add(
        dw + row[:, :, None] * hidden_size + feature[None, None, :],
        partial,
        valid[:, :, None] & (feature[None, None, :] < hidden_size),
        sem="relaxed",
    )
    bias_partial = tl.sum(tl.where(feature[None, None, :] == 8, partial, 0.0), axis=2)
    tl.atomic_add(db + row, bias_partial, valid, sem="relaxed")


def _bind_vjp(helper):
    # A private namespace changes only the output-VJP dependency. Preserve all
    # caller bytecode, pointer arguments, scan logic, and original global objects.
    fn = _original_kernel.fn
    impl = FunctionType(
        fn.__code__,
        {**fn.__globals__, "_output_weight_bias_vjp": helper},
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    impl.__annotations__ = fn.__annotations__.copy()
    impl.__kwdefaults__ = fn.__kwdefaults__
    impl.__module__ = __name__
    return triton.jit(impl)


def _bind_transpose():
    # Inductor emits JIT dependency .src without rewriting the caller's global
    # aliases. This shim preserves the canonical call name and emits a uniquely
    # named helper in generated code. Triton inlines the forwarding call.
    @triton.jit
    def _output_weight_bias_vjp(
        grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
    ):
        _effective_output_vjp_transpose(
            grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
        )

    return _bind_vjp(_output_weight_bias_vjp)


def _bind_rows():
    @triton.jit
    def _output_weight_bias_vjp(
        grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
    ):
        _effective_output_vjp_rows(
            grad_real, grad_imag, hidden, mode, channel, feature, hidden_size, modes, dw, db
        )

    return _bind_vjp(_output_weight_bias_vjp)


_transpose_kernel = _bind_transpose()
_rows_kernel = _bind_rows()
_VARIANTS = {1: _transpose_kernel, 2: _rows_kernel}


@cache
def vjp_revision(parent_revision):
    digest = hashlib.sha256(dot_revision(parent_revision).encode())
    for variant, kernel in sorted(_VARIANTS.items()):
        digest.update(str(variant).encode())
        digest.update(kernel.cache_key.encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def select_effective_vjp(scope, base):
    selected = select_output_dot(scope, base)
    shape = dict(scope.shape_items)
    if (
        scope.execution_regime != "compiled"
        or shape.get("operation") != VERTICAL_BACKWARD
        or shape.get("height") != 56
        or shape.get("width") != 56
        or shape.get("coarse") != 0
    ):
        return selected
    exact = replace(scope, kernel_revision=vjp_revision(scope.kernel_revision))
    marker = validated_geometry(VJP_NAME, exact, BUNDLE)
    if selected is not _original_kernel or marker is None:
        return selected
    return _VARIANTS.get(marker.blocks.get("VARIANT"), selected)


def vjp_graph_pass(previous):
    register_kernel_specializer(PROFILE_NAME, select_effective_vjp)
    signature = geometry_store_signature(VJP_NAME, BUNDLE, omit_empty=True)
    return _LayoutGraphPass(vjp_revision(signature), previous) if signature else previous
