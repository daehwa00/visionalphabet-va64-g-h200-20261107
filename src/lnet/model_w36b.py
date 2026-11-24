"""Opt-in stem and training compiler policies for Wave 36B."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn
from torch._inductor import list_mode_options

from .capture_safe_orthogonal import prepare_capture_safe_orthogonal_

_FEATURES = frozenset({"stem_cudnn", "capture_safe_ortho", "cuda_graphs", "max_autotune_gemm"})


def model_w36b_features() -> frozenset[str]:
    policy = os.environ.get("LNET_MODEL_W36B", "off")
    features = frozenset(part.strip() for part in policy.split(","))
    if features == {"off"}:
        return frozenset()
    if features - _FEATURES:
        message = "LNET_MODEL_W36B must be off or a comma-separated list of Wave 36B features"
        raise ValueError(message)
    return features


def prepare_model_w36b_(model: nn.Module) -> nn.Module:
    """Replace the exponential while retaining checkpoint and optimizer identities."""
    if model_w36b_features() & {"capture_safe_ortho", "cuda_graphs"}:
        # Graph capture depends on this replacement even if the legacy switch is off.
        prepare_capture_safe_orthogonal_(model, respect_env=False)
    return model


def compile_model_w36b(
    model: nn.Module, *, options: dict[str, Any] | None = None,
) -> nn.Module:
    """Prepare the model and retain compiler settings through lazy compilation."""
    features = model_w36b_features()
    prepare_model_w36b_(model)
    graphs = "cuda_graphs" in features
    autotune = "max_autotune_gemm" in features
    if options is None and not autotune:
        if graphs:
            return torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
        return model
    settings = dict(list_mode_options("reduce-overhead")) if graphs else {}
    settings.update(options or {})
    if autotune:
        settings.update(
            max_autotune_gemm=True,
            max_autotune_gemm_backends="ATEN,TRITON",
            aggressive_fusion=False,
        )
    return torch.compile(model, options=settings, fullgraph=graphs, dynamic=False)


def mark_model_w36b_step_begin() -> None:
    """Call before the first forward of each independent training iteration."""
    if "cuda_graphs" in model_w36b_features():
        torch.compiler.cudagraph_mark_step_begin()
