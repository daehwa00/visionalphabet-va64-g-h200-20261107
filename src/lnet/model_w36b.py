"""Opt-in training compiler policies for Wave 36B."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch._inductor import list_mode_options

from .capture_safe_orthogonal import prepare_capture_safe_orthogonal_

if TYPE_CHECKING:
    from collections.abc import Iterator

AUTOTUNE_FEATURES = frozenset({"max_autotune_gemm", "max_autotune_gemm_aten"})
_FEATURES = AUTOTUNE_FEATURES | {"capture_safe_ortho", "cuda_graphs"}


def model_w36b_features() -> frozenset[str]:
    policy = os.environ.get("LNET_MODEL_W36B", "off")
    features = frozenset(part.strip() for part in policy.split(","))
    if features == {"off"}:
        return frozenset()
    if features - _FEATURES or len(features & AUTOTUNE_FEATURES) > 1:
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
    autotune = bool(features & AUTOTUNE_FEATURES)
    if options is None and not autotune:
        if graphs:
            return torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
        return model
    settings = dict(list_mode_options("reduce-overhead")) if graphs else {}
    settings.update(options or {})
    if autotune:
        settings.update(
            max_autotune_gemm=True,
            max_autotune_gemm_backends=(
                "ATEN" if "max_autotune_gemm_aten" in features else "ATEN,TRITON"
            ),
            aggressive_fusion=False,
        )
    fullgraph = graphs or (autotune and "capture_safe_ortho" in features)
    return torch.compile(model, options=settings, fullgraph=fullgraph, dynamic=False)


def mark_model_w36b_step_begin() -> None:
    """Call before the first forward of each independent training iteration."""
    if "cuda_graphs" in model_w36b_features():
        torch.compiler.cudagraph_mark_step_begin()


@contextmanager
def count_cudagraph_replays() -> Iterator[list[int]]:
    """Collect CUDA Graph identities during a diagnostic step."""
    replays: list[int] = []
    replay = torch.cuda.CUDAGraph.replay

    def counted(graph: torch.cuda.CUDAGraph) -> None:
        replay(graph)
        replays.append(id(graph))

    torch.cuda.CUDAGraph.replay = counted
    try:
        yield replays
    finally:
        torch.cuda.CUDAGraph.replay = replay
