"""Opt-in training compiler policies for Wave 36B."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch._inductor import config as inductor_config
from torch._inductor import list_mode_options

from .capture_safe_orthogonal import prepare_capture_safe_orthogonal_

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_GEMM_FEATURES = frozenset({"max_autotune_gemm", "max_autotune_gemm_aten", "tma_gemm"})
AUTOTUNE_FEATURES = _GEMM_FEATURES | {
    "max_autotune_pointwise", "coord_descent", "aggressive_fusion",
}
_FEATURES = AUTOTUNE_FEATURES | {"capture_safe_ortho", "cuda_graphs", "compiled_update"}


def model_w36b_features() -> frozenset[str]:
    policy = os.environ.get("LNET_MODEL_W36B", "off")
    features = frozenset(part.strip() for part in policy.split(","))
    if features == {"off"}:
        return frozenset()
    if features - _FEATURES:
        message = "LNET_MODEL_W36B must be off or a comma-separated list of Wave 36B features"
        raise ValueError(message)
    if "max_autotune_gemm_aten" in features and features & {"max_autotune_gemm", "tma_gemm"}:
        raise ValueError("LNET_MODEL_W36B: ATEN-only GEMM tuning excludes Triton GEMM features")
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
    if features & _GEMM_FEATURES:
        settings.update(
            max_autotune_gemm=True,
            max_autotune_gemm_backends=(
                "ATEN" if "max_autotune_gemm_aten" in features else "ATEN,TRITON"
            ),
            aggressive_fusion=False,
        )
    if "tma_gemm" in features:
        settings["max_autotune_gemm_search_space"] = "DEFAULT"
        if hasattr(inductor_config.triton, "enable_persistent_tma_matmul"):
            settings["triton.enable_persistent_tma_matmul"] = True
    if "max_autotune_pointwise" in features:
        settings.update(max_autotune_pointwise=True, epilogue_fusion=True)
    if "coord_descent" in features:
        settings["coordinate_descent_tuning"] = True
    if "aggressive_fusion" in features:
        settings["aggressive_fusion"] = True
    fullgraph = graphs or (autotune and "capture_safe_ortho" in features)
    return torch.compile(model, options=settings, fullgraph=fullgraph, dynamic=False)


def compile_model_w36b_update(
    model: nn.Module, optimizer: torch.optim.Optimizer, *, backend: str | None = None,
) -> Callable[[], torch.Tensor] | None:
    """Build the opt-in post-backward update once per training loop."""
    if "compiled_update" not in model_w36b_features():
        return None
    parameters = tuple(model.parameters())

    def optimizer_step() -> None:
        optimizer.step()

    # Dynamo decomposes unfused AdamW and can change its rounding. Retain its
    # eager implementation while allowing the fused ATen step to be compiled.
    step = (
        optimizer_step
        if isinstance(optimizer, torch.optim.AdamW)
        and all(group.get("fused") for group in optimizer.param_groups)
        else torch.compiler.disable(optimizer_step)
    )

    def update() -> torch.Tensor:
        norm = nn.utils.clip_grad_norm_(parameters, 1.0)
        step()
        optimizer.zero_grad(set_to_none=True)
        return norm

    if backend is not None:
        return torch.compile(update, backend=backend)
    return torch.compile(update, mode="reduce-overhead")


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
