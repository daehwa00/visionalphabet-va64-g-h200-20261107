"""Opt-in training compiler policies for Wave 36B."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch._inductor import config as inductor_config
from torch._inductor import list_mode_options
from torch.optim.optimizer import _global_optimizer_post_hooks, _global_optimizer_pre_hooks

from .capture_safe_orthogonal import prepare_capture_safe_orthogonal_

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_GEMM_FEATURES = frozenset({"max_autotune_gemm", "max_autotune_gemm_aten", "tma_gemm"})
AUTOTUNE_FEATURES = _GEMM_FEATURES | {
    "max_autotune_pointwise", "coord_descent", "aggressive_fusion", "shape_padding",
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
    if "shape_padding" in features:
        settings["shape_padding"] = True
    fullgraph = graphs or (autotune and "capture_safe_ortho" in features)
    return torch.compile(model, options=settings, fullgraph=fullgraph, dynamic=False)


def _clip_static_grads(grads: list[torch.Tensor]) -> torch.Tensor:
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads, 2.0)), 2.0)  # noqa: SLF001
    torch._foreach_mul_(grads, (1.0 / (norm + 1e-6)).clamp(max=1.0))
    return norm


def _tensor_update(
    grads: list[torch.Tensor],
    groups: list[tuple],
    *,
    functional: bool = False,
) -> torch.Tensor:
    """Single-device, single-dtype fused AdamW update using stable inputs."""
    with torch.no_grad():
        norm = _clip_static_grads(grads)
        for params, active, avg, sq, max_sq, steps, settings in groups:
            torch._foreach_add_(steps, 1)
            if functional:
                # AOT functionalization cannot lower the in-place overload in 2.9.1.
                updated = torch.ops.aten._fused_adamw.default(  # noqa: SLF001
                    params, active, avg, sq, max_sq, steps, **settings,
                )
                for original, new in zip((params, active, avg, sq, max_sq), updated, strict=True):
                    if original:
                        torch._foreach_copy_(original, new)
            else:
                torch._fused_adamw_(params, active, avg, sq, max_sq, steps, **settings)
        torch._foreach_zero_(grads)
        return norm


def _adamw_settings(group: dict) -> dict:
    beta1, beta2 = group["betas"]
    return {
        "lr": group["lr"], "beta1": beta1, "beta2": beta2,
        "weight_decay": group["weight_decay"], "eps": group["eps"],
        "maximize": group["maximize"], "amsgrad": group["amsgrad"],
    }


class _StaticGradUpdate:
    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, backend: str | None,
    ) -> None:
        self.params = tuple(model.parameters())
        self.optimizer = optimizer
        self.backend = backend
        self.core = torch.compile(_tensor_update, backend=backend, fullgraph=True) if backend else (
            _tensor_update
        )
        self.zeroed: dict[nn.Parameter, tuple[torch.Tensor, int]] = {}
        self.graph: torch.cuda.CUDAGraph | None = None
        self.key: tuple | None = None
        self.norm: torch.Tensor | None = None
        self.graph_inputs: list[torch.Tensor] = []

    def _inputs(self, active: dict[nn.Parameter, torch.Tensor]) -> list[tuple] | None:  # noqa: C901
        opt = self.optimizer
        if (
            type(opt) is not torch.optim.AdamW
            or not active
            or hasattr(opt, "grad_scale")
            or hasattr(opt, "found_inf")
            or opt._optimizer_step_pre_hooks  # noqa: SLF001
            or opt._optimizer_step_post_hooks  # noqa: SLF001
            or _global_optimizer_pre_hooks
            or _global_optimizer_post_hooks
        ):
            return None
        first = next(iter(active.values()))
        device, dtype = first.device, first.dtype
        if (device.type != "cuda" and self.backend is None) or any(
            g.device != device or g.dtype != dtype or g.layout != torch.strided
            or g.requires_grad for g in active.values()
        ):
            return None
        groups = []
        for group in opt.param_groups:
            if not group["fused"] or group["differentiable"]:
                return None
            settings = _adamw_settings(group)
            if any(torch.is_tensor(value) for value in settings.values()):
                return None
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            if any(p not in active for p in params):
                return None
            states = [opt.state.get(p, {}) for p in params]
            avg, sq, max_sq, steps = [], [], [], []
            for state in states:
                if not state:
                    continue  # The first eager step initializes this state.
                avg.append(state["exp_avg"])
                sq.append(state["exp_avg_sq"])
                steps.append(state["step"])
                if settings["amsgrad"]:
                    max_sq.append(state["max_exp_avg_sq"])
            groups.append((params, [active[p] for p in params], avg, sq, max_sq, steps, settings))
        return groups if sum(len(group[0]) for group in groups) == len(active) else None

    def __call__(self) -> torch.Tensor:
        active = {}
        for param in self.params:
            grad = param.grad
            if grad is None:
                continue
            previous = self.zeroed.get(param)
            if previous is not None and grad is previous[0] and grad._version == previous[1]:  # noqa: SLF001
                # This parameter did not participate in the latest backward.
                param.grad = None
            else:
                active[param] = grad
        groups = self._inputs(active)
        warmup = not self.zeroed or (
            groups is not None and any(len(p) != len(steps) for p, *_, steps, _ in groups)
        )
        if groups is None or warmup:
            self.graph = self.key = self.norm = None
            self.graph_inputs = []
            norm = nn.utils.clip_grad_norm_(self.params, 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=groups is None)
            if groups is not None and self.backend is None:
                # AccumulateGrad may adopt the compiled backward output on the first
                # call. Give persistent grads storage outside the Inductor graph pool.
                for param in active:
                    param.grad = param.grad.clone()
                active = {p: p.grad for p in active}
        elif self.backend is not None:
            norm = self.core(list(active.values()), groups, functional=self.backend != "eager")
        else:
            norm = self._replay(list(active.values()), groups)
        self.zeroed = {p: (g, g._version) for p, g in active.items() if p.grad is g}  # noqa: SLF001
        return norm

    def _replay(self, grads: list[torch.Tensor], groups: list[tuple]) -> torch.Tensor:
        inputs = grads + [t for group in groups for tensors in group[:-1] for t in tensors]
        key = (
            tuple((id(t), t.data_ptr()) for t in inputs),
            tuple(tuple(group[-1].values()) for group in groups),
            tuple(len(group[0]) for group in groups),
        )
        if key != self.key and self.graph is not None:
            # Run the changed binding eagerly before considering it for capture.
            self.graph = None
            self.norm = None
            self.graph_inputs = inputs
            self.key = key
            return _tensor_update(grads, groups)
        if self.graph is None:
            # Capturing enqueues the kernels for later replay. Keep input storage alive
            # independently of .grad and optimizer state assignments.
            with torch.cuda.device(grads[0].device):
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.norm = _tensor_update(grads, groups)
            self.graph_inputs = inputs
            self.key = key
        if self.graph is None or self.norm is None:
            raise RuntimeError("Missing CUDA Graph optimizer state")
        self.graph.replay()
        torch.autograd.graph.increment_version(inputs)
        return self.norm.clone()


def compile_model_w36b_update(
    model: nn.Module, optimizer: torch.optim.Optimizer, *, backend: str | None = None,
) -> Callable[[], torch.Tensor] | None:
    """Build the opt-in post-backward update once per training loop."""
    if "compiled_update" not in model_w36b_features():
        return None
    return _StaticGradUpdate(model, optimizer, backend)


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
