"""Workload identity and compiled runtimes for training and evaluation."""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import importlib
import inspect
import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import torch
from imagenet100_checkpoint_runtime import atomic_json
from torch import nn
from torch.utils._pytree import tree_map

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from va_imagenet100_runtime import RunnerBindings

T = TypeVar("T")
TensorCapture = dict[str, dict[str, torch.Tensor]]

MODULE = Path(__file__).resolve()
PROJECT = MODULE.parents[1]


def digest(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cpu_tree[T](obj: T) -> T:
    return tree_map(
        lambda v: (
            v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        ),
        obj,
    )


def ident(fn: Callable[..., object]) -> list[str]:
    module = fn.__module__
    if module == "__main__":
        module = Path(inspect.getfile(fn)).stem
    return [module, fn.__qualname__]


def resolve(spec: Sequence[str]) -> Any:
    obj = importlib.import_module(spec[0])
    for name in spec[1].split("."):
        obj = getattr(obj, name)
    return obj


def rng(device: torch.device) -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 - snapshot the process-wide RNG
    }


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002 - restore the process-wide RNG


def state_hash(obj: object) -> str:
    h = hashlib.sha256()

    def visit(x: object) -> None:
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=repr):
                visit(k)
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for v in x:
                visit(v)
        else:
            h.update(repr(x).encode())

    visit(obj)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def compare_tensors(reference: TensorCapture, actual: TensorCapture) -> dict[str, Any]:
    result = {}
    passed = True
    for group in ("outputs", "grads", "updates"):
        a, b = reference[group], actual[group]
        if not a or a.keys() != b.keys():
            return {"passed": False, "keys_failed": group}
        max_abs = max_l2 = 0.0
        fail = []
        for name, x in a.items():
            y = b[name]
            if x.shape != y.shape or not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                fail.append(name)
                continue
            baseline = x.float()
            delta = y.float() - baseline
            absolute = delta.abs().max().item()
            rel = (delta.norm() / baseline.norm().clamp_min(1e-12)).item()
            limit = {"outputs": 0.05, "updates": 3e-5}.get(group)
            if limit is None:
                limit = 0.03 * baseline.abs().max().item() + 1e-6
            if absolute > limit or rel > (0.005 if group == "outputs" else 0.02):
                fail.append(name)
            max_abs = max(max_abs, absolute)
            max_l2 = max(max_l2, rel)
        result[group] = {
            "count": len(a),
            "max_abs": max_abs,
            "max_relative_l2": max_l2,
            "failed": fail,
        }
        passed &= not fail
    result["passed"] = passed
    return result


class AutomaticTrainingRuntime(nn.Module):
    """Share a compiled model for each activation microbatch size."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        recipe: dict[str, Any],
        model_config: dict[str, Any],
        variant: str,
        bindings: RunnerBindings,
        root: Path,
        training_generator: torch.Generator,
        mixup_generator: np.random.Generator,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.recipe = recipe
        self.model_config = model_config
        self.variant = variant
        self.training_generator = training_generator
        self.mixup_generator = mixup_generator
        factories = {
            "build": bindings.build_model,
            "config": bindings.model_config,
            "optimizer": bindings.build_optimizer,
            "prepare": bindings.prepare_model,
        }
        self.factories = {name: ident(fn) for name, fn in factories.items()}
        self.base = (
            Path(os.environ.get("LNET_LAUNCH_CACHE") or Path.home() / ".cache/lnet")
            / "effective-updates"
        )
        self.audit = root / "effective-update-calibration"
        self.runtimes = {}
        self.epoch_index = 0
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        self.schema = {
            "model": [(n, list(v.shape), str(v.dtype)) for n, v in model.state_dict().items()],
            "optimizer": ident(type(optimizer)),
            "groups": [
                {
                    "names": [names[id(v)] for v in group["params"]],
                    "options": {
                        k: v for k, v in group.items() if k not in {"params", "lr", "initial_lr"}
                    },
                }
                for group in optimizer.param_groups
            ],
            "factories": self.factories,
            "model_config": model_config,
            "variant": variant,
        }
        self.settings = {
            "threads": torch.get_num_threads(),
            "matmul": torch.get_float32_matmul_precision(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_bf16_reduced": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "cuda_fp16_reduced": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "emulate_precision_casts": bool(recipe.get("emulate_precision_casts", False)),
        }
        train_fn = bindings.train_epoch
        train_kwargs = train_fn.keywords if isinstance(train_fn, functools.partial) else {}
        train_fn = train_fn.func if isinstance(train_fn, functools.partial) else train_fn
        sources = {
            MODULE,
            *(
                MODULE.with_name(name + ".py")
                for name in (
                    "va_imagenet100_runtime",
                    "va_ablation_augment",
                    "va_model_builder",
                    "va_packed_backbone",
                    "effective_batch_runtime",
                    "effective_batch_search",
                    "effective_batch_update",
                    "effective_batch_epoch",
                    "imagenet100_data_runtime",
                )
            ),
            *(PROJECT / "src/lnet").glob("*.py"),
            *(Path(inspect.getfile(fn)) for fn in (*factories.values(), train_fn)),
        }
        revision = hashlib.sha256()
        for source in sorted(sources):
            try:
                revision.update(source.read_bytes())
            except FileNotFoundError:
                # An adjacent module may disappear during a source checkout.
                continue
        device = next(model.parameters()).device
        hardware = {"type": device.type}
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            hardware.update(
                name=props.name,
                capability=(props.major, props.minor),
                memory=props.total_memory,
                multiprocessors=props.multi_processor_count,
                cudnn=torch.backends.cudnn.version(),
            )
        self.identity = {
            "train_function": ident(train_fn),
            "train_options": train_kwargs,
            "objective": recipe.get("head_objective"),
            "device": hardware,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "model": digest(self.schema),
            "training_code": revision.hexdigest(),
            "compile_mode": os.environ.get("LNET_COMPILE_MODE") or recipe.get("compile_mode"),
            "precision": recipe["precision"],
            "channels_last": recipe.get("channels_last", False),
            "settings": self.settings,
            "mixup_alpha": recipe.get("mixup_alpha"),
            "reference_scan_forward": os.environ.get("LNET_GENERAL_SCAN_REF_FWD") == "1",
            "scan_env": {
                name: "0" if os.environ.get(name) == "0" else "1"
                for name in ("LNET_GENERAL_SCANS", "LNET_FUSED_FACTORIZED_READER")
            },
        }

    def runtime_for(self, key: str | int) -> nn.Module:
        from va_imagenet100_runtime import _build_runtime

        if key not in self.runtimes:
            self.runtimes[key] = _build_runtime(self.model, self.recipe)
        return self.runtimes[key]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.runtime_for("evaluation")(x)

    def event(self, key: str, status: str, **details: object) -> None:
        print(f"[training layout] {status} {key[:12]}", flush=True)
        atomic_json(self.audit / (key + ".json"), dict(key=key, status=status, **details))

    def live_state(self, device: torch.device) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gradients": {n: v.grad for n, v in self.model.named_parameters()},
            "rng": rng(device),
            "train_generator": self.training_generator.get_state(),
            "mixup": self.mixup_generator.bit_generator.state,
        }


def training_microbatch(
    runtime: nn.Module,
    x: torch.Tensor,
    _y: torch.Tensor,
    _other: torch.Tensor,
    _mixing: float | torch.Tensor,
    _size: int,
    _offset: int,
) -> contextlib.nullcontext[nn.Module]:
    if isinstance(runtime, AutomaticTrainingRuntime):
        runtime = runtime.runtime_for(len(x))
    return contextlib.nullcontext(runtime)
