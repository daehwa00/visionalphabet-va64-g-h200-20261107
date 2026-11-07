"""Exact external-model builders used by the ImageNet-100 baseline checkpoints.

The robustness evaluator consumes checkpoints made by the historical
``run_imagenet100_lightweight_zoo`` runner.  Those checkpoints contain the raw
official model state dict, so this module deliberately returns the raw model:
wrapping it would prepend a state-dict key namespace and make the checkpoints
unloadable.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import subprocess
import sys
import threading
import types
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from torch import nn

DEFAULT_SOURCE_ROOT = Path.home() / ".cache" / "lnet" / "imagenet100-baselines"
_IMPORT_LOCK = threading.RLock()
_ModelFactory = Callable[..., object]
_ModelRegistrar = Callable[[_ModelFactory], _ModelFactory]


@dataclass(frozen=True, slots=True)
class _Source:
    directory: str
    repository: str
    commit: str


# These are the immutable official source revisions used for the historical
# lightweight-zoo checkpoint campaign.  Sources are never fetched implicitly;
# a caller provides a checked-out source root or uses the local cache.
_SOURCES = {
    "tinynext": _Source(
        "tinynext",
        "https://github.com/yuffeenn/TinyNeXt.git",
        "3eb30a847f8e5916b975f139d101a0da1f0d7e67",
    ),
    "emov2": _Source(
        "emov2",
        "https://github.com/zhangzjn/EMOv2.git",
        "582a96fe63f169c99fab9cef627374f0a2d1ebb6",
    ),
    "parcnet": _Source(
        "parc_net",
        "https://github.com/hkzhang91/ParC-Net.git",
        "f6780d0ad795835df2181b8fd4737276deb6e211",
    ),
    "moganet": _Source(
        "moganet",
        "https://github.com/Westlake-AI/MogaNet.git",
        "c83e328b513289fddd2921e17a9143b0f966c0a1",
    ),
}

VARIANTS = (
    "tinynext_t",
    "tinynext_s",
    "emov2_1m",
    "parcnet_xs",
    "emov2_2m",
    "moganet_xt",
    "parcnet_s",
)
_SOURCE_BY_VARIANT = {
    "tinynext_t": "tinynext",
    "tinynext_s": "tinynext",
    "emov2_1m": "emov2",
    "emov2_2m": "emov2",
    "parcnet_xs": "parcnet",
    "parcnet_s": "parcnet",
    "moganet_xt": "moganet",
}


@dataclass(frozen=True, slots=True)
class LightweightZooConfig:
    """Historical zoo builder settings required by the saved checkpoints."""

    output_dim: int = 100
    source_root: Path | None = None


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_checkout(variant: str, source_root: Path | None) -> Path:
    try:
        source_name = _SOURCE_BY_VARIANT[variant]
    except KeyError as error:
        message = f"unknown ImageNet-100 baseline variant: {variant!r}; supported={VARIANTS}"
        raise ValueError(message) from error
    source = _SOURCES[source_name]
    root = source_root or Path(
        os.environ.get("IMAGENET100_BASELINE_SOURCE_ROOT", DEFAULT_SOURCE_ROOT)
    )
    checkout = root.expanduser().resolve() / source.directory
    if not (checkout / ".git").is_dir():
        message = (
            f"missing pinned source checkout for {variant!r}: {checkout}. "
            f"Clone {source.repository} at {source.commit} into that directory."
        )
        raise RuntimeError(message)
    if _git(checkout, "rev-parse", "--verify", "HEAD") != source.commit:
        message = f"source commit mismatch for {variant!r}: {checkout}"
        raise RuntimeError(message)
    if _git(checkout, "config", "--get", "remote.origin.url") != source.repository:
        message = f"source remote mismatch for {variant!r}: {checkout}"
        raise RuntimeError(message)
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        message = f"source checkout is dirty for {variant!r}: {checkout}"
        raise RuntimeError(message)
    return checkout


@contextlib.contextmanager
def _import_context(
    checkout: Path,
    prefixes: tuple[str, ...],
) -> Generator[None]:
    """Import one official module without leaking its generic package names."""
    # This module temporarily changes sys.path, sys.modules, and a timm
    # registry entry.  Serialize this module's own imports while that state is
    # live; it cannot coordinate unrelated imports elsewhere in the process.
    with _IMPORT_LOCK:
        saved = {
            name: module
            for name, module in sys.modules.items()
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        }
        for name in saved:
            del sys.modules[name]
        original_path = list(sys.path)
        original_dont_write_bytecode = sys.dont_write_bytecode
        registry_module: types.ModuleType | None = None
        original_register: _ModelRegistrar | None = None
        try:
            sys.path.insert(0, str(checkout))
            sys.dont_write_bytecode = True
            # Several archived sources use this legacy timm import only to
            # register a factory. Their factory is called directly below.
            try:
                registry_module = importlib.import_module("timm.models.registry")
                original_register = _registrar(registry_module)
                _set_registrar(registry_module, _identity_registrar)
            except (ImportError, TypeError):
                registry_module = None
            yield
        finally:
            if registry_module is not None and original_register is not None:
                _set_registrar(registry_module, original_register)
            for name in list(sys.modules):
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                    del sys.modules[name]
            sys.modules.update(saved)
            sys.path[:] = original_path
            sys.dont_write_bytecode = original_dont_write_bytecode


def _registrar(module: types.ModuleType) -> _ModelRegistrar:
    value = getattr(module, "register_model", None)
    if not callable(value):
        message = "timm.models.registry does not expose a callable register_model"
        raise TypeError(message)
    return cast("_ModelRegistrar", value)


def _set_registrar(module: types.ModuleType, registrar: _ModelRegistrar) -> None:
    setattr(module, "register_model", registrar)  # noqa: B010 - dynamic module attribute


def _identity_registrar(factory: _ModelFactory) -> _ModelFactory:
    return factory


def _module(
    checkout: Path,
    module_name: str,
    prefixes: tuple[str, ...],
) -> types.ModuleType:
    try:
        with _import_context(checkout, prefixes):
            return importlib.import_module(module_name)
    except Exception as error:
        message = f"failed to import {module_name!r} from pinned source {checkout}: {error}"
        raise RuntimeError(message) from error


def _parc_options(variant: str, output_dim: int) -> argparse.Namespace:
    options = argparse.Namespace()
    values: dict[str, object] = {
        "model.classification.n_classes": output_dim,
        "model.classification.classifier_dropout": 0.2,
        "model.layer.global_pool": "mean",
        "model.classification.edge.scale": "scale_xs"
        if variant == "parcnet_xs"
        else "scale_s",
        "model.classification.edge.mode": "outer_frame_v1",
        "model.classification.edge.kernel": "gcc_ca",
        "model.classification.edge.fusion": "add"
        if variant == "parcnet_xs"
        else "concat",
        "model.classification.edge.instance_kernel": "interpolation_bilinear",
        "model.classification.edge.mid_mix": False,
        "model.classification.edge.use_pe": True,
        "model.activation.name": "swish",
        "model.activation.inplace": False,
        "model.activation.neg_slope": 0.1,
        "model.normalization.name": "batch_norm_2d",
        "model.normalization.momentum": 0.1,
        "model.normalization.groups": 1,
        "model.layer.conv_init": "kaiming_normal",
        "model.layer.linear_init": "trunc_normal",
        "model.layer.linear_init_std_dev": 0.02,
    }
    for name, value in values.items():
        setattr(options, name, value)
    return options


def _build(variant: str, config: LightweightZooConfig) -> nn.Module:
    """Build the unwrapped topology that produced a saved baseline checkpoint."""
    if (
        isinstance(config.output_dim, bool) or config.output_dim <= 0
    ):
        message = f"output_dim must be a positive integer, got {config.output_dim!r}"
        raise ValueError(message)
    checkout = _source_checkout(variant, config.source_root)
    if variant.startswith("tinynext_"):
        module = _module(checkout / "classification", "models.tinynext", ("models",))
        constructor = getattr(module, variant)
        model = constructor(pretrained=False, num_classes=config.output_dim, distillation=False)
    elif variant.startswith("emov2_"):
        module = _module(
            checkout,
            "model.lib_emo.emo2",
            ("model", "util"),
        )
        constructor_name = (
            "EMO2_1M_k5_hybrid" if variant == "emov2_1m" else "EMO2_2M_k5_hybrid"
        )
        model = getattr(module, constructor_name)(pretrained=False, num_classes=config.output_dim)
    elif variant.startswith("parcnet_"):
        module = _module(checkout, "cvnets.models.classification.edgeformer", ("cvnets", "utils"))
        model = module.edgeformer(_parc_options(variant, config.output_dim))
    elif variant == "moganet_xt":
        module = _module(checkout, "models.moganet", ("models",))
        model = module.moganet_xtiny(pretrained=False, num_classes=config.output_dim)
    else:  # _source_checkout validates the public variant set before this branch.
        message = f"no builder registered for {variant!r}"
        raise ValueError(message)
    if not isinstance(model, nn.Module):
        message = f"official constructor for {variant!r} did not return torch.nn.Module"
        raise TypeError(message)
    return model


build_baseline_model = _build

__all__ = ["VARIANTS", "LightweightZooConfig", "_build", "build_baseline_model"]
