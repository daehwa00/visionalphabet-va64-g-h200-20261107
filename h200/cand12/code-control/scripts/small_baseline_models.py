"""Pinned public model constructors for the shared-recipe small baseline runs.

Official RepViT and EfficientVMamba sources remain unchanged on disk. The two
EfficientVMamba import substitutions below only adapt dependency/package names;
its CUDA scan, initialization, layers, and training forward are unchanged.
"""

from __future__ import annotations

# Public research modules are dynamically imported and have no type stubs.
# pyright: reportExplicitAny=false
import copy
import hashlib
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import timm

if TYPE_CHECKING:
    from torch import nn

MODEL_KEYS = ("efficientvmamba_t", "mobilenetv4_conv_small", "repvit_m0_9")
MODELS = MODEL_KEYS
UPSTREAM_COMMITS = {
    "EfficientVMamba": "0bc5ee288b402d648641f5494b73e9d152e0c62b",
    "RepViT": "298f42075eda5d2e6102559fad260c970769d34e",
}
PINNED_FILES = {
    "RepViT/model/repvit.py": "a9c8f0d8219a19049ad160e1d0e55351c41d92de19352af6ca65d76e100aef99",
    "EfficientVMamba/classification/models/vmamba_efficient.py": "59cae6a5d381e87d000015e4a356a43815435a6fdd2659eda48fa53afe901504",
    "EfficientVMamba/classification/lib/models/operations.py": "4bbe312a86e84f9f7a041defd5f3bc005d668617219fafb235d3c65fa2b512b8",
    "EfficientVMamba/classification/lib/models/mdconv.py": "d51f399e22690bc1186fb8620fbd7b70d85889404cee004e7b2035b31a4c4aad",
    "EfficientVMamba/classification/configs/vssm/vssm_efficient_tiny.yaml": "fa844bb8ea9eb798bb5cc11885d634b35b4ce36efe72fecb7780df4cc1828d16",
}
IMPORT_ADAPTERS = {
    "from lib.models.operations import InvertedResidual": "from _va_efficientvmamba.lib.models.operations import InvertedResidual",
    "from selective_scan_vmamba_pt202 import selective_scan_cuda_core": "import selective_scan_cuda_core",
    "from selective_scan_vmamba import selective_scan_cuda_core": "import selective_scan_cuda_core",
}
EFFICIENT_TINY_KWARGS = {
    "patch_size": 4,
    "in_chans": 3,
    "num_classes": 1000,
    "depths": [2, 2, 4, 2],
    "dims": 48,
    "d_state": 16,
    "dt_rank": "auto",
    "ssm_ratio": 2.0,
    "attn_drop_rate": 0.0,
    "shared_ssm": False,
    "softmax_version": False,
    "drop_rate": 0.0,
    "drop_path_rate": 0.2,
    "mlp_ratio": 0.0,
    "patch_norm": True,
    "downsample_version": "v1",
    "use_checkpoint": False,
    "window_size": 2,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_source(root: Path, prefix: str) -> None:
    for name, digest in PINNED_FILES.items():
        if name.startswith(prefix) and _sha(root / name) != digest:
            raise ValueError(f"pinned official source differs: {name}")


def _load(name: str, path: Path, source: str | None = None) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if source is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def _namespace(name: str, path: Path) -> None:
    existing = sys.modules.get(name)
    if existing is not None:
        if list(existing.__path__) != [str(path)]:
            raise ImportError(f"official namespace already bound to another source: {name}")
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def build_model(models_root: Path, model_key: str) -> nn.Module:
    """Create an unfused training model, never download pretrained weights."""
    root = Path(models_root).resolve()
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unsupported small baseline: {model_key}")
    if timm.__version__ != "1.0.26":
        raise ValueError("small baselines require pinned timm 1.0.26")
    if model_key == "mobilenetv4_conv_small":
        return timm.create_model(model_key, pretrained=False, num_classes=1000)
    if model_key == "repvit_m0_9":
        _check_source(root, "RepViT/")
        module = _load("_va_official_repvit", root / "RepViT/model/repvit.py")
        return module.repvit_m0_9(pretrained=False, num_classes=1000, distillation=False)
    _check_source(root, "EfficientVMamba/")
    try:
        importlib.import_module("selective_scan_cuda_core")
    except ImportError as exc:
        raise ImportError(
            "Build the pinned EfficientVMamba selective_scan CUDA extension in the experiment's isolated extension directory before training."
        ) from exc
    classification = root / "EfficientVMamba/classification"
    _namespace("_va_efficientvmamba", classification)
    _namespace("_va_efficientvmamba.lib", classification / "lib")
    _namespace("_va_efficientvmamba.lib.models", classification / "lib/models")
    path = classification / "models/vmamba_efficient.py"
    source = path.read_text()
    for original, replacement in IMPORT_ADAPTERS.items():
        if source.count(original) != 1:
            raise ValueError(f"official import adapter no longer applies: {original}")
        source = source.replace(original, replacement)
    module = _load("_va_efficientvmamba.vmamba_efficient", path, source)
    return module.EfficientVSSM(**EFFICIENT_TINY_KWARGS)


def make_source_manifest(models_root: Path) -> dict[str, Any]:
    """Freeze complete used Python/CUDA source trees, excluding Git/build outputs."""
    root = Path(models_root).resolve()
    _check_source(root, "")
    model_files: dict[str, str] = {}
    for name in PINNED_FILES:
        model_files[name] = _sha(root / name)
    cuda_root = root / "EfficientVMamba/selective_scan"
    for path in sorted(cuda_root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".cuh", ".h"}:
            model_files[path.relative_to(root).as_posix()] = _sha(path)
    timm_root = Path(timm.__file__).resolve().parent
    timm_files: dict[str, str] = {}
    for path in sorted(timm_root.rglob("*.py")):
        timm_files[path.relative_to(timm_root).as_posix()] = _sha(path)
    return {
        "schema": 1,
        "upstream_commits": dict(UPSTREAM_COMMITS),
        "timm_version": "1.0.26",
        "files": {},
        "model_files": model_files,
        "timm_files": timm_files,
        "efficientvmamba_import_adapters": dict(IMPORT_ADAPTERS),
        "efficientvmamba_tiny_constructor": copy.deepcopy(EFFICIENT_TINY_KWARGS),
        "mobilenetv4_provenance": {
            "implementation": "timm 1.0.26, mobilenetv4_conv_small(pretrained=False)",
            "paper": "https://doi.org/10.1007/978-3-031-73661-2_5",
            "note": "Paper acknowledges Ross Wightman's timm PyTorch reimplementation; official Google implementation is TensorFlow.",
        },
        "repvit_constructor": "repvit_m0_9(pretrained=False, num_classes=1000, distillation=False)",
        "selection_note": "RepViT-M0.9 is the smallest original-paper model. Later upstream M0.6 is outside this preselected comparison.",
    }


def source_binding(models_root: Path, source_manifest: Path) -> dict[str, str]:
    """Validate every frozen source before binding the checkpoint contract."""
    manifest = json.loads(Path(source_manifest).read_text())
    actual = make_source_manifest(models_root)
    # The runner owns its additional code-root-relative file bindings.
    comparable = dict(manifest)
    comparable["files"] = {}
    if comparable != actual:
        raise ValueError("small baseline source manifest differs from pinned files/settings")
    return {
        "small_baseline_source_manifest": _sha(Path(source_manifest)),
        "small_baseline_models": _sha(Path(__file__)),
        "timm_version": timm.__version__,
        **{f"upstream_{name}": commit for name, commit in UPSTREAM_COMMITS.items()},
    }


def parameter_inventory(model: nn.Module) -> dict[str, int]:
    return {
        "stored_elements": sum(p.numel() for p in model.parameters()),
        "trainable_elements": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stored_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }
