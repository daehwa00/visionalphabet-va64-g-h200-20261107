"""Content and device identity for measured VA inference plans."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import triton

if TYPE_CHECKING:
    from torch import Tensor, nn


def default_cache_dir() -> Path:
    configured = os.environ.get("VA_INFERENCE_CACHE_DIR")
    return Path(configured) if configured else Path.home() / ".cache" / "visionalphabet" / "inference"


def code_fingerprint(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in ("src", "scripts"):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        raw = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous().reshape(-1)
        digest.update(raw.view(torch.uint8).numpy().tobytes())
    for name, module in model.named_modules():
        scalars = {
            key: value for key, value in vars(module).items()
            if type(value) in (str, bool, int, float) and key != "training"
        }
        hooks = {key: len(value) for key, value in vars(module).items()
                 if "hook" in key and isinstance(value, dict)}
        digest.update(json.dumps(
            [name, type(module).__module__, type(module).__qualname__, scalars, hooks],
            sort_keys=True, allow_nan=False,
        ).encode())
    return digest.hexdigest()


def device_identity(device: torch.device) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    # Driver metadata is read-only. All visible UUIDs plus the CUDA ordinal keep
    # differently mapped multi-GPU processes from sharing an unrelated record.
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    return {
        "device": str(device), "name": props.name, "uuid": str(getattr(props, "uuid", "unavailable")),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "capability": [props.major, props.minor], "multiprocessors": props.multi_processor_count,
        "total_memory": props.total_memory,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver_devices": sorted(result.stdout.strip().splitlines()),
    }


def inference_signature(
    model: nn.Module, example: Tensor, *, hardware: dict[str, object] | None = None,
    code_hash: str | None = None,
) -> dict[str, object]:
    return {
        "schema": 1, "model_sha256": model_fingerprint(model),
        "code_sha256": code_hash or code_fingerprint(),
        "hardware": hardware if hardware is not None else device_identity(example.device),
        "shape": list(example.shape), "stride": list(example.stride()), "dtype": str(example.dtype),
        "torch": str(torch.__version__), "triton": triton.__version__, "cuda": torch.version.cuda,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "launch_autotune_disabled": os.environ.get("LNET_DISABLE_LAUNCH_AUTOTUNE"),
    }
