"""GPU defaults for training and reproducible benchmark presets."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REFERENCE_ENV = {
    "ALPHABET_VERTICAL_GRAD_STORE_MODE": "full_coverage_ro",
    "ALPHABET_VERTICAL_FORWARD_MODE": "batch4_2warp",
    "ALPHABET_HORIZONTAL_SCAN_MODE": "optimized_backward",
    "ALPHABET_STEM_STAGE0_MODE": "off",
    "ALPHABET_COMPILER_FEATURES": "cuda_graphs,tma_gemm,max_autotune_pointwise",
    "ALPHABET_READER_MODE": "off",
    "ALPHABET_CUDNN_BENCHMARK": "0",
    "ALPHABET_SCAN_FP32_PATH": "off",
    "ALPHABET_VERTICAL_SCAN_SHAPE_MODE": "off",
    "ALPHABET_FUSED_STEM_CONV": "auto"
}
COMPILED_ENV = {
    **REFERENCE_ENV,
    "ALPHABET_HORIZONTAL_SCAN_MODE": "hopper_specialized",
    "ALPHABET_STEM_STAGE0_MODE": "triton_m64_4warp",
    "ALPHABET_READER_MODE": "dense_nhwc",
    "ALPHABET_CUDNN_BENCHMARK": "1",
    "ALPHABET_SCAN_FP32_PATH": "fp32",
    "ALPHABET_VERTICAL_SCAN_SHAPE_MODE": "shape_specialized",
    "ALPHABET_FUSED_STEM_CONV": "on",
    "ALPHABET_SWIGLU_KERNEL": "vec8_tanh",
    "ALPHABET_RMSNORM_KERNEL": "cuda_spatial_vec8_4cta"
}

_SPLIT = {
    "ALPHABET_SCAN_SPLIT_BF16": "1",
    "ALPHABET_SCAN_SPLIT_FP8": "1",
    "ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS": "1",
}


def gpu_policy_environment() -> dict[str, str]:
    """Snapshot explicit and automatic kernel settings for benchmark and cache identity."""
    keys = (*COMPILED_ENV, *_SPLIT, "ALPHABET_SCAN_SMALL_SPATIAL_FP8")
    return {key: os.environ[key] for key in keys if key in os.environ}


def recommended_gpu_env(capability: tuple[int, int] | None) -> dict[str, str]:
    if capability is None:
        return {}
    if capability == (9, 0):
        return {**COMPILED_ENV, **_SPLIT}
    env = {**COMPILED_ENV, "ALPHABET_HORIZONTAL_SCAN_MODE": "optimized_backward"}
    if capability > (9, 0):
        env["ALPHABET_COMPILER_FEATURES"] = "cuda_graphs,max_autotune_pointwise"
    if capability != (8, 0) and capability < (8, 9):
        env.update(
            ALPHABET_READER_MODE="off",
            ALPHABET_COMPILER_FEATURES="cuda_graphs,max_autotune_pointwise",
        )
    return env


def device_capability(device: torch.device | str | int | None = None) -> tuple[int, int] | None:
    import torch  # noqa: PLC0415 -- defer runtime initialization until detection

    if device is not None and not isinstance(device, int):
        selected = torch.device(device)
        if selected.type != "cuda":
            return None
        device = selected.index
    return torch.cuda.get_device_capability(device) if torch.cuda.is_available() else None


def apply_gpu_policy(
    device: torch.device | str | int | None = None, *, overwrite: bool = False,
) -> dict[str, str]:
    """Apply defaults and return the effective values for the selected preset."""
    env = recommended_gpu_env(device_capability(device))
    if overwrite and env:
        for key in _SPLIT.keys() - env.keys():
            os.environ.pop(key, None)
        os.environ.update(env)
    return {key: os.environ.setdefault(key, value) for key, value in env.items()}
