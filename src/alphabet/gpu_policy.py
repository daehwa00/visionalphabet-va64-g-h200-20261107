"""GPU defaults for training and reproducible benchmark presets."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

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


def recommended_gpu_env(
    capability: tuple[int, int] | None, *, allow_fp8: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if capability is None:
        return {}
    if capability == (9, 0):
        if allow_fp8 is None:
            allow_fp8 = (os.environ if env is None else env).get("ALPHABET_ALLOW_FP8") == "1"
        split = _SPLIT if allow_fp8 else {"ALPHABET_SCAN_SPLIT_BF16": "1"}
        return {**COMPILED_ENV, **split}
    policy = {**COMPILED_ENV, "ALPHABET_HORIZONTAL_SCAN_MODE": "optimized_backward"}
    if capability > (9, 0):
        policy["ALPHABET_COMPILER_FEATURES"] = "cuda_graphs,max_autotune_pointwise"
    if capability != (8, 0) and capability < (8, 9):
        policy.update(
            ALPHABET_READER_MODE="off",
            ALPHABET_COMPILER_FEATURES="cuda_graphs,max_autotune_pointwise",
        )
    return policy


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
    allow_fp8: bool | None = None, env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply defaults and return the effective values for the selected preset."""
    target = os.environ if env is None else env
    policy = recommended_gpu_env(device_capability(device), allow_fp8=allow_fp8, env=target)
    if overwrite and policy:
        for key in _SPLIT.keys() - policy.keys():
            target.pop(key, None)
        target.update(policy)
    return {key: target.setdefault(key, value) for key, value in policy.items()}


def benchmark_arm_env(name: str) -> dict[str, str]:
    arm = name.removesuffix("_rev")
    if arm == "auto":
        return recommended_gpu_env(device_capability())
    if arm == "reference":
        return REFERENCE_ENV.copy()
    if arm == "hopper_fp8_resident":
        return recommended_gpu_env((9, 0), allow_fp8=True)
    if arm == "hopper_fp8_split":
        return {
            **COMPILED_ENV,
            "ALPHABET_SCAN_SPLIT_BF16": "1",
            "ALPHABET_SCAN_SPLIT_FP8": "1",
            "ALPHABET_SCAN_RESIDENT_BACKWARD_WEIGHTS": "0",
        }
    if arm == "hopper_bf16_split":
        return {**COMPILED_ENV, "ALPHABET_SCAN_SPLIT_BF16": "1"}
    if arm == "bf16_compiled":
        return COMPILED_ENV.copy()
    raise ValueError(f"Unknown arm {name!r}")


def apply_fixed_arm(arm: str) -> dict[str, str]:
    env = benchmark_arm_env(arm)
    if "ALPHABET_SCAN_SPLIT_FP8" not in env:
        os.environ.pop("ALPHABET_SCAN_SMALL_SPATIAL_FP8", None)
    for flag in _SPLIT.keys() - env.keys():
        os.environ.pop(flag, None)
    os.environ.update(env)
    return env
