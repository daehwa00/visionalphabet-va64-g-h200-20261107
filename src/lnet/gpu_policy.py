"""GPU defaults for training and reproducible benchmark presets."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REFERENCE_ENV = {
    "LNET_SCAN_SGS": "strict_ro",
    "LNET_SCAN_VFWD": "batch4_w2",
    "LNET_SCAN_HSCAN": "bwd_opt",
    "LNET_SCAN_VSCAN_SAVED": "off",
    "LNET_STEM_STAGE0_MODE": "off",
    "VA_READER_OPT_MODE": "off",
    "LNET_MODEL_W36B": "cuda_graphs,tma_gemm,max_autotune_pointwise",
    "LNET_READER_DENSE_NHWC": "off",
    "LNET_READER_EXEC_MODE": "off",
    "LNET_NORM_W50_MODE": "off",
    "LNET_CUDNN_BENCHMARK": "0",
    "LNET_SCAN_W51A": "off",
    "LNET_SCAN_W51B": "off",
    "LNET_SCAN_W52A": "off",
    "LNET_SCAN_W53A": "off",
    "LNET_FUSED_STEM_CONV": "auto"
}
LOCKIN_ENV = {
    **REFERENCE_ENV,
    "LNET_SCAN_HSCAN": "h55_lockin",
    "LNET_STEM_STAGE0_MODE": "triton_v3_m64_w4",
    "LNET_READER_DENSE_NHWC": "dense_nhwc",
    "LNET_READER_EXEC_MODE": "dense_nhwc",
    "LNET_CUDNN_BENCHMARK": "1",
    "LNET_SCAN_W51A": "fp32",
    "LNET_SCAN_W53A": "w57_lockin",
    "LNET_FUSED_STEM_CONV": "on",
    "LNET_MODEL_W56B": "swiglu_vec8_tanh",
    "LNET_MODEL_W57B_RMS": "cuda_sp_vec8_4cta"
}

_SPLIT = {"VA_W66_SPLIT_BF16": "1", "VA_W72_SPLIT_FP8": "1", "VA_W76_REG_WOBI": "1"}


def gpu_policy_environment() -> dict[str, str]:
    """Snapshot explicit and automatic kernel settings for benchmark and cache identity."""
    keys = (*LOCKIN_ENV, *_SPLIT, "VA_W74_SMALL_FP8")
    return {key: os.environ[key] for key in keys if key in os.environ}


def recommended_gpu_env(capability: tuple[int, int] | None) -> dict[str, str]:
    if capability is None:
        return {}
    if capability >= (9, 0):
        return {**LOCKIN_ENV, **_SPLIT}
    env = {**LOCKIN_ENV, "LNET_SCAN_HSCAN": "bwd_opt"}
    if capability != (8, 0) and capability < (8, 9):
        env.update(
            LNET_READER_DENSE_NHWC="off",
            LNET_READER_EXEC_MODE="off",
            LNET_MODEL_W36B="cuda_graphs,max_autotune_pointwise",
        )
    return env


def device_capability(device: torch.device | str | int | None = None) -> tuple[int, int] | None:
    import torch  # noqa: PLC0415 -- defer runtime initialization until detection

    if device is not None and not isinstance(device, int) and torch.device(device).type != "cuda":
        return None
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
