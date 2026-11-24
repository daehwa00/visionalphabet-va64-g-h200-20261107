"""Opt-in Wave 35C model and compiler experiments."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext

_MODES = {
    "off", "auto", "stem_v2", "swiglu_v3", "rmsnorm_bwd_inductor",
    "inductor_tune", "full_w35c",
}


def model_w35c_enabled(feature: str) -> bool:
    mode = os.environ.get("LNET_MODEL_W35C", "off")
    modes = mode.split(",")
    if not all(item in _MODES for item in modes):
        raise ValueError(f"Invalid LNET_MODEL_W35C mode: {mode}")
    return feature in modes or "full_w35c" in modes


def apply_inductor_tune() -> AbstractContextManager[None]:
    """Scope fusion settings around compilation and its lazy first execution."""
    if model_w35c_enabled("inductor_tune"):
        from torch._inductor import config  # noqa: PLC0415

        return config.patch(epilogue_fusion=True, aggressive_fusion=True)
    return nullcontext()
