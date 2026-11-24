"""Compatibility imports for the archived Wave 30C offline compiler probe.

LNET_MODEL_W30C is retired. Model execution uses the Wave 28C/29C paths.
"""

from .model_w29c import _swiglu_vec2d_kernel

__all__ = ["_swiglu_vec2d_kernel", "swiglu_tile"]


def swiglu_tile(hidden: int) -> tuple[int, int]:
    cols = min(64, hidden & -hidden)
    return 4096 // cols, cols
