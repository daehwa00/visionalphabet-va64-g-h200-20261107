"""Compatibility imports for the former rank-four-only factor kernels.

The active implementation is model-independent; no legacy tuned dispatch is
kept here. Historical source is stored alongside the benchmark records.
"""
from .triton_backward import (
    factor_grads_bshd as factor_grads_bshd_rank4,
    fused_backward_and_factor_grads_bshd as fused_backward_and_factor_grads_bshd_rank4,
)
