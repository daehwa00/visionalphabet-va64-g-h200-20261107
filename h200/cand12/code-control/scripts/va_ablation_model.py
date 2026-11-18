"""Scratch IN100 experiment with real feature widths 32/64/128/256."""

from __future__ import annotations

# Reuse the established model builders and frozen IN100 training callbacks.
# pyright: reportPrivateUsage=false, reportImplicitRelativeImport=false
# pyright: reportPrivateLocalImportUsage=false, reportExplicitAny=false
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import run_a2d_r2k3_capacity_insight_overnight_imagenet100 as capacity
import run_a2d_r2k3_stage_allocation_screen_imagenet100 as allocation
import run_va128_real_memory_imagenet100 as baseline
import torch
from torch import nn

from lnet.complex_scan import ComplexScanConfig
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_gated_post_fusion import GatedComplexPostFusion, GatedPoleExcitationS2DTransition

if TYPE_CHECKING:
    from argparse import Namespace

class PyramidRealMemoryIN100(baseline.RealMemoryIN100):
    architecture = "va-eight-ablation"

def build_backbone(config: ComplexScanConfig, channels, poles, depth) -> PyramidRealMemoryIN100:
    seed = torch.initial_seed()
    spec = allocation.StageAllocationSpec(
        cast("tuple[int, int, int, int]", tuple(c // 2 for c in channels)),
        poles,
        tuple(d - 2 for d in depth),
        family="real_memory_pyramid",
    )
    model = capacity._build_all_resolution(spec.as_insight_spec(), config)
    model = allocation._append_repeated_blocks(model, spec)
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, GatedComplexPostFusion):
                hidden = round(1.5 * child.modes)
                setattr(parent, name, GatedComplexPostFusion(child.modes, hidden))
                if isinstance(parent, GatedPoleExcitationS2DTransition):
                    parent.post_hidden = hidden
    # Preserve common-shaped initialization, including pole dynamics and the
    # fixed terminal head, without consuming the candidate's subsequent RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = baseline.anchor._build(baseline.anchor.D2262_P128_TERM, config)
    baseline.anchor._pair_common_initialization_(model, reference)
    for module in model.modules():
        if isinstance(module, FactorizedComplexConv2dReader):
            module.execution_mode = "factorized_triton"
    baseline.enable_packed_va128_(model)
    result = PyramidRealMemoryIN100(model)
    return result

