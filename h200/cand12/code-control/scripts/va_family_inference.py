"""Original VA K32/K48/K64/K96/K128 builders with optional packed execution."""

from __future__ import annotations

# The experiment builders intentionally use script-level imports.
# pyright: reportImplicitRelativeImport=false, reportPrivateUsage=false
from pathlib import Path

import a2d_r2k3_runtime as runtime
import run_a2d_r2k3_k_family_wave_a_imagenet100 as family
import torch
from run_lnet_k128_p160_160_160_128_d2262_imagenet1k import PrimaryLogitsAdapter
from run_va_retrain_imagenet1k import build_model
from va128_inference import FAMILY_PARAMETERS
from va128_packed_backbone import enable_packed_va128_

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader

VARIANTS = {32: "S-K32-U125", 48: "M-K48-U125", 64: "L-K64-U125", 96: "XL-K96-U125"}


def build_va_family(width: int, *, packed: bool = True) -> torch.nn.Module:
    if width == 128:
        if not packed:
            raise ValueError("use the preserved reviewer builder for the original K128 reference")
        return build_model("B", prepared=True)
    if width not in VARIANTS:
        raise ValueError("expected K32, K48, K64, K96 or K128")
    backbone = family._build(VARIANTS[width], runtime.model_config(output_dim=1000))
    if sum(parameter.numel() for parameter in backbone.parameters()) != FAMILY_PARAMETERS[width]:
        raise RuntimeError("family parameter count differs from the submitted architecture")
    for module in backbone.modules():
        if isinstance(module, FactorizedComplexConv2dReader):
            module.execution_mode = "factorized_triton" if packed else "dense"
    if packed:
        enable_packed_va128_(backbone)
    return PrimaryLogitsAdapter(backbone.prepare_for_compiled_training_())


def load_va_checkpoint(model: torch.nn.Module, path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    epochs = payload.get("completed_epochs")
    if epochs != 100:
        raise ValueError("the table requires the original 100-epoch trained checkpoint")
    state = {key.removeprefix("_orig_mod."): value for key, value in payload["model"].items()}
    model.load_state_dict(state, strict=True)
    return epochs
