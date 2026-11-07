# pyright: reportMissingImports=false, reportPrivateUsage=false
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_complex_scan_augmented_cifar100 as optimizer_source


def test_phase_gated_parameters_use_scale_safe_optimizer_contract() -> None:
    model = nn.ModuleDict(
        {
            "phase_gated": PhaseGatedComplexFFN(4, 8),
            "ordinary": nn.Linear(4, 4),
        }
    )
    optimizer = optimizer_source._build_optimizer(
        model,
        {
            "learning_rate": 3.0e-3,
            "weight_decay": 0.05,
            "modal_learning_rate_multiplier": 1.0 / 3.0,
            "pole_geometry_learning_rate_multiplier": 0.1,
            "fused_optimizer": False,
        },
    )
    settings = {
        id(parameter): (float(group["lr"]), float(group["weight_decay"]))
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    phase_gated = model["phase_gated"]
    assert isinstance(phase_gated, PhaseGatedComplexFFN)

    for projection in (
        phase_gated.input_projection,
        phase_gated.output_projection,
    ):
        for parameter in projection.parameters():
            assert settings[id(parameter)] == (3.0e-3, 0.0)

    for parameter in (
        phase_gated.alpha,
        phase_gated.gamma,
        *phase_gated.norm.parameters(),
    ):
        assert settings[id(parameter)] == (3.0e-3, 0.0)

    ordinary = model["ordinary"]
    assert isinstance(ordinary, nn.Linear)
    assert settings[id(ordinary.weight)] == (3.0e-3, 0.05)
    assert settings[id(ordinary.bias)] == (3.0e-3, 0.0)
    assert len(settings) == sum(1 for _ in model.parameters())


def test_optimizer_retracts_phase_gated_direction_rows_after_every_step() -> None:
    phase_gated = PhaseGatedComplexFFN(4, 8)
    phase_gated.enable_direction_output_gain_contract_(output_gain_max=0.5)
    optimizer = optimizer_source._build_optimizer(
        phase_gated,
        {
            "learning_rate": 3.0e-3,
            "weight_decay": 0.05,
            "modal_learning_rate_multiplier": 1.0 / 3.0,
            "pole_geometry_learning_rate_multiplier": 0.1,
            "fused_optimizer": False,
        },
    )
    real = torch.randn(3, 4)
    imag = torch.randn_like(real)
    output_real, output_imag = phase_gated(real, imag)
    (output_real.square().mean() + output_imag.square().mean()).backward()
    optimizer.step()

    for projection in (
        phase_gated.input_projection,
        phase_gated.output_projection,
    ):
        row_energy = (
            projection.weight_real.square().add(projection.weight_imag.square()).sum(dim=-1)
        )
        torch.testing.assert_close(row_energy, torch.ones_like(row_energy))
