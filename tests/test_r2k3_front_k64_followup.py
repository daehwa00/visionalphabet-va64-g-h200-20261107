from __future__ import annotations

# pyright: reportPrivateUsage=false
import torch

from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_front_k64_followup_d2262_imagenet100 as runner

EXPECTED_PARAMETERS = {
    runner.K64_P96: 4_183_972,
    runner.K64_P64: 4_138_788,
    runner.K128_P96: 4_598_500,
    runner.K96_P96: 4_361_732,
    runner.MID_POLE_REALLOCATION: 4_591_620,
    runner.POLE14_REALLOCATION: 4_534_148,
}


def test_k64_followup_prioritizes_two_front_controls() -> None:
    assert runner.MEMORY_SCALE_INITIAL == 1.0
    assert runner.SPECS[runner.K64_P96].excitation_modes == (64, 128, 128, 128)
    assert runner.SPECS[runner.K64_P96].pole_modes == (96, 160, 160, 160)
    assert runner.SPECS[runner.K64_P64].excitation_modes == (64, 128, 128, 128)
    assert runner.SPECS[runner.K64_P64].pole_modes == (64, 160, 160, 160)
    assert runner.JOBS_BY_GPU == {
        0: (runner.K64_P96, runner.K128_P96, runner.MID_POLE_REALLOCATION),
        1: (runner.K64_P64, runner.K96_P96, runner.POLE14_REALLOCATION),
    }
    assert all(spec.depth == (2, 2, 6, 2) for spec in runner.SPECS.values())
    assert all(spec.q4_dim == 640 for spec in runner.SPECS.values())


def test_k64_followup_builds_all_cells_with_exact_parameter_counts() -> None:
    config = runtime.model_config()
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, config)
        runner._assert_model(model, variant)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == (EXPECTED_PARAMETERS[variant])
        )
