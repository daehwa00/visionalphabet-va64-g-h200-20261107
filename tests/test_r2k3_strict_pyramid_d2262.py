from __future__ import annotations

# pyright: reportPrivateUsage=false
import torch

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_strict_pyramid_d2262_imagenet100 as runner

EXPECTED_PARAMETERS = 3_551_556


def test_strict_pyramid_contract_and_parameter_count() -> None:
    assert runner.SPEC.excitation_modes == (32, 64, 128, 128)
    assert runner.SPEC.pole_modes == (64, 128, 192, 128)
    assert runner.SPEC.depth == (2, 2, 6, 2)
    assert runner.SPEC.q4_dim == 512
    assert runner.MEMORY_SCALE_INITIAL == 1.0
    torch.manual_seed(501)
    model = runner._build(runner.VARIANT, runtime.model_config())
    runner._assert_model(model, runner.VARIANT)
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    readers = tuple(
        module for module in model.modules() if type(module) is FactorizedComplexConv2dReader
    )
    assert len(readers) == 12
    assert all(reader.rank == 2 and reader.kernel_size == 3 for reader in readers)
