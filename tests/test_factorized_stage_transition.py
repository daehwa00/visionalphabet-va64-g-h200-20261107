from __future__ import annotations

# pyright: reportMissingImports=false, reportPrivateUsage=false
import importlib
import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pytest
import torch
from torch.nn import functional

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig, ComplexScanStage
from lnet.pac_factorized_stage_transition import (
    FactorizedS2DPostFusionTransition,
    ModeResidualPathCollapse,
)
from lnet.pac_path_cffn import FactorizedQuadrantPathModeCFFNCombiner

if TYPE_CHECKING:
    from lnet.pac_grouped_path_cffn import GroupedWidelyLinear

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MPM8_RUNNER = "run_a2d_deep4_calibrated_uniform_p96_factorized_imagenet100"
STEM32_RUNNER = "run_a2d_deep4_calibrated_uniform_p96_stemres_imagenet100"


class _Runner(Protocol):
    __file__: str | None
    VARIANT: str
    P: int

    def _build(self, variant: str, config: ComplexScanConfig) -> ComplexScanBackbone: ...


@pytest.fixture(scope="module")
def runners() -> tuple[_Runner, _Runner]:
    sys.path.insert(0, str(SCRIPTS))
    try:
        candidate = cast("_Runner", cast("object", importlib.import_module(MPM8_RUNNER)))
        baseline = cast("_Runner", cast("object", importlib.import_module(STEM32_RUNNER)))
        assert candidate.__file__ is not None
        assert Path(candidate.__file__).resolve().parent == SCRIPTS.resolve()
        return candidate, baseline
    finally:
        sys.path.remove(str(SCRIPTS))


def _config() -> ComplexScanConfig:
    return ComplexScanConfig(output_dim=100, stem_strides=(2, 2))


def _stages(
    model: ComplexScanBackbone,
) -> tuple[ComplexScanStage, ComplexScanStage, ComplexScanStage]:
    return cast(
        "tuple[ComplexScanStage, ComplexScanStage, ComplexScanStage]",
        tuple(getattr(model, name) for name in ("stage1", "stage2", "stage3")),
    )


def test_mode_path_collapse_matches_composed_projection_semantics() -> None:
    torch.manual_seed(701)
    module = ModeResidualPathCollapse(6, mode_hidden=12, path_hidden=8).double()
    real = torch.randn(2, 3, 2, 4, 6, dtype=torch.double)
    imag = torch.randn_like(real)

    actual = module.forward_packed(real, imag)
    unit = module.mode_norm(real, imag)
    hidden = module.mode_input(*unit)
    update = module.mode_output(functional.silu(hidden[0]), functional.silu(hidden[1]))
    mixed = real + update[0], imag + update[1]
    path = module.path_input(*mixed)
    expected = module.path_output(functional.silu(path[0]), functional.silu(path[1]))

    assert actual[0].shape == (2, 3, 2, 1, 6)
    torch.testing.assert_close(actual, expected)


def test_factorized_postfusion_matches_its_residual_semantics() -> None:
    torch.manual_seed(703)
    module = FactorizedS2DPostFusionTransition(6, post_hidden=12).double()
    pole_real = torch.randn(2, 3, 2, 6, dtype=torch.double)
    pole_imag = torch.randn_like(pole_real)
    carry_real = torch.randn(2, 3, 2, 24, dtype=torch.double)
    carry_imag = torch.randn_like(carry_real)

    actual = module(pole_real, pole_imag, carry_real, carry_imag)
    carry = module._carry(carry_real, carry_imag)
    merged = carry[0] + pole_real, carry[1] + pole_imag
    unit = module.post_norm(*merged)
    hidden = module.post_input(*unit)
    update = module.post_output(functional.silu(hidden[0]), functional.silu(hidden[1]))
    expected = merged[0] + update[0], merged[1] + update[1]

    torch.testing.assert_close(actual, expected)


def test_production_mpm8_builder_installs_semantic_dispatch_without_fused_flag(
    runners: tuple[_Runner, _Runner],
) -> None:
    candidate, _baseline = runners
    model = candidate._build(candidate.VARIANT, _config())

    assert "fused" not in inspect.signature(ModeResidualPathCollapse).parameters
    for stage in _stages(model):
        assert isinstance(stage.quadrant_path_mode_combiner, ModeResidualPathCollapse)
        assert isinstance(stage.augmented, FactorizedS2DPostFusionTransition)
    terminal = model.terminal
    assert terminal.quadrant_path_mode_combiner is None


def test_mpm8_combiner_calls_the_public_dispatcher_automatically(
    runners: tuple[_Runner, _Runner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, _baseline = runners
    model = candidate._build(candidate.VARIANT, _config())
    combiner = _stages(model)[0].quadrant_path_mode_combiner
    assert isinstance(combiner, ModeResidualPathCollapse)
    calls: list[None] = []

    def observed(
        real: torch.Tensor,
        imag: torch.Tensor,
        *,
        input_projection: GroupedWidelyLinear,
        output_projection: GroupedWidelyLinear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(None)
        hidden_real, hidden_imag = input_projection(real, imag)
        return output_projection(functional.silu(hidden_real), functional.silu(hidden_imag))

    monkeypatch.setattr(
        "lnet.pac_factorized_stage_transition.grouped_cartesian_cffn",
        observed,
    )
    real = torch.randn(2, 3, 2, 4, candidate.P)
    imag = torch.randn_like(real)

    output_real, output_imag = combiner.forward_packed(real, imag)

    assert calls == [None]
    assert output_real.shape == output_imag.shape == (2, 3, 2, 1, candidate.P)


def test_production_mpm8_model_runs_end_to_end_with_gradients(
    runners: tuple[_Runner, _Runner],
) -> None:
    candidate, _baseline = runners
    model = candidate._build(candidate.VARIANT, _config())
    inputs = torch.randn(2, 3, 32, 32)

    descriptor = model.raw_descriptor(inputs)
    assert descriptor.shape == (2, 1536)
    descriptor.square().mean().backward()

    stage1 = _stages(model)[0]
    assert isinstance(stage1.quadrant_path_mode_combiner, ModeResidualPathCollapse)
    assert isinstance(stage1.augmented, FactorizedS2DPostFusionTransition)
    active = (*stage1.quadrant_path_mode_combiner.parameters(), *stage1.augmented.parameters())
    assert all(parameter.grad is not None for parameter in active)


def test_standard_stem32_builder_keeps_the_standard_four_path_semantics(
    runners: tuple[_Runner, _Runner],
) -> None:
    _candidate, baseline = runners
    model = baseline._build(baseline.VARIANT, _config())

    for stage in _stages(model):
        assert isinstance(
            stage.quadrant_path_mode_combiner,
            FactorizedQuadrantPathModeCFFNCombiner,
        )
        assert stage.quadrant_path_mode_combiner.output_paths == 4
