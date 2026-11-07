from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
import math
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_pole_optimization import (
    EXPECTED_K128_BANKS,
    install_log_damping_,
    pole_snapshot,
    project_log_damping_,
)


class _PoleHolder(nn.Module):
    def __init__(self, grid_size: int, *, terminal: bool = False) -> None:
        super().__init__()
        self.pole_scan = _make_stage(grid_size, terminal=terminal)


class _K128BankFixture(nn.Module):
    """Small K128 path fixture with two radial levels and eight orientations.

    Multiscale banks use ``((index // 8 + index % 8) % 2 == 1)`` for their
    long-memory half.  Production K128 widths have 16 or 20 radial levels; this
    fixture uses two so every orientation gets one long and one short pole.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = _make_stage(56)
        self.stage2 = _make_stage(28)
        self.stage3 = _make_stage(14)
        self.terminal = _make_stage(7, terminal=True)
        self.same_resolution_blocks = nn.ModuleDict(
            {
                str(size): _PoleHolder(size, terminal=size == 7)
                for size in (56, 28, 14, 7)
            }
        )
        self.extra_same_resolution_blocks = nn.ModuleDict(
            {
                "14": nn.ModuleList(
                    [_PoleHolder(14) for _ in range(4)]
                )
            }
        )
        self.probe = nn.Linear(16, 16)


def _make_stage(grid_size: int, *, terminal: bool = False) -> ComplexScanStage:
    terminal = terminal or grid_size == 7
    return ComplexScanStage(
        16,
        maximum_phase=0.8,
        output_modes=None,
        damping_min=0.01,
        damping_max=2.0 if terminal else 0.7,
    )


def _banks(model: nn.Module) -> dict[str, ComplexScanStage]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ComplexScanStage)
    }


def _damping_values(stage: ComplexScanStage) -> tuple[Tensor, Tensor]:
    damping_x, damping_y = stage._damping_fields()
    return damping_x.detach().reshape(-1), damping_y.detach().reshape(-1)


def _sum_squared_means(values: tuple[Tensor, ...]) -> Tensor:
    total = values[0].square().mean()
    for value in values[1:]:
        total = total + value.square().mean()
    return total


def test_ordinary_stage_keeps_legacy_state_schema_and_parameter_order() -> None:
    torch.manual_seed(31)
    stage = _make_stage(8)

    assert list(stage.state_dict()) == [
        "damping_logits_x",
        "damping_logits_y",
        "phase_x",
        "phase_y",
    ]
    assert [name for name, _ in stage.named_parameters()] == [
        "damping_logits_x",
        "damping_logits_y",
        "phase_x",
        "phase_y",
    ]
    assert stage.log_damping_x is None
    assert stage.log_damping_y is None


def test_log_conversion_preserves_damping_scan_outputs_and_shared_gradients() -> None:
    torch.manual_seed(120)
    source = _K128BankFixture()
    converted = _K128BankFixture()
    converted.load_state_dict(source.state_dict(), strict=True)
    original_x, original_y = _damping_values(source.stage1)

    real = torch.randn(1, 3, 4, 16)
    imag = torch.randn(1, 3, 4, 16)
    shape = cast("tuple[int, int, int, int]", tuple(real.shape))
    source_x, source_y = source.stage1.pole_coefficients(shape)
    expected_coefficients = (*source_x, *source_y)
    expected_loss = _sum_squared_means(expected_coefficients)
    expected_phase_gradients = torch.autograd.grad(
        expected_loss,
        (source.stage1.phase_x, source.stage1.phase_y),
    )

    before_parameters = sum(parameter.numel() for parameter in converted.parameters())
    rng_before = torch.random.get_rng_state()
    install_log_damping_(converted)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert sum(parameter.numel() for parameter in converted.parameters()) == before_parameters

    converted_x, converted_y = _damping_values(converted.stage1)
    torch.testing.assert_close(converted_x, original_x, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(converted_y, original_y, rtol=1e-6, atol=1e-7)
    actual_x, actual_y = converted.stage1.pole_coefficients(shape)
    actual_coefficients = (*actual_x, *actual_y)
    for expected, actual in zip(expected_coefficients, actual_coefficients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    actual_loss = _sum_squared_means(actual_coefficients)
    actual_phase_gradients = torch.autograd.grad(
        actual_loss,
        (converted.stage1.phase_x, converted.stage1.phase_y),
    )
    for expected, actual in zip(expected_phase_gradients, actual_phase_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)

    source_real = real.detach().clone().requires_grad_()
    source_imag = imag.detach().clone().requires_grad_()
    converted_real = real.detach().clone().requires_grad_()
    converted_imag = imag.detach().clone().requires_grad_()
    source_output = source.stage1(source_real, source_imag)[1]
    converted_output = converted.stage1(converted_real, converted_imag)[1]
    torch.testing.assert_close(converted_output, source_output, rtol=1e-6, atol=2e-7)
    source_forward_gradients = torch.autograd.grad(
        source_output.square().mean(),
        (source_real, source_imag, source.stage1.phase_x, source.stage1.phase_y),
    )
    converted_forward_gradients = torch.autograd.grad(
        converted_output.square().mean(),
        (converted_real, converted_imag, converted.stage1.phase_x, converted.stage1.phase_y),
    )
    for expected, actual in zip(
        source_forward_gradients,
        converted_forward_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=3e-7)

    keys = set(converted.stage1.state_dict())
    assert "damping_logits_x" not in keys
    assert "damping_logits_y" not in keys
    assert {"log_damping_x", "log_damping_y"}.issubset(keys)
    parameter_names = dict(converted.stage1.named_parameters())
    assert "damping_logits_x" not in parameter_names
    assert "damping_logits_y" not in parameter_names
    assert {"log_damping_x", "log_damping_y"}.issubset(parameter_names)


def test_multiscale_initialization_balances_orientations_and_is_rng_free() -> None:
    torch.manual_seed(991)
    baseline = _K128BankFixture()
    torch.manual_seed(991)
    improved = _K128BankFixture()
    baseline_state = baseline.state_dict()
    parameter_count = sum(parameter.numel() for parameter in improved.parameters())
    rng_before = torch.random.get_rng_state()

    install_log_damping_(improved, multiscale_initialization=True)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert len(_banks(improved)) == EXPECTED_K128_BANKS
    assert sum(parameter.numel() for parameter in improved.parameters()) == parameter_count
    assert all(stage.log_damping_x is not None for stage in _banks(improved).values())
    for name, value in baseline_state.items():
        if "damping_logits_" in name or "log_damping_" in name:
            continue
        torch.testing.assert_close(improved.state_dict()[name], value, rtol=0.0, atol=0.0)

    before_snapshot = pole_snapshot(baseline)
    after_snapshot = pole_snapshot(improved)
    assert len(after_snapshot["banks"]) == EXPECTED_K128_BANKS
    assert after_snapshot["initialization"]["name"] == "balanced_short_long_v1"
    assert after_snapshot["initialization"]["long_index_mask"] == (
        "((index // 8 + index % 8) % 2 == 1)"
    )
    by_name = {bank["name"]: bank for bank in after_snapshot["banks"]}
    before_by_name = {bank["name"]: bank for bank in before_snapshot["banks"]}
    for name, bank in by_name.items():
        original = before_by_name[name]
        assert bank["grid_size"] == original["grid_size"]
        modes = len(bank["effective_damping_x"])
        radial_levels = modes // 8
        assert radial_levels % 2 == 0
        long = [
            index
            for index in range(modes)
            if (index // 8 + index % 8) % 2 == 1
        ]
        short = [index for index in range(modes) if index not in long]
        assert len(long) == len(short) == modes // 2
        for orientation in range(8):
            orientation_indices = [
                radial * 8 + orientation for radial in range(radial_levels)
            ]
            assert sum(index in long for index in orientation_indices) == (
                radial_levels // 2
            )
        for index in short:
            assert bank["effective_damping_x"][index] == pytest.approx(
                original["effective_damping_x"][index], rel=1e-6, abs=1e-7
            )
            assert bank["effective_damping_y"][index] == pytest.approx(
                original["effective_damping_y"][index], rel=1e-6, abs=1e-7
            )
        x_long = [bank["effective_damping_x"][index] for index in long]
        y_long = [bank["effective_damping_y"][index] for index in long]
        assert y_long == pytest.approx(list(reversed(x_long)), rel=1e-6, abs=1e-7)
        assert all(bank["bounds"]["min"] < value < bank["bounds"]["max"] for value in x_long)
        assert all(bank["bounds"]["min"] < value < bank["bounds"]["max"] for value in y_long)
        assert after_snapshot["initialization"]["banks"][name]["long_index_mask"] == (
            "((index // 8 + index % 8) % 2 == 1)"
        )

    round_trip = json.loads(json.dumps(after_snapshot, sort_keys=True))
    assert round_trip == after_snapshot


def test_projection_clamps_eta_without_killing_boundary_gradients() -> None:
    model = _K128BankFixture()
    install_log_damping_(model)
    for stage in _banks(model).values():
        assert stage.log_damping_x is not None
        assert stage.log_damping_y is not None
        with torch.no_grad():
            stage.log_damping_x[0] = math.log(stage.damping_min) - 2.0
            stage.log_damping_y[0] = math.log(stage.damping_max) + 2.0

    project_log_damping_(model)

    for stage in _banks(model).values():
        assert stage.log_damping_x is not None
        assert stage.log_damping_y is not None
        assert stage.log_damping_x[0].item() == pytest.approx(math.log(stage.damping_min))
        assert stage.log_damping_y[0].item() == pytest.approx(math.log(stage.damping_max))
    boundary_bank = model.stage1
    assert boundary_bank.log_damping_x is not None
    assert boundary_bank.log_damping_y is not None
    damping_x, damping_y = boundary_bank._damping_fields()
    (damping_x.sum() + damping_y.sum()).backward()
    assert boundary_bank.log_damping_x.grad is not None
    assert boundary_bank.log_damping_y.grad is not None
    assert boundary_bank.log_damping_x.grad[0].item() > 0.0
    assert boundary_bank.log_damping_y.grad[0].item() > 0.0


def test_multiscale_custom_paths_require_explicit_grid_sizes() -> None:
    model = _K128BankFixture()
    custom_stage = model.stage1
    delattr(model, "stage1")
    model.custom_bank = custom_stage

    with pytest.raises(ValueError, match="explicit grid size"):
        install_log_damping_(model, multiscale_initialization=True)

    install_log_damping_(
        model,
        multiscale_initialization=True,
        grid_sizes={"custom_bank": 56},
    )
    snapshot = pole_snapshot(model, grid_sizes={"custom_bank": 56})
    custom_snapshot = next(
        bank for bank in snapshot["banks"] if bank["name"] == "custom_bank"
    )
    assert custom_snapshot["grid_size"] == 56
