# pyright: reportExplicitAny=false, reportMissingImports=false, reportPrivateUsage=false
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import pytest
import torch
from torch import nn

from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig
from lnet.pac_factorized_stage_transition import (
    FactorizedS2DPostFusionTransition,
    ModeResidualPathCollapse,
)
from lnet.pac_grouped_path_cffn import grouped_cartesian_cffn
from lnet.pac_mean_one_magnitude_gate import mean_one_magnitude_gate
from lnet.pac_phase_gated_cffn import PhaseGatedComplexFFN
from lnet.pac_phase_gated_transition import (
    AveragePoolMagnitudeGateTransition,
    AveragePoolPhaseGatedRefinementTransition,
    DirectPathCollapse,
    MemoryOnlyMagnitudeGateTransition,
    PathOnlyCollapse,
    PathPhaseGatedCollapse,
    PhaseGatedModeResidualPathCollapse,
)
from lnet.pac_triton_grouped_path_cffn import d4_grouped_path_collapse_reference

if TYPE_CHECKING:
    from lnet.complex_scan import ComplexScanStage


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RUNNER = "run_a2d_deep4_calibrated_uniform_p96_phase_gated_imagenet100"


class _Runner(Protocol):
    VARIANT: str

    def _build(self, variant: str, config: ComplexScanConfig) -> ComplexScanBackbone: ...

    def _variant_config(self) -> dict[str, object]: ...

    def _wandb_model_metrics(self, model: torch.nn.Module) -> dict[str, float]: ...


class _TrainingDiagnostics(Protocol):
    _latest_training_diagnostics: dict[str, float]


class _OptimizerSource(Protocol):
    def _build_optimizer(
        self,
        model: torch.nn.Module,
        recipe: dict[str, float | bool],
    ) -> torch.optim.Optimizer: ...


def _runner() -> _Runner:
    sys.path.insert(0, str(SCRIPTS))
    try:
        return cast("_Runner", cast("object", importlib.import_module(RUNNER)))
    finally:
        sys.path.remove(str(SCRIPTS))


def _optimizer_source() -> _OptimizerSource:
    sys.path.insert(0, str(SCRIPTS))
    try:
        return cast(
            "_OptimizerSource",
            cast("object", importlib.import_module("run_complex_scan_augmented_cifar100")),
        )
    finally:
        sys.path.remove(str(SCRIPTS))


def test_phase_gated_mode_reuses_exact_mpm8_path_block() -> None:
    torch.manual_seed(811)
    baseline = ModeResidualPathCollapse(6, mode_hidden=12, path_hidden=8).double()
    module = PhaseGatedModeResidualPathCollapse(6, mode_hidden=10, path_hidden=8).double()
    module.copy_path_from(baseline)
    real = torch.randn(2, 3, 2, 4, 6, dtype=torch.double)
    imag = torch.randn_like(real)

    actual = module.forward_packed(real, imag)
    mixed = module.mode(real, imag)
    expected = grouped_cartesian_cffn(
        *mixed,
        input_projection=baseline.path_input,
        output_projection=baseline.path_output,
    )

    assert actual[0].shape == actual[1].shape == (2, 3, 2, 1, 6)
    torch.testing.assert_close(actual, expected)


def test_path_only_collapse_is_exactly_the_retained_path_block() -> None:
    torch.manual_seed(31)
    baseline = PhaseGatedModeResidualPathCollapse(7, mode_hidden=5, path_hidden=3)
    path_only = PathOnlyCollapse(7, path_hidden=3)
    path_only.copy_path_from(baseline)
    real = torch.randn(2, 4, 6, 4, 7)
    imag = torch.randn_like(real)

    expected = grouped_cartesian_cffn(
        real,
        imag,
        input_projection=baseline.path_input,
        output_projection=baseline.path_output,
    )
    actual = path_only.forward_packed(real, imag)

    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(right, left, rtol=0, atol=0)


def test_direct_path_collapse_fused_encoding_is_exact_and_differentiable() -> None:
    torch.manual_seed(35)
    direct = DirectPathCollapse(7)
    encoded = DirectPathCollapse(7)
    encoded.load_state_dict(direct.state_dict())
    direct_real = torch.randn(2, 3, 2, 4, 7, requires_grad=True)
    direct_imag = torch.randn_like(direct_real, requires_grad=True)
    encoded_real = direct_real.detach().clone().requires_grad_()
    encoded_imag = direct_imag.detach().clone().requires_grad_()

    actual = direct.forward_packed(direct_real, direct_imag)
    expected = d4_grouped_path_collapse_reference(
        encoded_real,
        encoded_imag,
        *encoded.packed_parameters(),
    )
    actual_loss = actual[0].square().mean() + actual[1].square().mean()
    expected_loss = expected[0].square().mean() + expected[1].square().mean()
    actual_loss.backward()
    expected_loss.backward()

    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(direct_real.grad, encoded_real.grad, rtol=5.0e-5, atol=2.0e-6)
    torch.testing.assert_close(direct_imag.grad, encoded_imag.grad, rtol=5.0e-5, atol=2.0e-6)
    for direct_parameter, encoded_parameter in zip(
        direct.parameters(),
        encoded.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            direct_parameter.grad,
            encoded_parameter.grad,
            rtol=5.0e-5,
            atol=2.0e-6,
        )


@pytest.mark.parametrize("activation", ["silu", "identity"])
def test_path_phase_gated_collapse_places_one_shared_pg_between_path_projections(
    activation: str,
) -> None:
    apply_cartesian_silu = activation == "silu"
    torch.manual_seed(37)
    baseline = PhaseGatedModeResidualPathCollapse(7, mode_hidden=5, path_hidden=3)
    module = PathPhaseGatedCollapse(
        7,
        mode_hidden=5,
        path_hidden=3,
        apply_cartesian_silu=apply_cartesian_silu,
    )
    module.copy_from(baseline)
    real = torch.randn(2, 4, 6, 4, 7)
    imag = torch.randn_like(real)

    hidden = module.path_input(real, imag)
    mixed = module.mode(hidden[0].contiguous(), hidden[1].contiguous())
    if apply_cartesian_silu:
        mixed = tuple(torch.nn.functional.silu(coordinate) for coordinate in mixed)
    expected = module.path_output(
        mixed[0],
        mixed[1],
    )
    actual = module.forward_packed(real, imag)

    assert hidden[0].shape == (2, 4, 6, 3, 7)
    assert module.apply_cartesian_silu is apply_cartesian_silu
    assert sum(isinstance(child, PhaseGatedComplexFFN) for child in module.modules()) == 1
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(right, left, rtol=0, atol=0)


def test_average_pool_phase_gated_refinement_matches_the_declared_equation() -> None:
    torch.manual_seed(43)
    modes = 6
    module = AveragePoolPhaseGatedRefinementTransition(modes, refine_hidden=9)
    memory_real = torch.randn(2, 3, 2, modes)
    memory_imag = torch.randn_like(memory_real)
    source_real = torch.randn(2, 6, 4, modes)
    source_imag = torch.randn_like(source_real)
    carry_real = (
        source_real.reshape(2, 3, 2, 2, 2, modes)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(2, 3, 2, 4 * modes)
    )
    carry_imag = (
        source_imag.reshape(2, 3, 2, 2, 2, modes)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(2, 3, 2, 4 * modes)
    )

    actual = module(memory_real, memory_imag, carry_real, carry_imag)
    local_real, local_imag = module._carry(carry_real, carry_imag)
    expected_local_real = torch.nn.functional.avg_pool2d(
        source_real.permute(0, 3, 1, 2),
        kernel_size=2,
        stride=2,
    ).permute(0, 2, 3, 1)
    expected_local_imag = torch.nn.functional.avg_pool2d(
        source_imag.permute(0, 3, 1, 2),
        kernel_size=2,
        stride=2,
    ).permute(0, 2, 3, 1)
    expected = module.refine(
        memory_real + expected_local_real,
        memory_imag + expected_local_imag,
    )

    assert module.refine.modes == modes
    assert module.refine.hidden_modes == 9
    assert module.refine.gamma is None
    assert not any("carry_weight" in name for name, _ in module.named_parameters())
    torch.testing.assert_close(local_real, expected_local_real, rtol=0, atol=0)
    torch.testing.assert_close(local_imag, expected_local_imag, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected)
    (actual[0].square().mean() + actual[1].square().mean()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_average_pool_pure_gate_is_exactly_the_identity_centered_equation() -> None:
    torch.manual_seed(47)
    modes = 6
    module = AveragePoolMagnitudeGateTransition(modes)
    memory_real = torch.randn(2, 3, 2, modes, requires_grad=True)
    memory_imag = torch.randn_like(memory_real, requires_grad=True)
    carry_real = torch.randn(2, 3, 2, 4 * modes, requires_grad=True)
    carry_imag = torch.randn_like(carry_real, requires_grad=True)

    local_real, local_imag = module._carry(carry_real, carry_imag)
    merged_real = memory_real + local_real
    merged_imag = memory_imag + local_imag
    normalized_real, normalized_imag = module.norm(merged_real, merged_imag)
    gain = mean_one_magnitude_gate(
        normalized_real,
        normalized_imag,
        module.gate.alpha,
        redistribution=module.gate.redistribution,
    )
    expected = merged_real * gain, merged_imag * gain
    residual_form = (
        merged_real + merged_real * (gain - 1.0),
        merged_imag + merged_imag * (gain - 1.0),
    )
    actual = module(memory_real, memory_imag, carry_real, carry_imag)

    torch.testing.assert_close(gain.mean(dim=-1), torch.ones_like(gain[..., 0]))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, residual_form)
    assert set(dict(module.named_parameters())) == {"norm.weight", "gate.alpha"}
    (actual[0].square().mean() + actual[1].square().mean()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (memory_real, memory_imag, carry_real, carry_imag)
    )


def test_average_pool_pure_gate_is_exact_identity_when_gate_is_one() -> None:
    module = AveragePoolMagnitudeGateTransition(5)
    with torch.no_grad():
        module.gate.alpha.zero_()
    memory_real = torch.randn(2, 3, 2, 5)
    memory_imag = torch.randn_like(memory_real)
    carry_real = torch.randn(2, 3, 2, 20)
    carry_imag = torch.randn_like(carry_real)

    actual = module(memory_real, memory_imag, carry_real, carry_imag)
    local = module._carry(carry_real, carry_imag)
    expected = memory_real + local[0], memory_imag + local[1]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_memory_only_pure_gate_removes_the_carry_branch_exactly() -> None:
    torch.manual_seed(59)
    modes = 6
    module = MemoryOnlyMagnitudeGateTransition(modes)
    memory_real = torch.randn(2, 3, 2, modes, requires_grad=True)
    memory_imag = torch.randn_like(memory_real, requires_grad=True)
    carry_real = torch.randn(2, 3, 2, 4 * modes, requires_grad=True)
    carry_imag = torch.randn_like(carry_real, requires_grad=True)

    normalized_real, normalized_imag = module.norm(memory_real, memory_imag)
    gain = mean_one_magnitude_gate(
        normalized_real,
        normalized_imag,
        module.gate.alpha,
        redistribution=module.gate.redistribution,
    )
    expected = memory_real * gain, memory_imag * gain
    actual = module(memory_real, memory_imag, carry_real, carry_imag)
    changed_carry = module(
        memory_real,
        memory_imag,
        100.0 * carry_real,
        -100.0 * carry_imag,
    )

    assert not hasattr(module, "carry_input_modes")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(changed_carry, expected)
    (actual[0].square().mean() + actual[1].square().mean()).backward()
    assert memory_real.grad is not None
    assert memory_imag.grad is not None
    assert carry_real.grad is None
    assert carry_imag.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_memory_only_pure_gate_is_identity_when_gate_is_one() -> None:
    module = MemoryOnlyMagnitudeGateTransition(5)
    with torch.no_grad():
        module.gate.alpha.zero_()
    memory_real = torch.randn(2, 3, 2, 5)
    memory_imag = torch.randn_like(memory_real)

    actual = module(memory_real, memory_imag)

    torch.testing.assert_close(actual, (memory_real, memory_imag), rtol=0, atol=0)


def test_average_pool_pure_gate_is_globally_phase_equivariant() -> None:
    torch.manual_seed(61)
    module = AveragePoolMagnitudeGateTransition(5).double().eval()
    memory_real = torch.randn(2, 3, 2, 5, dtype=torch.double)
    memory_imag = torch.randn_like(memory_real)
    carry_real = torch.randn(2, 3, 2, 20, dtype=torch.double)
    carry_imag = torch.randn_like(carry_real)
    angle = 0.41
    cosine = math.cos(angle)
    sine = math.sin(angle)

    expected_real, expected_imag = module(
        memory_real,
        memory_imag,
        carry_real,
        carry_imag,
    )
    actual_real, actual_imag = module(
        cosine * memory_real - sine * memory_imag,
        sine * memory_real + cosine * memory_imag,
        cosine * carry_real - sine * carry_imag,
        sine * carry_real + cosine * carry_imag,
    )

    torch.testing.assert_close(
        actual_real,
        cosine * expected_real - sine * expected_imag,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        actual_imag,
        sine * expected_real + cosine * expected_imag,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_average_pool_refinement_restores_the_bf16_packed_pg_boundary() -> None:
    module = AveragePoolPhaseGatedRefinementTransition(96, refine_hidden=96).cuda()
    memory_real = torch.randn(2, 3, 2, 96, device="cuda", requires_grad=True)
    memory_imag = torch.randn_like(memory_real, requires_grad=True)
    carry_real = torch.randn(2, 3, 2, 384, device="cuda", requires_grad=True)
    carry_imag = torch.randn_like(carry_real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = module(
            memory_real,
            memory_imag,
            carry_real,
            carry_imag,
        )
        loss = output_real.float().square().mean() + output_imag.float().square().mean()
    loss.backward()

    assert output_real.dtype is torch.bfloat16
    assert output_imag.dtype is torch.bfloat16
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (memory_real, memory_imag, carry_real, carry_imag)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_average_pool_pure_gate_keeps_the_bf16_compiled_boundary() -> None:
    module = AveragePoolMagnitudeGateTransition(96).cuda()
    memory_real = torch.randn(2, 3, 2, 96, device="cuda", requires_grad=True)
    memory_imag = torch.randn_like(memory_real, requires_grad=True)
    carry_real = torch.randn(2, 3, 2, 384, device="cuda", requires_grad=True)
    carry_imag = torch.randn_like(carry_real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = module(
            memory_real,
            memory_imag,
            carry_real,
            carry_imag,
        )
        loss = output_real.float().square().mean() + output_imag.float().square().mean()
    loss.backward()

    assert output_real.dtype is torch.bfloat16
    assert output_imag.dtype is torch.bfloat16
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (memory_real, memory_imag, carry_real, carry_imag)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_memory_only_pure_gate_keeps_the_bf16_boundary() -> None:
    module = MemoryOnlyMagnitudeGateTransition(96).cuda()
    memory_real = torch.randn(2, 3, 2, 96, device="cuda", requires_grad=True)
    memory_imag = torch.randn_like(memory_real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = module(memory_real, memory_imag)
        loss = output_real.float().square().mean() + output_imag.float().square().mean()
    loss.backward()

    assert output_real.dtype is torch.bfloat16
    assert output_imag.dtype is torch.bfloat16
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (memory_real, memory_imag)
    )


def test_production_runner_installs_phase_gated_mode_without_manual_flag() -> None:
    runner = _runner()
    model = runner._build(
        runner.VARIANT,
        ComplexScanConfig(output_dim=100, stem_strides=(2, 2)),
    )

    for name in ("stage1", "stage2", "stage3"):
        stage = cast("ComplexScanStage", getattr(model, name))
        assert isinstance(
            stage.quadrant_path_mode_combiner,
            PhaseGatedModeResidualPathCollapse,
        )
        assert type(stage.augmented) is FactorizedS2DPostFusionTransition
    assert model.terminal.quadrant_path_mode_combiner is None


def test_phase_gated_projections_do_not_decay_while_gamma_suppresses_gradients() -> None:
    module = PhaseGatedComplexFFN(6, 9)
    recipe = {
        "learning_rate": 3.0e-3,
        "weight_decay": 0.05,
        "modal_learning_rate_multiplier": 1.0 / 3.0,
        "pole_geometry_learning_rate_multiplier": 0.1,
        "fused_optimizer": False,
    }
    optimizer = _optimizer_source()._build_optimizer(module, recipe)
    groups = {
        id(parameter): (float(group["lr"]), float(group["weight_decay"]))
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    projection_parameters = tuple(module.input_projection.parameters()) + tuple(
        module.output_projection.parameters()
    )
    for parameter in projection_parameters:
        assert groups[id(parameter)] == (3.0e-3, 0.0)

    before = tuple(parameter.detach().clone() for parameter in projection_parameters)
    for parameter in module.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for expected, actual in zip(before, projection_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_phase_gated_optimizer_change_does_not_disable_global_matrix_decay() -> None:
    model = nn.Sequential(PhaseGatedComplexFFN(6, 9), nn.Linear(6, 4, bias=False))
    recipe = {
        "learning_rate": 3.0e-3,
        "weight_decay": 0.05,
        "modal_learning_rate_multiplier": 1.0 / 3.0,
        "pole_geometry_learning_rate_multiplier": 0.1,
        "fused_optimizer": False,
    }
    optimizer = _optimizer_source()._build_optimizer(model, recipe)
    groups = {
        id(parameter): float(group["weight_decay"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert groups[id(model[1].weight)] == 0.05


def test_production_runner_uses_requested_main_and_auxiliary_heads() -> None:
    runner = _runner()
    model = runner._build(
        runner.VARIANT,
        ComplexScanConfig(output_dim=100, stem_strides=(2, 2)),
    )
    classifier = cast("Any", model.classifier)
    main = classifier.fusion
    auxiliary = classifier.affine

    assert model.descriptor_dim == 1536
    assert isinstance(main.standardizer, nn.BatchNorm1d)
    assert main.standardizer.affine is False
    assert (main.fusion.in_features, main.fusion.out_features) == (1536, 384)
    assert isinstance(main.activation, nn.GELU)
    assert isinstance(main.norm, nn.RMSNorm)
    assert (main.refinement.in_features, main.refinement.out_features) == (384, 256)
    assert isinstance(main.refinement_activation, nn.GELU)
    assert isinstance(main.refinement_norm, nn.RMSNorm)
    assert (main.classifier.in_features, main.classifier.out_features) == (256, 100)
    assert isinstance(auxiliary.standardizer, nn.BatchNorm1d)
    assert auxiliary.standardizer.affine is False
    assert (auxiliary.linear.in_features, auxiliary.linear.out_features) == (1536, 100)
    assert classifier.affine_auxiliary_weight == 0.5


def test_production_objective_weights_affine_mixup_ce_by_one_half() -> None:
    runner = _runner()
    model = runner._build(
        runner.VARIANT,
        ComplexScanConfig(output_dim=100, stem_strides=(2, 2)),
    )
    output = model.classifier(torch.randn(4, 1536))
    targets = torch.tensor([1, 2, 3, 4])
    permuted_targets = torch.tensor([4, 3, 2, 1])
    _, loss, terms = cast("Any", runner).head_runner._training_objective(
        model,
        output,
        targets,
        permuted_targets,
        0.7,
    )

    torch.testing.assert_close(
        loss,
        terms["joint_ce"] + 0.5 * terms["affine_aux_ce"],
    )


def test_production_contract_describes_the_actual_p1_workspace() -> None:
    runner = _runner()
    payload = runner._variant_config()
    backbone = cast("dict[str, object]", payload["backbone"])
    transition = cast("dict[str, object]", backbone["stage_transition"])
    gate = cast("dict[str, object]", transition["mode_gate"])

    assert gate["value_hidden_width"] == 192
    assert gate["gate_hidden_width"] == 192
    assert gate["input_projection_complex_width"] == 384
    assert gate["gate_statistics_dtype"] == "float32"
    assert gate["beta"] == "removed"
    assert gate["gate_mean"] == 1.0
    assert gate["redistribution"] == 0.5
    assert gate["residual_scale"] == {
        "name": "gamma",
        "initial": 0.01,
        "learnable": True,
    }
    head = cast("dict[str, object]", payload["head"])
    assert head["affine_auxiliary_weight"] == 0.5


def test_production_metrics_expose_phase_gate_branch_gradients() -> None:
    runner = _runner()
    model = runner._build(
        runner.VARIANT,
        ComplexScanConfig(output_dim=100, stem_strides=(2, 2)),
    )
    for name in ("stage1", "stage2", "stage3"):
        stage = cast("ComplexScanStage", getattr(model, name))
        mixer = cast("PhaseGatedModeResidualPathCollapse", stage.quadrant_path_mode_combiner)
        for parameter in mixer.mode.parameters():
            parameter.grad = torch.ones_like(parameter)

    metrics = runner._wandb_model_metrics(model)

    for index in (1, 2, 3):
        prefix = f"phase_gated/stage{index}/mode_"
        assert metrics[f"{prefix}input_value_weight_grad_norm"] > 0.0
        assert metrics[f"{prefix}input_gate_weight_grad_norm"] > 0.0
        assert metrics[f"{prefix}output_weight_grad_norm"] > 0.0
        assert metrics[f"{prefix}alpha_grad_norm"] > 0.0
        assert metrics[f"{prefix}gamma_grad_norm"] > 0.0


def test_production_metrics_preserve_separate_training_loss_terms() -> None:
    runner = _runner()
    model = runner._build(
        runner.VARIANT,
        ComplexScanConfig(output_dim=100, stem_strides=(2, 2)),
    )
    cast("_TrainingDiagnostics", cast("object", model))._latest_training_diagnostics = {
        "joint_ce": 3.25,
        "affine_aux_ce": 4.75,
    }

    metrics = runner._wandb_model_metrics(model)

    assert metrics["train/joint_ce"] == 3.25
    assert metrics["train/affine_aux_ce"] == 4.75


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_fp32_autocast_connects_phase_mode_directly_to_grouped_path() -> None:
    torch.manual_seed(817)
    module = PhaseGatedModeResidualPathCollapse(
        96,
        mode_hidden=192,
        path_hidden=8,
    ).cuda()
    real = torch.randn(2, 3, 2, 4, 96, device="cuda", requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = module.forward_packed(real, imag)
    (output_real.float().square().mean() + output_imag.float().square().mean()).backward()

    assert output_real.dtype is output_imag.dtype is torch.bfloat16
    assert real.grad is not None
    assert imag.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())
