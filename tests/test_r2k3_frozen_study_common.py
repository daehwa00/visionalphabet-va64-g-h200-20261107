from __future__ import annotations

# pyright: reportMissingImports=false
import torch
from r2k3_frozen_study_common import (
    PredictionResult,
    balanced_indices,
    bootstrap_mean_ci,
    coefficient_intervention,
    paired_effect,
    path_intervention,
    pole_mask_transform,
    recurrence_off_transform,
)

from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_phase_gated_transition import PathOnlyCollapse


def _prediction(logits: torch.Tensor, labels: torch.Tensor) -> PredictionResult:
    probability = logits.softmax(-1)
    confidence, prediction = probability.max(-1)
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return PredictionResult(
        indices=torch.arange(len(labels)),
        labels=labels,
        logits=logits,
        losses=losses,
        margins=true - masked.max(-1).values,
        confidences=confidence,
        predictions=prediction,
    )


def test_balanced_indices_are_deterministic_and_class_balanced() -> None:
    targets = [0] * 10 + [1] * 10 + [2] * 10
    first = balanced_indices(targets, 3)
    second = balanced_indices(targets, 3)
    assert first == second
    assert [targets[index] for index in first] == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_paired_effect_uses_aligned_per_example_values() -> None:
    labels = torch.tensor([0, 1, 0, 1])
    baseline = _prediction(torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]]), labels)
    candidate = _prediction(torch.tensor([[0.0, 4.0], [0.0, 4.0], [3.0, 0.0], [4.0, 0.0]]), labels)
    effect = paired_effect(baseline, candidate, bootstrap_repeats=200, seed=7)
    assert effect["accuracy_delta_pp"] == -50.0
    assert effect["correct_to_wrong"] == 2
    assert effect["prediction_flips"] == 2


def test_bootstrap_mean_ci_is_reproducible() -> None:
    values = torch.linspace(-1, 1, 30)
    labels = torch.arange(30) % 3
    assert bootstrap_mean_ci(values, labels, repeats=200, seed=13) == bootstrap_mean_ci(
        values, labels, repeats=200, seed=13
    )


def test_recurrence_off_avoids_zero_power_corner() -> None:
    pole = tuple(torch.ones(1, 1, 1, 4) for _ in range(4))
    transformed = recurrence_off_transform(
        "stage",
        ComplexScanStage(4, maximum_phase=1.0, output_modes=None),
        pole,
        "x",
    )
    assert torch.all(transformed[0] == 1.0e-6)
    assert torch.all(transformed[1] == 0)
    assert transformed[2] is pole[2]
    assert transformed[3] is pole[3]


def test_coefficient_intervention_restores_method() -> None:
    module = ComplexScanStage(4, maximum_phase=1.0, output_modes=None)
    original = module.pole_coefficients
    selected = {"stage": [0, 1]}
    with coefficient_intervention({"stage": module}, pole_mask_transform(selected)):
        pole_x, pole_y = module.pole_coefficients((1, 4, 4, 4))
        assert torch.all(pole_x[2][..., :2] == 0)
        assert torch.all(pole_y[3][..., :2] == 0)
    assert module.pole_coefficients == original


def test_path_intervention_restores_all_complex_weights() -> None:
    module = PathOnlyCollapse(4)
    before = {name: value.detach().clone() for name, value in module.state_dict().items()}
    with path_intervention({"path": module}, (1, 3)):
        assert torch.all(module.path_input.weight_real[..., [1, 3]] == 0)
        assert torch.all(module.path_input.conjugate_imag[..., [1, 3]] == 0)
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, before[name])
