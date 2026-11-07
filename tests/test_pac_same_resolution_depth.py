from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from lnet import pac_same_resolution_depth as same_resolution_depth
from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_directional import direction_aligned_cells
from lnet.pac_gated_post_fusion import (
    GatedPoleExcitationS2DTransition,
    resized_gated_transition,
)
from lnet.pac_phase_gated_transition import PathCoherenceGateCollapse, PathOnlyCollapse
from lnet.pac_product_scan_reference import (
    direction_relative_cells_to_full_resolution,
    product_scan_full4_reference,
)
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock


def _template(modes: int) -> ComplexScanStage:
    return ComplexScanStage(
        modes,
        maximum_phase=1.0,
        output_modes=None,
        scan_memory_policy="recompute",
    )


def test_full16_cell_reconstruction_is_exact_for_every_direction() -> None:
    source = torch.randn(2, 6, 8, 5)
    directions = ((1, 1), (-1, 1), (1, -1), (-1, -1))
    cells = torch.stack(
        tuple(
            direction_aligned_cells(
                source,
                source,
                direction_x=direction_x,
                direction_y=direction_y,
            )[0]
            for direction_x, direction_y in directions
        ),
        dim=-3,
    )
    expected = torch.stack((source,) * 4, dim=-2)
    torch.testing.assert_close(
        direction_relative_cells_to_full_resolution(cells),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_odd_same_resolution_collapse_matches_explicit_reference() -> None:
    torch.manual_seed(97)
    modes = 8
    reference = SameResolutionPoleScanBlock(
        modes,
        reader_rank=2,
        kernel_size=3,
        pole_template=_template(modes),
        post_hidden=9,
    )
    candidate = deepcopy(reference)
    reference_input = tuple(torch.randn(2, 7, 7, modes, requires_grad=True) for _ in range(2))
    candidate_input = tuple(value.detach().clone().requires_grad_() for value in reference_input)

    source = reference.reader(*reference_input)
    pole_x, pole_y = reference.pole_scan.pole_coefficients(tuple(source[0].shape))
    paths = product_scan_full4_reference(
        pole_x,
        pole_y,
        source,
        gain_normalization=reference.pole_scan.product_gain_normalization,
    )
    expected = reference.collapse_paths(*paths)
    actual = candidate.collapsed_memory(*candidate_input)
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    cotangent = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(
        expected,
        (*reference_input, *reference.parameters()),
        cotangent,
        allow_unused=True,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        (*candidate_input, *candidate.parameters()),
        cotangent,
        allow_unused=True,
    )
    for value, target in zip(actual_gradients, expected_gradients, strict=True):
        if value is None or target is None:
            assert value is target
            continue
        torch.testing.assert_close(value, target, rtol=1.0e-4, atol=1.0e-4)


def test_odd_packed_collapse_uses_canonical_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes = 8
    block = SameResolutionPoleScanBlock(
        modes,
        reader_rank=2,
        kernel_size=3,
        pole_template=_template(modes),
        post_hidden=9,
    )
    original = same_resolution_depth.run_product_scan_path_collapse_pipeline
    calls = 0

    def tracked(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        same_resolution_depth,
        "run_product_scan_path_collapse_pipeline",
        tracked,
    )
    real = torch.randn(2, 7, 7, modes)
    imag = torch.randn_like(real)

    output = block.collapsed_memory(real, imag)

    assert output[0].shape == real.shape
    assert calls == 1


def test_nonpacked_research_collapse_uses_explicit_reference_path() -> None:
    modes = 8
    block = SameResolutionPoleScanBlock(
        modes,
        reader_rank=2,
        kernel_size=3,
        pole_template=_template(modes),
        post_hidden=9,
    )
    block.path_collapse = PathCoherenceGateCollapse(modes)
    real = torch.randn(2, 7, 7, modes, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    source = block.reader(real, imag)
    pole_x, pole_y = block.pole_scan.pole_coefficients(tuple(source[0].shape))
    paths = product_scan_full4_reference(
        pole_x,
        pole_y,
        source,
        gain_normalization=block.pole_scan.product_gain_normalization,
    )
    expected = block.collapse_paths(*paths)

    actual = block.collapsed_memory(real, imag)

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    (actual[0].square().mean() + actual[1].square().mean()).backward()
    assert block.path_collapse.coherence_weight.grad is not None


@pytest.mark.parametrize(("height", "width"), [(6, 8), (5, 7), (6, 7), (5, 8), (1, 1)])
def test_same_resolution_scan_preserves_arbitrary_grid_and_uses_full_transition(
    height: int,
    width: int,
) -> None:
    modes = 8
    block = SameResolutionPoleScanBlock(
        modes,
        reader_rank=2,
        kernel_size=3,
        pole_template=_template(modes),
        post_hidden=12,
    )
    real = torch.randn(2, height, width, modes, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    assert isinstance(block.path_collapse, PathOnlyCollapse)
    assert block.pole_scan.scan_memory_policy == "retain"

    memory = block.collapsed_memory(real, imag)
    expected = block.post_fusion(real + memory[0], imag + memory[1])
    output = block(real, imag)
    torch.testing.assert_close(output[0], expected[0], rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(output[1], expected[1], rtol=1.0e-6, atol=1.0e-6)
    assert block.post_fusion.hidden_modes == 12
    assert not hasattr(block, "residual_scale")
    assert not hasattr(block, "output_projection")

    (output[0].square().mean() + output[1].square().mean()).backward()
    assert all(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in block.parameters()
    )


def test_pole_template_is_copied_not_shared() -> None:
    modes = 8
    template = _template(modes)
    block = SameResolutionPoleScanBlock(
        modes,
        reader_rank=2,
        kernel_size=3,
        pole_template=template,
        post_hidden=12,
    )
    for name in ("damping_logits_x", "damping_logits_y", "phase_x", "phase_y"):
        source = getattr(template, name)
        target = getattr(block.pole_scan, name)
        torch.testing.assert_close(source, target, rtol=0.0, atol=0.0)
        assert source is not target


def test_same_resolution_block_separates_excitation_and_pole_widths() -> None:
    block = SameResolutionPoleScanBlock(
        6,
        pole_modes=16,
        reader_rank=2,
        kernel_size=3,
        pole_template=_template(16),
        post_hidden=12,
    )
    real = torch.randn(2, 6, 8, 6, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    memory = block.collapsed_memory(real, imag)
    assert memory[0].shape == real.shape
    output = block(real, imag)
    assert output[0].shape == real.shape
    (output[0].square().mean() + output[1].square().mean()).backward()
    assert block.memory_projection is not None
    assert all(parameter.grad is not None for parameter in block.parameters())


def test_gated_transition_resize_preserves_both_merge_branches() -> None:
    source = GatedPoleExcitationS2DTransition(6, 4, 8, post_hidden=16)
    with torch.no_grad():
        source.carry_weight.copy_(torch.randn_like(source.carry_weight))
    replacement = resized_gated_transition(source, 12)

    assert replacement.post_hidden == replacement.post_fusion.hidden_modes == 12
    torch.testing.assert_close(replacement.carry_weight, source.carry_weight)
    assert replacement.memory_projection is not None
    assert source.memory_projection is not None
    assert replacement.carry_projection is not None
    assert source.carry_projection is not None
    for target, current in (
        (replacement.memory_projection, source.memory_projection),
        (replacement.carry_projection, source.carry_projection),
    ):
        for target_parameter, source_parameter in zip(
            target.parameters(),
            current.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(target_parameter, source_parameter)
