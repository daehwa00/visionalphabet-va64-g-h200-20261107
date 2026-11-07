from __future__ import annotations

import torch

from scripts.run_r2k3_followup_rotation_paths import (
    EXPECTED_PERMUTATION,
    PERMUTATIONS,
    LayerAccumulator,
    align_rotated_field,
    expected_ccw90_permutation,
)


def _rotated_observation(
    clean_real: torch.Tensor,
    clean_imag: torch.Tensor,
    *,
    conjugate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = torch.tensor(EXPECTED_PERMUTATION)
    aligned_real = clean_real.index_select(-2, permutation)
    aligned_imag = clean_imag.index_select(-2, permutation)
    if conjugate:
        aligned_imag = -aligned_imag
    return (
        torch.rot90(aligned_real, k=1, dims=(1, 2)),
        torch.rot90(aligned_imag, k=1, dims=(1, 2)),
    )


def test_ccw90_expected_path_permutation_uses_documented_convention() -> None:
    assert expected_ccw90_permutation() == (1, 3, 0, 2)
    assert EXPECTED_PERMUTATION in PERMUTATIONS
    assert len(PERMUTATIONS) == 24


def test_spatial_alignment_and_expected_permutation_recover_exact_field() -> None:
    generator = torch.Generator().manual_seed(7)
    clean_real = torch.randn(3, 5, 5, 4, 6, generator=generator)
    clean_imag = torch.randn(3, 5, 5, 4, 6, generator=generator)
    rotated = _rotated_observation(clean_real, clean_imag)
    expected = torch.tensor(EXPECTED_PERMUTATION)
    assert torch.equal(align_rotated_field(rotated[0]), clean_real.index_select(-2, expected))
    assert torch.equal(align_rotated_field(rotated[1]), clean_imag.index_select(-2, expected))

    accumulator = LayerAccumulator()
    accumulator.add((clean_real, clean_imag), rotated)
    aggregate = accumulator.aggregate_metrics()
    expected_index = PERMUTATIONS.index(EXPECTED_PERMUTATION)
    assert torch.isclose(
        aggregate["direct_complex_cosine"][expected_index],
        aggregate["direct_complex_cosine"].new_tensor(1.0),
        atol=1.0e-6,
    )
    # Cross moments are accumulated from float32 GEMMs; the energy identity
    # has the expected small cancellation error before conversion to float64.
    assert aggregate["direct_symmetric_nrmse"][expected_index] < 1.0e-3
    assert int(aggregate["direct_complex_cosine"].argmax()) == expected_index


def test_global_conjugation_is_scored_as_fixed_sensitivity_convention() -> None:
    generator = torch.Generator().manual_seed(11)
    clean_real = torch.randn(2, 4, 4, 4, 5, generator=generator)
    clean_imag = torch.randn(2, 4, 4, 4, 5, generator=generator)
    rotated = _rotated_observation(clean_real, clean_imag, conjugate=True)
    accumulator = LayerAccumulator()
    accumulator.add((clean_real, clean_imag), rotated)
    aggregate = accumulator.aggregate_metrics()
    expected_index = PERMUTATIONS.index(EXPECTED_PERMUTATION)
    assert torch.isclose(
        aggregate["conjugate_complex_cosine"][expected_index],
        aggregate["conjugate_complex_cosine"].new_tensor(1.0),
        atol=1.0e-6,
    )
    assert aggregate["conjugate_symmetric_nrmse"][expected_index] < 1.0e-3
    assert aggregate["direct_complex_cosine"][expected_index] < 0.5
