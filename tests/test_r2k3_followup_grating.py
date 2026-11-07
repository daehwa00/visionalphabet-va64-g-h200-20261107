from __future__ import annotations

# pyright: reportMissingImports=false, reportPrivateUsage=false
import math

import torch
from run_r2k3_followup_grating import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Stimulus,
    _module_summary,
    circular_orientation_error_degrees,
    parse_frequencies,
    render_stimuli,
    spearman_correlation,
    stimulus_grid,
)


def test_stimulus_grid_is_complete_and_deterministic() -> None:
    first = stimulus_grid((1.0, 2.0), orientations=4, phases=4, contrast=0.9)
    second = stimulus_grid((1.0, 2.0), orientations=4, phases=4, contrast=0.9)
    assert first == second
    assert len(first) == 2 * 4 * 4 + 5
    assert len({stimulus.stimulus_id for stimulus in first}) == len(first)
    assert [stimulus.kind for stimulus in first[-5:]] == [
        "gray",
        "black",
        "white",
        "white_noise",
        "checkerboard",
    ]


def test_frequency_parser_rejects_aliases_and_unsorted_values() -> None:
    assert parse_frequencies("1,2,4") == (1.0, 2.0, 4.0)
    for invalid in ("", "0,1", "2,1", "1,1", "113"):
        try:
            parse_frequencies(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid frequency grid was accepted: {invalid}")


def test_opposite_phases_are_symmetric_around_gray_after_rendering() -> None:
    stimuli = [
        Stimulus("phase0", "grating", 4.0, 30.0, 0.0, 0.8),
        Stimulus("phase180", "grating", 4.0, 30.0, 180.0, 0.8),
    ]
    normalized = render_stimuli(stimuli, size=32, edge_taper_fraction=0.1)
    mean = normalized.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    standard_deviation = normalized.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    raw = normalized * standard_deviation + mean
    torch.testing.assert_close((raw[0] + raw[1]) / 2.0, torch.full_like(raw[0], 0.5))
    assert torch.isfinite(normalized).all()


def test_circular_orientation_error_is_modulo_180_degrees() -> None:
    first = torch.tensor([0.0, 170.0, 20.0, 90.0])
    second = torch.tensor([180.0, 10.0, 170.0, 0.0])
    torch.testing.assert_close(
        circular_orientation_error_degrees(first, second),
        torch.tensor([0.0, 20.0, 30.0, 90.0]),
    )


def test_spearman_handles_monotone_values_and_ties() -> None:
    increasing = spearman_correlation(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])
    )
    assert increasing is not None
    assert math.isclose(increasing, 1.0)
    descending = spearman_correlation(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([6.0, 5.0, 4.0])
    )
    assert descending is not None
    assert math.isclose(descending, -1.0)
    tied = spearman_correlation(torch.tensor([1.0, 1.0, 2.0]), torch.tensor([2.0, 2.0, 3.0]))
    assert tied is not None
    assert math.isclose(tied, 1.0)


def test_module_summary_recovers_constructed_frequency_and_orientation_tuning() -> None:
    frequencies = (2.0, 4.0, 8.0)
    orientations = 4
    phases = 2
    stimuli = stimulus_grid(frequencies, orientations, phases, contrast=0.9)
    modes = 3
    response = torch.zeros(len(stimuli), modes)
    preferred_orientations = (0, 1, 2)
    for pole, (frequency_index, orientation_index) in enumerate(
        zip(range(modes), preferred_orientations, strict=True)
    ):
        for phase_index in range(phases):
            flat_index = (
                frequency_index * orientations * phases
                + orientation_index * phases
                + phase_index
            )
            response[flat_index, pole] = 10.0
    resolution = 14
    learned_frequency = torch.tensor(frequencies) / resolution
    learned_angle = torch.tensor(preferred_orientations) * (math.pi / orientations)
    phase_x = 2.0 * math.pi * learned_frequency * learned_angle.cos()
    phase_y = 2.0 * math.pi * learned_frequency * learned_angle.sin()

    summary, rows, evoked = _module_summary(
        "stage3",
        "scan_state",
        response,
        stimuli,
        frequencies,
        orientations,
        phases,
        phase_x,
        phase_y,
        permutations=100,
    )

    assert math.isclose(summary["frequency_spearman_rho"], 1.0)
    assert summary["median_orientation_error_degrees"] == 0.0
    assert math.isclose(summary["orientation_pairing_alignment_mean_cos_2delta"], 1.0)
    assert evoked.shape == (3, 4, 3)
    assert [row["observed_preferred_frequency_cycles_per_image"] for row in rows] == [
        2.0,
        4.0,
        8.0,
    ]
