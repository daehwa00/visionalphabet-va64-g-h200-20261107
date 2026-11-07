from __future__ import annotations

# pyright: reportPrivateUsage=false
from PIL import Image

from scripts.run_r2k3_frozen_study_robustness import (
    PerturbedImageFolder,
    _condition_grid,
    _cross_grid,
)


def test_resolution_views_reach_the_model_at_requested_size() -> None:
    dataset = object.__new__(PerturbedImageFolder)
    image = Image.new("RGB", (480, 360), color=(127, 127, 127))

    for size in (192, 256, 288):
        value = dataset._resolution_view(image, size)
        assert value.shape == (3, size, size)


def test_robustness_protocol_has_unique_conditions_and_seven_crosses() -> None:
    base_ids = [condition_id for condition_id, _kind, _severity in _condition_grid()]
    cross_ids = [condition_id for condition_id, _kind, _severity in _cross_grid()]

    assert len(base_ids) == 28
    assert len(base_ids) == len(set(base_ids))
    assert len(cross_ids) == 7
    assert set(cross_ids).issubset(base_ids)
