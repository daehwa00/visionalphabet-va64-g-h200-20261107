from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

# pyright: reportArgumentType=false, reportPrivateUsage=false
import pytest
import torch

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from scripts import a2d_r2k3_diagnostics as diagnostics
from scripts import analyze_a2d_r2k3_checkpoint_capacity as analysis

if TYPE_CHECKING:
    from pathlib import Path


def test_factorial_binding_exposes_joint_k_and_p_width() -> None:
    binding = analysis._binding("factorial", "K96-PF15K-SR56-28-14-7")
    assert binding.excitation_modes == (96, 96, 96, 96)
    assert binding.pole_modes == (96, 96, 96, 96)


def test_balanced_indices_are_deterministic_and_class_balanced() -> None:
    targets = [0] * 5 + [1] * 5
    assert analysis._balanced_indices(targets, 2) == [1, 6, 3, 8]
    assert analysis._balanced_indices(targets, 0) == [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    with pytest.raises(ValueError, match="fewer than"):
        analysis._balanced_indices(targets, 6)


def test_balanced_indices_keep_every_stats_prefix_class_representative() -> None:
    targets = [target for target in range(100) for _ in range(10)]
    indices = analysis._balanced_indices(targets, 10)
    prefix_counts = Counter(targets[index] for index in indices[:256])

    assert len(prefix_counts) == 100
    assert max(prefix_counts.values()) - min(prefix_counts.values()) <= 1


def test_dataset_manifest_changes_with_validation_file_metadata(tmp_path: Path) -> None:
    split_root = tmp_path / "val"
    image = split_root / "class-a" / "image.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"first")
    samples = [(str(image), 0)]

    first = analysis._dataset_manifest(split_root, {"class-a": 0}, samples)
    image.write_bytes(b"second-version")
    second = analysis._dataset_manifest(split_root, {"class-a": 0}, samples)

    assert first != second


def test_descriptor_cache_rejects_a_mismatched_protocol_key(tmp_path: Path) -> None:
    cache = tmp_path / "descriptor.pt"
    torch.save(
        {
            "descriptor": torch.zeros(1, 4),
            "labels": torch.zeros(1, dtype=torch.long),
            "checkpoint_sha256": "checkpoint",
            "split": "old-protocol",
        },
        cache,
    )

    with pytest.raises(RuntimeError, match="stale descriptor cache"):
        diagnostics.descriptor_cache(
            torch.nn.Identity(),
            [],
            torch.device("cpu"),
            cache,
            "checkpoint",
            "new-protocol",
            None,
        )


def test_rank2_weight_metrics_detect_independent_components() -> None:
    reader = FactorizedComplexConv2dReader(2, 2, rank=2, kernel_size=3)
    with torch.no_grad():
        for parameter in (
            reader.point_weight_real,
            reader.point_weight_imag,
            reader.spatial_weight_real,
            reader.spatial_weight_imag,
        ):
            parameter.zero_()
        reader.point_weight_real[:, 0, 0] = 1
        reader.point_weight_real[:, 1, 1] = 1
        reader.spatial_weight_real[:, :, 1, 1] = 1

    metrics = analysis._rank2_weight_metrics(reader)
    assert metrics["component_cosine_median"] == pytest.approx(0.0)
    assert metrics["minor_component_energy_share_median"] == pytest.approx(0.5)
    assert metrics["independent_second_fraction_median"] == pytest.approx(0.5)


def test_duplicate_clusters_include_anti_correlated_poles() -> None:
    correlation = torch.tensor(
        [
            [0.0, -0.99, 0.0],
            [-0.99, 0.0, 0.2],
            [0.0, 0.2, 0.0],
        ]
    )

    assert diagnostics.union_find_clusters(correlation, 0.95) == [[0, 1]]


def test_selection_map_deduplicates_identical_all_group() -> None:
    reader = FactorizedComplexConv2dReader(2, 2, rank=2, kernel_size=3)
    readers = {
        "main.stage1": reader,
        "main.stage2": reader,
    }
    selections = analysis._selection_map(
        readers,
        individual=False,
    )
    assert selections == {"main.all": ("main.stage1", "main.stage2")}
