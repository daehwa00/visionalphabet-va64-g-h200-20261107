from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.finalize_va_retrain import finalize
from scripts.queue_va_retrain import aggregate


def test_final_report_requires_all_seeds_and_preserves_confounds(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    template = root / "docs/rebuttal-2026-11-06/response-draft.en.md"
    (tmp_path / "campaign.json").write_text(json.dumps({"batch_size": 128}))
    (tmp_path / "preflight").mkdir()
    for v in "BPA":
        (tmp_path / "preflight" / f"{v}-b128.json").write_text(
            json.dumps(
                {
                    "train_images_per_second": 500.0,
                    "inference": {"1": {"median_ms": 3.0}, "16": {"images_per_second": 2000.0}},
                }
            )
        )
    aggregate(tmp_path)
    with pytest.raises(ValueError, match="all nine"):
        finalize(tmp_path, template)
    assert not (tmp_path / "rebuttal/response-draft-with-retraining.en.md").exists()
    for v in "BPA":
        for seed in range(3):
            path = tmp_path / "runs" / f"va_k128_retrain_{v}" / f"seed_{seed}"
            path.mkdir(parents=True)
            (path / "result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "completed_epochs": 100,
                        "training_seconds": 3600.0,
                        "final_validation": {
                            "accuracy": 0.75,
                            "top5_accuracy": 0.9,
                            "cross_entropy": 1.0,
                            "examples": 50000,
                        },
                    }
                )
            )
            (path / "parameter-inventory.json").write_text(
                json.dumps(
                    {
                        "total": {"stored_elements": 5083176, "trainable_elements": 5083176},
                        "head": {"stored_elements": 513000},
                        "phase": {"trainable_elements": 0 if v == "P" else 3712},
                    }
                )
            )
            (path / "cost.json").write_text(
                json.dumps(
                    {"gpu_hours": 1.0, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}
                )
            )
    aggregate(tmp_path)
    finalize(tmp_path, template)
    draft = (tmp_path / "rebuttal/response-draft-with-retraining.en.md").read_text()
    assert "No new training results are included." not in draft
    assert "all nine completed B/P/A retraining runs" in draft
    assert "head-capacity and optimization effects" in draft
    assert "not a true-real baseline" in draft
    assert len((tmp_path / "rebuttal/seed-results-and-cost.csv").read_text().splitlines()) == 10
