from __future__ import annotations

# pyright: reportExplicitAny=false
import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from scripts import validate_r2k3_checkpoint_study as validator


def _file_record(root: Path, relative: str, content: bytes) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": relative,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _valid_study(root: Path) -> tuple[dict[str, Any], str]:
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable epoch 100 checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    metric_bytes = json.dumps({"metrics": {"finite": 1.25}}, sort_keys=True).encode()
    metric_record = _file_record(root, "artifacts/metrics.json", metric_bytes)
    metric_record.update({"artifact_id": "A_metrics", "media_type": "application/json"})

    figures: list[dict[str, Any]] = []
    for index, figure_id in enumerate(validator.FIGURE_IDS, start=1):
        figure_record = _file_record(root, f"figures/figure-{index}.png", f"png-{index}".encode())
        sidecar_content = json.dumps(
            {"figure_id": figure_id, "panels": {"A": {"metric": float(index)}}},
            sort_keys=True,
        ).encode()
        sidecar_record = _file_record(root, f"figures/figure-{index}.json", sidecar_content)
        figures.append(
            {
                "figure_id": figure_id,
                **figure_record,
                "sidecar_path": sidecar_record["path"],
                "sidecar_sha256": sidecar_record["sha256"],
                "sidecar_bytes": sidecar_record["bytes"],
            }
        )

    report = (
        "# 체크포인트 연구 보고서\n\n"
        "역사적 최고 기록은 epoch 99의 88.34%이다. "
        "이 값과 별개로 지정한 epoch 100 체크포인트 기록은 88.30%이다.\n"
    ).encode()
    report_record = _file_record(root, "report_ko.md", report)

    def family(family_id: str) -> dict[str, Any]:
        condition_ids = list(validator.EXPECTED_CONDITION_IDS[family_id])
        conditions = []
        for condition_id in condition_ids:
            if condition_id in validator.ALLOWED_UNSUPPORTED.get(family_id, frozenset()):
                conditions.append(
                    {
                        "condition_id": condition_id,
                        "status": "unsupported",
                        "reason": "production interface cannot express this intervention",
                        "metrics": {},
                    }
                )
                continue
            condition: dict[str, Any] = {
                "condition_id": condition_id,
                "status": "complete",
                "metrics": {"score": 1.0},
            }
            if family_id == "F04_pole_targeted" and condition_id in validator.F04_TARGETS:
                condition["matched_control"] = {
                    "repeats": 20,
                    "condition_ids": [
                        f"{condition_id}_random_{seed:02d}" for seed in range(20)
                    ],
                }
            conditions.append(condition)
        result: dict[str, Any] = {
            "family_id": family_id,
            "status": "complete",
            "required_condition_ids": sorted(condition_ids),
            "observed_condition_ids": sorted(condition_ids),
            "conditions": conditions,
        }
        if family_id == "F04_pole_targeted":
            result["matched_control_contrasts"] = [
                {
                    "target_condition_id": target_id,
                    "accuracy_delta_pp_vs_random_mean": 0.0,
                    "accuracy_delta_pp_vs_random_mean_ci95": [-0.1, 0.1],
                    "empirical_p_more_harmful": 1.0,
                    "empirical_q_more_harmful_bh": 1.0,
                }
                for target_id in validator.F04_TARGETS
            ]
        if family_id == "F11_robustness_invariance":
            result["intervention_cross"] = {
                "status": "complete",
                "condition_ids": [
                    f"half_life_cap_2_x_{condition_id}"
                    for condition_id in validator.F11_CROSS_BASES
                ],
            }
        return result

    families: dict[str, Any] = {
        family_id: family(family_id) for family_id in validator.FAMILY_IDS
    }
    last_family_id = validator.FAMILY_IDS[-1]
    family_path = root / "families/F12_calibration_error_statistics.json"
    family_path.parent.mkdir(parents=True, exist_ok=True)
    family_path.write_text(json.dumps(families[last_family_id]), encoding="utf-8")
    families[last_family_id] = "families/F12_calibration_error_statistics.json"

    manifest: dict[str, Any] = {
        "schema": validator.SCHEMA,
        "study": {
            "variant": validator.VARIANT,
            "status": "complete_pending_validation",
        },
        "provenance": {
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "bytes": checkpoint.stat().st_size,
                "payload": {"variant": validator.VARIANT, "seed": 501, "epoch": 100},
                "parent_inventory_sha256_before": "a" * 64,
                "parent_inventory_sha256_after": "a" * 64,
            },
            "training_record": {
                "best_accuracy": 0.8834,
                "best_epoch": 99,
                "final_accuracy": 0.8830,
                "final_epoch": 100,
            },
            "no_training_guard": {
                "optimizer_step_calls": 0,
                "parameter_update_calls": 0,
                "linear_probe_runs": 0,
                "checkpoint_sha256_before": checkpoint_sha,
                "checkpoint_sha256_after": checkpoint_sha,
            },
        },
        "baseline": {"metrics": {"samples": 5000, "correct": 4415, "accuracy": 0.883}},
        "families": families,
        "artifacts": [metric_record],
        "figures": figures,
        "report": report_record,
        "validation": {},
    }
    _write_manifest(root, manifest)
    return manifest, checkpoint_sha


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / validator.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _messages(summary: dict[str, Any]) -> str:
    return "\n".join(summary["errors"])


def test_minimal_complete_study_passes(tmp_path: Path) -> None:
    _manifest, checkpoint_sha = _valid_study(tmp_path)

    summary = validator.validate_study(tmp_path, checkpoint_sha)

    assert summary["ok"] is True
    assert summary["errors"] == []
    assert summary["checked_families"] == 12
    assert summary["checked_figures"] == 6
    assert summary["checked_artifacts"] == 1


def test_finalize_records_validation_and_complete_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _manifest, checkpoint_sha = _valid_study(tmp_path)

    return_code = validator.main(
        [
            "--root",
            str(tmp_path),
            "--checkpoint-sha256",
            checkpoint_sha,
            "--finalize",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    finalized = json.loads((tmp_path / validator.MANIFEST_NAME).read_text())

    assert return_code == 0
    assert summary["ok"] is True
    assert finalized["study"]["status"] == "complete_validated"
    assert finalized["validation"]["ok"] is True


def test_cli_prints_json_and_returns_nonzero_for_invalid_study(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, checkpoint_sha = _valid_study(tmp_path)
    manifest["provenance"]["no_training_guard"]["optimizer_step_calls"] = 1
    _write_manifest(tmp_path, manifest)

    return_code = validator.main(["--root", str(tmp_path), "--checkpoint-sha256", checkpoint_sha])
    output = json.loads(capsys.readouterr().out)

    assert return_code == 1
    assert output["ok"] is False
    assert "optimizer_step_calls" in _messages(output)


def test_rejects_baseline_arithmetic_and_missing_family_condition(tmp_path: Path) -> None:
    manifest, checkpoint_sha = _valid_study(tmp_path)
    manifest["baseline"]["metrics"]["accuracy"] = 0.99
    manifest["families"]["F03_recurrence_causality"]["observed_condition_ids"] = [
        "other"
    ]
    _write_manifest(tmp_path, manifest)

    summary = validator.validate_study(tmp_path, checkpoint_sha)

    assert summary["ok"] is False
    assert "correct / samples" in _messages(summary)
    assert "required_condition_ids must exactly equal" in _messages(summary)


def test_rejects_self_declared_dummy_family_contract(tmp_path: Path) -> None:
    manifest, checkpoint_sha = _valid_study(tmp_path)
    manifest["families"]["F08_terminal_q"] = {
        "family_id": "F08_terminal_q",
        "status": "complete",
        "required_condition_ids": ["dummy"],
        "observed_condition_ids": ["dummy"],
        "conditions": [
            {"condition_id": "dummy", "status": "complete", "metrics": {"score": 1.0}}
        ],
    }
    _write_manifest(tmp_path, manifest)

    summary = validator.validate_study(tmp_path, checkpoint_sha)

    assert summary["ok"] is False
    assert "independent protocol contract" in _messages(summary)


def test_rejects_non_finite_json_metrics(tmp_path: Path) -> None:
    manifest, checkpoint_sha = _valid_study(tmp_path)
    manifest["families"]["F08_terminal_q"]["conditions"][0]["metrics"]["score"] = float(
        "nan"
    )
    _write_manifest(tmp_path, manifest)

    summary = validator.validate_study(tmp_path, checkpoint_sha)

    assert summary["ok"] is False
    assert "non-finite JSON number" in _messages(summary)


def test_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    manifest, checkpoint_sha = _valid_study(tmp_path)
    artifact_path = tmp_path / manifest["artifacts"][0]["path"]
    artifact_path.write_bytes(b"tampered")

    summary = validator.validate_study(tmp_path, checkpoint_sha)

    assert summary["ok"] is False
    assert "sha256 mismatch" in _messages(summary)


def test_rejects_artifact_symlink(tmp_path: Path) -> None:
    manifest, checkpoint_sha = _valid_study(tmp_path)
    artifact_path = tmp_path / manifest["artifacts"][0]["path"]
    target = tmp_path / "unregistered-target.json"
    target.write_bytes(artifact_path.read_bytes())
    artifact_path.unlink()
    artifact_path.symlink_to(target)
    _write_manifest(tmp_path, manifest)

    summary = validator.validate_study(tmp_path, checkpoint_sha)

    assert summary["ok"] is False
    assert "contains a symlink" in _messages(summary)


def test_rejects_checkpoint_after_hash_even_when_other_hashes_match(tmp_path: Path) -> None:
    manifest, checkpoint_sha = _valid_study(tmp_path)
    manifest["provenance"]["no_training_guard"]["checkpoint_sha256_after"] = "b" * 64
    _write_manifest(tmp_path, manifest)

    summary = validator.validate_study(tmp_path, checkpoint_sha)

    assert summary["ok"] is False
    assert "SHA256 after" in _messages(summary)


def test_rejects_report_that_conflates_historical_and_epoch100_records(tmp_path: Path) -> None:
    manifest, checkpoint_sha = _valid_study(tmp_path)
    report_path = tmp_path / manifest["report"]["path"]
    report_path.write_text("한국어 보고서: 88.34%와 88.30%를 참고한다.\n", encoding="utf-8")
    report_bytes = report_path.read_bytes()
    manifest["report"]["sha256"] = hashlib.sha256(report_bytes).hexdigest()
    manifest["report"]["bytes"] = len(report_bytes)
    _write_manifest(tmp_path, manifest)

    summary = validator.validate_study(tmp_path, checkpoint_sha)

    assert summary["ok"] is False
    messages = _messages(summary)
    assert "historical/best" in messages
    assert "epoch-100 checkpoint" in messages
