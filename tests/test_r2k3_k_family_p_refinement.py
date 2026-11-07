from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from h200.k_family_p_refinement import validate_prepared
from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k_family_p_refinement_imagenet100 as runner
from scripts import run_a2d_r2k3_k_family_wave_a_imagenet100 as family
from scripts import run_h200_imagenet100_k_family_p_refinement as worker
from scripts import run_h200_k_family_p_refinement_queue as queue

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

PARAMETER_COUNTS = {
    runner.M_K48_P80: 857_124,
    runner.L_K64_P2: 1_290_308,
    runner.L_K64_P23: 1_339_652,
    runner.S_K32_P3: 430_404,
    runner.XL_K96_P2_LITE: 2_814_116,
}


def test_refinement_order_matches_requested_priority() -> None:
    assert runner.VARIANTS == (
        runner.M_K48_P80,
        runner.L_K64_P2,
        runner.L_K64_P23,
        runner.S_K32_P3,
        runner.XL_K96_P2_LITE,
    )
    assert runner.JOBS_BY_GPU == {0: runner.VARIANTS}
    assert queue.VARIANTS == runner.VARIANTS
    assert worker.PARAMETER_COUNTS == PARAMETER_COUNTS


def test_all_refinement_models_match_exact_counts_and_identity_carries() -> None:
    for variant in runner.VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        runner._assert_model(model, variant)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == PARAMETER_COUNTS[variant]
        )
        width = runner.POLICIES[variant][0]
        assert runner.SPECS[variant].excitation_modes == (width,) * 4
        assert runner.SPECS[variant].depth == (2, 2, 6, 2)
        for name in family.STAGE_NAMES[:3]:
            transition = getattr(model, name).augmented
            assert type(transition) is GatedPoleExcitationS2DTransition
            assert transition.carry_projection is None


def test_refinement_campaign_remains_prepared_and_stopped() -> None:
    payload = validate_prepared.validate_prepared()
    assert payload["state"] == "prepared_only"
    assert payload["control"] == "stop"
    assert payload["runtime_manifest"] == "absent"
    assert tuple(payload["variants"]) == runner.VARIANTS


def test_refinement_queue_continues_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "schema": queue.RUNTIME_SCHEMA,
                "campaign_id": "test-refinement",
                "parameter_counts": PARAMETER_COUNTS,
                "training": {
                    "variants": list(runner.VARIANTS),
                    "seed": 501,
                    "epochs": 100,
                    "batch_size": 128,
                    "precision": "bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "run"
    completed: set[str] = set()
    launched: list[str] = []

    def fake_complete(_root: Path, variant: str, _seed: int, _epochs: int) -> bool:
        return variant in completed

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        variant = command[command.index("--variants") + 1]
        launched.append(variant)
        if variant == runner.L_K64_P2:
            return SimpleNamespace(returncode=17)
        completed.add(variant)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv(queue.RUNTIME_ENV_VAR, str(runtime_path))
    monkeypatch.setenv("H200_EXPECTED_COMMIT", "a" * 40)
    monkeypatch.setenv("H200_CONTROL_STOP_MARKER", str(tmp_path / "stop.json"))
    monkeypatch.setattr(queue, "_complete_result", fake_complete)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_h200_k_family_p_refinement_queue.py",
            "--root",
            str(root),
            "--data-root",
            str(tmp_path / "data"),
            "--workers",
            "8",
        ],
    )
    assert queue.main() == 1
    assert tuple(launched) == runner.VARIANTS
    status = json.loads(
        (root / "k-family-p-refinement-queue.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "COMPLETE_WITH_FAILURES"
    assert status["jobs"][runner.L_K64_P2]["status"] == "FAILED"
