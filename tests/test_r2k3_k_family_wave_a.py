from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportPrivateUsage=false
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from h200.k_family_xl import validate_prepared
from lnet.pac_gated_post_fusion import GatedPoleExcitationS2DTransition
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_k_family_wave_a_h200_imagenet100 as h200_runner
from scripts import run_a2d_r2k3_k_family_wave_a_imagenet100 as runner
from scripts import run_h200_imagenet100_k_family_xl as h200_worker
from scripts import run_h200_k_family_xl_queue as h200_queue
from scripts import smoke_h200_k_family_xl as h200_smoke

if TYPE_CHECKING:
    from pathlib import Path


PARAMETER_COUNTS = {
    "XL-K96-U1": 2_313_892,
    "XL-K96-U125": 2_791_524,
    "XL-K96-Shaped": 2_996_740,
    "XL-K96-Rich": 3_200_068,
}


def test_wave_a_partitions_every_new_candidate_once() -> None:
    queued = tuple(variant for lane in runner.JOBS_BY_GPU.values() for variant in lane)
    assert len(runner.VARIANTS) == 11
    assert len(queued) == len(set(queued)) == len(runner.VARIANTS)
    assert set(queued) == set(runner.VARIANTS)
    assert set(runner.SMOKE_VARIANTS).issubset(runner.VARIANTS)
    assert len(runner.SMOKE_VARIANTS) == 4
    assert len(h200_runner.VARIANTS) == 4
    assert set(runner.VARIANTS).isdisjoint(h200_runner.VARIANTS)
    assert set(runner.VARIANTS) | set(h200_runner.VARIANTS) == set(runner.POLICIES)


def test_h200_queue_contains_only_xl_candidates() -> None:
    assert h200_runner.VARIANTS == runner.XL_VARIANTS
    assert all(variant.startswith("XL-K96-") for variant in h200_runner.VARIANTS)
    assert h200_runner.JOBS_BY_GPU == {0: h200_runner.VARIANTS}
    assert h200_smoke.VARIANT == "XL-K96-Rich"
    assert PARAMETER_COUNTS["XL-K96-Rich"] == h200_smoke.EXPECTED_PARAMETERS


def test_wave_a_specs_are_uniform_k_d2262() -> None:
    for variant, spec in runner.SPECS.items():
        width = runner.POLICIES[variant][0]
        assert spec.excitation_modes == (width,) * 4
        assert spec.depth == (2, 2, 6, 2)
        assert all(pole % 16 == 0 for pole in spec.pole_modes)


def test_representatives_build_with_identity_carries() -> None:
    for variant in runner.SMOKE_VARIANTS:
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        runner._assert_model(model, variant)
        assert sum(parameter.numel() for parameter in model.parameters()) > 0
        for name in runner.STAGE_NAMES[:3]:
            transition = getattr(model, name).augmented
            assert type(transition) is GatedPoleExcitationS2DTransition
            assert transition.carry_projection is None


def test_all_xl_models_match_exact_parameter_and_identity_carry_contracts() -> None:
    for variant in h200_runner.VARIANTS:
        torch.manual_seed(501)
        model = h200_runner._build(variant, runtime.model_config())
        runner._assert_model(model, variant)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == PARAMETER_COUNTS[variant]
        )
        assert runner.SPECS[variant].excitation_modes == (96, 96, 96, 96)
        assert runner.SPECS[variant].depth == (2, 2, 6, 2)
        for name in runner.STAGE_NAMES[:3]:
            transition = getattr(model, name).augmented
            assert type(transition) is GatedPoleExcitationS2DTransition
            assert transition.carry_projection is None


def test_h200_xl_campaign_remains_prepared_and_fail_closed() -> None:
    payload = validate_prepared.validate_prepared()
    assert payload == {
        "campaign_id": "h200-imagenet100-k-family-xl-s501",
        "state": "prepared_only",
        "variants": list(h200_runner.VARIANTS),
        "control": "stop",
        "runtime_manifest": "absent",
    }
    with pytest.raises(RuntimeError, match="H200_K_FAMILY_XL_WANDB_RUNTIME"):
        h200_worker._runtime()


def test_h200_xl_queue_is_ordered_restart_safe_and_failure_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "schema": h200_queue.RUNTIME_SCHEMA,
                "campaign_id": "test-xl-campaign",
                "parameter_counts": PARAMETER_COUNTS,
                "training": {
                    "variants": list(h200_runner.VARIANTS),
                    "seed": 501,
                    "epochs": 100,
                    "batch_size": 128,
                    "precision": "bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )
    stop_marker = tmp_path / "stop.json"
    root = tmp_path / "run"
    launched: list[str] = []
    completed: set[str] = set()

    def fake_complete(_root: Path, variant: str, _seed: int, _epochs: int) -> bool:
        return variant in completed

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        variant = command[command.index("--variants") + 1]
        launched.append(variant)
        if variant == "XL-K96-U125":
            return SimpleNamespace(returncode=17)
        completed.add(variant)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("H200_K_FAMILY_XL_WANDB_RUNTIME", str(runtime_path))
    monkeypatch.setenv("H200_EXPECTED_COMMIT", "a" * 40)
    monkeypatch.setenv("H200_CONTROL_STOP_MARKER", str(stop_marker))
    monkeypatch.setattr(h200_queue, "_complete_result", fake_complete)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_h200_k_family_xl_queue.py",
            "--root",
            str(root),
            "--data-root",
            str(tmp_path / "data"),
            "--workers",
            "8",
        ],
    )
    assert h200_queue.main() == 1
    assert tuple(launched) == h200_runner.VARIANTS
    status = json.loads((root / "k-family-xl-queue.json").read_text(encoding="utf-8"))
    assert status["status"] == "COMPLETE_WITH_FAILURES"
    assert status["jobs"]["XL-K96-U125"]["status"] == "FAILED"
    assert all(
        status["jobs"][variant]["status"] == "COMPLETED"
        for variant in h200_runner.VARIANTS
        if variant != "XL-K96-U125"
    )


def test_h200_xl_queue_quarantines_invalid_results_and_skips_exact_completions(
    tmp_path: Path,
) -> None:
    variant = "XL-K96-U1"
    result = tmp_path / "results" / f"{variant}__seed501.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"variant":"XL-K96-U1"}\n', encoding="utf-8")
    assert not h200_queue._complete_result(tmp_path, variant, 501, 100)
    h200_queue._quarantine_incomplete_result(tmp_path, variant, 501, 100)
    assert not result.exists()
    assert len(list(result.parent.glob(f"{result.stem}.invalid-*.json"))) == 1
    result.write_text(
        json.dumps(
            {
                "variant": variant,
                "seed": 501,
                "parameters": PARAMETER_COUNTS[variant],
                "history": [{"epoch": epoch} for epoch in range(1, 101)],
            }
        ),
        encoding="utf-8",
    )
    assert h200_queue._complete_result(tmp_path, variant, 501, 100)


def test_h200_xl_wandb_proxy_adds_throughput_peak_vram_and_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}
            self.url = "https://wandb.invalid/run"
            self.logged: list[tuple[dict[str, object], int]] = []
            self.finished = False

        def log(self, metrics: dict[str, object], *, step: int) -> None:
            self.logged.append((metrics, step))

        def finish(self) -> None:
            self.finished = True

    fake = FakeRun()
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 12_345)
    proxy = h200_worker._WandbTelemetryProxy(
        fake,
        training_examples=130_000,
        resume_state={"checkpoint_found": True, "resumed_from_epoch": 7},
    )
    proxy.log({"time/training_seconds": 10.0, "train/loss": 1.0}, step=8)
    assert fake.logged[0][0]["throughput/images_per_second"] == 13_000.0
    assert fake.logged[0][0]["memory/peak_vram_bytes"] == 12_345.0
    assert proxy.summary["checkpoint_found_at_start"] is True
    assert proxy.summary["resumed_from_epoch"] == 7
    proxy.finish()
    assert proxy.summary["peak_vram_bytes"] == 12_345.0
    assert fake.finished is True
