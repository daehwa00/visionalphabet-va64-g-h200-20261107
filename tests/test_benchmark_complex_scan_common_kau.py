# pyright: reportExplicitAny=false, reportPrivateUsage=false
from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

import benchmark_complex_scan_common_kau as benchmark
import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path


def _result(role: str, throughput: float, memory: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": benchmark.SCHEMA,
        "role": role,
        "device": "NVIDIA GeForce RTX 4090",
        "device_capability": [8, 9],
        "device_total_memory_bytes": 24_000_000_000,
        "torch_version": "test",
        "batch_size": benchmark.EXPECTED_BATCH_SIZE,
        "precision": "bfloat16 autocast",
        "input_shape": [benchmark.EXPECTED_BATCH_SIZE, 3, 224, 224],
        "compile_mode": "default",
        "fused_adamw": True,
        "parameters": benchmark.EXPECTED_PARAMETERS,
        "config": benchmark._config_payload(),
        "model_fixture_sha256": "fixture",
        "optimizer_group_signature": [{"name": "all"}],
        "scan_pipeline": "associative_product",
        "median_step_throughput_images_per_second": throughput,
        "peak_allocated_bytes": memory,
    }
    if role == "candidate":
        payload["parity"] = {
            "logits_max_abs": 1.0e-3,
            "loss_abs": 1.0e-4,
            "gradient_max_abs": 2.0e-3,
            "gradient_relative_rmse": 1.0e-3,
            "gradient_keys_equal": True,
            "optimizer_parameter_max_abs": 2.0e-3,
            "optimizer_parameter_relative_rmse": 1.0e-3,
            "optimizer_parameter_keys_equal": True,
        }
    return payload


def _arguments(
    tmp_path: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tests_status: str = "pass",
) -> argparse.Namespace:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    return argparse.Namespace(
        baseline=baseline_path,
        candidate=candidate_path,
        output=tmp_path / "evaluation.json",
        tests_status=tests_status,
        minimum_speedup=1.15,
        maximum_memory_ratio=1.0,
        maximum_logits_error=2.0e-2,
        maximum_loss_error=5.0e-3,
        maximum_gradient_relative_rmse=2.0e-2,
        maximum_update_relative_rmse=2.0e-2,
    )


def test_exact_model_configuration_matches_imagenet_zero_init_lrq64() -> None:
    config = benchmark._config()

    assert config.stem_strides == (2, 2)
    assert config.modes == (32, 32, 32)
    assert config.augmented_widths == (64, 64)
    assert config.carry_bases == ("s2d", "s2d")
    assert config.quadratic_rank == 64


def test_fixture_contract_allows_new_default_config_fields(tmp_path: Path) -> None:
    active = benchmark._config_payload()
    legacy = dict(active)
    legacy.pop("dual_fusion_lrq_head", None)
    fixture_path = tmp_path / "fixture.pt"
    torch.save(
        {
            "schema": benchmark.FIXTURE_SCHEMA,
            "batch_size": benchmark.EXPECTED_BATCH_SIZE,
            "config": legacy,
        },
        fixture_path,
    )

    loaded = benchmark._load_fixture(fixture_path)
    assert loaded["config"] == legacy

    legacy["quadratic_rank"] = int(legacy["quadratic_rank"]) + 1
    torch.save(
        {
            "schema": benchmark.FIXTURE_SCHEMA,
            "batch_size": benchmark.EXPECTED_BATCH_SIZE,
            "config": legacy,
        },
        fixture_path,
    )
    with pytest.raises(ValueError, match="fixed batch/configuration"):
        benchmark._load_fixture(fixture_path)


def test_config_extension_preserves_legacy_contract() -> None:
    active = benchmark._config_payload()
    legacy = dict(active)
    legacy.pop("dual_fusion_lrq_head", None)

    assert benchmark._config_extends(legacy, active)

    incompatible = dict(active)
    incompatible["quadratic_rank"] += 1
    assert not benchmark._config_extends(incompatible, active)


def test_evaluator_accepts_contract_boundary(tmp_path: Path) -> None:
    baseline = _result("baseline", throughput=1000.0, memory=10_000)
    candidate = _result("candidate", throughput=1150.0, memory=10_000)
    args = _arguments(tmp_path, baseline, candidate)

    assert benchmark._evaluate(args) == 0
    evaluation = json.loads(args.output.read_text())
    assert evaluation["status"] == "pass"
    assert evaluation["speedup"] == 1.15
    assert evaluation["memory_ratio"] == 1.0


def test_evaluator_rejects_speed_memory_parity_and_test_regressions(tmp_path: Path) -> None:
    baseline = _result("baseline", throughput=1000.0, memory=10_000)
    candidate = _result("candidate", throughput=1100.0, memory=10_001)
    candidate["parity"]["gradient_relative_rmse"] = 2.1e-2
    args = _arguments(tmp_path, baseline, candidate, tests_status="fail")

    assert benchmark._evaluate(args) == 1
    failures = "\n".join(json.loads(args.output.read_text())["failures"])
    assert "speedup" in failures
    assert "memory ratio" in failures
    assert "gradient_relative_rmse" in failures
    assert "regression tests failed" in failures
