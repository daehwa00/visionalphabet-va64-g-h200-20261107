"""Deterministic evidence for subprocess microbatch decisions and receipt replay."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import effective_batch_search as search
import pytest
import torch
from automatic_training_layout import TensorCapture, digest, read_json
from imagenet100_checkpoint_runtime import atomic_json

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("samples", "candidates", "baselines"),
    [
        (1024, (1024, 512, 256, 128, 64, 32), (128, 64, 32)),
        (256, (256, 128, 64, 32), (128, 64, 32)),
        (128, (128, 64, 32), (128, 64, 32)),
        (96, (96, 64, 32), (96, 64, 32)),
        (17, (17, 16, 8, 4), (17, 16, 8, 4)),
        (1, (1,), (1,)),
    ],
)
def test_candidate_sizes(
    samples: int, candidates: tuple[int, ...], baselines: tuple[int, ...]
) -> None:
    assert search.candidate_microbatches(samples) == candidates
    assert search.baseline_microbatches(samples) == baselines


def captures(value: float) -> TensorCapture:
    return {group: {"value": torch.tensor([value])} for group in ("outputs", "grads", "updates")}


def test_search_uses_all_sizes_and_replays_alternating_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = {"effective_samples": 1024}
    key = digest(workload)
    seen = []

    def arm(
        _work: Path, _label: str, microbatch: int
    ) -> tuple[dict[str, Any], TensorCapture | None]:
        seen.append(microbatch)
        if microbatch in {1024, 128}:
            return {"status": "oom"}, None
        ms = {512: 1.0, 256: 2.0, 64: 4.0, 32: 8.0}[microbatch]
        return {
            "status": "ok",
            "ms": ms,
            "images_per_second": 1024000 / ms,
            "peak_mib": 20,
        }, captures(2.0 if microbatch == 512 else 1.0)

    monkeypatch.setattr(search, "_run_arm", arm)
    receipt = search.measure({}, key, workload, tmp_path)
    assert seen == [128, 64, 1024, 512, 256, 32, 64, 256, 256, 64, 64, 256]
    assert receipt["baseline"] == 64
    assert receipt["microbatch"] == 256
    assert receipt["status"] == "confirmed"
    assert receipt["assessed"]["512"]["parity"]["passed"] is False
    assert receipt["assessed"]["1024"]["status"] == "oom"
    path = tmp_path / "receipt.json"
    atomic_json(path, receipt)
    assert search.valid_receipt(read_json(path), key)

    invalid = deepcopy(receipt)
    invalid["confirmations"][0]["pairs"][0]["order"].reverse()
    assert not search.valid_receipt(invalid, key)
    invalid = deepcopy(receipt)
    invalid["confirmations"][0]["pairs"][2]["candidate"]["ms"] = 5.0
    assert not search.valid_receipt(invalid, key)
    invalid = deepcopy(receipt)
    del invalid["assessed"]["32"]
    assert not search.valid_receipt(invalid, key)


def test_confirmation_rejects_fast_candidate_with_numerical_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = {"effective_samples": 256}
    seen = []

    def arm(_work: Path, _label: str, microbatch: int) -> tuple[dict[str, Any], TensorCapture]:
        seen.append(microbatch)
        ms = {256: 1.0, 128: 2.0, 64: 3.0, 32: 4.0}[microbatch]
        value = 2.0 if microbatch == 256 and seen.count(256) > 1 else 1.0
        return {
            "status": "ok",
            "ms": ms,
            "images_per_second": 256000 / ms,
            "peak_mib": 20,
        }, captures(value)

    monkeypatch.setattr(search, "_run_arm", arm)
    key = digest(workload)
    receipt = search.measure({}, key, workload, tmp_path)
    assert receipt["microbatch"] == 128
    assert receipt["status"] == "baseline"
    assert not receipt["confirmations"][0]["pairs"][0]["candidate"]["parity"]["passed"]
    assert search.valid_receipt(receipt, key)


def test_all_baselines_oom_yields_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(search, "_run_arm", lambda *_args: ({"status": "oom"}, None))
    workload = {"effective_samples": 128}
    with pytest.raises(RuntimeError, match="No feasible baseline"):
        search.measure({}, digest(workload), workload, tmp_path)


@pytest.mark.parametrize(
    ("error", "status"), [(torch.cuda.OutOfMemoryError, "oom"), (FloatingPointError, "nonfinite")]
)
def test_disposable_worker_reports_typed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: type[Exception],
    status: str,
) -> None:
    def fail(_work: Path, _microbatch: int) -> None:
        raise error("benchmark failed")

    monkeypatch.setattr(search, "_benchmark", fail)
    job = tmp_path / "trial.job"
    atomic_json(job, {"microbatch": 128})
    search.worker(job)
    assert read_json(job.with_suffix(".result.json")) == {"status": status}
    assert not job.with_suffix(".pt").exists()


def test_typed_oom_recognizes_wrapped_cuda_error() -> None:
    wrapped = RuntimeError("compile failed")
    wrapped.__cause__ = torch.cuda.OutOfMemoryError()
    assert search._is_cuda_oom(wrapped)
    assert not search._is_cuda_oom(RuntimeError("CUDA out of memory"))
