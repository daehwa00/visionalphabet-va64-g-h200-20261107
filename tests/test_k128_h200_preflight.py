"""Keep both arms on one measured direct batch size, including after restore."""

import json
from types import SimpleNamespace

import k128_h200_preflight as preflight
import pytest
import run_k128_optimization_campaign as campaign
from k128_h200_preflight import EXECUTION_VERSION, choose_batch, saved_selection


def test_batch_512_requires_a_measured_gain_and_handles_oom() -> None:
    small = {"status": "passed", "images_per_second": 1000.0}
    assert choose_batch({256: small, 512: {"status": "passed", "images_per_second": 1040.0}}) == 512
    assert choose_batch({256: small, 512: {"status": "passed", "images_per_second": 1010.0}}) == 256
    assert choose_batch({256: small, 512: {"status": "oom"}}) == 256
    with pytest.raises(RuntimeError, match="did not complete"):
        choose_batch({256: small, 512: {"status": "failed"}})


def test_restored_selection_is_shared_and_cannot_silently_change(tmp_path) -> None:
    assert saved_selection(tmp_path) is None
    baseline = tmp_path / "baseline/contracts/execution-selection.json"
    improved = tmp_path / "improved/contracts/execution-selection.json"
    selection = {
        "execution_version": EXECUTION_VERSION,
        "selected_batch": 512,
        "effective_batch": 512,
        "gradient_accumulation_steps": 1,
    }
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps(selection))
    assert saved_selection(tmp_path) == selection
    improved.parent.mkdir(parents=True)
    improved.write_text(json.dumps({**selection, "selected_batch": 256, "effective_batch": 256}))
    with pytest.raises(RuntimeError, match="conflicting"):
        saved_selection(tmp_path)
    improved.write_text(json.dumps({**selection, "gradient_accumulation_steps": 2}))
    with pytest.raises(RuntimeError, match="incompatible"):
        saved_selection(tmp_path)


def test_fast_task_requires_expected_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("H200_AGENT_TOKEN", raising=False)
    origin = tmp_path / "origin.json"
    origin.write_text(json.dumps({"baseline": {"completed_epochs": 10}}))
    args = SimpleNamespace(
        root=tmp_path, data_root=tmp_path / "data", variant="baseline", fast=True,
        batch_size=512, workers=8, epochs=200, resume_origin=origin,
    )
    task = campaign.make_task(args, "baseline")
    assert task.batch_size == task.effective_batch_size == 512
    assert task.gradient_accumulation_steps == 1
    assert task.execution.compile_model
    assert task.execution.gpu_mixup
    with pytest.raises(RuntimeError, match="refusing a fresh restart"):
        campaign.run_variant(args)


def test_pipeline_timer_excludes_warmup_and_worker_shutdown(monkeypatch) -> None:
    class Batches:
        index = 0
        closed = False

        def __iter__(self) -> "Batches":
            return self

        def __next__(self) -> int:
            self.index += 1
            return self.index

        def _shutdown_workers(self) -> None:
            self.closed = True

    batches = Batches()
    synchronized = []
    monkeypatch.setattr(
        preflight.torch.cuda, "synchronize", lambda: synchronized.append(batches.index)
    )
    monkeypatch.setattr(preflight.time, "perf_counter", lambda: 123.0)
    limited = preflight._TimedBatches(batches, warmup=2, measured=3)
    assert list(limited) == [1, 2, 3, 4, 5]
    assert synchronized == [2]
    assert limited.started == 123.0
    assert not batches.closed
    limited.close()
    assert batches.closed
