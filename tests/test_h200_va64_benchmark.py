"""CPU-only checks for isolated H200 benchmark accounting and selection."""

import json
import signal
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h200.va64 import benchmark_g
from h200.va64.benchmark_g import BoundedBatches, select_candidate, state_digest


def result(batch: int, speed: float) -> dict:
    return {
        "status": "passed",
        "physical_batch": batch,
        "gradient_accumulation_steps": 256 // batch,
        "measure_updates": 100,
        "initial_state": {"model_sha256": "model", "optimizer_sha256": "adam", "rng_sha256": "rng"},
        "measurement": {"optimizer_updates": 100, "examples": 25600, "images_per_second": speed},
    }


def test_select_faster_batch_with_same_global_batch() -> None:
    selected = select_candidate([result(128, 100), result(256, 150)])
    assert selected["selected_batch_size"] == 256
    assert selected["gradient_accumulation_steps"] == 1
    assert selected["global_batch_size"] == 256
    assert not selected["training_checkpoint_modified"]
    assert not selected["training_rng_modified"]


def test_slow_256_retains_128_and_oom_falls_back() -> None:
    assert select_candidate([result(128, 150), result(256, 100)])["selected_batch_size"] == 128
    assert (
        select_candidate([result(128, 150), {"status": "oom", "physical_batch": 256}])[
            "selected_batch_size"
        ]
        == 128
    )
    with pytest.raises(RuntimeError, match="neither"):
        select_candidate([{"status": "oom", "physical_batch": 128}])


def test_reject_mismatched_initial_rng_and_partial_measurement() -> None:
    changed = result(256, 150)
    changed["initial_state"]["rng_sha256"] = "different"
    with pytest.raises(ValueError, match="identical"):
        select_candidate([result(128, 100), changed])
    partial = result(256, 150)
    partial["measurement"]["optimizer_updates"] = 99
    with pytest.raises(ValueError, match="invalid"):
        select_candidate([result(128, 100), partial])


def test_warmup_and_measurement_share_one_iterator_and_stop(tmp_path: Path) -> None:
    batches = iter(range(6))
    assert list(BoundedBatches(batches, 2, None)) == [0, 1]
    assert list(BoundedBatches(batches, 3, None)) == [2, 3, 4]
    stop = tmp_path / "STOP"
    stop.touch()
    with pytest.raises(InterruptedError, match="stop"):
        list(BoundedBatches(batches, 1, stop))


def test_state_digest_ignores_tensor_storage_identity_and_tracks_values() -> None:
    state = {"scalar": torch.tensor(1.0), "complex": torch.tensor([1 + 2j]), "state": [1, (2, 3)]}
    assert state_digest(state) == state_digest(deepcopy(state))
    changed = deepcopy(state)
    changed["complex"][0] = 2 + 2j
    assert state_digest(state) != state_digest(changed)


def test_driver_isolates_candidates_and_keeps_checkpoint_and_parent_rng(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"original checkpoint")
    args = Namespace(
        data_root=tmp_path,
        output=tmp_path,
        checkpoint=checkpoint,
        stop_file=tmp_path / "STOP",
        workers=8,
        warmup_updates=30,
        measure_updates=100,
    )
    calls = []

    def launch(command, *, env, check):
        calls.append((command, env))
        candidate = Path(command[command.index("--output") + 1])
        batch = int(command[command.index("--physical-batch") + 1])
        candidate.mkdir()
        (candidate / "result.json").write_text(json.dumps(result(batch, batch)))
        assert command[command.index("--checkpoint") + 1] == str(checkpoint)
        assert not check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark_g.subprocess, "run", launch)
    before = torch.get_rng_state().clone()
    benchmark_g.compare_candidates(args)
    assert torch.equal(torch.get_rng_state(), before)
    assert checkpoint.read_bytes() == b"original checkpoint"
    assert len(calls) == 2
    assert calls[0][1]["TORCHINDUCTOR_CACHE_DIR"] != calls[1][1]["TORCHINDUCTOR_CACHE_DIR"]
    assert json.loads((tmp_path / "selection.json").read_text())["selected_batch_size"] == 256


def test_supervisor_signals_touch_shared_stop_without_exiting(tmp_path, monkeypatch) -> None:
    handlers = {}
    monkeypatch.setattr(
        benchmark_g.signal, "signal", lambda signum, handler: handlers.update({signum: handler})
    )
    stop = tmp_path / "STOP"
    benchmark_g.install_stop_handlers(stop)
    assert set(handlers) == {signal.SIGTERM, signal.SIGINT}
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert stop.exists()
    handlers[signal.SIGINT](signal.SIGINT, None)
    assert stop.exists()
