from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from torch import nn

from scripts import queue_va64_retrain as queue
from scripts import run_va64_retrain_imagenet1k as runner
from scripts import transition_va128_to_va64 as switch

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def threads() -> Generator[None]:
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_va64_baseline_regression_and_pairing() -> None:
    torch.manual_seed(0)
    original = runner.PrimaryLogitsAdapter(
        runner.anchor._build("L-K64-U125", runner.ancestry.model_config(output_dim=1000))
    ).eval()
    torch.manual_seed(0)
    baseline = runner.build_model("B", prepared=False).eval()
    torch.manual_seed(0)
    phase = runner.build_model("P", prepared=False).eval()
    torch.manual_seed(0)
    averaged = runner.build_model("A", prepared=False).eval()
    assert runner.parameter_inventory(baseline)["total"]["stored_elements"] == 1_562_760
    assert runner.parameter_inventory(phase)["total"]["trainable_elements"] == 1_560_840
    assert runner.parameter_inventory(averaged)["total"]["stored_elements"] == 1_322_760
    runner.assert_zero_phases(phase)
    assert averaged.model.classifier.q4_dim == 80
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
        if ".classifier." not in name:
            torch.testing.assert_close(value, averaged.state_dict()[name], rtol=0, atol=0)
        if not name.endswith((".phase_x", ".phase_y")):
            torch.testing.assert_close(value, phase.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(2, 3, 32, 32)
    target = torch.tensor([1, 2])
    left = nn.functional.cross_entropy(original(x), target)
    right = nn.functional.cross_entropy(baseline(x), target)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.backward()
    right.backward()
    for a, b in zip(original.parameters(), baseline.parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    for model in (phase, averaged):
        out = model(x)
        assert out.shape == (2, 1000)
        assert torch.isfinite(out).all()
        out.square().mean().backward()
    runner.assert_zero_phases(phase)


def test_va64_training_modes_and_cooperative_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps({"root": str(tmp_path), "counts": {"train": 1281167, "val": 50000}})
    )
    args = argparse.Namespace(
        data_root=tmp_path,
        data_manifest=manifest,
        output_root=tmp_path / "runs",
        variant="B",
        seed=0,
        batch_size=128,
        workers=0,
        resume=False,
        stop_file=tmp_path / "STOP",
    )
    tasks = []
    for v in "BPA":
        args.variant = v
        task = runner.make_task(args)
        tasks.append(task)
        assert task.seed == 0
        assert task.epochs == 100
        assert task.effective_batch_size == 256
        assert task.gradient_accumulation_steps == 2
        assert runner.DEFINITIONS[v]["phase_mode"] == ("zero" if v == "P" else "learned")
        assert runner.DEFINITIONS[v]["readout_mode"] == (
            "mean_energy" if v == "A" else "directional"
        )
    recipes = [runner.runtime._contract(t, torch.device("cpu"))["recipe"] for t in tasks]
    assert recipes[0] == recipes[1] == recipes[2]
    assert len({t.model_key for t in tasks}) == 3
    assert len({t.source_sha256["ablation_definition"] for t in tasks}) == 3
    captured = {}

    def fake_run(task: runner.runtime.BaselineTask, **kwargs: object) -> dict:
        assert not kwargs["stop_requested"]()
        args.stop_file.touch()
        assert kwargs["stop_requested"]()
        runner.runtime._atomic_json(
            task.output_dir / "latest-epoch.json", {"training_seconds": 12.0}
        )
        captured["task"] = task
        return {"status": "paused", "completed_epochs": 1}

    monkeypatch.setattr(runner.runtime, "run_task", fake_run)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    assert runner.train(args)["status"] == "paused"
    assert (
        json.loads((captured["task"].output_dir / "cost.json").read_text())["training_seconds"]
        == 12.0
    )


def test_transition_rejects_changed_identity_without_signals() -> None:
    current = switch.identity(os.getpid())
    assert current is not None
    wrong = {**current, "start_time": "wrong"}
    with pytest.raises(RuntimeError, match="identity"):
        switch.same_process(wrong)


@pytest.mark.parametrize("relative_parent", [False, True])
def test_transition_fences_only_parent_and_preserves_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_parent: bool
) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    run = old / "runs/va_k128_retrain_B/seed_0"
    (run / "contracts").mkdir(parents=True)
    (old / "code-003").mkdir()
    (old / "code-003/SOURCE_MANIFEST.json").write_text("{}")
    (old / "status.json").write_text(
        json.dumps({"state": "running", "variant": "B", "seed": 0, "pid": 22})
    )
    (old / "campaign.json").write_text("{}")
    (run / "steps.jsonl").write_text(json.dumps({"epoch": 18}) + "\n")
    (run / "latest-epoch.json").write_text(json.dumps({"epoch": 18}))
    contract = {"schema": "test"}
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    (run / "contracts/task.json").write_bytes(raw)
    torch.save(
        {
            "contract_sha256": hashlib.sha256(raw).hexdigest(),
            "completed_epochs": 18,
            "global_step": 18 * 5005,
            "history": [{}] * 18,
            "model": {"x": torch.ones(1)},
            "optimizer": {},
            "rng": {},
        },
        run / "checkpoint.pt",
    )
    states = {
        11: {
            "pid": 11,
            "ppid": 1,
            "start_time": "a",
            "state": "S",
            "cmdline": str(old / "code-003/scripts/prepare_va_retrain_campaign.py"),
        },
        22: {
            "pid": 22,
            "ppid": 11,
            "start_time": "b",
            "state": "R",
            "cmdline": str(old / "code-003/scripts/run_va_retrain_imagenet1k.py"),
        },
    }
    if relative_parent:
        states[11]["cmdline"] = f"python -u scripts/prepare_va_retrain_campaign.py --root {old}"
        states[11]["cwd"] = str(old / "code-003")
    calls = []

    def fake_kill(pid, sig):
        calls.append((pid, sig))
        if sig == signal.SIGSTOP:
            states[pid]["state"] = "T"
        elif sig == signal.SIGINT:
            states[pid]["state"] = "Z"
        elif sig == signal.SIGKILL:
            states.pop(pid)

    monkeypatch.setattr(
        switch, "identity", lambda pid: dict(states[pid]) if pid in states else None
    )
    monkeypatch.setattr(switch.os, "kill", fake_kill)
    switch.transition(old, new, 10)
    receipt = json.loads((new / "legacy-pause.json").read_text())
    assert receipt["completed_epoch"] == 18
    assert (Path(receipt["backup"])).read_bytes() == (run / "checkpoint.pt").read_bytes()
    assert calls == [(11, signal.SIGSTOP), (22, signal.SIGINT), (11, signal.SIGKILL)]
    assert json.loads((old / "status.json").read_text())["state"] == "paused_for_va64"


def test_va64_queue_resume_config_and_single_seed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "preflight").mkdir()
    code = tmp_path / "code"
    code.mkdir()
    (code / "fixture.py").write_text("# snapshot\n")
    manifest = {
        "sha256": {"fixture.py": hashlib.sha256((code / "fixture.py").read_bytes()).hexdigest()}
    }
    (code / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest))
    sha = hashlib.sha256((code / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()
    (root / "legacy-pause.json").write_text(json.dumps({"state": "paused_for_va64"}))
    for v in "BPA":
        (root / "preflight" / f"{v}-b128.json").write_text(
            json.dumps({"status": "passed", "source_manifest_sha256": sha})
        )
        run = root / "runs" / f"va_k64_retrain_{v}" / "seed_0"
        run.mkdir(parents=True)
        (run / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "completed_epochs": 100,
                    "final_validation": {
                        "accuracy": 0.67 if v == "B" else 0.66,
                        "top5_accuracy": 0.85,
                        "cross_entropy": 1.0,
                        "examples": 50000,
                    },
                }
            )
        )
        (run / "cost.json").write_text(
            json.dumps({"gpu_hours": 1.0, "peak_allocated_bytes": 1, "peak_reserved_bytes": 2})
        )
        (run / "parameter-inventory.json").write_text(
            json.dumps({"total": {"stored_elements": 1562760, "trainable_elements": 1562760}})
        )
    (root / "preflight/unit-tests.json").write_text(
        json.dumps({"status": "passed", "source_manifest_sha256": sha})
    )

    def no_launch(*_args: object, **_kwargs: object) -> None:
        pytest.fail("completed runs must not launch again")

    monkeypatch.setattr(queue.subprocess, "Popen", no_launch)
    queue.queue(root, code, Path("/python"))
    queue.queue(root, code, Path("/python"), resume=True)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["completed"] == 3
    assert summary["seed_sample_sd"] is None
    assert summary["seed_confidence_interval"] is None
    assert summary["B-P_pp"] == pytest.approx(1.0)
    assert "single-seed observations" in (root / "rebuttal-single-seed.en.md").read_text()
    ledger = json.loads((root / "ownership.json").read_text())
    ledger["runs"]["B0"]["owner"] = "foreign"
    (root / "ownership.json").write_text(json.dumps(ledger))
    with pytest.raises(RuntimeError, match="owned by a different"):
        queue.update_claim(root, "B", "running")
