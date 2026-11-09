"""CPU validation of G replicate initialization and seed-local contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import run_va64_g_h200 as entry
from scripts import run_va64_retrain_imagenet1k as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def canonical_path(seed):
    public = ROOT / "initialization" / f"initial-B{seed}.pt"
    if public.exists():
        return public
    return (
        ROOT
        / "artifacts"
        / "va64-followup-seeds-20261108"
        / "initialization"
        / f"seed_{seed}"
        / f"initial-B{seed}.pt"
    )


@pytest.mark.parametrize("seed", [1, 2])
def test_actual_canonical_backbone_and_fresh_head(seed):
    path = canonical_path(seed)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry.INITIAL_SHAS[seed]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    torch.manual_seed(seed + 17)
    model = runner.build_model("G", prepared=False)
    head = {
        name: value.clone() for name, value in model.state_dict().items() if ".classifier." in name
    }
    assert entry.initialize_common_backbone(model, path, seed=seed) is model
    assert sum(p.numel() for p in model.parameters()) == 1_882_760
    for name, value in model.state_dict().items():
        assert torch.equal(value, head[name] if name in head else payload["model"][name]), name


def test_replicates_have_different_backbones():
    one = torch.load(canonical_path(1), map_location="cpu", weights_only=True)["model"]
    two = torch.load(canonical_path(2), map_location="cpu", weights_only=True)["model"]
    assert any(not torch.equal(one[name], two[name]) for name in one if ".classifier." not in name)


def test_wrong_seed_and_hash_rejected(tmp_path):
    model = runner.build_model("G", prepared=False)
    with pytest.raises(ValueError, match="seeds 0, 1, and 2"):
        entry.initialize_common_backbone(model, seed=3)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        entry.initialize_common_backbone(model, canonical_path(1), seed=2)
    damaged = tmp_path / "damaged.pt"
    damaged.write_bytes(canonical_path(1).read_bytes() + b"corruption")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        entry.initialize_common_backbone(model, damaged, seed=1)


@pytest.mark.parametrize("field,value", [("seed", 2), ("variant", "G"), ("trained", True)])
def test_canonical_metadata_rejected_even_if_digest_matches(tmp_path, monkeypatch, field, value):
    payload = torch.load(canonical_path(1), map_location="cpu", weights_only=True)
    payload[field] = value
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    monkeypatch.setitem(entry.INITIAL_SHAS, 1, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="untrained B1"):
        entry.initialize_common_backbone(runner.build_model("G", prepared=False), path, seed=1)


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("recovery", ["resume", "VA64_G_RECOVERY", "VA64_G_RESUME_PLAN"])
def test_replicates_reject_seed0_recovery(seed, recovery, monkeypatch):
    monkeypatch.delenv("VA64_G_RECOVERY", raising=False)
    monkeypatch.delenv("VA64_G_RESUME_PLAN", raising=False)
    args = argparse.Namespace(variant="G", seed=seed, resume=recovery == "resume")
    if recovery != "resume":
        monkeypatch.setenv(recovery, "seed0-only")
    with pytest.raises(ValueError, match="fresh start"):
        entry.validate_run(args)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_supported_fresh_run(seed, monkeypatch):
    monkeypatch.delenv("VA64_G_RECOVERY", raising=False)
    monkeypatch.delenv("VA64_G_RESUME_PLAN", raising=False)
    entry.validate_run(argparse.Namespace(variant="G", seed=seed, resume=False))


@pytest.mark.parametrize("seed", [1, 2])
def test_entrypoint_seed_contract_and_output(seed, tmp_path, monkeypatch):
    for name in ("VA64_G_MANAGED_ROOT", "VA64_G_RECOVERY", "VA64_G_RESUME_PLAN"):
        monkeypatch.delenv(name, raising=False)
    manifest = tmp_path / "data.json"
    manifest.write_text(
        json.dumps({"root": str(tmp_path), "counts": {"train": 1281167, "val": 50000}})
    )
    args = argparse.Namespace(
        variant="G",
        seed=seed,
        data_root=tmp_path,
        data_manifest=manifest,
        output_root=tmp_path / "out",
        batch_size=128,
        workers=0,
        resume=False,
    )
    monkeypatch.setattr(entry.experiment, "arguments", lambda: args)
    # Register mutable globals with monkeypatch so main's hooks do not leak.
    monkeypatch.setattr(entry.experiment, "build_model", entry.experiment.build_model)
    monkeypatch.setattr(entry.experiment, "make_task", entry.experiment.make_task)
    real_initialize = entry.initialize_common_backbone
    monkeypatch.setattr(
        entry,
        "initialize_common_backbone",
        lambda model, path=None, seed=0: real_initialize(model, canonical_path(seed), seed=seed),
    )

    def train(arguments):
        task = entry.experiment.make_task(arguments)
        assert task.seed == seed
        assert task.output_dir == (tmp_path / "out" / "va_k64_retrain_G" / f"seed_{seed}")
        assert task.execution.gpu_mixup
        assert task.source_sha256["canonical_backbone_initialization"] == entry.INITIAL_SHAS[seed]
        assert (
            task.source_sha256["h200_g_entrypoint"]
            == hashlib.sha256(Path(entry.__file__).read_bytes()).hexdigest()
        )
        model = entry.experiment.build_model("G", prepared=False)
        original = torch.load(canonical_path(seed), map_location="cpu", weights_only=True)["model"]
        for name, value in model.state_dict().items():
            if ".classifier." not in name:
                assert torch.equal(value, original[name])
        return {"status": "completed"}

    monkeypatch.setattr(entry.experiment, "train", train)
    entry.main()


@pytest.mark.parametrize("seed", [1, 2])
def test_managed_stop_uses_seed_job_and_output(seed, tmp_path, monkeypatch):
    import va64_runtime_operations as operations

    args = argparse.Namespace(variant="G", seed=seed, resume=False)
    monkeypatch.setenv("VA64_G_MANAGED_ROOT", str(tmp_path))
    monkeypatch.setenv("VA64_G_JOB", "va64-g-seed0")
    monkeypatch.delenv("VA64_G_RECOVERY", raising=False)
    monkeypatch.delenv("VA64_G_RESUME_PLAN", raising=False)
    monkeypatch.setattr(entry.experiment, "arguments", lambda: args)
    monkeypatch.setattr(entry.experiment, "build_model", entry.experiment.build_model)
    monkeypatch.setattr(entry.experiment, "make_task", entry.experiment.make_task)
    monkeypatch.setattr(operations, "install_training", lambda *args: None)
    calls = []
    monkeypatch.setattr(operations, "current", lambda *args: calls.append(args))

    def interrupt(arguments):
        assert entry.os.environ["VA64_G_JOB"] == f"va64-g-seed{seed}"
        raise entry.experiment.runtime.TrainingInterrupted()

    monkeypatch.setattr(entry.experiment, "train", interrupt)
    with pytest.raises(SystemExit) as stopped:
        entry.main()
    assert stopped.value.code == 130
    assert calls == [
        (
            tmp_path,
            f"va64-g-seed{seed}",
            "stopped_without_backup",
            tmp_path / "runs" / "va_k64_retrain_G" / f"seed_{seed}",
        )
    ]
