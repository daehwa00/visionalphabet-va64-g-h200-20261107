from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from typing import TYPE_CHECKING
from unittest.mock import patch

from scripts import k128_h200_observer as observer
from scripts import k128_h200_transport as transport

if TYPE_CHECKING:
    from pathlib import Path

CODE_SHA = "b" * 40
JOBS = transport.JOBS


def _archive_for_job(root: Path, job: str, epoch: int, step: int) -> bytes:
    output = root / ("source-" + job)
    output.mkdir(parents=True)
    (output / "checkpoint.pt").write_bytes((job + "-checkpoint").encode())
    (output / "dataset.json").write_text('{"train_images":128116}\n')
    (output / "poles").mkdir()
    (output / "poles" / "epoch-0000.json.gz").write_bytes(gzip.compress(b"initial"))
    (output / "poles" / ("epoch-" + str(epoch).zfill(4) + ".json.gz")).write_bytes(
        gzip.compress(("poles-" + str(epoch)).encode())
    )
    path = transport._make_archive(output, job, epoch, step)
    data = path.read_bytes()
    shutil.rmtree(output)
    return data


def _metadata(data: bytes, epoch: int, step: int) -> dict:
    return {
        "sha": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "chunks": (len(data) + transport.CHUNK_BYTES - 1) // transport.CHUNK_BYTES,
        "epoch": epoch,
        "global_step": step,
        "code_sha": CODE_SHA,
        "complete": True,
        "verified": False,
    }


class _ObserverAPI:
    def __init__(self, archives: dict[str, bytes], metadata: dict[str, dict]) -> None:
        self.authorization_value = "owner-secret-for-test"
        self.token = self.authorization_value
        self.archives = archives
        self.metadata = metadata
        self.acks: list[tuple[str, str]] = []

    def call(
        self,
        path: str,
        _value: object | None = None,
        _method: str | None = None,
        *,
        raw: bool = False,
        **_kwargs: object,
    ) -> object:
        if path.startswith("/snapshot?after="):
            return {
                "session": {"id": "session", "code": CODE_SHA, "ended": True},
                "control": {"finished_jobs": list(JOBS)},
                "events": [
                    {
                        "id": 1,
                        "value": json.dumps(
                            {
                                "seq": 0,
                                "logs": "Authorization: Bearer " + self.authorization_value,
                                "status": {
                                    "current": {
                                        "campaign_id": "k128-optimization-20260929",
                                        "job": JOBS[0],
                                    }
                                },
                            }
                        ),
                    }
                ],
            }
        if path == "/command":
            return {"ok": True}
        for job in JOBS:
            if path == "/artifact/" + job + "/latest":
                return self.metadata[job]
            prefix = "/artifact/" + job + "/" + self.metadata[job]["sha"] + "/"
            if path.startswith(prefix) and raw:
                part = int(path[len(prefix) :])
                start = part * transport.CHUNK_BYTES
                data = self.archives[job]
                return data[start : start + transport.CHUNK_BYTES]
            if path == "/artifact/" + job + "/" + self.metadata[job]["sha"] + "/ack":
                self.acks.append((job, self.metadata[job]["sha"]))
                return {"verified": True}
        raise AssertionError("Unexpected relay request: " + path)


def test_metrics_keep_checkpoint_step_when_resume_progress_resets() -> None:
    metrics = observer._metrics(
        {
            "progress": {"epoch": 0, "global_step": 0, "loss": None},
            "checkpoint": {
                "epoch": 100,
                "global_step": 5000,
                "train": {"loss": 0.7},
                "validation": {"accuracy": 0.5},
            },
        }
    )
    assert metrics["optimizer_updates"] == 5000
    assert metrics["epoch"] == 100
    assert metrics["train/loss"] == 0.7
    assert metrics["validation/top1_percent"] == 50.0


def test_observer_downloads_hash_checks_and_acks_both_final_archives(tmp_path: Path) -> None:
    archives = {
        job: _archive_for_job(tmp_path, job, 200, 10_000)
        for job in JOBS
    }
    metadata = {job: _metadata(archives[job], 200, 10_000) for job in JOBS}
    api = _ObserverAPI(archives, metadata)
    secrets = tmp_path / "owner-secrets.json"
    secrets.write_text("{}")
    destination = tmp_path / "observer"

    with patch.object(observer, "owner_api_from_secrets", return_value=api):
        observer.observe(destination, secrets, once=False, wandb_enabled=False)

    assert set(api.acks) == {(job, metadata[job]["sha"]) for job in JOBS}
    for job in JOBS:
        backup = observer._backup_path(destination, job, metadata[job])
        assert backup.is_file()
        receipt = json.loads(backup.with_suffix(".receipt.json").read_text())
        assert receipt["observer_managed"] is True
        assert receipt["sha"] == metadata[job]["sha"]
    event_log = (destination / observer.EVENTS_NAME).read_text()
    assert "[REDACTED]" in event_log
    assert api.authorization_value not in event_log


def test_retention_deletes_only_observer_owned_old_archives_and_keeps_final(tmp_path: Path) -> None:
    job = JOBS[0]
    directory = tmp_path / observer.BACKUP_ROOT / job
    directory.mkdir(parents=True)
    managed: dict[int, Path] = {}
    for epoch in (10, 20, 30, 200):
        sha = str(epoch).zfill(64)
        archive = directory / (
            "epoch-" + str(epoch).zfill(4) + "-step-0000000100-" + sha[:12] + ".tar.gz"
        )
        archive.write_bytes(("managed-" + str(epoch)).encode())
        managed[epoch] = archive
        observer.atomic_json(
            archive.with_suffix(".receipt.json"),
            {
                "observer_managed": True,
                "job": job,
                "sha": sha,
                "epoch": epoch,
                "global_step": 100,
                "verified_at": float(epoch),
                "archive": str(archive),
            },
        )

    unowned = directory / "manual.tar.gz"
    unowned.write_bytes(b"manual")
    external = tmp_path / "outside.tar.gz"
    external.write_bytes(b"outside")
    observer.atomic_json(
        directory / "external.receipt.json",
        {
            "observer_managed": True,
            "job": job,
            "sha": "c" * 64,
            "epoch": 1,
            "global_step": 1,
            "verified_at": 1.0,
            "archive": str(external),
        },
    )

    observer._cleanup_managed_archives(tmp_path, job)

    assert not managed[10].exists()
    assert not managed[20].exists()
    assert managed[30].is_file()
    assert managed[200].is_file()
    assert unowned.is_file()
    assert external.is_file()
    assert managed[10].with_suffix(".receipt.json").is_file()
    assert managed[20].with_suffix(".receipt.json").is_file()
