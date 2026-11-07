from __future__ import annotations

# pyright: reportPrivateLocalImportUsage=false, reportPrivateUsage=false
# pyright: reportUnknownLambdaType=false
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from scripts import chain_r2k3_queue as chain


def test_complete_requires_every_job_terminal(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"jobs": {"a": {"status": "COMPLETE"}}}))
    assert chain._complete(path)
    path.write_text(json.dumps({"jobs": {"a": {"status": "COMPLETE"}, "b": {"status": "QUEUED"}}}))
    assert not chain._complete(path)


def test_main_waits_then_executes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"jobs": {"a": {"status": "QUEUED"}}}))
    sleeps = []
    executed = []

    def finish(_seconds: float) -> None:
        sleeps.append(True)
        path.write_text(json.dumps({"jobs": {"a": {"status": "COMPLETE"}}}))

    monkeypatch.setattr(chain.time, "sleep", finish)
    monkeypatch.setattr(
        chain.os,
        "execvpe",
        lambda program, command, _environment: executed.append((program, command)),
    )
    chain.main(
        [
            "--status",
            str(path),
            "--poll-seconds",
            "0.01",
            "--",
            "python",
            "worker.py",
        ]
    )

    assert sleeps == [True]
    assert executed == [("python", ["python", "worker.py"])]
