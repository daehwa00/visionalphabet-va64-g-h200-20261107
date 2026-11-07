from __future__ import annotations

import errno
import gzip
import hashlib
import io
import json
import os
import tarfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Self
from unittest.mock import patch

import pytest

from scripts import va64_h200_transport as transport

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

CODE_SHA = "a" * 40
JOB = "va64-g-seed0"


def _write_output(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoint.pt").write_bytes(b"checkpoint-payload")
    (root / "dataset.json").write_text('{"train_images":128116}\n')
    (root / "checkpoint-meta.json").write_text('{"epoch":10,"global_step":500}\n')
    (root / "result.json").write_text('{"status":"running"}\n')
    (root / "owner-secrets.json").write_text('{"IN10_OWNER_TOKEN":"must-not-upload"}\n')
    (root / "contracts").mkdir()
    (root / "contracts" / "source.json").write_text('{"code_sha":"abc"}\n')
    (root / "poles").mkdir()
    (root / "poles" / "epoch-0000.json.gz").write_bytes(gzip.compress(b"initial-poles"))
    (root / "poles" / "epoch-0010.json.gz").write_bytes(gzip.compress(b"trained-poles"))
    (root / "telemetry").mkdir()
    (root / "telemetry" / "metrics.jsonl").write_text('{"epoch":10}\n')
    (root / "telemetry" / "console.log").write_text("excluded console output\n")


class _UploadAPI:
    def __init__(self) -> None:
        self.base = transport.DEFAULT_RELAY_URL
        self.record = None
        self.chunks: list[bytes] = []

    def call(
        self,
        path: str,
        value: bytes | dict[str, object] | None = None,
        method: str | None = None,
        **_kwargs: object,
    ) -> dict[str, bool]:
        if path == "/artifact/" + JOB + "/begin":
            self.record = value
            return {"complete": False, "created": 1234}
        if path.endswith("/finish"):
            return {"complete": True}
        if method == "PUT":
            assert isinstance(value, bytes)
            self.chunks.append(value)
            return {"ok": True}
        raise AssertionError("Unexpected relay request")


class _DownloadAPI:
    def __init__(self, metadata: dict, archive_bytes: bytes) -> None:
        self.metadata = metadata
        self.archive_bytes = archive_bytes
        self.calls: list[tuple[str, object, str | None]] = []

    def call(
        self,
        path: str,
        value: object | None = None,
        method: str | None = None,
        *,
        raw: bool = False,
        **_kwargs: object,
    ) -> object:
        self.calls.append((path, value, method))
        if path == "/artifact/" + JOB + "/latest":
            return self.metadata
        prefix = "/artifact/" + JOB + "/" + self.metadata["sha"] + "/"
        if path.startswith(prefix) and raw:
            part = int(path[len(prefix) :])
            start = part * transport.CHUNK_BYTES
            return self.archive_bytes[start : start + transport.CHUNK_BYTES]
        raise AssertionError("Unexpected relay request")


class _JSONResponse:
    def __init__(self, payload: bytes = b'{"ok":true}') -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class _FakeRequestsResponse:
    def __init__(
        self,
        status_code: int = 200,
        content: bytes = b'{"ok":true}',
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.reason = "test"
        self.headers = headers or {}
        self.content = content
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeRequestsSession:
    def __init__(
        self,
        response_factory: Callable[[], _FakeRequestsResponse],
        barrier: threading.Barrier | None = None,
    ) -> None:
        self.response_factory = response_factory
        self.barrier = barrier
        self.calls: list[dict[str, object]] = []
        self.trust_env = True

    def request(self, **kwargs: object) -> _FakeRequestsResponse:
        self.calls.append(kwargs)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        return self.response_factory()


class _TransientOpener:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def open(self, _request: object, *, timeout: float) -> _JSONResponse:
        assert timeout > 0
        self.calls += 1
        if self.calls <= self.failures:
            cause = ConnectionResetError(errno.ECONNRESET, "private transport detail")
            raise urllib.error.URLError(cause)
        return _JSONResponse()


class _ResumableUploadAPI:
    def __init__(self, failing_part: int | None = None, server_created: int = 1234) -> None:
        self.base = transport.DEFAULT_RELAY_URL
        self.failing_part = failing_part
        self.server_created = server_created
        self.uploaded_parts: list[int] = []
        self.failed = False

    def call(
        self,
        path: str,
        _value: bytes | dict[str, object] | None = None,
        method: str | None = None,
        **_kwargs: object,
    ) -> dict[str, bool]:
        if path.startswith("/artifact/" + JOB + "/begin"):
            return {"complete": False, "created": self.server_created}
        if method == "PUT":
            part = int(path.rsplit("/", maxsplit=1)[-1])
            self.uploaded_parts.append(part)
            if part == self.failing_part and not self.failed:
                self.failed = True
                raise transport.TransportError(
                    "relay connection failed",
                    failure_type="ConnectionResetError",
                    errno=errno.ECONNRESET,
                    attempts=8,
                    retry_elapsed_seconds=120.0,
                    retry_deadline_exceeded=True,
                )
            return {"ok": True}
        if path.endswith("/finish"):
            return {"complete": True}
        raise AssertionError("Unexpected relay request")


def _artifact_metadata(data: bytes) -> dict:
    return {
        "sha": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "chunks": (len(data) + transport.CHUNK_BYTES - 1) // transport.CHUNK_BYTES,
        "epoch": 10,
        "global_step": 500,
        "code_sha": CODE_SHA,
        "complete": True,
        "verified": False,
    }


def test_backup_archive_whitelists_only_recovery_files_and_uploads_metadata(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _write_output(output)
    api = _UploadAPI()

    with patch.object(transport, "_agent_api", return_value=api):
        result = transport.backup_run(JOB, output, 10, 500, CODE_SHA, wait=True)

    assert result["job"] == JOB
    assert result["epoch"] == 10
    assert api.record["global_step"] == 500
    assert api.record["code_sha"] == CODE_SHA
    uploaded = b"".join(api.chunks)
    archive_path = tmp_path / "uploaded.tar.gz"
    archive_path.write_bytes(uploaded)
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert names == {
        "checkpoint.pt",
        "checkpoint-meta.json",
        "dataset.json",
        "result.json",
        "contracts/source.json",
        "poles/epoch-0000.json.gz",
        "poles/epoch-0010.json.gz",
        "telemetry/metrics.jsonl",
    }
    assert b"must-not-upload" not in uploaded
    assert b"excluded console output" not in uploaded
    assert (output / ".va64-backup" / (JOB + "-backup-state.json")).is_file()


def test_restore_downloads_and_merges_without_overwriting_epoch_zero_poles(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_output(source)
    archive_path = transport._make_archive(source, JOB, 10, 500)
    archive_bytes = archive_path.read_bytes()
    metadata = _artifact_metadata(archive_bytes)
    api = _DownloadAPI(metadata, archive_bytes)
    destination = tmp_path / "restored"
    destination.mkdir()
    initial_poles = gzip.compress(b"owner-initial-poles")
    (destination / "poles").mkdir()
    (destination / "poles" / "epoch-0000.json.gz").write_bytes(initial_poles)

    with patch.object(transport, "_agent_api", return_value=api):
        restored = transport.restore_backup(
            JOB,
            destination,
            tmp_path / "owner-cache" / "latest.tar.gz",
        )

    assert restored
    assert (destination / "checkpoint.pt").read_bytes() == b"checkpoint-payload"
    assert (destination / "dataset.json").is_file()
    assert (destination / "contracts" / "source.json").is_file()
    assert (destination / "poles" / "epoch-0010.json.gz").is_file()
    assert (destination / "poles" / "epoch-0000.json.gz").read_bytes() == initial_poles
    assert (tmp_path / "owner-cache" / "latest.tar.gz").is_file()
    assert not any(path[0].endswith("/ack") for path in api.calls)


def test_restore_prefers_an_existing_local_checkpoint_without_network_access(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "checkpoint.pt").write_bytes(b"local")

    with patch.object(transport, "_agent_api", side_effect=AssertionError("network used")):
        assert transport.restore_backup(JOB, output, tmp_path / "archive.tar.gz") is False


def test_archive_validation_rejects_traversal_and_symlink_sources(tmp_path: Path) -> None:
    malicious = tmp_path / "malicious.tar.gz"
    with tarfile.open(malicious, "w:gz") as archive:
        checkpoint = tarfile.TarInfo("checkpoint.pt")
        checkpoint.size = 1
        archive.addfile(checkpoint, io.BytesIO(b"x"))
        escape = tarfile.TarInfo("../escaped.txt")
        escape.size = 1
        archive.addfile(escape, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe path"):
        transport.validate_archive(malicious)

    output = tmp_path / "linked"
    _write_output(output)
    (output / "telemetry" / "linked.jsonl").symlink_to(output / "owner-secrets.json")
    with pytest.raises(ValueError, match="symlink"):
        transport._make_archive(output, JOB, 10, 500)


def test_relay_base_must_select_the_isolated_va64_namespace() -> None:
    assert transport._relay_base() == transport.DEFAULT_RELAY_URL
    with pytest.raises(ValueError, match="/va64g"):
        transport._relay_base("https://example.workers.dev/in10")


def test_api_retries_more_than_four_transient_connection_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = transport.API("private-test-token", base="https://example.workers.dev/va64g")
    opener = _TransientOpener(failures=5)
    api._opener = opener
    monkeypatch.setattr(transport.time, "sleep", lambda _seconds: None)

    response = api.call(
        "/artifact/" + JOB + "/begin",
        {"sha": "a" * 64},
        attempts=8,
        retry_timeout=60,
    )

    assert response == {"ok": True}
    assert opener.calls == 6


def test_api_does_not_retry_permanent_http_errors_or_expose_response_data() -> None:
    api = transport.API("private-test-token", base="https://example.workers.dev/va64g")

    class ForbiddenOpener:
        calls = 0

        def open(self, _request: object, *, timeout: float) -> _JSONResponse:
            assert timeout > 0
            self.calls += 1
            raise urllib.error.HTTPError(
                "https://private.invalid/path?credential=hidden",
                403,
                "forbidden",
                {},
                io.BytesIO(b"private response body"),
            )

    opener = ForbiddenOpener()
    api._opener = opener
    with pytest.raises(transport.TransportError) as captured:
        api.call("/artifact/" + JOB + "/latest", attempts=8, retry_timeout=60)

    error = captured.value
    assert opener.calls == 1
    assert error.safe_details()["http_status"] == 403
    assert "private.invalid" not in str(error)
    assert "hidden" not in str(error)
    assert "private response body" not in str(error)


def test_api_retry_deadline_is_bounded_and_keeps_safe_connection_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = transport.API("private-test-token", base="https://example.workers.dev/va64g")
    opener = _TransientOpener(failures=99)
    api._opener = opener
    clock = [0.0]
    monkeypatch.setattr(transport.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        transport.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    with pytest.raises(transport.TransportError) as captured:
        api.call("/artifact/" + JOB + "/begin", {}, attempts=8, retry_timeout=2.5)

    details = captured.value.safe_details()
    assert opener.calls == 2
    assert clock[0] == 2.5
    assert details["retry_deadline_exceeded"] is True
    assert details["retry_elapsed_seconds"] == 2.5
    assert details["failure_type"] == "ConnectionResetError"
    assert details["errno"] == errno.ECONNRESET


def test_requests_opener_reuses_a_session_and_preserves_request_headers() -> None:
    sessions: list[_FakeRequestsSession] = []

    def create_session() -> _FakeRequestsSession:
        session = _FakeRequestsSession(_FakeRequestsResponse)
        sessions.append(session)
        return session

    opener = transport._RequestsSessionOpener(session_factory=create_session)
    request = urllib.request.Request(
        "https://example.workers.dev/va64g/events",
        data=b'{"seq":0}',
        headers={
            "Authorization": "Bearer private-test-token",
            "X-Session-ID": "private-test-session",
            "Content-Type": "application/json",
        },
        method="PUT",
    )

    for _ in range(2):
        with opener.open(request, timeout=7) as response:
            assert response.read() == b'{"ok":true}'

    assert len(sessions) == 1
    assert len(sessions[0].calls) == 2
    call = sessions[0].calls[0]
    assert call["method"] == "PUT"
    assert call["url"] == request.full_url
    assert call["data"] == b'{"seq":0}'
    assert call["timeout"] == 7
    assert call["allow_redirects"] is False
    assert call["auth"] is transport._PRESERVE_AUTH
    assert call["headers"]["Authorization"] == "Bearer private-test-token"
    assert sessions[0].trust_env is True
    default_session = transport._RequestsSessionOpener()._session()
    assert default_session.trust_env is True
    default_session.close()


def test_requests_auth_keeps_the_explicit_bearer_and_skips_netrc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = transport.requests.Session()
    monkeypatch.setattr(
        transport.requests.sessions,
        "get_netrc_auth",
        lambda _url: pytest.fail("netrc must not override relay authentication"),
    )
    request = transport.requests.Request(
        "GET",
        "https://example.workers.dev/va64g/health",
        headers={"Authorization": "Bearer explicit-test-token"},
        auth=transport._PRESERVE_AUTH,
    )

    prepared = session.prepare_request(request)

    assert prepared.headers["Authorization"] == "Bearer explicit-test-token"
    session.close()


def test_requests_opener_uses_one_session_per_thread() -> None:
    barrier = threading.Barrier(2)
    sessions: list[_FakeRequestsSession] = []
    sessions_lock = threading.Lock()

    def create_session() -> _FakeRequestsSession:
        session = _FakeRequestsSession(
            _FakeRequestsResponse,
            barrier=barrier,
        )
        with sessions_lock:
            sessions.append(session)
        return session

    opener = transport._RequestsSessionOpener(session_factory=create_session)
    request = urllib.request.Request("https://example.workers.dev/va64g/health")

    def read_health() -> bytes:
        with opener.open(request, timeout=7) as response:
            return response.read()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: read_health(), range(2)))

    assert results == [b'{"ok":true}', b'{"ok":true}']
    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]


@pytest.mark.parametrize("status_code", [302, 403, 429, 503])
def test_requests_opener_maps_redirect_and_non_2xx_responses_to_http_error(
    status_code: int,
) -> None:
    sessions: list[_FakeRequestsSession] = []

    def create_session() -> _FakeRequestsSession:
        session = _FakeRequestsSession(
            lambda: _FakeRequestsResponse(
                status_code=status_code,
                content=b"body must not be exposed",
                headers={"Location": "https://elsewhere.invalid/"},
            )
        )
        sessions.append(session)
        return session

    opener = transport._RequestsSessionOpener(session_factory=create_session)
    request = urllib.request.Request("https://example.workers.dev/va64g/artifact")

    with pytest.raises(urllib.error.HTTPError) as captured:
        opener.open(request, timeout=7)

    assert captured.value.code == status_code
    assert captured.value.headers.get("Location") == "https://elsewhere.invalid/"
    assert captured.value.read() == b""
    assert sessions[0].calls[0]["allow_redirects"] is False


def test_resume_progress_is_bound_to_relay_and_server_artifact_version(
    tmp_path: Path,
) -> None:
    record = {
        "sha": "c" * 64,
        "bytes": 100,
        "chunks": 10,
        "epoch": 10,
        "global_step": 500,
        "code_sha": CODE_SHA,
    }
    progress_path = tmp_path / "progress.json"
    state = transport._transfer_progress(
        JOB,
        record,
        5,
        "uploading",
        transport.DEFAULT_RELAY_URL,
        1234,
    )
    transport.atomic_json(progress_path, state)

    assert (
        transport._read_transfer_progress(
            progress_path, record, transport.DEFAULT_RELAY_URL, 1234
        )
        == 5
    )
    assert (
        transport._read_transfer_progress(
            progress_path, record, transport.DEFAULT_RELAY_URL, 1235
        )
        == 0
    )
    assert (
        transport._read_transfer_progress(
            progress_path, record, "https://other.workers.dev/va64g", 1234
        )
        == 0
    )
    state.pop("server_created")
    transport.atomic_json(progress_path, state)
    assert (
        transport._read_transfer_progress(
            progress_path, record, transport.DEFAULT_RELAY_URL, 1234
        )
        == 0
    )


def test_failed_archive_upload_logs_checkpoint_and_resumes_at_last_acknowledged_chunk(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    _write_output(output)
    (output / "checkpoint.pt").write_bytes(os.urandom(6 * transport.CHUNK_BYTES))
    archive_path = transport._make_archive(output, JOB, 10, 500)
    archive_data = archive_path.read_bytes()
    sha = hashlib.sha256(archive_data).hexdigest()
    progress_path = transport._transfer_progress_path(archive_path, JOB, sha)
    api = _ResumableUploadAPI(failing_part=5)

    with patch.object(transport, "_agent_api", return_value=api):
        with pytest.raises(transport.TransportError):
            transport._upload_archive(JOB, archive_path, 10, 500, CODE_SHA)

        progress = json.loads(progress_path.read_text())
        failures = [
            json.loads(line)
            for line in (archive_path.parent / (JOB + "-transfer-failures.jsonl"))
            .read_text()
            .splitlines()
        ]
        assert progress["acknowledged_chunks"] == 5
        assert progress["status"] == "failed"
        assert failures[-1]["part"] == 5
        assert failures[-1]["error"]["errno"] == errno.ECONNRESET
        assert archive_path.is_file()

        api.failing_part = None
        result = transport._upload_archive(JOB, archive_path, 10, 500, CODE_SHA)

    remaining_parts = api.uploaded_parts[6:]
    assert remaining_parts[0] == 5
    assert not set(remaining_parts) & {0, 1, 2, 3, 4}
    assert result["sha"] == sha
    assert not archive_path.exists()
    assert json.loads(progress_path.read_text())["status"] == "complete"
