"""Private K128 H200 checkpoint transport and safe archive recovery.

The H200 process gets only its per-session agent token. Owner credentials are
read by the local observer from the existing owner-secrets file.
"""

from __future__ import annotations

import base64
import concurrent.futures
import email.utils
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, Self

import requests

DEFAULT_RELAY_URL = (
    "https://visionalphabet-k128-opt-20260929.gpupulse-monitor.workers.dev/k128opt"
)
RELAY_URL_ENV = "K128_OPT_RELAY_URL"
AGENT_TOKEN_ENV = "H200_AGENT_TOKEN"  # noqa: S105 - this is an environment variable name.
SESSION_ID_ENV = "H200_SESSION_ID"
OWNER_TOKEN_KEY = "IN10_OWNER_TOKEN"  # noqa: S105 - this is a JSON key name.
JOBS = ("k128-baseline-501", "k128-improved-501")
CHUNK_BYTES = 32 * 1024
MAX_ARCHIVE_BYTES = 384 * 1024**2
MAX_ARCHIVE_CHUNKS = (MAX_ARCHIVE_BYTES + CHUNK_BYTES - 1) // CHUNK_BYTES
MAX_ARCHIVE_FILES = 20_000
MAX_UNPACKED_BYTES = 2 * 1024**3
UPLOAD_CHUNK_ATTEMPTS = 8
UPLOAD_CHUNK_RETRY_TIMEOUT = 120.0
UPLOAD_PROGRESS_INTERVAL = 4
_POLE_FILE = re.compile(r"poles/epoch-\d{4}\.json\.gz\Z")
_CODE_SHA = re.compile(r"[a-f0-9]{40}\Z")


class TransportError(RuntimeError):
    """A relay failure that never includes request headers or response bodies."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str | None = None,
        errno: int | None = None,
        http_status: int | None = None,
        attempts: int | None = None,
        retry_elapsed_seconds: float | None = None,
        retry_deadline_exceeded: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type or type(self).__name__
        self.errno = errno
        self.http_status = http_status
        self.attempts = attempts
        self.retry_elapsed_seconds = retry_elapsed_seconds
        self.retry_deadline_exceeded = retry_deadline_exceeded

    def safe_details(self) -> dict[str, Any]:
        """Return diagnostics without request URL, headers, body, or exception text."""
        return {
            "failure_type": self.failure_type,
            "errno": self.errno,
            "http_status": self.http_status,
            "attempts": self.attempts,
            "retry_elapsed_seconds": self.retry_elapsed_seconds,
            "retry_deadline_exceeded": self.retry_deadline_exceeded,
        }


def _safe_network_details(error: BaseException) -> tuple[str, int | None]:
    reason = getattr(error, "reason", None)
    cause = reason if isinstance(reason, BaseException) else error
    errno = getattr(cause, "errno", None)
    return type(cause).__name__, errno if type(errno) is int else None


def _elapsed(started: float) -> float:
    return round(max(0.0, time.monotonic() - started), 3)


def _retry_deadline_error(
    started: float,
    attempts: int,
    *,
    failure_type: str | None = None,
    errno: int | None = None,
    http_status: int | None = None,
) -> TransportError:
    return TransportError(
        "relay retry deadline exceeded",
        failure_type=failure_type or "RetryDeadline",
        errno=errno,
        http_status=http_status,
        attempts=attempts,
        retry_elapsed_seconds=_elapsed(started),
        retry_deadline_exceeded=True,
    )

def _relay_base(value: str | None = None) -> str:
    base = (value or os.environ.get(RELAY_URL_ENV) or DEFAULT_RELAY_URL).rstrip("/")
    parsed = urllib.parse.urlsplit(base)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.rstrip("/").endswith("/k128opt")
    ):
        raise ValueError("K128_OPT_RELAY_URL must be an HTTPS /k128opt endpoint")
    return base


class _RequestsResponse:
    def __init__(self, response: requests.Response) -> None:
        self._response = response

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self._response.close()

    def read(self) -> bytes:
        try:
            return self._response.content
        except requests.RequestException as error:
            raise urllib.error.URLError(error) from None


class _PreserveAuthorization(requests.auth.AuthBase):
    """Keep the explicit Bearer header and prevent requests from consulting netrc."""

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        return request


_PRESERVE_AUTH = _PreserveAuthorization()


class _RequestsSessionOpener:
    """urllib Request opener backed by one persistent requests session per thread."""

    def __init__(
        self,
        session_factory: Callable[[], requests.Session] = requests.Session,
    ) -> None:
        self._session_factory = session_factory
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._session_factory()
            self._local.session = session
        return session

    def open(self, request: urllib.request.Request, timeout: float) -> _RequestsResponse:
        try:
            response = self._session().request(
                method=request.get_method(),
                url=request.full_url,
                data=request.data,
                headers=dict(request.header_items()),
                auth=_PRESERVE_AUTH,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as error:
            raise urllib.error.URLError(error) from None

        if not 200 <= response.status_code < 300:
            status = response.status_code
            reason = response.reason
            response_headers = response.headers
            response.close()
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                reason,
                response_headers,
                io.BytesIO(),
            )
        return _RequestsResponse(response)


class API:
    """urllib-compatible authenticated transport for the isolated K128 namespace."""

    def __init__(self, token: str, session: str = "", base: str | None = None) -> None:
        if not isinstance(token, str) or not token:
            raise ValueError("relay token is required")
        self.token = token
        self.session = session
        self.base = _relay_base(base)
        self._lock = threading.Lock()
        self._next_chunk = 0.0
        self._opener = _RequestsSessionOpener()

    @staticmethod
    def _retry_after(error: urllib.error.HTTPError) -> float:
        value = error.headers.get("Retry-After") if error.headers else None
        if not value:
            return 0.0
        try:
            return min(300.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            try:
                seconds = email.utils.parsedate_to_datetime(value).timestamp() - time.time()
                return min(300.0, max(0.0, seconds))
            except (TypeError, ValueError, OverflowError):
                return 0.0

    def call(
        self,
        path: str,
        value: Any = None,
        method: str | None = None,
        *,
        raw: bool = False,
        timeout: float = 15.0,
        attempts: int = 4,
        retry_timeout: float | None = None,
    ) -> Any:
        if not path.startswith("/") or "://" in path:
            raise ValueError("relay path must be relative to the configured namespace")
        if not isinstance(attempts, int) or attempts < 1:
            raise ValueError("attempts must be a positive integer")
        if retry_timeout is not None and retry_timeout <= 0:
            raise ValueError("retry_timeout must be positive")
        started = time.monotonic()
        deadline = started + retry_timeout if retry_timeout is not None else None

        binary = isinstance(value, bytes)
        if binary:
            body = json.dumps(
                {"base64": base64.b64encode(value).decode("ascii")},
                separators=(",", ":"),
            ).encode("utf-8")
        elif value is None:
            body = None
        else:
            body = json.dumps(value, separators=(",", ":")).encode("utf-8")

        headers = {
            "Authorization": "Bearer " + self.token,
            "X-Session-ID": self.session,
            "User-Agent": "visionalphabet-k128-transport/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # The validated HTTPS namespace and no-redirect opener restrict this request.
        request = urllib.request.Request(  # noqa: S310
            self.base + path,
            data=body,
            headers=headers,
            method=method or ("POST" if body is not None else "GET"),
        )

        for attempt in range(attempts):
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise _retry_deadline_error(started, attempt)
            if binary or raw:
                with self._lock:
                    time.sleep(max(0.0, self._next_chunk - time.monotonic()))
                    self._next_chunk = time.monotonic() + 0.125
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise _retry_deadline_error(started, attempt)
            request_timeout = min(timeout, remaining) if remaining is not None else timeout
            try:
                with self._opener.open(
                    request, timeout=request_timeout
                ) as response:
                    result = json.loads(response.read())
            except urllib.error.HTTPError as error:
                retry = error.code == 429 or 500 <= error.code <= 599
                attempts_made = attempt + 1
                if not retry or attempt == attempts - 1:
                    raise TransportError(
                        "relay returned HTTP " + str(error.code),
                        failure_type="HTTPError",
                        http_status=error.code,
                        attempts=attempts_made,
                        retry_elapsed_seconds=_elapsed(started),
                    ) from None
                delay = min(60.0, 2.0**attempt)
                delay = max(delay, self._retry_after(error))
                remaining = deadline - time.monotonic() if deadline is not None else None
                if remaining is not None and remaining <= 0:
                    raise _retry_deadline_error(
                        started, attempts_made, failure_type="HTTPError", http_status=error.code
                    ) from None
                if remaining is not None and delay >= remaining:
                    time.sleep(remaining)
                    raise _retry_deadline_error(
                        started, attempts_made, failure_type="HTTPError", http_status=error.code
                    ) from None
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as error:
                failure_type, errno = _safe_network_details(error)
                attempts_made = attempt + 1
                if attempt == attempts - 1:
                    raise TransportError(
                        "relay connection failed",
                        failure_type=failure_type,
                        errno=errno,
                        attempts=attempts_made,
                        retry_elapsed_seconds=_elapsed(started),
                    ) from None
                delay = min(30.0, 2.0**attempt)
                remaining = deadline - time.monotonic() if deadline is not None else None
                if remaining is not None and remaining <= 0:
                    raise _retry_deadline_error(
                        started, attempts_made, failure_type=failure_type, errno=errno
                    ) from None
                if remaining is not None and delay >= remaining:
                    time.sleep(remaining)
                    raise _retry_deadline_error(
                        started, attempts_made, failure_type=failure_type, errno=errno
                    ) from None
                time.sleep(delay)
            else:
                if raw:
                    if not isinstance(result, dict) or not isinstance(result.get("base64"), str):
                        raise TransportError("relay returned an invalid encoded chunk")
                    try:
                        return base64.b64decode(result["base64"], validate=True)
                    except (ValueError, base64.binascii.Error) as error:
                        raise TransportError("relay returned an invalid encoded chunk") from error
                return result
        raise TransportError(
            "relay request failed",
            attempts=attempts,
            retry_elapsed_seconds=_elapsed(started),
        )


def _agent_api() -> API:
    token = os.environ.get(AGENT_TOKEN_ENV, "")
    session = os.environ.get(SESSION_ID_ENV, "")
    if not token or not session:
        raise RuntimeError("H200 agent session is unavailable")
    return API(token, session)


def owner_api_from_secrets(path: str | Path) -> API:
    """Read the existing owner token without printing or returning it."""
    try:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("owner secrets file could not be read") from error
    token = values.get(OWNER_TOKEN_KEY) if isinstance(values, dict) else None
    if not isinstance(token, str) or not token:
        raise RuntimeError("owner token is missing from the owner secrets file")
    return API(token)


def atomic_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _check_job(job_id: str) -> None:
    if job_id not in JOBS:
        raise ValueError("job ID is outside the K128 campaign")


def _safe_relative_name(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
        raise ValueError("archive contains an unsafe path")
    parts = name.rstrip("/").split("/")
    if not parts or any(part in ("", ".", "..") or part.startswith(".") for part in parts):
        raise ValueError("archive contains an unsafe path")
    path = PurePosixPath(*parts)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError("archive contains an unsafe path")
    return path


def _allowed_archive_path(name: str, *, directory: bool = False) -> bool:
    path = _safe_relative_name(name)
    text = path.as_posix()
    parts = path.parts
    if len(parts) == 1 and text in {
        "checkpoint.pt",
        "result.json",
        "dataset.json",
        "checkpoint-meta.json",
    }:
        return not directory
    if directory:
        return text in {"contracts", "poles", "telemetry"} or any(
            text == folder or text.startswith(folder + "/")
            for folder in ("contracts", "poles", "telemetry")
        )
    if parts[0] == "contracts" and len(parts) > 1:
        return Path(parts[-1]).suffix.lower() in {
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".txt",
            ".sha256",
            ".sig",
            ".signature",
        }
    if parts[0] == "poles":
        return bool(_POLE_FILE.fullmatch(text))
    if parts[0] == "telemetry" and len(parts) > 1:
        suffix = "".join(Path(part).suffix.lower() for part in parts[-2:])
        return suffix in {".json", ".jsonl", ".csv", ".json.gz"}
    return False


def _source_files(output_dir: Path) -> list[tuple[str, Path, os.stat_result]]:
    if output_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")
    root = output_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("output directory must be a directory")
    selected: list[tuple[str, Path, os.stat_result]] = []

    for name in (
        "checkpoint.pt",
        "result.json",
        "dataset.json",
        "checkpoint-meta.json",
    ):
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("archive source must be a regular file")
        selected.append((name, path, info))

    for folder in ("contracts", "poles", "telemetry"):
        top = root / folder
        if not top.exists() and not top.is_symlink():
            continue
        if top.is_symlink() or not top.is_dir():
            raise ValueError("archive directory must be a real directory")
        for current, directories, files in os.walk(top, topdown=True, followlinks=False):
            current_path = Path(current)
            directories.sort()
            files.sort()
            for directory in list(directories):
                candidate = current_path / directory
                if candidate.is_symlink():
                    raise ValueError("archive source contains a symlink")
            for filename in files:
                path = current_path / filename
                relative = path.relative_to(root).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError("archive source contains a symlink")
                if not stat.S_ISREG(info.st_mode):
                    continue
                if any(part.startswith(".") for part in PurePosixPath(relative).parts):
                    continue
                if _allowed_archive_path(relative):
                    selected.append((relative, path, info))

    selected.sort(key=lambda item: item[0])
    if len(selected) > MAX_ARCHIVE_FILES:
        raise ValueError("archive contains too many files")
    if not any(name == "checkpoint.pt" for name, _, _ in selected):
        raise FileNotFoundError("checkpoint.pt is required for backup")
    return selected


def _make_archive(output_dir: Path, job_id: str, epoch: int, global_step: int) -> Path:
    files = _source_files(output_dir)
    backup_dir = output_dir / ".k128-backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    archive_path = backup_dir / (job_id + "-e" + str(epoch) + "-s" + str(global_step) + ".tar.gz")
    temporary = archive_path.with_name(archive_path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as raw:
            compressed = gzip.GzipFile(
                fileobj=raw, filename="", mode="wb", compresslevel=6, mtime=0
            )
            try:
                with tarfile.open(
                    fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
                ) as archive:
                    for relative, path, expected in files:
                        with path.open("rb") as stream:
                            actual = os.fstat(stream.fileno())
                            signature = (actual.st_ino, actual.st_size, actual.st_mtime_ns)
                            expected_signature = (expected.st_ino, expected.st_size, expected.st_mtime_ns)
                            if signature != expected_signature:
                                raise RuntimeError("archive source changed while preparing backup")
                            entry = tarfile.TarInfo(relative)
                            entry.size = actual.st_size
                            entry.mode = 0o600
                            entry.mtime = 0
                            entry.uid = 0
                            entry.gid = 0
                            entry.uname = ""
                            entry.gname = ""
                            archive.addfile(entry, stream)
            finally:
                compressed.close()
            raw.flush()
            os.fsync(raw.fileno())
        if temporary.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("backup archive exceeds the K128 relay size limit")
        temporary.replace(archive_path)
        _fsync_directory(backup_dir)
        return archive_path
    finally:
        temporary.unlink(missing_ok=True)


def _file_signature(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": str(path),
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def _transfer_progress_path(archive_path: Path, job_id: str, sha: str) -> Path:
    return archive_path.parent / (job_id + "-" + sha + "-transfer-progress.json")


def _read_transfer_progress(
    path: Path,
    record: dict[str, Any],
    relay_base: str,
    server_created: int | None,
) -> int:
    reusable = type(server_created) is int and server_created > 0
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        progress = {}
    if not reusable or not isinstance(progress, dict):
        progress = {}
    identity = ("sha", "epoch", "global_step", "code_sha", "chunks")
    identity_matches = (
        not any(progress.get(key) != record[key] for key in identity)
        and progress.get("relay_base") == relay_base
        and progress.get("server_created") == server_created
    )
    acknowledged = progress.get("acknowledged_chunks")
    valid_acknowledgement = (
        isinstance(acknowledged, int) and 0 <= acknowledged <= record["chunks"]
    )
    return acknowledged if identity_matches and valid_acknowledgement else 0


def _transfer_progress(
    job_id: str,
    record: dict[str, Any],
    acknowledged_chunks: int,
    status: str,
    relay_base: str,
    server_created: int | None,
) -> dict[str, Any]:
    return {
        "job": job_id,
        "sha": record["sha"],
        "epoch": record["epoch"],
        "global_step": record["global_step"],
        "code_sha": record["code_sha"],
        "relay_base": relay_base,
        "server_created": server_created,
        "bytes": record["bytes"],
        "chunks": record["chunks"],
        "acknowledged_chunks": acknowledged_chunks,
        "status": status,
        "updated_at": time.time(),
    }


def _record_upload_failure(
    archive_path: Path,
    job_id: str,
    record: dict[str, Any],
    acknowledged_chunks: int,
    phase: str,
    error: TransportError,
    relay_base: str,
    server_created: int | None,
    part: int | None = None,
) -> None:
    detail = {
        "job": job_id,
        "sha": record["sha"],
        "epoch": record["epoch"],
        "global_step": record["global_step"],
        "code_sha": record["code_sha"],
        "phase": phase,
        "part": part,
        "acknowledged_chunks": acknowledged_chunks,
        "chunks": record["chunks"],
        "error": error.safe_details(),
        "time": time.time(),
    }
    log_path = archive_path.parent / (job_id + "-transfer-failures.jsonl")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(detail, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    progress_path = _transfer_progress_path(archive_path, job_id, record["sha"])
    state = _transfer_progress(
        job_id,
        record,
        acknowledged_chunks,
        "finish_pending" if phase == "finish" else "failed",
        relay_base,
        server_created,
    )
    state["last_failure"] = detail["error"]
    atomic_json(progress_path, state)


def _upload_archive(
    job_id: str,
    archive_path: Path,
    epoch: int,
    global_step: int,
    code_sha: str,
) -> dict[str, Any]:
    api = _agent_api()
    signature = _file_signature(archive_path)
    with archive_path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            info.st_ino != signature["inode"]
            or info.st_size != signature["bytes"]
            or info.st_mtime_ns != signature["mtime_ns"]
        ):
            raise RuntimeError("backup archive changed before upload")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        sha = digest.hexdigest()
        stream.seek(0)
        chunk_count = (info.st_size + CHUNK_BYTES - 1) // CHUNK_BYTES
        record = {
            "sha": sha,
            "bytes": info.st_size,
            "chunks": chunk_count,
            "epoch": epoch,
            "global_step": global_step,
            "code_sha": code_sha,
            "workers": 0,
        }
        progress_path = _transfer_progress_path(archive_path, job_id, sha)
        relay_base = api.base
        try:
            existing = api.call("/artifact/" + job_id + "/begin", record, timeout=60)
        except TransportError as error:
            _record_upload_failure(
                archive_path,
                job_id,
                record,
                0,
                "begin",
                error,
                relay_base,
                None,
            )
            raise
        server_created = existing.get("created") if isinstance(existing, dict) else None
        if type(server_created) is not int or server_created <= 0:
            server_created = None
        acknowledged_chunks = _read_transfer_progress(
            progress_path, record, relay_base, server_created
        )
        atomic_json(
            progress_path,
            _transfer_progress(
                job_id,
                record,
                acknowledged_chunks,
                "begin_pending",
                relay_base,
                server_created,
            ),
        )

        if isinstance(existing, dict) and existing.get("complete"):
            acknowledged_chunks = chunk_count
        else:
            stream.seek(acknowledged_chunks * CHUNK_BYTES)
            for part in range(acknowledged_chunks, chunk_count):
                block = stream.read(CHUNK_BYTES)
                try:
                    api.call(
                        "/artifact/" + job_id + "/" + sha + "/" + str(part),
                        block,
                        "PUT",
                        timeout=60,
                        attempts=UPLOAD_CHUNK_ATTEMPTS,
                        retry_timeout=UPLOAD_CHUNK_RETRY_TIMEOUT,
                    )
                except TransportError as error:
                    _record_upload_failure(
                        archive_path,
                        job_id,
                        record,
                        acknowledged_chunks,
                        "chunk",
                        error,
                        relay_base,
                        server_created,
                        part,
                    )
                    raise
                acknowledged_chunks = part + 1
                if (
                    acknowledged_chunks % UPLOAD_PROGRESS_INTERVAL == 0
                    or acknowledged_chunks == chunk_count
                ):
                    atomic_json(
                        progress_path,
                        _transfer_progress(
                            job_id,
                            record,
                            acknowledged_chunks,
                            "uploading",
                            relay_base,
                            server_created,
                        ),
                    )
            try:
                api.call("/artifact/" + job_id + "/" + sha + "/finish", {}, timeout=60)
            except TransportError as error:
                _record_upload_failure(
                    archive_path,
                    job_id,
                    record,
                    acknowledged_chunks,
                    "finish",
                    error,
                    relay_base,
                    server_created,
                )
                raise

        atomic_json(
            progress_path,
            _transfer_progress(
                job_id,
                record,
                chunk_count,
                "complete",
                relay_base,
                server_created,
            ),
        )

    result = {
        "job": job_id,
        "sha": sha,
        "bytes": info.st_size,
        "chunks": chunk_count,
        "epoch": epoch,
        "global_step": global_step,
        "code_sha": code_sha,
        "uploaded_at": time.time(),
    }
    state_path = archive_path.parent / (job_id + "-backup-state.json")
    atomic_json(state_path, result)
    archive_path.unlink(missing_ok=True)
    return result


_UPLOAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="k128-h200-backup"
)
_UPLOAD_LOCK = threading.Lock()
_UPLOADS: dict[str, list[concurrent.futures.Future[dict[str, Any]]]] = {}


def backup_run(
    job_id: str,
    output_dir: str | Path,
    epoch: int,
    global_step: int,
    code_sha: str,
    *,
    wait: bool = False,
) -> dict[str, Any]:
    """Archive one recovery point and send it through the agent session.

    A nonblocking call queues the upload on one background worker. A final or
    pause checkpoint should use wait=True so the training process drains it.
    """
    _check_job(job_id)
    if not isinstance(epoch, int) or not 0 <= epoch <= 200:
        raise ValueError("epoch must be between 0 and 200")
    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step must be non-negative")
    if not isinstance(code_sha, str) or not _CODE_SHA.fullmatch(code_sha):
        raise ValueError("code_sha must be a 40-character lowercase commit hash")

    root = Path(output_dir)
    archive_path = _make_archive(root, job_id, epoch, global_step)
    archive_bytes = archive_path.stat().st_size
    future = _UPLOAD_POOL.submit(
        _upload_archive, job_id, archive_path, epoch, global_step, code_sha
    )
    with _UPLOAD_LOCK:
        pending = [item for item in _UPLOADS.get(job_id, []) if not item.done()]
        pending.append(future)
        _UPLOADS[job_id] = pending

    if not wait:
        return {
            "status": "queued",
            "job": job_id,
            "epoch": epoch,
            "global_step": global_step,
            "archive_bytes": archive_bytes,
        }

    with _UPLOAD_LOCK:
        drain = list(_UPLOADS.get(job_id, []))
    results = [item.result() for item in drain]
    with _UPLOAD_LOCK:
        _UPLOADS[job_id] = []
    return results[-1]


def _int_in_range(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and minimum <= value <= maximum


def _int_at_least(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and value >= minimum


def _is_bytes(value: Any) -> bool:
    return isinstance(value, bytes)


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def validate_metadata(metadata: Any) -> dict[str, Any]:
    if not _is_dict(metadata):
        raise ValueError("relay returned no complete artifact")
    sha = metadata.get("sha")
    size = metadata.get("bytes")
    chunks = metadata.get("chunks")
    epoch = metadata.get("epoch")
    step = metadata.get("global_step")
    code_sha = metadata.get("code_sha")
    if (
        not isinstance(sha, str)
        or not re.fullmatch(r"[a-f0-9]{64}", sha)
        or not _int_in_range(size, 1, MAX_ARCHIVE_BYTES)
        or not isinstance(chunks, int)
        or chunks != (size + CHUNK_BYTES - 1) // CHUNK_BYTES
        or chunks > MAX_ARCHIVE_CHUNKS
        or not _int_in_range(epoch, 0, 200)
        or not _int_at_least(step, 0)
        or not isinstance(code_sha, str)
        or not _CODE_SHA.fullmatch(code_sha)
        or metadata.get("complete") is False
    ):
        raise ValueError("relay artifact metadata is invalid")
    return metadata


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def _download_archive(api: API, job_id: str, metadata: dict[str, Any], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("archive target must not be a symlink")
    if target.is_file():
        sha, size = _sha256_file(target)
        if sha == metadata["sha"] and size == metadata["bytes"]:
            return target

    part_path = target.with_name(target.name + "." + uuid.uuid4().hex + ".part")
    digest = hashlib.sha256()
    total = 0
    try:
        with part_path.open("xb") as stream:
            for start in range(0, metadata["chunks"], 4):
                last = min(start + 4, metadata["chunks"])
                for part in range(start, last):
                    block = api.call(
                        "/artifact/" + job_id + "/" + metadata["sha"] + "/" + str(part),
                        raw=True,
                        timeout=60,
                    )
                    if not _is_bytes(block):
                        raise ValueError("relay returned an invalid checkpoint chunk")
                    stream.write(block)
                    digest.update(block)
                    total += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        if total != metadata["bytes"] or digest.hexdigest() != metadata["sha"]:
            raise ValueError("backup archive SHA256 or size mismatch")
        part_path.replace(target)
        _fsync_directory(target.parent)
        return target
    finally:
        part_path.unlink(missing_ok=True)


def _inspect_archive(archive_path: Path) -> list[tarfile.TarInfo]:
    members: list[tarfile.TarInfo] = []
    seen: set[str] = set()
    total = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                path = _safe_relative_name(member.name)
                normalized = path.as_posix()
                if normalized in seen:
                    raise ValueError("archive contains duplicate paths")
                seen.add(normalized)
                if member.isdir():
                    if not _allowed_archive_path(member.name, directory=True):
                        raise ValueError("archive contains an unapproved directory")
                elif member.isfile():
                    if not _allowed_archive_path(member.name):
                        raise ValueError("archive contains an unapproved file")
                    if member.size < 0:
                        raise ValueError("archive contains an invalid file size")
                    total += member.size
                    if total > MAX_UNPACKED_BYTES:
                        raise ValueError("archive expands beyond the recovery limit")
                    contents = archive.extractfile(member)
                    if contents is None:
                        raise ValueError("archive file contents are unavailable")
                    read = 0
                    while block := contents.read(1024 * 1024):
                        read += len(block)
                    if read != member.size:
                        raise ValueError("archive file length is inconsistent")
                else:
                    raise ValueError("archive contains a link or special file")
                members.append(member)
                if len(members) > MAX_ARCHIVE_FILES + 3:
                    raise ValueError("archive contains too many files")
    except (OSError, tarfile.TarError) as error:
        raise ValueError("backup archive is corrupt") from error
    if not any(member.name == "checkpoint.pt" and member.isfile() for member in members):
        raise ValueError("backup archive has no checkpoint.pt")
    return members


def validate_archive(archive_path: str | Path) -> dict[str, Any]:
    """Validate all archive headers and contents without extracting files."""
    members = _inspect_archive(Path(archive_path))
    return {
        "files": sum(member.isfile() for member in members),
        "unpacked_bytes": sum(member.size for member in members if member.isfile()),
    }


def _stage_archive_members(
    archive_path: Path, members: list[tarfile.TarInfo], stage: Path
) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in members:
            if not member.isfile():
                continue
            relative = _safe_relative_name(member.name)
            staged = stage.joinpath(*relative.parts)
            staged.parent.mkdir(parents=True, exist_ok=True)
            contents = archive.extractfile(member)
            if contents is None:
                raise ValueError("archive file contents are unavailable")
            with staged.open("xb") as stream:
                shutil.copyfileobj(contents, stream, length=1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())


def _install_staged_members(
    stage: Path,
    output_dir: Path,
    members: list[tarfile.TarInfo],
    installed: list[Path],
) -> None:
    for member in members:
        if not member.isfile():
            continue
        relative = _safe_relative_name(member.name)
        staged = stage.joinpath(*relative.parts)
        destination = output_dir.joinpath(*relative.parts)
        cursor = output_dir
        for part in relative.parts[:-1]:
            cursor = cursor / part
            if cursor.exists() or cursor.is_symlink():
                if cursor.is_symlink() or not cursor.is_dir():
                    raise ValueError("restore destination contains an unsafe path")
            else:
                cursor.mkdir()
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file():
                raise ValueError("restore destination contains an unsafe file")
            if relative.as_posix() == "poles/epoch-0000.json.gz":
                continue
            if relative.parts[0] in {"poles", "telemetry"}:
                continue
            if _sha256_file(destination) == _sha256_file(staged):
                continue
            if relative.parts[0] in {"contracts", "dataset.json"}:
                raise ValueError("restore contract conflicts with local run files")
            continue

        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".restore")
        try:
            shutil.copyfile(staged, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, destination)
            installed.append(destination)
        except FileExistsError:
            continue
        finally:
            temporary.unlink(missing_ok=True)


def _safe_extract(archive_path: Path, output_dir: Path) -> bool:
    if output_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output directory must be a directory")
    if (output_dir / "checkpoint.pt").exists() or (output_dir / "checkpoint.pt").is_symlink():
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    members = _inspect_archive(archive_path)
    stage = Path(tempfile.mkdtemp(prefix=".k128-restore-", dir=output_dir.parent))
    installed: list[Path] = []
    try:
        _stage_archive_members(archive_path, members, stage)
        _install_staged_members(stage, output_dir, members, installed)
        _fsync_directory(output_dir)
        return (output_dir / "checkpoint.pt").is_file()
    except Exception:
        for path in reversed(installed):
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def restore_backup(
    job_id: str,
    output_dir: str | Path,
    archive_path: str | Path,
) -> bool:
    """Restore the latest completed relay archive into an empty run output.

    Local checkpoints take precedence. Existing epoch-zero poles are preserved,
    so recovery never regenerates or overwrites the initialization record.
    """
    _check_job(job_id)
    output = Path(output_dir)
    if output.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if (output / "checkpoint.pt").exists() or (output / "checkpoint.pt").is_symlink():
        return False
    api = _agent_api()
    metadata = api.call("/artifact/" + job_id + "/latest")
    if metadata is None:
        return False
    metadata = validate_metadata(metadata)
    target = Path(archive_path)
    archive = _download_archive(api, job_id, metadata, target)
    validate_archive(archive)
    return _safe_extract(archive, output)


def download_latest(
    api: API,
    job_id: str,
    archive_path: str | Path,
) -> dict[str, Any] | None:
    """Download and validate one job's latest artifact; caller decides when to ACK."""
    _check_job(job_id)
    metadata = api.call("/artifact/" + job_id + "/latest")
    if metadata is None:
        return None
    metadata = validate_metadata(metadata)
    archive = _download_archive(api, job_id, metadata, Path(archive_path))
    validate_archive(archive)
    return metadata


def download_artifact(
    api: API,
    job_id: str,
    metadata: dict[str, Any],
    archive_path: str | Path,
) -> Path:
    """Download a previously fetched artifact after validating its metadata."""
    _check_job(job_id)
    checked = validate_metadata(metadata)
    archive = _download_archive(api, job_id, checked, Path(archive_path))
    validate_archive(archive)
    return archive
