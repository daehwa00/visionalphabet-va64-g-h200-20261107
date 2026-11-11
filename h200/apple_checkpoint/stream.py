"""Bounded uint16 stream for a pinned, preshuffled Pythia corpus.

Only the immutable manifest order is used; this deliberately differs from the
paper trainer's random sampling. Partial pages cannot verify a whole-shard LFS
hash: HTTPS, pinned revision, exact byte ranges and token bounds are checked;
SHA256 of each completed page is available in ``cached_pages`` for auditing.
"""
from __future__ import annotations

import bisect
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
import hashlib
import http.client
import json
import operator
from pathlib import Path, PurePosixPath
import re
import threading
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np


class StreamError(RuntimeError):
    """A sanitized streaming error; never includes a signed download URL."""


class StreamStopped(StreamError):
    """The local stop file exists or the stream has been closed."""


class TokenStream:
    def __init__(self, manifest_path, cache_root, cache_bytes=2 * 1024**3,
                 stop_file=None, *, page_bytes=64 * 1024**2,
                 opener: Callable | None = None, retries=3, timeout=10.0):
        manifest = json.loads(Path(manifest_path).read_text())
        self.manifest = manifest.get("dataset", manifest)
        revision = self.manifest.get("revision", "")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("A fixed 40-character dataset revision is required")
        repo = self.manifest.get("repo", "EleutherAI/pile-standard-pythia-preshuffled")
        if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("Invalid dataset repository")
        self.shards = self.manifest.get("shards", [])
        if not isinstance(self.shards, list) or not self.shards:
            raise ValueError("Manifest has no ordered shards")
        self._ends = []
        self._urls = []
        names = set()
        total_bytes = 0
        for shard in self.shards:
            name = shard.get("name", "")
            size = shard.get("bytes")
            sha = shard.get("lfs_sha256", "")
            if (not isinstance(name, str) or not name or name in names
                    or PurePosixPath(name).is_absolute()
                    or ".." in PurePosixPath(name).parts or "\\" in name):
                raise ValueError("Invalid or duplicate shard name")
            if type(size) is not int or size <= 0 or size % 2:
                raise ValueError("Each shard must have a positive, even byte size")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("Each shard requires its LFS SHA256")
            names.add(name)
            total_bytes += size
            self._ends.append(total_bytes)
            encoded = urllib.parse.quote(name, safe="/")
            self._urls.append(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{encoded}")
        self.total_tokens = total_bytes // 2
        self.page_bytes = operator.index(page_bytes)
        self.cache_bytes = operator.index(cache_bytes)
        if self.page_bytes < 2 or self.page_bytes % 2:
            raise ValueError("page_bytes must be positive and uint16 aligned")
        if self.cache_bytes < 2 * self.page_bytes:
            raise ValueError("cache_bytes must hold at least two pages for prefetch")
        if type(retries) is not int or retries < 1 or not 0 < timeout <= 60:
            raise ValueError("Invalid retry count or timeout")
        self._opener = opener or urllib.request.urlopen
        self._retries, self._timeout = retries, timeout
        self._stop_file = Path(stop_file) if stop_file else None
        self._closed = threading.Event()
        self._lock = threading.RLock()
        self._batch_lock = threading.Lock()
        self._entries = OrderedDict()
        self._futures: dict[tuple[int, int], Future] = {}
        self._disk_bytes = 0
        self.stats = {"bytes_downloaded": 0, "cache_hits": 0,
                      "wait_seconds": 0.0, "page_downloads": 0,
                      "peak_cache_bytes": 0}
        canonical = json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode()
        self.manifest_sha256 = hashlib.sha256(canonical).hexdigest()
        root = Path(cache_root)
        root.mkdir(parents=True, exist_ok=True)
        self.cache_dir = root / f"apple-{self.manifest_sha256[:16]}-{uuid.uuid4().hex}"
        self.cache_dir.mkdir(mode=0o700)
        (self.cache_dir / ".apple-stream-owner.json").write_text(json.dumps({
            "manifest_sha256": self.manifest_sha256, "revision": revision,
            "page_bytes": self.page_bytes, "cache_bytes": self.cache_bytes}))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apple-stream")

    def _check_stop(self):
        if self._closed.is_set() or (self._stop_file and self._stop_file.exists()):
            raise StreamStopped("Token stream stopped")

    def _key_at(self, byte_offset):
        shard = bisect.bisect_right(self._ends, byte_offset)
        base = self._ends[shard - 1] if shard else 0
        return shard, (byte_offset - base) // self.page_bytes

    def _page_bounds(self, key):
        shard, page = key
        offset = page * self.page_bytes
        return offset, min(self.page_bytes, self.shards[shard]["bytes"] - offset)

    def _next_key(self, key):
        shard, page = key
        if (page + 1) * self.page_bytes < self.shards[shard]["bytes"]:
            return shard, page + 1
        return (shard + 1, 0) if shard + 1 < len(self.shards) else None

    @property
    def cached_pages(self):
        with self._lock:
            return [{"shard": self.shards[k[0]]["name"], "offset": k[1] * self.page_bytes,
                     "bytes": e[1], "sha256": e[2]} for k, e in self._entries.items()]

    def _reserve(self, size):
        # A single worker owns all downloads; reads hold this lock until copied.
        with self._lock:
            while self._disk_bytes + size > self.cache_bytes:
                if not self._entries:
                    raise StreamError("Cache budget cannot hold the requested page")
                _, (path, old_size, _) = self._entries.popitem(last=False)
                path.unlink()
                self._disk_bytes -= old_size
            self._disk_bytes += size  # Includes a partial page, so disk use stays bounded.
            self.stats["peak_cache_bytes"] = max(self.stats["peak_cache_bytes"], self._disk_bytes)

    def _download(self, key):
        shard, page = key
        offset, size = self._page_bounds(key)
        path = self.cache_dir / f"{shard:05d}-{page:09d}.bin"
        partial = path.with_suffix(".part")
        self._check_stop()
        self._reserve(size)
        completed = False
        try:
            for attempt in range(self._retries):
                self._check_stop()
                try:
                    req = urllib.request.Request(self._urls[shard], headers={
                        "Range": f"bytes={offset}-{offset + size - 1}",
                        "Accept-Encoding": "identity", "User-Agent": "APPLE70M-stream/1"})
                    with self._opener(req, timeout=self._timeout) as response:
                        # Refuse a whole-file response BEFORE reading its body.
                        if response.status != 206:
                            raise StreamError("Server refused bounded HTTP Range (expected 206)")
                        expected = f"bytes {offset}-{offset + size - 1}/{self.shards[shard]['bytes']}"
                        if response.headers.get("Content-Range") != expected:
                            raise StreamError("Unexpected Content-Range")
                        length = response.headers.get("Content-Length")
                        if length is not None and length != str(size):
                            raise StreamError("Unexpected Content-Length")
                        if response.headers.get("Content-Encoding", "identity") != "identity":
                            raise StreamError("Unexpected Content-Encoding")
                        digest = hashlib.sha256()
                        received = 0
                        read = getattr(response, "read1", response.read)
                        with partial.open("xb") as output:
                            while received < size:
                                self._check_stop()
                                block = read(min(64 * 1024, size - received))
                                if not block or len(block) > size - received:
                                    raise StreamError("Truncated or oversized range body")
                                output.write(block)
                                digest.update(block)
                                received += len(block)
                                with self._lock:
                                    self.stats["bytes_downloaded"] += len(block)
                            # Bounded extra read detects a larger body without buffering it.
                            if read(1):
                                raise StreamError("Oversized range body")
                        self._check_stop()
                    values = np.memmap(partial, mode="r", dtype="<u2")
                    valid = bool(np.all(values <= 50276))
                    del values
                    if not valid:
                        raise StreamError("Out-of-vocabulary uint16 token in stream page")
                    self._check_stop()
                    partial.rename(path)
                    with self._lock:
                        self._entries[key] = (path, size, digest.hexdigest())
                        self.stats["page_downloads"] += 1
                    completed = True
                    return
                except StreamStopped:
                    raise
                except (StreamError, OSError, urllib.error.URLError, http.client.HTTPException, ValueError):
                    partial.unlink(missing_ok=True)
                    if attempt + 1 == self._retries:
                        raise StreamError(f"Range download failed after {self._retries} attempts "
                                          f"(shard {shard}, page {page})") from None
                    # Event waits interrupt close; poll the external stop file as well.
                    deadline = time.monotonic() + min(2 ** attempt, 4)
                    while time.monotonic() < deadline:
                        self._check_stop()
                        self._closed.wait(min(0.1, max(0., deadline - time.monotonic())))
        finally:
            partial.unlink(missing_ok=True)
            if not completed:
                with self._lock:
                    self._disk_bytes -= size

    def _future(self, key):
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.stats["cache_hits"] += 1
                return None
            future = self._futures.get(key)
            if future is None:
                self._check_stop()
                future = self._executor.submit(self._download, key)
                self._futures[key] = future
            return future

    def _copy(self, key, offset, count, destination):
        future = self._future(key)
        if future is not None:
            started = time.monotonic()
            try:
                while True:
                    self._check_stop()
                    try:
                        future.result(timeout=0.1)
                        break
                    except FutureTimeout:
                        continue
            finally:
                with self._lock:
                    self.stats["wait_seconds"] += time.monotonic() - started
                    if future.done():
                        self._futures.pop(key, None)
        with self._lock:
            # Only one batch reader exists, and prefetch is submitted after copying.
            path, _, _ = self._entries[key]
            self._entries.move_to_end(key)
            with path.open("rb") as source:
                source.seek(offset)
                raw = source.read(count * 2)
            if len(raw) != count * 2:
                raise StreamError("Cached page length changed")
            destination[:] = np.frombuffer(raw, dtype="<u2")

    def _prefetch(self, key):
        if key is None:
            return
        with self._lock:
            # Completed failures remain until demanded so errors are not silently lost.
            # Successful futures can be forgotten: their page is in _entries.
            for old, future in list(self._futures.items()):
                if future.done() and future.exception() is None:
                    self._futures.pop(old)
                elif old != key and future.cancel():
                    self._futures.pop(old)
            # At most one active prefetch; demand reads may queue behind that page.
            if not any(not f.done() for f in self._futures.values()):
                self._future(key)

    def batch(self, start_token, batch_size, seq_len):
        """Read deterministic adjacent sequences; labels use the last target token.

        Row i starts at start_token+i*seq_len. Output is a writable, contiguous
        int64 [batch_size, seq_len+1] array (adjacent rows share a token value).
        """
        start_token, batch_size, seq_len = map(operator.index, (start_token, batch_size, seq_len))
        if start_token < 0 or batch_size <= 0 or seq_len <= 0:
            raise ValueError("Invalid token cursor or batch shape")
        token_count = batch_size * seq_len + 1
        if start_token + token_count > self.total_tokens:
            raise EOFError("Batch would exceed the manifest corpus")
        with self._batch_lock:
            self._check_stop()
            tokens = np.empty(token_count, dtype=np.int64)
            position, written = start_token * 2, 0
            while written < token_count:
                key = self._key_at(position)
                base = self._ends[key[0] - 1] if key[0] else 0
                page_offset = position - base - key[1] * self.page_bytes
                _, page_size = self._page_bounds(key)
                count = min((page_size - page_offset) // 2, token_count - written)
                self._copy(key, page_offset, count, tokens[written:written + count])
                position += count * 2
                written += count
                self._prefetch(self._next_key(key))
            return np.lib.stride_tricks.sliding_window_view(tokens, seq_len + 1)[::seq_len].copy()

    def close(self):
        """Stop prefetch, wait for its bounded socket timeout, preserve own cache."""
        self._closed.set()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
