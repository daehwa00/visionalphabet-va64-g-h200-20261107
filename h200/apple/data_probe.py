#!/usr/bin/env python3
"""Bounded, stdlib-only preflight for official Pythia token data.

This does not download the corpus or prove cross-job persistence. Existing
outputs and source data are never overwritten. Only the small JSON report is
stored in output-root; downloaded data stays under cache-root.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid

DATASET = "EleutherAI/pile-standard-pythia-preshuffled"
TOKENIZER = "EleutherAI/gpt-neox-20b"
SHARDS = [f"document-{i:05d}-of-00020.bin" for i in range(21)]
TARGET_TOKENS = 100_000_000_000
CACHE_BUDGET_BYTES = 1024**3
RESERVE_BYTES = 1024**3
MAX_SAMPLE_BYTES = 64 * 1024**2
MAX_METADATA_BYTES = 8 * 1024**2


class ProbeError(RuntimeError):
    pass


class Stopped(ProbeError):
    pass


def check_stop(stop_file: Path | None) -> None:
    if stop_file is not None and stop_file.exists():
        raise Stopped("STOP file present")


def open_url(url: str, *, stop_file: Path | None, headers: dict | None = None):
    check_stop(stop_file)
    request = urllib.request.Request(url, headers={"User-Agent": "apple70m-data-probe/1", **(headers or {})})
    try:
        return urllib.request.urlopen(request, timeout=30)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Redirect URLs and exceptions can contain signed query parameters.
        raise ProbeError(f"HTTP request failed ({type(exc).__name__})") from None


def read_bounded(response, limit: int, stop_file: Path | None) -> bytes:
    chunks = []
    total = 0
    while True:
        check_stop(stop_file)
        chunk = response.read(min(64 * 1024, limit + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise ProbeError("HTTP body exceeds allowed size")


def metadata(kind: str, repo: str, stop_file: Path | None) -> dict:
    with open_url(f"https://huggingface.co/api/{kind}/{repo}?blobs=true", stop_file=stop_file) as response:
        raw = read_bounded(response, MAX_METADATA_BYTES, stop_file)
    try:
        result = json.loads(raw)
        if not re.fullmatch(r"[0-9a-f]{40}", result["sha"]):
            raise ValueError("invalid revision")
        if not isinstance(result["siblings"], list):
            raise ValueError("invalid siblings")
    except (ValueError, KeyError, TypeError):
        raise ProbeError("Invalid HF repository metadata") from None
    return result


def file_record(entry: dict, *, require_lfs: bool = False) -> dict:
    try:
        size = int(entry["size"])
        name = entry["rfilename"]
        blob = entry["blobId"]
        lfs = entry.get("lfs", {})
        digest = lfs.get("sha256")
        if size <= 0 or not re.fullmatch(r"[0-9a-f]{40}", blob):
            raise ValueError("invalid file metadata")
        if require_lfs and (not re.fullmatch(r"[0-9a-f]{64}", digest or "") or lfs.get("size") != size):
            raise ValueError("invalid LFS metadata")
    except (KeyError, ValueError, TypeError):
        raise ProbeError("Missing or invalid HF file size/hash") from None
    return {"name": name, "bytes": size, "git_blob_sha1": blob, "lfs_sha256": digest}


def dataset_manifest(info: dict) -> dict:
    entries = {e["rfilename"]: e for e in info["siblings"]}
    if sorted(n for n in entries if n.endswith(".bin")) != SHARDS:
        raise ProbeError("Expected exactly the 21 official bin shards")
    try:
        shards = [file_record(entries[name], require_lfs=True) for name in SHARDS]
        idx = file_record(entries["document.idx"], require_lfs=True)
    except KeyError:
        raise ProbeError("Official document.idx missing") from None
    if any(x["bytes"] % 2 for x in shards):
        raise ProbeError("Odd-sized uint16 shard")
    return {"repo": DATASET, "revision": info["sha"], "shards": shards, "index": idx}


def storage_plan(manifest: dict, free_bytes: int) -> dict:
    exact_bytes = 2 * TARGET_TOKENS
    covered = 0
    needed = []
    for shard in manifest["shards"]:
        if covered >= exact_bytes:
            break
        take = min(shard["bytes"], exact_bytes - covered)
        needed.append({**shard, "prefix_bytes_used": take})
        covered += take
    if covered < exact_bytes:
        raise ProbeError("Corpus is smaller than the 100B token target")
    full_bytes = sum(s["bytes"] for s in manifest["shards"])
    full_shard_bytes = sum(s["bytes"] for s in needed)
    idx_bytes = manifest["index"]["bytes"]
    return {
        "target_unique_tokens": TARGET_TOKENS,
        "token_dtype": "little-endian uint16",
        "prefix_bytes": exact_bytes,
        "prefix_shards": needed,
        "complete_prefix_shards_bytes": full_shard_bytes,
        "full_corpus_bin_bytes": full_bytes,
        "full_corpus_tokens": full_bytes // 2,
        "index_bytes": idx_bytes,
        "reserve_bytes": RESERVE_BYTES,
        "available_bytes": free_bytes,
        "cache_budget_bytes": CACHE_BUDGET_BYTES,
        "cache_space_ok": free_bytes >= CACHE_BUDGET_BYTES + RESERVE_BYTES,
        "storage_mode": "bounded streaming cache; corpus sizes are informational only",
        "staged_merge_note": "A separate merged copy needs additional staging space. No merge is performed.",
        "prefix_index_note": "The original index describes the full corpus; a truncated prefix requires a validated truncated/rebuilt index or a contiguous-token reader.",
        "unique_prefix_note": "100B is a unique-token prefix budget, not proof of the trainer's sampling order or number of predicted tokens.",
    }


def sample_range(url: str, size: int, total: int, stop_file: Path | None, offset: int = 0) -> bytes:
    if size <= 0 or offset < 0 or offset + size > total:
        raise ProbeError("Range is outside the declared file size")
    with open_url(url, stop_file=stop_file, headers={"Range": f"bytes={offset}-{offset + size - 1}", "Accept-Encoding": "identity"}) as response:
        if response.status != 206:
            # Do not read even one byte if Range was ignored (potential 30GB body).
            raise ProbeError("Server refused bounded HTTP Range (expected 206)")
        expected = f"bytes {offset}-{offset + size - 1}/{total}"
        if response.headers.get("Content-Range") != expected:
            raise ProbeError("Unexpected Content-Range")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ProbeError("Encoded range response is unsupported")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and content_length != str(size):
            raise ProbeError("Unexpected ranged Content-Length")
        body = read_bounded(response, size, stop_file)
    if len(body) != size:
        raise ProbeError("Short ranged response")
    return body


def validate_tokens(raw: bytes, allowed_ids: set[int]) -> dict:
    if not raw or len(raw) % 2:
        raise ProbeError("Token sample must contain complete uint16 values")
    values = array.array("H")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    invalid = sorted(set(values) - allowed_ids)
    if invalid:
        raise ProbeError(f"Sample has token IDs outside tokenizer vocabulary ({len(invalid)} distinct)")
    return {"bytes": len(raw), "tokens": len(values), "min_token_id": min(values), "max_token_id": max(values), "sha256": hashlib.sha256(raw).hexdigest(), "valid_token_ids": True}


def atomic_write(path: Path, raw: bytes, stop_file: Path | None) -> None:
    check_stop(stop_file)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    check_stop(stop_file)
    os.replace(tmp, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def mount_report(output: Path) -> dict:
    raw = Path("/proc/self/mountinfo").read_text() if Path("/proc/self/mountinfo").exists() else ""
    matches = []
    for line in raw.splitlines():
        fields = line.split()
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
        if output == Path(mount) or Path(mount) in output.parents:
            matches.append((len(mount), line))
    return {"matching_mountinfo": max(matches)[1] if matches else None, "mountinfo": raw, "cross_job_persistence": "unverified; this process can only verify write/read in its current mount namespace"}


def tokenizer_probe(destination: Path, stop_file: Path | None) -> tuple[dict, set[int]]:
    info = metadata("models", TOKENIZER, stop_file)
    entries = {e["rfilename"]: e for e in info["siblings"]}
    records = []
    tokenizer = None
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if name not in entries:
            raise ProbeError(f"Tokenizer metadata missing {name}")
        record = file_record(entries[name])
        if record["bytes"] > MAX_METADATA_BYTES:
            raise ProbeError("Tokenizer file exceeds bounded download size")
        url = f"https://huggingface.co/{TOKENIZER}/resolve/{info['sha']}/{name}"
        with open_url(url, stop_file=stop_file) as response:
            body = read_bounded(response, record["bytes"], stop_file)
        if len(body) != record["bytes"]:
            raise ProbeError("Tokenizer file size differs from metadata")
        git_digest = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        if git_digest != record["git_blob_sha1"]:
            raise ProbeError("Tokenizer Git blob hash mismatch")
        atomic_write(destination / name, body, stop_file)
        record["download_sha256"] = hashlib.sha256(body).hexdigest()
        records.append(record)
        if name == "tokenizer.json":
            tokenizer = json.loads(body)
    try:
        allowed = set(tokenizer["model"]["vocab"].values())
        allowed.update(t["id"] for t in tokenizer.get("added_tokens", []))
        if not allowed or any(not isinstance(i, int) or not 0 <= i <= 65535 for i in allowed):
            raise ValueError("bad vocabulary")
    except (KeyError, TypeError, ValueError):
        raise ProbeError("Tokenizer vocabulary is incompatible with uint16") from None
    return {"repo": TOKENIZER, "revision": info["sha"], "files": records, "distinct_token_ids": len(allowed), "max_token_id": max(allowed)}, allowed


def run_probe(output_root: Path, cache_root: Path, stop_file: Path | None, sample_bytes: int) -> dict:
    check_stop(stop_file)
    if sample_bytes <= 0 or sample_bytes % 2 or sample_bytes > MAX_SAMPLE_BYTES:
        raise ProbeError("sample-bytes must be positive, even, and at most 64MiB")
    output_root = output_root.resolve()
    cache_root = cache_root.resolve()
    if cache_root == output_root or output_root in cache_root.parents:
        raise ProbeError("cache-root must be outside output-root; data is temporary")
    destination = output_root / "data-probe"
    destination.mkdir(parents=True, exist_ok=False)
    cache = cache_root / ("data-probe-" + uuid.uuid4().hex)
    report = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "output": str(destination), "cache": str(cache), "status": "in_progress", "bulk_corpus_downloaded": False}
    try:
        cache.mkdir(parents=True, exist_ok=False)
        free = shutil.disk_usage(cache).free
        report["storage"] = {"cache_free_bytes_before": free, "cache_mount": mount_report(cache), "output_mount": mount_report(destination)}
        if free < CACHE_BUDGET_BYTES + RESERVE_BYTES:
            report["status"] = "insufficient_space_for_streaming_cache"
            atomic_write(destination / "report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(), stop_file)
            return report
        nonce = os.urandom(4096)
        atomic_write(cache / "write-read.bin", nonce, stop_file)
        if (cache / "write-read.bin").read_bytes() != nonce:
            raise ProbeError("Atomic write/read mismatch")
        report["atomic_fsync_write_read"] = "passed"
        manifest = dataset_manifest(metadata("datasets", DATASET, stop_file))
        report["dataset"] = manifest
        report["capacity_plan"] = storage_plan(manifest, free)
        tokenizer, allowed = tokenizer_probe(cache, stop_file)
        report["tokenizer"] = tokenizer
        first = manifest["shards"][0]
        if sample_bytes > first["bytes"]:
            raise ProbeError("Sample is larger than the first shard")
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{manifest['revision']}/{first['name']}"
        regions = [("first", 0), ("middle", ((first["bytes"] - sample_bytes) // 2) // 2 * 2), ("last", first["bytes"] - sample_bytes)]
        report["samples"] = []
        for name, offset in regions:
            check_stop(stop_file)
            started = time.perf_counter()
            raw = sample_range(url, sample_bytes, first["bytes"], stop_file, offset=offset)
            elapsed = time.perf_counter() - started
            sample = validate_tokens(raw, allowed)
            sample.update({"region": name, "offset_bytes": offset, "shard": first["name"], "http_elapsed_seconds": elapsed, "http_bytes_per_second": len(raw) / max(elapsed, 1e-9), "http_mib_per_second": len(raw) / max(elapsed, 1e-9) / 1024**2})
            report["samples"].append(sample)
            atomic_write(cache / ("document-" + name + "-sample.bin"), raw, stop_file)
        report["network_note"] = "Three bounded first/middle/last requests to one shard; includes HTTP setup latency. This is not a sustained training-stream throughput measurement."
        report["status"] = "passed"
    except Stopped:
        # STOP means stop immediately: no extra persistence, network, or checks.
        raise
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc) if isinstance(exc, ProbeError) else type(exc).__name__
        atomic_write(destination / "report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(), stop_file)
        raise
    check_stop(stop_file)
    atomic_write(destination / "report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(), stop_file)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="Unique run directory; data-probe must not already exist")
    parser.add_argument("--cache-root", type=Path, required=True, help="Temporary data cache, normally /app/scratch/<run>; never output-root")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--sample-bytes", type=int, default=1024**2)
    args = parser.parse_args()
    try:
        report = run_probe(args.output_root, args.cache_root, args.stop_file, args.sample_bytes)
    except Stopped:
        print(json.dumps({"status": "stopped"}), flush=True)
        return 130
    except Exception as exc:
        # Never echo HTTP exceptions or redirect URLs with signed credentials.
        print(json.dumps({"status": "failed", "error": str(exc) if isinstance(exc, ProbeError) else type(exc).__name__}), flush=True)
        return 1
    print(json.dumps({"status": report["status"], "report": str(Path(report["output"]) / "report.json"), "cache": report["cache"], "cache_free_bytes": report["storage"]["cache_free_bytes_before"], "dataset_revision": report.get("dataset", {}).get("revision"), "cross_job_persistence": "unverified"}), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
