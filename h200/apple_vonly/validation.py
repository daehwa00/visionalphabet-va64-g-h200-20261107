"""Prepare a bounded, independent Pile validation prefix without storing raw Pile.

The source SHA is the repository's published SHA, not a claim that the entire
470 MB source was downloaded and hashed. Only the requested token prefix is read.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

SOURCE_REPO = "mit-han-lab/pile-val-backup"
SOURCE_REVISION = "2f5e46ae6a69cf0dce4b12f78241c408936ca0e4"
SOURCE_FILE = "val.jsonl.zst"
SOURCE_SHA256 = "264c875d8bbd355d8daa9d032b75fd8fb91606218bb84dd1155b203fcd5fab92"
TOKENIZER_REPO = "EleutherAI/gpt-neox-20b"
TOKENIZER_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"
TOKENIZER_SHA256 = "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
DEFAULT_TARGET_TOKENS = 2_097_153
HTTP_TIMEOUT = 30


class PreparationStopped(RuntimeError):
    """The supervisor's stop file was observed."""


def check_stop(stop_file: Path | None) -> None:
    if stop_file is not None and stop_file.exists():
        raise PreparationStopped("validation preparation stopped")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_source_metadata() -> None:
    url = f"https://huggingface.co/api/datasets/{SOURCE_REPO}/tree/{SOURCE_REVISION}"
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
        payload = response.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise RuntimeError("unexpectedly large source metadata")
    files = json.loads(payload)
    source = next((row for row in files if row.get("path") == SOURCE_FILE), None)
    if source is None or source.get("lfs", {}).get("oid") != SOURCE_SHA256:
        raise RuntimeError("pinned validation source metadata differs")


def load_tokenizer(cache_root: Path, stop_file: Path | None):
    from tokenizers import Tokenizer

    path = cache_root / "gpt-neox-20b-tokenizer.json"
    if not path.exists() or sha256(path) != TOKENIZER_SHA256:
        url = (f"https://huggingface.co/{TOKENIZER_REPO}/resolve/"
               f"{TOKENIZER_REVISION}/tokenizer.json")
        with tempfile.NamedTemporaryFile("wb", dir=cache_root, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
                    total = 0
                    while True:
                        check_stop(stop_file)
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > 3 * 1024 * 1024:
                            raise RuntimeError("tokenizer exceeds expected size")
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
                if sha256(temporary) != TOKENIZER_SHA256:
                    raise RuntimeError("tokenizer SHA256 mismatch")
                check_stop(stop_file)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.token_to_id("<|endoftext|>") != 0:
        raise RuntimeError("unexpected tokenizer EOS id")
    return tokenizer


def write_token_prefix(compressed_stream, tokenizer, target: Path,
                       target_tokens: int, stop_file: Path | None = None) -> dict:
    """Read the JSONL.zst prefix and atomically publish little-endian uint16 IDs.

    Empty documents are skipped, matching the existing official-validation
    preparation script. Every nonempty document gets exactly one appended EOS.
    """
    import zstandard

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    check_stop(stop_file)
    count = documents = lines = 0
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with zstandard.ZstdDecompressor().stream_reader(
                compressed_stream, read_size=64 * 1024, closefd=False
            ) as reader, io.TextIOWrapper(reader, encoding="utf-8") as text:
                while count < target_tokens:
                    check_stop(stop_file)
                    line = text.readline()
                    if not line:
                        raise RuntimeError(f"validation source ended at {count} tokens")
                    lines += 1
                    document = json.loads(line)
                    if not isinstance(document, dict) or not isinstance(document.get("text"), str):
                        raise ValueError(f"invalid validation document {lines}")
                    if not document["text"]:
                        continue
                    ids = tokenizer.encode(document["text"], add_special_tokens=False).ids
                    ids = list(ids) + [0]
                    if min(ids) < 0 or max(ids) > 65535:
                        raise ValueError(f"token outside uint16 at document {lines}")
                    check_stop(stop_file)
                    values = array("H", ids[: target_tokens - count])
                    if sys.byteorder != "little":
                        values.byteswap()
                    data = values.tobytes()
                    handle.write(data)
                    digest.update(data)
                    count += len(values)
                    documents += 1
            handle.flush()
            os.fsync(handle.fileno())
            check_stop(stop_file)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return {"tokens": count, "docs_read": documents, "jsonl_lines_read": lines,
            "token_file_sha256": digest.hexdigest()}


def prepare_validation(cache_root: Path, output_root: Path,
                       stop_file: Path | None = None,
                       target_tokens: int = DEFAULT_TARGET_TOKENS) -> dict:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    cache_root = Path(cache_root).resolve()
    output_root = Path(output_root).resolve()
    stop_file = None if stop_file is None else Path(stop_file)
    check_stop(stop_file)
    cache_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"pile-validation-{target_tokens}.uint16"
    manifest_path = output_root / "validation-manifest.json"
    contract = {
        "dataset": "The Pile validation", "split": "validation",
        "source_repo": SOURCE_REPO, "source_file": SOURCE_FILE,
        "source_resolved_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "source_sha256_verification": "published LFS metadata; full source not downloaded",
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_resolved_revision": TOKENIZER_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "append_eos": True, "eos_token_id": 0, "dtype": "uint16_le",
        "token_file": str(target), "tokens": target_tokens,
        "selection": "first target_tokens after per-document tokenization and EOS",
    }
    if manifest_path.exists() and target.exists():
        previous = json.loads(manifest_path.read_text())
        if (all(previous.get(key) == value for key, value in contract.items())
                and target.stat().st_size == target_tokens * 2
                and sha256(target) == previous.get("token_file_sha256")):
            return previous
    verify_source_metadata()
    check_stop(stop_file)
    tokenizer = load_tokenizer(cache_root, stop_file)
    url = (f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/"
           f"{SOURCE_REVISION}/{SOURCE_FILE}")
    check_stop(stop_file)
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected validation response {response.status}")
        result = write_token_prefix(response, tokenizer, target, target_tokens, stop_file)
    manifest = {**contract, **result}
    check_stop(stop_file)
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    args = parser.parse_args()
    try:
        manifest = prepare_validation(**vars(args))
    except PreparationStopped:
        raise SystemExit(130)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
