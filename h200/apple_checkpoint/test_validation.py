"""No-network validation-prefix tests with a real compressed byte stream."""
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import zstandard

import validation


class FakeTokenizer:
    def __init__(self, hook=lambda: None):
        self.hook = hook

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        self.hook()
        return types.SimpleNamespace(ids=[int(x) for x in text.split()])


def stream(documents):
    raw = "".join(json.dumps({"text": text}) + "\n" for text in documents).encode()
    return io.BytesIO(zstandard.ZstdCompressor().compress(raw))


class ValidationTests(unittest.TestCase):
    def test_prefix_eos_truncation_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "tokens"
            source = stream(["1 2", "", "3 4 5", "6"])
            result = validation.write_token_prefix(source, FakeTokenizer(), target, 6)
            expected = bytes.fromhex("010002000000030004000500")
            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(result["tokens"], 6)
            self.assertEqual(result["docs_read"], 2)
            self.assertEqual(result["jsonl_lines_read"], 3)
            self.assertEqual(result["token_file_sha256"], validation.sha256(target))
            second = validation.write_token_prefix(stream(["1 2", "", "3 4 5", "6"]),
                                                   FakeTokenizer(), target, 6)
            self.assertEqual(second, result)
            self.assertFalse(source.closed)  # Caller closes the HTTP connection.

    def test_stop_during_tokenization_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, stop = root / "tokens", root / "STOP"
            target.write_bytes(b"old")
            tokenizer = FakeTokenizer(lambda: stop.touch())
            with self.assertRaises(validation.PreparationStopped):
                validation.write_token_prefix(stream(["1 2"]), tokenizer, target, 2, stop)
            self.assertEqual(target.read_bytes(), b"old")
            self.assertEqual(set(root.iterdir()), {target, stop})

    def test_short_source_and_invalid_token_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "tokens"
            with self.assertRaises(RuntimeError):
                validation.write_token_prefix(stream(["1"]), FakeTokenizer(), target, 10)
            self.assertFalse(target.exists())
            with self.assertRaises(ValueError):
                validation.write_token_prefix(stream(["65536"]), FakeTokenizer(), target, 1)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_only_requested_prefix_is_tokenized(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            tokenizer, "encode", wraps=tokenizer.encode
        ) as encode:
            validation.write_token_prefix(stream(["1 2", "3 4"]), tokenizer,
                                          Path(directory) / "tokens", 3)
            self.assertEqual(encode.call_count, 1)

    def test_preexisting_stop_prevents_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop = root / "STOP"
            stop.touch()
            with patch.object(validation.urllib.request, "urlopen") as request:
                with self.assertRaises(validation.PreparationStopped):
                    validation.prepare_validation(root / "cache", root / "out", stop, 4)
                request.assert_not_called()

    def test_contract_manifest_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = stream(["1 2", "3"])
            response.status = 200
            with patch.object(validation, "verify_source_metadata"), patch.object(
                validation, "load_tokenizer", return_value=FakeTokenizer()
            ), patch.object(validation.urllib.request, "urlopen", return_value=response) as request:
                manifest = validation.prepare_validation(root / "cache", root / "out", target_tokens=4)
                self.assertEqual(manifest["split"], "validation")
                self.assertEqual(manifest["dtype"], "uint16_le")
                self.assertTrue(Path(manifest["token_file"]).is_absolute())
                self.assertEqual(manifest["append_eos"], True)
                self.assertEqual(request.call_args.kwargs["timeout"], 30)
                self.assertTrue(response.closed)
                request.reset_mock()
                self.assertEqual(validation.prepare_validation(root / "cache", root / "out",
                                                               target_tokens=4), manifest)
                request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
