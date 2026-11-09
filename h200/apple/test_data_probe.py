"""No network or large files: unit tests for the bounded data probe."""
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import data_probe as probe


class Response(io.BytesIO):
    def __init__(self, body=b"", status=206, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.read_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        return super().read(size)


def dataset_info():
    siblings = [
        {"rfilename": name, "size": 30_000_000_000 if i < 20 else 78_336_000,
         "blobId": "a" * 40,
         "lfs": {"sha256": "b" * 64, "size": 30_000_000_000 if i < 20 else 78_336_000}}
        for i, name in enumerate(probe.SHARDS)
    ]
    siblings.append({"rfilename": "document.idx", "size": 1_757_184_042, "blobId": "c" * 40,
                     "lfs": {"sha256": "d" * 64, "size": 1_757_184_042}})
    return {"sha": "e" * 40, "siblings": siblings}


class ProbeTests(unittest.TestCase):
    def test_range_refusal_never_reads_full_body(self):
        response = Response(b"potentially unbounded body", status=200)
        with patch.object(probe, "open_url", return_value=response):
            with self.assertRaisesRegex(probe.ProbeError, "refused"):
                probe.sample_range("https://example.invalid/bin", 4, 30_000_000_000, None)
        self.assertEqual(response.read_calls, 0)
        self.assertTrue(response.closed)

    def test_range_size_and_total_are_enforced(self):
        for header in ("bytes 1-4/30000000000", "bytes 0-3/5", None):
            response = Response(b"abcd", headers={"Content-Range": header})
            with patch.object(probe, "open_url", return_value=response):
                with self.assertRaisesRegex(probe.ProbeError, "Content-Range"):
                    probe.sample_range("https://example.invalid/bin", 4, 30_000_000_000, None)
            self.assertEqual(response.read_calls, 0)

    def test_range_bounded_read_and_short_read(self):
        for body in (b"abc", b"abcde"):
            response = Response(body, headers={"Content-Range": "bytes 0-3/20"})
            with patch.object(probe, "open_url", return_value=response):
                with self.assertRaises(probe.ProbeError):
                    probe.sample_range("https://example.invalid/bin", 4, 20, None)
        response = Response(b"abcd", headers={"Content-Range": "bytes 0-3/20", "Content-Length": "4"})
        with patch.object(probe, "open_url", return_value=response):
            self.assertEqual(probe.sample_range("https://example.invalid/bin", 4, 20, None), b"abcd")

    def test_invalid_token_bytes_and_vocabulary(self):
        for raw in (b"", b"x", b"\xff\xff"):
            with self.assertRaises(probe.ProbeError):
                probe.validate_tokens(raw, {0, 1, 50276})
        result = probe.validate_tokens(b"\x00\x00\x01\x00\x64\xc4", {0, 1, 50276})
        self.assertEqual(result["tokens"], 3)
        self.assertEqual(result["max_token_id"], 50276)

    def test_complete_metadata_and_space_boundary(self):
        manifest = probe.dataset_manifest(dataset_info())
        required = probe.CACHE_BUDGET_BYTES + probe.RESERVE_BYTES
        low = probe.storage_plan(manifest, required - 1)
        enough = probe.storage_plan(manifest, required)
        self.assertFalse(low["cache_space_ok"])
        self.assertTrue(enough["cache_space_ok"])
        self.assertLess(required, 200_000_000_000)
        self.assertEqual(enough["full_corpus_bin_bytes"], 600_078_336_000)
        self.assertEqual(len(enough["prefix_shards"]), 7)
        self.assertEqual(enough["prefix_shards"][-1]["prefix_bytes_used"], 20_000_000_000)

    def test_missing_hash_or_extra_shard_rejected(self):
        info = dataset_info()
        del info["siblings"][0]["lfs"]
        with self.assertRaises(probe.ProbeError):
            probe.dataset_manifest(info)
        info = dataset_info()
        info["siblings"].append({"rfilename": "unexpected.bin"})
        with self.assertRaises(probe.ProbeError):
            probe.dataset_manifest(info)

    def test_atomic_write_read_and_no_overwrite_of_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = b"test-atomic"
            probe.atomic_write(root / "file.bin", raw, None)
            self.assertEqual((root / "file.bin").read_bytes(), raw)
            self.assertFalse((root / "file.bin.part").exists())
            (root / "data-probe").mkdir()
            with patch.object(probe, "metadata") as fetch:
                with self.assertRaises(FileExistsError):
                    probe.run_probe(root, root.parent / "cache", None, 4)
                fetch.assert_not_called()

    def test_stop_prevents_directory_network_and_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stop = root / "STOP"
            stop.touch()
            with patch.object(probe, "metadata") as fetch:
                with self.assertRaises(probe.Stopped):
                    probe.run_probe(root / "new-run", root / "cache", stop, 4)
                fetch.assert_not_called()
            self.assertFalse((root / "new-run").exists())
            with self.assertRaises(probe.Stopped):
                probe.atomic_write(root / "must-not-write", b"no", stop)
            self.assertFalse((root / "must-not-write").exists())

    def test_stop_during_stream_interrupts_next_read(self):
        with tempfile.TemporaryDirectory() as temp:
            stop = Path(temp) / "STOP"
            class StopResponse(Response):
                def read(self, size=-1):
                    result = super().read(size)
                    stop.touch()
                    return result
            response = StopResponse(b"abcdef")
            with self.assertRaises(probe.Stopped):
                probe.read_bounded(response, 6, stop)
            self.assertEqual(response.read_calls, 1)

    def test_full_probe_mocked_and_low_space_reported(self):
        for free, expected in ((1000, "insufficient_space_for_streaming_cache"), (3 * 1024**3, "passed")):
            with self.subTest(free=free), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "run"
                usage = type("Disk", (), {"free": free})()
                with patch.object(probe.shutil, "disk_usage", return_value=usage), \
                     patch.object(probe, "metadata", return_value=dataset_info()), \
                     patch.object(probe, "tokenizer_probe", return_value=({"revision": "a" * 40}, {0, 1})), \
                     patch.object(probe, "sample_range", return_value=b"\x00\x00\x01\x00"):
                    result = probe.run_probe(root, root.parent / "cache", None, 4)
                self.assertEqual(result["status"], expected)
                report = json.loads((root / "data-probe" / "report.json").read_text())
                if expected == "passed":
                    self.assertEqual(len(report["samples"]), 3)
                    self.assertTrue(all(s["tokens"] == 2 for s in report["samples"]))
                    self.assertTrue(all(s["http_bytes_per_second"] > 0 for s in report["samples"]))
                self.assertFalse(report["bulk_corpus_downloaded"])
                self.assertIn("unverified", report["storage"]["output_mount"]["cross_job_persistence"])
                self.assertEqual([p.name for p in (root / "data-probe").iterdir()], ["report.json"])
                self.assertTrue(Path(report["cache"]).is_dir())

    def test_bad_sample_size_before_directory_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new"
            for size in (0, -2, 3, probe.MAX_SAMPLE_BYTES + 2):
                with self.assertRaises(probe.ProbeError):
                    probe.run_probe(root, root.parent / "cache", None, size)
            self.assertFalse(root.exists())

    def test_offsets_are_bounded_and_sent_to_local_http(self):
        payload = bytes(range(256)) * 4
        requested = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                value = self.headers["Range"]
                requested.append(value)
                start, end = map(int, value.removeprefix("bytes=").split("-"))
                body = payload[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/bin"
            for offset in (0, 504, 1008):
                self.assertEqual(probe.sample_range(url, 16, 1024, None, offset=offset), payload[offset:offset + 16])
            self.assertEqual(requested, ["bytes=0-15", "bytes=504-519", "bytes=1008-1023"])
            for offset in (-1, 1010):
                with self.assertRaises(probe.ProbeError):
                    probe.sample_range(url, 16, 1024, None, offset=offset)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_data_cache_cannot_be_inside_output_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "output"
            with self.assertRaisesRegex(probe.ProbeError, "outside"):
                probe.run_probe(root, root / "cache", None, 4)
            self.assertFalse(root.exists())

    def test_http_exception_does_not_expose_url(self):
        with patch.object(probe.urllib.request, "urlopen", side_effect=probe.urllib.error.URLError("https://example/?secret=token")):
            with self.assertRaises(probe.ProbeError) as caught:
                probe.open_url("https://example.invalid", stop_file=None)
            self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
