"""CPU-only range/cache tests; no external network or corpus downloads."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

spec = importlib.util.spec_from_file_location("apple_token_stream", Path(__file__).with_name("stream.py"))
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


class Response(io.BytesIO):
    def __init__(self, payload, *, status=206, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}
        self.reads = 0

    def read(self, *args):
        self.reads += 1
        return super().read(*args)

    def read1(self, *args):
        return self.read(*args)


class Corpus:
    def __init__(self, root, lengths=(23, 19, 31)):
        self.values = np.arange(sum(lengths), dtype=np.uint16)
        self.payloads = {}
        shards = []
        cursor = 0
        for i, length in enumerate(lengths):
            name = f"document-{i:05d}.bin"
            payload = self.values[cursor:cursor + length].astype("<u2").tobytes()
            self.payloads[name] = payload
            shards.append({"name": name, "bytes": len(payload),
                           "lfs_sha256": hashlib.sha256(payload).hexdigest()})
            cursor += length
        self.manifest = {"repo": "EleutherAI/pile-standard-pythia-preshuffled",
                         "revision": "a" * 40, "shards": shards}
        self.path = Path(root) / "manifest.json"
        self.path.write_text(json.dumps(self.manifest))
        self.calls = []
        self.lock = threading.Lock()

    def opener(self, request, timeout):
        name = request.full_url.rsplit("/", 1)[1]
        payload = self.payloads[name]
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        with self.lock:
            self.calls.append((name, start, end, timeout))
        assert request.get_header("Accept-encoding") == "identity"
        return Response(payload[start:end + 1], headers={
            "Content-Range": f"bytes {start}-{end}/{len(payload)}",
            "Content-Length": str(end - start + 1)})

    def reader(self, root, **kwargs):
        return stream.TokenStream(self.path, root, page_bytes=16, cache_bytes=32,
                                  opener=kwargs.pop("opener", self.opener), retries=1, **kwargs)


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.corpus = Corpus(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_batch(self, reader, start, batch, seq):
        result = reader.batch(start, batch, seq)
        expected = np.array([self.corpus.values[start + row * seq:start + row * seq + seq + 1]
                             for row in range(batch)], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.int64)
        self.assertTrue(result.flags.c_contiguous and result.flags.writeable)

    def test_shapes_page_and_shard_boundaries_random_resume(self):
        with self.corpus.reader(self.root / "cache") as reader:
            for start, batch, seq in [(0, 2, 4), (17, 3, 5), (3, 1, 2),
                                      (41, 3, 6), (0, 7, 10), (72, 1, 1)]:
                if start == 72:
                    with self.assertRaises(EOFError):
                        reader.batch(start, batch, seq)
                else:
                    self.assert_batch(reader, start, batch, seq)
            self.assertLessEqual(reader.stats["peak_cache_bytes"], 32)
            self.assertLessEqual(sum(p.stat().st_size for p in reader.cache_dir.glob("*.bin")), 32)
            self.assertTrue(reader.cached_pages)
            self.assertTrue(all(len(page["sha256"]) == 64 for page in reader.cached_pages))
        self.assertTrue(all(future.done() for future in reader._futures.values()))

    def test_deterministic_new_reader_and_fixed_namespace(self):
        with self.corpus.reader(self.root / "cache") as first:
            expected = first.batch(15, 2, 9)
        with self.corpus.reader(self.root / "cache") as second:
            np.testing.assert_array_equal(second.batch(15, 2, 9), expected)
            self.assertEqual(first.manifest_sha256, second.manifest_sha256)
            self.assertNotEqual(first.cache_dir, second.cache_dir)
            self.assertTrue((second.cache_dir / ".apple-stream-owner.json").exists())

    def test_existing_files_never_modified(self):
        cache = self.root / "cache"
        cache.mkdir()
        existing = cache / "document-00000.bin"
        existing.write_bytes(b"must remain")
        with self.corpus.reader(cache) as reader:
            for cursor in range(0, 60, 3):
                self.assert_batch(reader, cursor, 1, 3)
        self.assertEqual(existing.read_bytes(), b"must remain")

    def test_no_download_on_init_and_cache_hit(self):
        with self.corpus.reader(self.root / "cache") as reader:
            self.assertEqual(self.corpus.calls, [])
            self.assert_batch(reader, 0, 1, 2)
            self.assert_batch(reader, 1, 1, 2)
            first = [call for call in self.corpus.calls if call[:3] == ("document-00000.bin", 0, 15)]
            self.assertEqual(len(first), 1)
            self.assertGreater(reader.stats["cache_hits"], 0)

    def test_next_page_prefetch_runs_without_second_batch(self):
        prefetched = threading.Event()
        def opener(request, timeout):
            if request.get_header("Range") == "bytes=16-31":
                prefetched.set()
            return self.corpus.opener(request, timeout)
        with self.corpus.reader(self.root / "cache", opener=opener) as reader:
            self.assert_batch(reader, 0, 1, 2)
            self.assertTrue(prefetched.wait(1))

    def test_invalid_manifest_and_configuration(self):
        mutations = [lambda x: x.update(revision="main"),
                     lambda x: x["shards"][0].update(bytes=3),
                     lambda x: x["shards"][0].update(lfs_sha256=""),
                     lambda x: x["shards"][0].update(name="../outside.bin"),
                     lambda x: x.update(repo="https://example/private"),
                     lambda x: x["shards"].append(dict(x["shards"][0]))]
        for mutation in mutations:
            document = json.loads(json.dumps(self.corpus.manifest))
            mutation(document)
            self.corpus.path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                self.corpus.reader(self.root / "cache")
        self.corpus.path.write_text(json.dumps(self.corpus.manifest))
        for kwargs in [{"page_bytes": 3}, {"cache_bytes": 2}, {"timeout": 0}, {"retries": 0}]:
            with self.assertRaises(ValueError):
                stream.TokenStream(self.corpus.path, self.root / "cache", **kwargs)

    def test_probe_report_wrapper_accepted(self):
        self.corpus.path.write_text(json.dumps({"dataset": self.corpus.manifest, "status": "passed"}))
        with self.corpus.reader(self.root / "cache") as reader:
            self.assert_batch(reader, 0, 1, 1)

    def test_invalid_batch_and_eof_before_fetch(self):
        with self.corpus.reader(self.root / "cache") as reader:
            for args in [(-1, 1, 2), (0, 0, 2), (0, 1, 0)]:
                with self.assertRaises(ValueError):
                    reader.batch(*args)
            for args in [(73, 1, 1), (0, 8, 10)]:
                with self.assertRaises(EOFError):
                    reader.batch(*args)
            self.assertEqual(self.corpus.calls, [])

    def test_whole_file_200_refused_before_body(self):
        response = Response(b"would be a 30GB response", status=200)
        with self.corpus.reader(self.root / "cache", opener=lambda *a, **k: response) as reader:
            with self.assertRaises(stream.StreamError):
                reader.batch(0, 1, 1)
            self.assertEqual(response.reads, 0)
            self.assertEqual(reader.stats["bytes_downloaded"], 0)
            self.assertEqual(reader._disk_bytes, 0)

    def test_bad_range_length_encoding_truncation_oversize_and_token(self):
        good = {"Content-Range": "bytes 0-15/46", "Content-Length": "16"}
        cases = [({**good, "Content-Range": "bytes 1-16/46"}, b"\0" * 16),
                 ({**good, "Content-Range": "bytes 0-15/999"}, b"\0" * 16),
                 ({**good, "Content-Length": "17"}, b"\0" * 16),
                 ({**good, "Content-Encoding": "gzip"}, b"\0" * 16),
                 (good, b"\0" * 14),
                 ({"Content-Range": "bytes 0-15/46"}, b"\0" * 17),
                 (good, np.array([50277] * 8, dtype="<u2").tobytes())]
        for i, (headers, payload) in enumerate(cases):
            with self.corpus.reader(self.root / f"cache{i}", opener=lambda *a, **k: Response(payload, headers=headers)) as reader:
                with self.assertRaises(stream.StreamError):
                    reader.batch(0, 1, 1)
                self.assertEqual(reader._disk_bytes, 0)
                self.assertFalse(list(reader.cache_dir.glob("*.part")))
                self.assertFalse(list(reader.cache_dir.glob("*.bin")))

    def test_signed_url_never_in_error(self):
        def opener(*args, **kwargs):
            raise urllib.error.URLError("https://signed.example/?secret=NEVERPRINT")
        with self.corpus.reader(self.root / "cache", opener=opener) as reader:
            with self.assertRaises(stream.StreamError) as error:
                reader.batch(0, 1, 1)
            self.assertNotIn("NEVERPRINT", str(error.exception))
            self.assertIsNone(error.exception.__cause__)
            self.assertTrue(error.exception.__suppress_context__)

    def test_bounded_retry_then_success(self):
        calls = []
        def opener(request, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise urllib.error.URLError("temporary")
            return self.corpus.opener(request, timeout)
        with stream.TokenStream(self.corpus.path, self.root / "cache", page_bytes=16,
                                cache_bytes=32, opener=opener, retries=2) as reader:
            self.assert_batch(reader, 0, 1, 1)
            self.assertGreaterEqual(len(calls), 2)

    def test_stop_file_before_download_and_closed_reader(self):
        stop = self.root / "STOP"
        stop.touch()
        reader = self.corpus.reader(self.root / "cache", stop_file=stop)
        try:
            with self.assertRaises(stream.StreamStopped):
                reader.batch(0, 1, 1)
            self.assertEqual(self.corpus.calls, [])
            stop.unlink()
        finally:
            reader.close()
        with self.assertRaises(stream.StreamStopped):
            reader.batch(0, 1, 1)
        reader.close()

    def test_stop_during_response_drops_partial_without_retry(self):
        stop = self.root / "STOP"
        corpus = self.corpus
        class StopsResponse(Response):
            def read(self, count):
                result = super().read(min(count, 2))
                stop.touch()
                return result
        def opener(request, timeout):
            response = corpus.opener(request, timeout)
            return StopsResponse(response.getvalue(), headers=response.headers)
        with self.corpus.reader(self.root / "cache", opener=opener, stop_file=stop) as reader:
            with self.assertRaises(stream.StreamStopped):
                reader.batch(0, 1, 1)
            self.assertEqual(len(corpus.calls), 1)
            self.assertFalse(list(reader.cache_dir.glob("*.part")))

    def test_close_interrupts_retry_wait(self):
        failed = threading.Event()
        def opener(*args, **kwargs):
            failed.set()
            raise urllib.error.URLError("temporary")
        reader = stream.TokenStream(self.corpus.path, self.root / "cache", page_bytes=16,
                                    cache_bytes=32, opener=opener, retries=3)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(reader.batch, 0, 1, 1)
            self.assertTrue(failed.wait(1))
            started = time.monotonic()
            reader.close()
            self.assertLess(time.monotonic() - started, 0.5)
            with self.assertRaises(stream.StreamStopped):
                future.result(timeout=1)

    def test_real_http_range_transport_locally(self):
        payloads = self.corpus.payloads
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                payload = payloads[self.path.rsplit("/", 1)[1]]
                start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                self.wfile.write(payload[start:end + 1])
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def opener(request, timeout):
            local = f"http://127.0.0.1:{server.server_port}/{request.full_url.rsplit('/', 1)[1]}"
            return urllib.request.urlopen(urllib.request.Request(local, headers=dict(request.headers)), timeout=timeout)
        try:
            with self.corpus.reader(self.root / "cache", opener=opener) as reader:
                self.assert_batch(reader, 17, 3, 8)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
