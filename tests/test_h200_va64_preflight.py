import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from h200.va64.preflight import verify_source


class SourceVerificationTests(unittest.TestCase):
    def test_verified_file_and_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'model.py').write_bytes(b'model')
            (root / 'SOURCE_MANIFEST.json').write_text(json.dumps({'sha256': {'model.py': hashlib.sha256(b'model').hexdigest()}}))
            self.assertEqual(verify_source(root), 1)
            (root / 'model.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_source(root)

    def test_manifest_cannot_escape_source_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('../outside.py', '/absolute.py'):
                (root / 'SOURCE_MANIFEST.json').write_text(json.dumps({'sha256': {name: 'unused'}}))
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    verify_source(root)
