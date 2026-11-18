"""Safely restore and verify the original train/val files used by the ordinary CLI."""
import hashlib,json,os
from pathlib import Path
import tarfile

root=Path(os.environ["RUN_ROOT"])
fixture=root/'private-input'
manifest=json.loads((fixture/'manifest.json').read_text())
expected={f"data/{r['relative']}":r['sha256'] for group in ('train','val') for r in manifest[group]}
assert len(manifest['train'])==1024 and len(manifest['val'])==100 and len(expected)==1124
assert len({r['sha256'] for r in manifest['train']})==1024
seen=set()
with tarfile.open(fixture/'data.tar','r') as archive:
    for m in archive:
        n=Path(m.name)
        assert not n.is_absolute() and '..' not in n.parts and n.parts and n.parts[0]=='data'
        if m.isdir(): continue
        assert m.isfile() and n.as_posix() in expected and n.as_posix() not in seen
        target=root/n;target.parent.mkdir(exist_ok=True,parents=True)
        src=archive.extractfile(m);assert src is not None
        with target.open('xb') as dest:
            h=hashlib.sha256()
            for piece in iter(lambda:src.read(1024*1024),b''):h.update(piece);dest.write(piece)
        assert h.hexdigest()==expected[n.as_posix()]
        seen.add(n.as_posix())
assert seen==set(expected)
print("REAL_DATASET_VERIFIED",len(seen),len(manifest['train']),len(manifest['val']))
