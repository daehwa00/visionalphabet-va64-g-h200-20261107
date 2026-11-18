"""Bounded numeric handoff from the fixed frontier; raw captures remain local."""
import base64,json,lzma,os
from pathlib import Path
root=Path(os.environ['RUN_ROOT'])
def marker(name,d):
    blob=base64.b64encode(lzma.compress(json.dumps(d,separators=(',',':')).encode(),preset=9)).decode()
    print(name,blob)
catalog=root/'qualified-catalog.json'
if catalog.is_file():marker('R3_QUALIFIED_PROPOSAL_XZ',json.loads(catalog.read_text()))
ordinary=root/'ordinary-result.json'
if ordinary.is_file():
    r=json.loads(ordinary.read_text())
    marker('R3_ORDINARY_STATUS_XZ',{k:r[k] for k in ('winner','micro','valid_receipt','receipt_sha256','actual_fixed_candidate_screens','catalog_only_warm_check') if k in r})
p=root/'matched-table.json'
if p.is_file():marker('MATCHED_FIXED_TABLE_XZ',json.loads(p.read_text()))
p=root/'fixed-continuation.json'
if p.is_file():
    a=json.loads(p.read_text())
    d={k:a[k] for k in ('initial','anchor_718_digest','observed_shapes','frontier','frontier_screens','winners','alternatives_exhausted','fixed_best','needs_official','errors') if k in a}
    d['ranks']=[dict(tag=r['tag'],ms=r['ms'],families=r['trace']['families_ms'],top=r['trace']['top'][:5]) for r in a['ranks']]
    d['pairs']=[dict(control=v['control'],candidate=v['candidate'],passed=v['passed'],
        pairs=[{k:p[k] for k in ('order','i','control','candidate','control_peak','candidate_peak','numeric')} for p in v['pairs']]) for v in a['pairs']]
    d['finite_counts']={'outputs':5,'grads':193,'updates':193}
    marker('H200_FRONTIER_XZ',d)
