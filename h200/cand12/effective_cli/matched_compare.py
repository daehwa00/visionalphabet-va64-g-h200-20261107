"""Numerical checks apply only to identical microbatch/composition arms."""
import argparse
import json
from pathlib import Path
import torch
import sys

if len(sys.argv)==4 and sys.argv[1].endswith('.capture.pt'):
    sys.argv[1:]=[p.removesuffix('.capture.pt')+'.json' for p in sys.argv[1:3]]+['--output',sys.argv[3]]

p=argparse.ArgumentParser()
p.add_argument('reference',type=Path)
p.add_argument('candidate',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
ref_doc=json.loads(a.reference.read_text())
cand_doc=json.loads(a.candidate.read_text())
for key in ('microbatch','checkpoint_sha256','fixture_sha256','primitive_sha256'):
    assert ref_doc[key]==cand_doc[key], key
ref=torch.load(a.reference.with_suffix('.capture.pt'),weights_only=False,map_location='cpu')
cand=torch.load(a.candidate.with_suffix('.capture.pt'),weights_only=False,map_location='cpu')
report={'passed':True,'groups':{}}
for group,expected in ref['checks'].items():
    actual=cand['checks'][group]
    assert actual.keys()==expected.keys()
    values={}
    for name,x in expected.items():
        y=actual[name]
        delta=y.float()-x.float()
        absolute=delta.abs().max().item()
        relative=(delta.norm()/x.float().norm().clamp_min(1e-12)).item()
        limit=0.03*x.float().abs().max().item()+1e-6 if group=='grads' else {'outputs':0.05,'updates':3e-5}[group]
        passed=bool(torch.isfinite(y).all()) and absolute<=limit and relative<=(0.005 if group=='outputs' else 0.02)
        values[name]={'max_abs':absolute,'relative_l2':relative,'max_abs_limit':limit,'passed':passed}
        report['passed'] &= passed
    report['groups'][group]=values
for k in ('samples','microbatches','optimizer_steps'):
    assert ref[k]==cand[k],k
report['scalar_checks']={}
for k in ('loss','clip_norm'):
    x,y=float(ref[k]),float(cand[k])
    passed=abs(y-x)<=0.02*abs(x)+1e-6
    report['scalar_checks'][k]={'reference':x,'candidate':y,'passed':passed}
    report['passed'] &= passed
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'counts':{k:len(v) for k,v in report['groups'].items()}}))
raise SystemExit(0 if report['passed'] else 1)
