"""Return numerical evidence. Model, images, RNG bytes and optimizer tensors stay in the job."""
from pathlib import Path
import base64,json,lzma,os,hashlib
root=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(p.read_text()) if p.exists() else {}
result=read(root/'ordinary-result.json')
result.pop('source_closure',None)
print("ORDINARY_SUMMARY",json.dumps(result,separators=(',',':')))
lines=(root/'cli-e2e.log').read_text().splitlines() if (root/'cli-e2e.log').is_file() else []
receipt=next((v for v in reversed(lines) if v.startswith('FULL_RECEIPT_TRANSFER ')),None)
if receipt:print(receipt)
if not receipt:
    print("NO_TRANSFERRED_RECEIPT")
    for n in ('cold','warm','bundled'):
        p=root/n/'train.log'
        if p.is_file():
            print("CLI_TAIL",n,json.dumps([l[-500:] for l in p.read_text().splitlines()[-12:]]))
a=read(root/'fixed-continuation.json')
if a:
    for label,r in a['runs'].items():
        if 'trace' in r:
            r['trace']['top']=r['trace']['top'][:6]
        if 'applied' in r:
            r['compiled']=[dict(phase=v['phase'],shape=v['shape'],count=len(v['launches']),variant=v['launches'][0].get('output_vjp_variant')) for v in r.pop('applied')]
        if 'screen' in label:r.pop('groups_ms',None)
    s=base64.b64encode(lzma.compress(json.dumps(a,separators=(',',':')).encode(),preset=9)).decode()
    if len(s)+len(receipt or '')>46500:
        # Pair timings and numerical evidence remain; collapse repetitive screenings.
        a['runs']={k:v for k,v in a['runs'].items() if 'screen' not in k}
        s=base64.b64encode(lzma.compress(json.dumps(a,separators=(',',':')).encode(),preset=9)).decode()
    print("FOLLOWUP_XZ",s)
