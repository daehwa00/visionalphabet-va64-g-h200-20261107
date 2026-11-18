from pathlib import Path
import base64,hashlib,json,os,zlib

root=Path(os.environ['RUN_ROOT']);p=root/'audit.json'
if not p.exists():raise SystemExit("no audit")
r=json.loads(p.read_text())
small={k:v for k,v in r.items() if k not in ('runs','rankings','errors')}
small['rankings']=[{**x,'trace':{**x['trace'],'top':x['trace']['top'][:9]}} if 'trace' in x else x for x in r.get('rankings',[])]
small['errors']=[dict(label=x['label'],attempt=x['attempt'],detail=x['detail'][-380:]) for x in r.get('errors',[])]
small['runs']={}
for key,v in r['runs'].items():
    fields=('status','plan','ms','img_s','peak_mib','groups','same_micro','update','primitive_sha256','source')
    row={k:v[k] for k in fields if k in v}
    if 'applied' in v:
        row['applied']=[dict(phase=x['phase'],shape=x['shape'],count=len(x['launches']),first=x['launches'][0]) for x in v['applied']]
    if key.startswith("reference-") and 'trace' in v:row['top']=v['trace']['top'][:5]
    small['runs'][key]=row
def packed(obj):return base64.b64encode(zlib.compress(json.dumps(obj,separators=(',',':')).encode(),9)).decode()
blob=packed(small)
if len(blob)>43000:
    # Retain all numerical gates, means and complete pair results; collapse repetitive per-run group timings/check details on screens.
    for k,v in small['runs'].items():
        if k.startswith('screen-'):
            v.pop('groups',None)
            if 'same_micro' in v:v['same_micro']={'passed':v['same_micro']['passed']}
        if 'update' in v and not k.startswith('reference-'):v.pop('update')
        if 'source' in v and not k.startswith('reference-'):v.pop('source')
    blob=packed(small)
print("EFFECTIVE_FIXED_AUDIT_Z",blob)
print("END_STATUS",r.get('boundary'),r.get('winner'),r.get('winner_micro'),len(r['runs']),len(r.get('unresolved_runs',[])))
