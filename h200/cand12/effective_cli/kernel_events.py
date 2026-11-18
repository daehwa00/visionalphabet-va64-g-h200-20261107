"""Return ordered Kineto CUDA kernel events. Shape comes only from correlated CPU Input Dims."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

FIELDS=['name','duration_us','grid','block','registers_per_thread','shape']
LIMIT=32*1024**2

def dimensions(value):
    if type(value) is int and value>=0:return value
    if not isinstance(value,list):return None
    items=[dimensions(x) for x in value]
    return items if all(x is not None for x in items) else None

def extract(path,arm):
    raw=path.read_bytes();proof=arm['profile_outside_timer']
    sha=hashlib.sha256(raw).hexdigest();assert sha==proof['trace_sha256']
    events=json.loads(raw)['traceEvents']
    shapes={}
    for e in events:
        a=e.get('args',{});key=a.get('External id')
        if e.get('cat')=='cpu_op' and key is not None and 'Input Dims' in a:
            dims=dimensions(a['Input Dims'])
            if key not in shapes:shapes[key]=dims
            elif shapes[key]!=dims:shapes[key]=None
    rows=[];missing={k:0 for k in FIELDS[2:]}
    for e in events:
        if e.get('cat')!='kernel':continue
        a=e.get('args',{});duration=e['dur'];assert math.isfinite(duration) and duration>=0
        row=[e['name'],duration,dimensions(a.get('grid')),dimensions(a.get('block')),
             a.get('registers per thread'),shapes.get(a.get('External id'))]
        assert row[4] is None or type(row[4]) is int
        for k,val in zip(FIELDS[2:],row[2:]):
            if val is None or val==[]:missing[k]+=1
        rows.append(row)
    ms=math.fsum(x[1] for x in rows)/1000
    assert len(rows)==proof['kernel_event_count'] and math.isclose(ms,proof['total_kernel_ms'],rel_tol=1e-9)
    return dict(trace_sha256=sha,kernel_event_count=len(rows),total_kernel_ms=ms,unavailable_fields=missing,events=rows)

def pack(root,limit=LIMIT):
    plans=json.loads((root/'matched-table.json').read_text())['plans']
    data=dict(schema=1,fields=FIELDS,duration_unit='us',raw_trace_retrieved=False,
              shape_definition='All Input Dims of the CPU operator matched by External id; null when unavailable or ambiguous.',
              unprofiled_pairs=[2],traces=[])
    for plan in plans:
        for i in (1,3):
            for side in ('control','winner'):
                role=plan['role'];path=root/('matched-'+role)/f'{i}-{side}.trace.json'
                info=extract(path,plan['pairs'][i-1]['arms'][side])
                data['traces'].append(dict(role=role,micro=plan['microbatch'],pair=i,arm=side,**info))
    def encode():return gzip.compress(json.dumps(data,separators=(',',':'),allow_nan=False).encode(),compresslevel=6,mtime=0)
    blob=encode()
    # Prefer keeping complete control/winner pairs for the native winner if storage is constrained.
    for role,micro,i in reversed(list(dict.fromkeys((v['role'],v['micro'],v['pair']) for v in data['traces']))):
        if len(blob)<=limit:break
        for v in data['traces']:
            if (v['role'],v['micro'],v['pair'])==(role,micro,i):
                del v['events'];v['omitted_reason']='result channel byte limit'
        blob=encode()
    assert len(blob)<=limit
    manifest=[{k:v for k,v in t.items() if k!='events'}|dict(recovered_events=len(t.get('events',[]))) for t in data['traces']]
    return blob,manifest

def main():
    root=Path(os.environ['RUN_ROOT'])
    status=dict(raw_trace_retrieved=False,format='gzip JSON CUDA kernel event table')
    try:
        blob,traces=pack(root)
        status.update(traces=traces,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
        from return_receipt import upload
        upload(blob,'kernel-events');status['transfer']='finished'
    except Exception as exc:
        status.update(transfer='incomplete',error_type=type(exc).__name__)
    (root/'kernel-events-status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('KERNEL_EVENTS_STATUS',json.dumps(status,separators=(',',':')),flush=True)
if __name__=='__main__':main()
