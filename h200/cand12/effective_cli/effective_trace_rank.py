from collections import defaultdict
import json
from pathlib import Path

def rank(path):
    es=json.loads(Path(path).read_text())['traceEvents']
    cpu={e["args"]["External id"]:e["args"] for e in es if e.get("cat")=="cpu_op" and "External id" in e.get("args",{})}
    bins=defaultdict(lambda:[0,0.]);fam=defaultdict(float)
    for e in es:
        if e.get('cat')!='kernel':continue
        a=e['args'];c=cpu.get(a.get('External id'),{});name=e['name'];dims=c.get('Input Dims',[])
        # Kineto also serializes List[Tensor] as nested dimension lists.
        shape=[]
        def visit(v):
            if not isinstance(v,(list,tuple)):return
            if len(v) in (4,5) and all(type(i) is int for i in v) and v[0]>1:
                shape.append(list(v))
            elif len(shape)<2:
                for child in v:
                    if len(shape)>=2:break
                    visit(child)
        visit(dims)
        shape=shape[:2]
        role=('vback' if 'associative_backward' in name or name=='_path_backward_kernel' else
              'hback' if 'bidirectional' in name.lower() and 'backward' in name.lower() or 'horizontal_path_backward' in name else
              'vfwd' if 'associative_forward' in name or name=='_path_forward_kernel' else
              'hfwd' if 'bidirectional' in name.lower() and 'forward' in name.lower() else 'other')
        key=(role,name[:160],json.dumps(shape),tuple(a.get('grid',[])),tuple(a.get('block',[])))
        bins[key][0]+=1;bins[key][1]+=e['dur']/1000;fam[role]+=e['dur']/1000
    rows=[dict(family=k[0],name=k[1],shape=json.loads(k[2]),grid=k[3],block=k[4],calls=v[0],ms_update=v[1]) for k,v in sorted(bins.items(),key=lambda x:-x[1][1])]
    return dict(total_kernel_ms=sum(fam.values()),kernel_event_count=sum(v[0] for v in bins.values()),families_ms=dict(fam),top=rows[:18])
