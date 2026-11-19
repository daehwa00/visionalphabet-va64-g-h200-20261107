"""Stage a qualified geometry-only proposal and verify actual ordinary screens against the measured fixed plan."""
import hashlib
import json
import os
from pathlib import Path

from effective_batch_proposals import _map_proposal, proposal_candidates
from automatic_training_layout import digest

def load(p):
    return json.loads(p.read_text())
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');tmp.replace(p)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def portable(records,tag):
    packed={}
    for name,tab in sorted(records.items()):
        for key,g in sorted(tab.items()):
            s=json.loads(key)
            assert s['dtype']=='bfloat16' and s['execution_regime']=='compiled'
            shape={k:s['shape'][k] for k in ('height','width','modes','operation','coarse') if k in s['shape']}
            item=dict(family=name,shape=shape,geometry=g)
            k=(name,json.dumps(shape,sort_keys=True))
            assert k not in packed or item==packed[k]
            packed[k]=item
    settings=list(packed.values())
    return dict(name='h200-fixed-final-'+tag,origin=tag,dtype='bfloat16',effective_samples=1024,
                settings=settings,settings_count=len(settings),settings_digest=digest(settings))

def effective(overrides,defaults):
    out={}
    for n in set(defaults)|set(overrides):
        for k in set(defaults.get(n,{}))|set(overrides.get(n,{})):
            g=overrides.get(n,{}).get(k,defaults.get(n,{}).get(k))
            if g and g.get('blocks'):out.setdefault(n,{})[k]=g
    return out

def actual_records(records,observed):
    out={}
    for n,tab in records.items():
        for key,g in tab.items():
            s=json.loads(key)
            keys=[k for k in observed.get(n,[]) if all(json.loads(k)[a]==s[a] for a in ('dtype','shape','execution_regime'))]
            assert len(keys)==1,(n,s['shape'],len(keys))
            out.setdefault(n,{})[keys[0]]=g
    return out

def stage(root):
    src=root/'source'
    path=src/'src/lnet/effective_update_proposals.json'
    before=sha(path);catalog=load(path)
    fixed=load(root/'fixed-continuation.json'); best=fixed['fixed_best']; m=best['micro']
    qualifier=load(root/'matched-table.json')['plans'][0]
    assert (qualifier['role'],qualifier['status'],qualifier['microbatch'])==('native','complete',m)
    assert len(qualifier['pairs'])==3 and all(p['numeric']['passed'] for p in qualifier['pairs'])==qualifier['numeric_passed']
    discovery=load(root/'final-discovery.json')
    assert any(x['name']=='h200-e1024-718' for x in catalog['proposals'])
    selected=portable(best['records'],best['tag'])
    target=effective(best['records'],discovery['defaults'])
    matched=[]
    for p in catalog['proposals']:
        try:other=_map_proposal(p,discovery['observed'])
        except ValueError:continue
        if effective(other,discovery['defaults'])==target:matched.append(p['name'])
    if qualifier['qualified'] and not matched:
        assert not any(p['name']==selected['name'] for p in catalog['proposals'])
        assert effective(_map_proposal(selected,discovery['observed']),discovery['defaults'])==target
        catalog['proposals'].append(selected);write(path,catalog);matched=[selected['name']]
    spec=dict(catalog_before=before,catalog_after=sha(path),effective_records_digest=digest(target),
              final_qualified=qualifier['qualified'],final_proposal_names=matched,fixture_epoch=18,
              fixture_global_step=18270,origin=best['tag'],source_commit=os.environ['R3_JOB_COMMIT'],
              final_proposal=selected if qualifier['qualified'] else None,anchor='h200-e1024-718',
              native_micro=m,strict_frontier_complete=fixed.get('alternatives_exhausted',False))
    write(root/'qualified-catalog.json',spec)
    return spec

def inspect_receipt(doc,root):
    fixed=load(root/'fixed-continuation.json')['fixed_best']
    result={}
    for label,model in [('fixed_final',fixed)]:
        m=model['micro']; discovery=doc['assessed_microbatches'][str(m)]
        assert discovery['status']=='ok'
        defaults=discovery['defaults']
        target=effective(actual_records(model['records'],discovery['observed']),defaults)
        hits=[];confirm=[]
        for t in doc['trials']:
            if t.get('microbatch')!=m or 'records' not in t:continue
            if effective(t['records'],defaults)!=target:continue
            if t.get('screen') and t['screen']['status']=='ok':
                hits.append(dict(name=t['plan'],status=t['status'],numeric=t['reference_checks']['passed'],ms=t['screen']['ms']))
            if t['status']=='confirmed':
                confirm.append(dict(name=t['plan'],pairs=len(t['pairs'])))
        default_hit=target==effective({},defaults) and discovery['reference_checks']['passed']
        result[label]=dict(micro=m,effective_records_digest=digest(target),screens=hits,confirmed=confirm,
                           default_measured=default_hit,ordinary_winner=(doc['microbatch']==m and effective(doc['records'],defaults)==target))
        result[label]['present']=bool(hits or default_hit)
    return result

if __name__=='__main__':
    root=Path(os.environ['RUN_ROOT'])
    result=stage(root)
    print('QUALIFIED_CATALOG',result['catalog_before'],result['catalog_after'],result['final_qualified'],result['final_proposal_names'])
