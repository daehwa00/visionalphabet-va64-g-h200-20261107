"""Explore the missing families against a frozen, numerically qualified H200 incumbent."""
import copy
import json
import os
import statistics
from pathlib import Path
import fixed_vjp as f
from effective_batch_search import VJP_NAME, _vjp_contract
from automatic_training_layout import LAYOUT_NAME, PROFILE_NAME, SEGMENT_NAME, DOT_NAME, digest

DISABLED=dict(blocks={},num_warps=4,num_stages=1)
history=[]
transfers=[]
screen_results=[]

def canonical(m,records):
    defaults=f.disc[m]['defaults']
    merged=f.complete(m,records)
    for name in list(merged):
        merged[name]={k:g for k,g in merged[name].items() if g.get('blocks') or defaults.get(name,{}).get(k,{}).get('blocks')}
        if not merged[name]:del merged[name]
    return merged

def register(m,records,label):
    return f.put(m,canonical(m,records),label)

def mapped(m,records,exact):
    out=f.transplant(records,m)
    if exact:
        for n,tab in records.items():
            assert len(out.get(n,{}))==len(tab),(n,m,'unmapped incumbent')
            for src,g in tab.items():
                s=json.loads(src)
                assert any(g==v and all(json.loads(k)[field]==s[field] for field in ('execution_regime','dtype','shape')) for k,v in out[n].items())
    return out

def keys(m,n,h,op=None,coarse=None):
    for k in f.disc[m]['observed'].get(n,[]):
        s=json.loads(k)['shape']
        if s['height']==h and (op is None or s.get('operation')==op) and (coarse is None or s.get('coarse')==coarse):yield k

def write_scope(plan,m,n,h,op,coarse,geometry):
    found=list(keys(m,n,h,op,coarse))
    if found:
        for k in found:plan.setdefault(n,{})[k]=geometry
    return len(found)

def strip_marker(plan,m):
    for k in keys(m,VJP_NAME,56,1,0):plan.get(VJP_NAME,{}).pop(k,None)

def proposals(best,family):
    m,record,_=f.PLANS[best]
    if family=='vjp':
        for variant in (0,2,1): # no marker, two-row gather, column transpose
            plan=copy.deepcopy(record);strip_marker(plan,m)
            if variant:
                if not write_scope(plan,m,VJP_NAME,56,1,0,dict(blocks=dict(VARIANT=variant),num_stages=1,num_warps=4)):continue
                try:_vjp_contract(plan)
                except (KeyError,ValueError):continue
            yield register(m,plan,f'vjp-{variant}')
    if family=='vback':
        # Include #718's complete post-promotion settings, plus adjacent width/order options.
        for coarse in (0,1):
            for bm,w,g,seg,dot in ((4,2,4,1,1),(8,4,4,1,1),(4,2,4,1,0),(8,4,4,1,0),
                    (4,2,1,1,1),(4,2,4,0,0),(4,2,8,1,1),(4,4,4,1,1),(2,2,4,1,1),
                    (8,4,1,1,1),(8,2,4,1,1)):
                geom=dict(blocks=dict(BM=bm,GROUP=g,DOUBLING=1),num_warps=w,num_stages=1)
                plan=copy.deepcopy(record)
                if not write_scope(plan,m,PROFILE_NAME,56,1,coarse,geom):continue
                write_scope(plan,m,SEGMENT_NAME,56,1,coarse,geom if seg else DISABLED)
                write_scope(plan,m,DOT_NAME,56,1,coarse,geom if dot else DISABLED)
                if not coarse and not (dot and seg):strip_marker(plan,m)
                yield register(m,plan,f'vback-c{coarse}-m{bm}w{w}g{g}-s{seg}d{dot}')
    if family in ('vfwd','hback'):
        op=0 if family=='vfwd' else 2
        configs=((8,4,1),(8,4,4),(4,4,4),(4,2,4)) if op==0 else ((16,4,1),(32,4,1),(32,8,1))
        for coarse in (0,1):
            for bm,w,g in configs:
                b=dict(BM=bm,GROUP=g)
                if op==0:b['DOUBLING']=0
                plan=copy.deepcopy(record)
                if write_scope(plan,m,PROFILE_NAME,56,op,coarse,dict(blocks=b,num_warps=w,num_stages=1)):
                    yield register(m,plan,f'{family}-c{coarse}-m{bm}w{w}g{g}')
    if family=='legacy':
        for h in (28,14,7):
            for bm,w,g in ((4,2,4),(8,2,4),(8,4,4),(8,4,1)):
                plan=copy.deepcopy(record)
                geo=dict(blocks=dict(BLOCK_MODES=bm,LINES_PER_GROUP=g,COMPACT_CHANNELS=3),num_warps=w,num_stages=1)
                if write_scope(plan,m,LAYOUT_NAME,h,None,None,geo):
                    yield register(m,plan,f'legacy-{h}-m{bm}w{w}g{g}')
    if family=='micro':
        for micro in transfers:
            if micro!=m and f.observed(micro):
                projected=mapped(micro,record,False)
                try:_vjp_contract(projected)
                except (ValueError,KeyError):continue
                yield register(micro,projected,f'repartition-{micro}')

def align_prior_trace(prior,tag):
    label=prior.get('source')
    r=f.measure(tag,'exact-prior-rank',profile=True,confirm=True)
    assert r is not None and r[0]['reference']['passed'] and r[0]['trace']['top']
    f.audit['initial']=dict(tag=tag,source=label,source_tag=prior.get('source_tag'),trace=r[0]['trace'])

def measured_screen(tag,label):
    ans=f.measure(tag,label)
    if ans is None:return None
    row=ans[0]
    return dict(ms=row['ms'],numeric=row['reference']['passed'],peak_mib=row['peak_mib'])

def run():
    f.snapshot()
    ctx=json.loads((Path(os.environ['COLD_INPUTS'])/'previous-fixed.json').read_text())
    assert ctx['anchor_718']['micro']==1024 and digest(ctx['anchor_718']['records']).startswith('2dd78beafd09f8')
    prior=ctx['incumbent']; m=prior['micro']; assert f.observed(m)
    best=register(m,mapped(m,prior['records'],True),'verified incumbent')
    align_prior_trace(prior,best)
    prior_rejections={tuple(p) for p in ctx.get('closed_719_edges',[])}
    initial_tag=best
    f.champions.add(best)
    transfers.extend(ctx['next_micros'])
    f.audit.update(frontier=history,frontier_screens=screen_results)
    f.audit['anchor_718_digest']=digest(ctx['anchor_718']['records'])
    f.save()
    # Fixed data only. The ordinary winner may use a different augmentation sequence.
    ordinary=ctx.get('ordinary_winner')
    if ordinary and f.observed(ordinary['micro']):
        trial=register(ordinary['micro'],mapped(ordinary['micro'],ordinary['records'],True),'ordinary on fixed group')
        if f.confirm(best,trial):best=trial
    while True:
        start=best
        latest_trace=(f.audit['ranks'][-1] if f.audit['ranks'] else f.audit['initial'])['trace']
        order=['vjp','vback']+sorted(('vfwd','hback'), key=lambda n:-latest_trace['families_ms'].get(n,0))+['legacy','micro']
        group=dict(control=best,families=[]);history.append(group);f.save()
        tested=set()
        for family in order:
            tags=list(dict.fromkeys(proposals(best,family)))
            tags=[t for t in tags if t!=best and t not in tested and t not in f.champions]
            source_tag=prior.get('source_tag') if best==initial_tag else best
            covered=[t for t in tags if (source_tag,t) in prior_rejections]
            tags=[t for t in tags if t not in covered]
            entry=dict(family=family,generated=len(tags),previously_closed=covered,completed=0,confirmed=[]);group['families'].append(entry);f.save()
            if not tags:continue
            tested.update(tags)
            reference=measured_screen(best,f'control-{len(history)}-{family}')
            if not reference or not reference['numeric']:raise RuntimeError('frontier control failed')
            scored=[]
            for tag in tags:
                v=measured_screen(tag,f'screen-{len(history)}-{family}-{tag}')
                record=dict(control=best,tag=tag,label=f.PLANS[tag][2],family=family,control_ms=reference['ms'],outcome=v)
                screen_results.append(record)
                if v:entry['completed']+=1
                if v and v['numeric'] and v['ms']<=reference['ms']*1.007:scored.append((v['ms'],tag))
                f.save()
            for ms,tag in sorted(scored):
                entry['confirmed'].append(tag)
                if f.confirm(best,tag):
                    best=tag;group['promoted']=tag
                    break
            if best!=start:break
        f.save()
        if start==best:break
    assert f.profile_winner(best)
    m,records,label=f.PLANS[best]
    f.audit['observed_shapes']={str(m):{n:[json.loads(k)['shape'] for k in keys] for n,keys in v['observed'].items()} for m,v in f.disc.items()}
    (f.root/'final-discovery.json').write_text(json.dumps(dict(observed=f.disc[m]['observed'],defaults=f.disc[m]['defaults'])))
    last=history[-1]['families']
    f.audit['alternatives_exhausted']=(len(last)==len(order) and all(e['generated']==e['completed'] for e in last)
            and not f.audit['errors'] and not any(p.get('capacity_failed') or p.get('reason') for p in f.audit['pairs']))
    f.audit['fixed_best']=dict(micro=m,records=records,name=label,tag=best)
    f.audit['comparison_plans']=[dict(role='native',micro=m,winner=records,defaults=f.complete(m,{}))]
    f.audit['needs_official']=best!=f.audit['initial']['tag'] or not ctx.get('prior_official_complete',False)
    if m!=128 and f.observed(128):
        p=mapped(128,records,False)
        f.audit['comparison_plans'].append(dict(role='matched_4090_micro_trial',micro=128,winner=canonical(128,p),defaults=f.complete(128,{}),
                 mapped_scopes=sum(map(len,p.values())),parent_scopes=sum(map(len,records.values()))))
    f.save()

if __name__=='__main__':
    try:run()
    except Exception as exc:f.audit['errors'].append(repr(exc)[-400:]);raise
    finally:f.save()
