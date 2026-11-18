"""Independently test registered fixed-group choices and both common output VJPs at effective 1024."""
from pathlib import Path
import base64,copy,hashlib,json,os,random,statistics,sys,zlib
import numpy as np
import torch

root=Path(os.environ['RUN_ROOT']);source=root/'source';sys.path[:0]=[str(source/'scripts'),str(source/'src')]
from automatic_training_layout import candidates,compare_tensors,digest
from effective_batch_search import _run_arm,vjp_candidates,_vjp_applied,_vjp_contract,VJP_NAME
work=root/'fixed-continuation';work.mkdir()
os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(root/'ind-fixed'),TRITON_CACHE_DIR=str(root/'tri-fixed'),LNET_LAUNCH_CACHE=str(root/'fixed-local'))
torch.set_num_threads(4)
audit=dict(runs={},screen=[],pairs=[],ranks=[],winners=[],errors=[]);champions=set()
ref={};disc={};count=0
PLANS={};seen=set()
def save():
    (root/'fixed-continuation.json').write_text(json.dumps(audit))
def put(m,records,label):
    k=f'{m}-'+digest(records)[:16]
    if k not in PLANS:PLANS[k]=(m,records,label)
    return k
def measure(tag,name,discover=False,confirm=False,profile=False):
    global count
    m,records,label=PLANS[tag];count+=1
    suffix=f'{count:03d}-{name}'
    try:
        res,data=_run_arm(work,suffix,m,records,discover=discover,confirm=confirm,profile=profile)
    except (RuntimeError,ValueError) as exc:
        audit['errors'].append(dict(arm=suffix,micro=m,detail=str(exc)[-1300:]))
        save();return None
    row=dict(status=res['status'],micro=m,label=label,tag=tag)
    audit['runs'][suffix]=row
    if res['status']=='oom':save();return None
    assert res['capture_schema']['counts']=={'outputs':5,'grads':193,'updates':193} and res['capture_schema']['finite']
    if discover and m not in ref:disc[m]=res;ref[m]=data
    checks=compare_tensors(ref[m],data)
    row.update(ms=res['ms'],peak_mib=res['peak_mib'],img_s=1024000/res['ms'],groups_ms=res['groups_ms'],reference=checks)
    if profile and records:
        trace=work/(suffix+'.trace.json')
        from effective_trace_rank import rank
        row['trace']=rank(trace)
        row['applied']=res['applied'];assert _vjp_applied(records,res['applied'])
    save();print("FIXED",suffix,m,round(row['ms'],3),checks['passed'],flush=True)
    return row,data
def observed(m):
    if m not in disc:
        if measure(put(m,{},'default'),f'discover-{m}',discover=True) is None:return False
    return True
def transplant(records,m):
    result={}
    for n,table in records.items():
        if n not in disc[m]['observed']:continue
        for orig,g in table.items():
            source_fields=json.loads(orig)
            matching=[key for key in disc[m]['observed'][n] if json.loads(key)['dtype']==source_fields['dtype'] and
                      {k:v for k,v in json.loads(key)['shape'].items() if k!='batch'}=={k:v for k,v in source_fields['shape'].items() if k!='batch'}]
            if len(matching)==1:result.setdefault(n,{})[matching[0]]=g
    return result

def complete(m,records):
    # Inherited bundle parents also matter when verifying the compiled direct VJP.
    # Preserve disabled overrides, and include inherited positive parents for the trace checker.
    effective={n:{k:v for k,v in table.items() if v.get('blocks')} for n,table in disc[m]['defaults'].items()}
    for n,table in records.items():effective.setdefault(n,{}).update(table)
    for n in list(effective):
        if not effective[n]:del effective[n]
    return effective

def confirm(best,candidate,advance=True):
    if best==candidate or (advance and candidate in champions):return False
    if (best,candidate) in seen:
        existing=next(q for q in audit['pairs'] if q['control']==best and q['candidate']==candidate)
        if existing['passed'] and advance and profile_winner(candidate):
            audit['winners'].append(candidate);champions.add(candidate);save();return True
        return existing['passed'] and not advance
    seen.add((best,candidate));q=dict(control=best,candidate=candidate,pairs=[],passed=False);audit['pairs'].append(q)
    for i in (1,2,3):
        by={}
        for side in (('candidate','control') if i==2 else ('control','candidate')):
            tag=best if side=='control' else candidate
            ans=measure(tag,f'{tag}-p{i}-{side}',confirm=True,profile=(i==1 and bool(PLANS[tag][1])))
            if ans is None:q['capacity_failed']=True;save();return False
            by[side]=ans
        c,a=[by[side][0] for side in ('control','candidate')]
        x=compare_tensors(by['control'][1],by['candidate'][1]) if c['micro']==a['micro'] else None
        q['pairs'].append(dict(i=i,order='candidate-control' if i==2 else 'control-candidate',control=c['ms'],candidate=a['ms'],
                                control_peak=c['peak_mib'],candidate_peak=a['peak_mib'],cross=x,
                                numeric=c['reference']['passed'] and a['reference']['passed'] and (x is None or x['passed'])))
        save()
    q['passed']=all(p['numeric'] and p['candidate']<p['control'] for p in q['pairs']) and max(p['candidate'] for p in q['pairs'])<min(p['control'] for p in q['pairs'])
    if q['passed'] and advance:
        if not profile_winner(candidate):
            q['passed']=False;q['reason']='trace or validation unavailable'
        else:
            audit['winners'].append(candidate);champions.add(candidate)
    save();return q['passed']
def profile_winner(tag):
    # The upstream worker profiles explicit records. For a default use a measured, matching parent reassertion.
    m,recs,label=PLANS[tag];profile_tag=tag
    if not recs:
        options=disc[m]['defaults'];active={name:{k:g for k,g in tab.items() if g.get('blocks')} for name,tab in options.items()}
        active={n:t for n,t in active.items() if t}
        profile_tag=put(m,active or {'retained_path_step':{}},'explicit default for profiling')
    ans=measure(profile_tag,'rank-'+tag,profile=True,confirm=True)
    if not ans or not ans[0].get('trace') or not ans[0]['reference']['passed']:return False
    audit['ranks'].append(dict(tag=tag,ms=ans[0]['ms'],trace=ans[0]['trace']));return True
def local_neighbours(best):
    # Recheck launch choices against the new full-update winner, with its VJP
    # fixed. Earlier measurements used the previous parent geometry and search.
    m,plan,_=PLANS[best]
    for op,coarse,variants in ((1,0,[(4,2,1),(4,2,8),(8,4,4),(2,2,4),(4,4,4)]),
                               (1,1,[(4,2,4),(8,4,4)]),
                               (0,0,[(8,4,1),(8,4,4)]),
                               (2,0,[(16,4,1),(32,4,1)])):
        for bm,warps,group in variants:
            trial=copy.deepcopy(plan)
            for n in ('retained_path_step','retained_four_row_prefix','retained_output_weight_bias_dot'):
                for key in disc[m]['observed'][n]:
                    sh=json.loads(key)['shape']
                    if (sh.get('height'),sh.get('operation'),sh.get('coarse'))!=(56,op,coarse):continue
                    # Preserve the incumbent enablement and algorithm, changing only its tile.
                    base=trial.get(n,{}).get(key,disc[m]['defaults'][n].get(key,{}))
                    if not base.get('blocks'):continue
                    g=copy.deepcopy(base);g['blocks'].update(BM=bm,GROUP=group);g['num_warps']=warps
                    trial.setdefault(n,{})[key]=g
            yield put(m,complete(m,trial),f'neighbor-op{op}-c{coarse}-tile{bm}-w{warps}-g{group}')

    for h in (28,14):
        for bm,warps,g in ((4,2,4),(8,2,4),(8,4,1)):
            test=copy.deepcopy(plan)
            for key in disc[m]['observed']['product_scan_direct_path_backward_layout']:
                if json.loads(key)['shape']['height']==h:
                    test.setdefault('product_scan_direct_path_backward_layout',{})[key]=dict(
                        blocks=dict(BLOCK_MODES=bm,LINES_PER_GROUP=g,COMPACT_CHANNELS=3),num_warps=warps,num_stages=1)
            yield put(m,complete(m,test),f'neighbor-{h}-m{bm}w{warps}g{g}')

def snapshot():
    src=root/'private-input';meta=json.loads((src/'fixture.json').read_text())
    im,lab,perm=[torch.load(src/n,weights_only=True,map_location='cpu',mmap=True) for n in ('images.pt','labels.pt','permutation.pt')]
    lam=float(meta['mixup']['lambda'])
    mixed=(lam*im+(1-lam)*im[perm]).contiguous(memory_format=torch.channels_last)
    assert hashlib.sha256(mixed.contiguous().numpy()).hexdigest()=='d24e2c1b32e1d6312705a08b9100ab4f1e94db9d349e661c35d79d3009143805'
    cp=torch.load(src/'checkpoint.pt',weights_only=True,map_location='cpu')
    assert (cp['epoch'],cp['global_step'],len(cp['optimizer']['state']))==(18,18270,193)
    recipe=json.loads(Path(os.environ['CONTRACT']).read_text())['recipe']
    recipe.update(batch_size=128, gradient_accumulation_steps=8, effective_batch_size=1024)
    rng=dict(cpu=cp['torch_rng_state'],cuda=cp['cuda_rng_state'][0],python=cp['python_rng_state'],numpy=np.random.RandomState(809).get_state())
    torch.save(dict(index=0,settings=dict(threads=4,matmul='high',cudnn_benchmark=True,cudnn_tf32=True,cudnn_deterministic=False,
                     deterministic=False,warn_only=False,emulate_precision_casts=False),
                    recipe=recipe,model=cp['model'],optimizer=cp['optimizer'],rng=rng,
                    variant=cp['variant'],model_config={},batch=(mixed,lab,lab[perm],lam),factories={
                         'build':['run_va_eight_ablations','_build'],'config':['va_model_builder','model_config'],
                         'prepare':['va_imagenet100_runtime','prepare_model'],'optimizer':['run_va_eight_ablations','_build_optimizer']}),work/'snapshot.pt')
def main():
    snapshot()
    prior=json.loads((Path(os.environ['COLD_INPUTS'])/'previous-fixed.json').read_text());pm=prior['micro']
    assert observed(pm)
    best=put(pm,complete(pm,transplant(prior['records'],pm)),'previous fixed-data winner')
    champions.add(best)
    profile_winner(best)
    ordinary=root/'winning-receipt.json';d=json.loads(ordinary.read_text()) if ordinary.is_file() else {}
    if d and observed(d['microbatch']):
        candidate=put(d['microbatch'],complete(d['microbatch'],transplant(d['records'],d['microbatch'])),'ordinary winner on fixed group')
        if confirm(best,candidate):best=candidate
    m=PLANS[best][0];ds=disc[m]
    normal_parent=copy.deepcopy(PLANS[best][1]);normal_parent.pop(VJP_NAME,None)
    contract_options=vjp_candidates(ds['observed'],ds['defaults'],{'output-dot-direct-prior':normal_parent,**candidates(ds['observed'])})
    if contract_options:
        first=next(iter(contract_options.values()));no_marker=copy.deepcopy(first);no_marker.pop(VJP_NAME,None)
        base_tag=put(m,complete(m,no_marker),'head-to-head parent')
        eligible=[]
        for variant in (1,2):
            plan=copy.deepcopy(first)
            for marker in plan[VJP_NAME].values():marker['blocks']['VARIANT']=variant
            tag=put(m,complete(m,plan),f'head-to-head VJP {variant}')
            if confirm(base_tag,tag,advance=False):eligible.append(tag)
        if eligible:
            eligible.sort(key=lambda tag: statistics.median(p['candidate'] for p in next(q for q in audit['pairs'] if q['control']==base_tag and q['candidate']==tag)['pairs']))
            top=eligible[0]
            if len(eligible)==2:
                contest=confirm(eligible[1],top,advance=False)
                audit['vjp_head_to_head']=dict(control=eligible[1],candidate=top,strict_three_pair_win=contest)
            if confirm(best,top):best=top
    effective_micros=list(dict.fromkeys([PLANS[best][0],pm,d.get('microbatch',pm),*prior.get('next_micros',[])]))
    round_i=0
    while True:
        round_i+=1
        start=best;scored=[]
        for m in effective_micros:
            if not observed(m):continue
            discovery=disc[m]
            parent=transplant(PLANS[best][1],m)
            # User's new VJPs must be tested on the same geometries as the validated parent. Donor choices remain trials.
            parents={'output-dot-direct-prior':parent,**candidates(discovery['observed'])}
            options=vjp_candidates(discovery['observed'],discovery['defaults'],parents)
            if not options:
                audit.setdefault('vjp_unavailable',{})[str(m)]='no matching direct-dot parent in observed scopes'
                save()
                continue
            dedup=set()
            for label,recs in options.items():
                tag=put(m,complete(m,recs),label)
                if tag in dedup or tag==best or tag in champions:continue
                dedup.add(tag)
                v=measure(tag,f'vjp-screen-{round_i}-{tag}',profile=True)
                if v:
                    audit['screen'].append(dict(round=round_i,tag=tag,ms=v[0]['ms'],passed=v[0]['reference']['passed']))
                    if v[0]['reference']['passed']:scored.append((v[0]['ms'],tag))
        control=measure(best,'control-screen-'+str(round_i))
        if not control:raise RuntimeError('Current fixed winner unavailable')
        for ms,tag in sorted(set(scored)):
            if ms>control[0]['ms']*1.007:continue
            if confirm(best,tag):best=tag;break
        if best==start:
            audit['termination']='no further confirmed full-update improvement among eligible VJP parents/partitions';break
    # Revisit a different mechanism (tile/order and repartitioning) after a
    # layout change, then stop only after a pass yields no further qualified plan.
    tested=set()
    while True:
        old=best
        targets=list(local_neighbours(best))
        for micro in effective_micros:
            if micro!=PLANS[best][0] and observed(micro):
                recs=complete(micro,transplant(PLANS[best][1],micro))
                targets.append(put(micro,recs,'repartition-final'))
        ranked=[]
        for tag in dict.fromkeys(targets):
            if tag in champions or (best,tag) in tested:continue
            tested.add((best,tag))
            trial=measure(tag,'neighbor-screen-'+tag,profile=True)
            if trial and trial[0]['reference']['passed']:ranked.append((trial[0]['ms'],tag))
        c=measure(best,'neighbor-screen-control')
        if c is None:raise RuntimeError('follow-up control unavailable')
        for ms,tag in sorted(ranked):
            if ms>c[0]['ms']*1.005:break
            if confirm(best,tag):best=tag;break
        if best==old:break
    if not profile_winner(best):
        raise RuntimeError('Final fixed-input profile or numerical verification failed')
    audit['alternatives_exhausted']=not audit['errors'] and not audit.get('vjp_unavailable')
    m, records, label = PLANS[best]
    audit['fixed_best']=dict(micro=m,records=records,name=label,tag=best)
    audit['comparison_plans']=[dict(role='native',micro=m,winner=records,defaults=complete(m,{}))]
    save()
    if m != 128 and observed(128):
        mapped=transplant(records,128)
        audit['comparison_plans'].append(dict(role='matched_4090_micro_trial',micro=128,
            winner=complete(128,mapped),defaults=complete(128,{}),
            mapped_scopes=sum(map(len,mapped.values())),parent_scopes=sum(map(len,records.values()))))
        save()
if __name__=='__main__':
    try:main()
    except Exception as exc:audit['errors'].append(str(exc)[-1500:]);raise
    finally:save()
