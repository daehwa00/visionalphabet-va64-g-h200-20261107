"""Adaptive H200 qualification on one immutable 1,024-sample optimizer update.

Discovery, candidate screens and confirmations are separate processes. Each
microbatch size has its own numerical reference; BatchNorm is allowed to differ.
"""
from pathlib import Path
import copy, hashlib, json, math, os, statistics, subprocess, sys, time, zlib, base64
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['RUN_ROOT'])
OUT=ROOT/'measurements';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'source'
sys.path[:0]=[str(SOURCE/'scripts'),str(SOURCE/'src')]
from automatic_training_layout import compare_tensors
from effective_trace_rank import rank

torch.set_num_threads(2)
LAY="product_scan_direct_path_backward_layout"
RET="retained_path_step"
SEG="retained_four_row_prefix"
DOT="retained_output_weight_bias_dot"
FILES={LAY:'validated_path_layouts',RET:'validated_retained_steps',SEG:'validated_segmented_scans',DOT:'validated_output_dots'}
audit=dict(runs={},screens=[],confirmations=[],winners=[],coverage=[],rankings=[],errors=[],plans={})
refs={};observed={};plans={};seen=set(); promoted=set()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(): (ROOT/'audit.json').write_text(json.dumps(audit))
def geom(bm,warps,group,op):
    b={"BM":bm,"GROUP":group}
    if op!=2:b["DOUBLING"]=int(op==1)
    return dict(blocks=b,num_warps=warps,num_stages=1)
def setgeom(plan,n,k,g):
    plan.setdefault(n,{})[k]=g
def getscopes(m,n):return [(k,json.loads(k)['shape']) for k in observed[m][n]]
def newplan(m,records,label):
    # Omit explicit values already in effect in the unmodified local bundle.
    cfg={n:{k:g for k,g in table.items() if g!=observed[m][n].get(k)} for n,table in records.items()}
    cfg={n:d for n,d in cfg.items() if d}
    tag=f'{m}-'+digest(cfg)[:14]
    if tag not in plans:
        plans[tag]=dict(micro=m,records=cfg,label=label)
        audit["plans"][tag]=dict(micro=m,label=label,records_digest=digest(cfg))
    return tag
def run(tag,label,*,discover=False,profile=False,confirm=False):
    plan=plans[tag];m=plan['micro'];base=OUT/(label+'.json')
    args=dict(label=label,records=plan['records'],micro=m,discover=discover,profile=profile,output=str(base),
              warmup=3 if confirm else 2,groups=5 if confirm else 3,samples=3 if confirm else 2)
    job=OUT/(label+'.job');job.write_text(json.dumps(args))
    env=dict(os.environ,PYTHONHASHSEED="0",PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(SOURCE/'scripts')+':'+str(SOURCE/'src')+':'+str(HERE),
             TORCHINDUCTOR_CACHE_DIR=str(ROOT/f'ind-{m}'),TRITON_CACHE_DIR=str(ROOT/'tri'),LNET_LAUNCH_CACHE=str(ROOT/'empty-layouts'))
    for attempt in range(2):
        with base.with_suffix('.log').open('w') as log:
            proc=subprocess.run([sys.executable,str(HERE/'isolated.py'),str(job)],env=env,stdout=log,stderr=log)
        data=json.loads(base.read_text()) if base.exists() else {}
        if proc.returncode==0:break
        if data.get('status')=='oom':break
        audit['errors'].append(dict(label=label,attempt=attempt,rc=proc.returncode,detail=base.with_suffix('.log').read_text()[-1200:]))
        if attempt==0:env['TORCHINDUCTOR_COMPILE_THREADS']='1'
    if proc.returncode or data.get('status')!='passed':
        status='oom' if data.get('status')=='oom' else 'execution_error'
        audit['runs'][label]=dict(plan=tag,status=status);save();return None
    assert data['effective_batch']==1024 and data['epoch']==18 and data['global_step']==18270
    assert data['checkpoint_sha256']=='af07043f8407221dfe0d75c0d689d76e573119ad8708ee8df5b29cc564ed1fc6'
    assert data['fixture_tensors_sha256']=={n:FIXTURE['files'][n]['sha256'] for n in ('images.pt','labels.pt','permutation.pt')}
    counts=data['capture_counts']; assert data['capture_finite'] and counts==dict(outputs=5,grads=193,updates=193)
    a=json.loads(base.with_suffix('.audit.json').read_text());assert a['zero']==a['clip']==a['adamw']==1 and a['samples']==1024
    if discover:
        fixed=json.loads((HERE/'content-hashes.json').read_text())
        assert a['mixed_input_sha256']==fixed['mixed'] and a['paired_labels_sha256']==fixed['other_labels']
    row=dict(plan=tag,status='passed',ms=data['median_ms'],img_s=data['images_s'],peak_mib=data['peak_allocated_mib'],groups=data['wall_samples_ms'],
             update=a,primitive_sha256=data['primitive_sha256'],source=data['source_closure_sha256'])
    audit['runs'][label]=row
    if discover: observed[m]=json.loads(base.with_suffix('.scopes.json').read_text())
    if profile:
        row['trace']=rank(base.with_suffix('.trace.json'))
        if plan['records']:
            path=base.with_suffix('.applied.json');assert path.is_file()
            row['applied']=json.loads(path.read_text())
    c=torch.load(base.with_suffix('.capture.pt'),weights_only=False,map_location='cpu')
    assert c['samples']==1024 and c['microbatches']==1024//m and c['optimizer_steps']==1
    if m not in refs:
        assert discover
        refs[m]=c['checks']
    proof=compare_tensors(refs[m],c['checks'])
    row['same_micro']=proof
    base.with_suffix('.capture.pt').unlink()
    print("EFF",label,m,round(row['ms'],2),round(row['peak_mib'],1),proof['passed'],flush=True)
    save()
    return row,c['checks']
def rank_winner(tag,round_name):
    run_result=run(tag,'rank-'+round_name,profile=True)
    if run_result is None:
        audit['rankings'].append(dict(round=round_name,plan=tag,status='trace_failed'));return False
    row,_=run_result;assert row['same_micro']['passed']
    audit['rankings'].append(dict(round=round_name,plan=tag,ms=row['ms'],trace=row['trace']));save();return True
def confirm(best,candidate,name):
    if best==candidate or candidate in promoted:return False
    entry=dict(test=name,control=best,candidate=candidate,pairs=[],passed=False)
    audit['confirmations'].append(entry)
    for i in (1,2,3):
        items={};tensors={}
        for side in (('candidate','control') if i==2 else ('control','candidate')):
            t=candidate if side=='candidate' else best
            v=run(t,f'{name}-{i}-{side}',confirm=True)
            if v is None:
                entry['error']=side;save();return False
            items[side],tensors[side]=v
        c,a=items['control'],items['candidate']
        both=c['same_micro']['passed'] and a['same_micro']['passed']
        cross=compare_tensors(tensors['control'],tensors['candidate']) if plans[best]['micro']==plans[candidate]['micro'] else None
        entry['pairs'].append(dict(i=i,order='candidate-control' if i==2 else 'control-candidate',control_ms=c['ms'],candidate_ms=a['ms'],
                                   numeric=bool(both and (cross is None or cross['passed'])),within_micro=cross))
        save()
    entry['passed']=all(v['numeric'] and v['candidate_ms']<v['control_ms'] for v in entry['pairs'])
    entry['latency_pct']=100*(1-statistics.median(v['candidate_ms'] for v in entry['pairs'])/statistics.median(v['control_ms'] for v in entry['pairs']))
    entry['three_pairs_faster']=entry['passed']
    if entry['passed']:
        entry['passed']=max(v['candidate_ms'] for v in entry['pairs'])<min(v['control_ms'] for v in entry['pairs'])
    save()
    if entry['passed']:
        promoted.add(candidate)
        audit['winners'].append(dict(plan=candidate,against=best,test=name));rank_winner(candidate,name)
    return entry['passed']

def donor(m,device,oldbatch=128):
    records={}
    for name,file in FILES.items():
        data=json.loads((SOURCE/'src/lnet'/(file+'.json')).read_text()).get('records',{}).get(device,{}).get(name,{})
        lookup={digest({k:v for k,v in json.loads(i)['shape'].items() if k!='batch'}):g for i,g in data.items() if json.loads(i)['shape']['batch']==oldbatch}
        for i,sh in getscopes(m,name):
            x=lookup.get(digest({k:v for k,v in sh.items() if k!='batch'}))
            if x is not None:setgeom(records,name,i,x)
    return records

def alternatives(tag):
    """One-factor changes against the current winner, ordered by its measured stage ranking."""
    m=plans[tag]['micro'];base=plans[tag]['records']
    ranked=next((v['trace']['top'] for v in reversed(audit['rankings']) if 'trace' in v),[])
    order=[]
    for z in ranked:
        family=z['family']
        if family not in order and family in ('vback','vfwd','hback'):order.append(family)
    for f in ('vback','vfwd','hback'):
        if f not in order:order.append(f)
    for family in order:
        configs=[]
        if family=='vback':
            for coarse in (False,True):
                # Prefix arithmetic vs output-matrix reduction vs tile/warp.
                for tile,warps,group,segment,dot in [(4,2,4,True,True),(8,4,4,True,True),(4,2,4,True,False),(8,4,4,True,False),
                                                       (4,2,1,True,True),(4,2,4,False,False)]:
                    configs.append((1,int(coarse),tile,warps,group,segment,dot))
        elif family=='vfwd':
            for tile,warps,group in ((8,4,1),(8,4,4),(4,4,4)):configs.append((0,0,tile,warps,group,False,False))
        else:
            for coarse in (0,1):
                for tile,warps in ((16,4),(32,4),(32,8)):
                    configs.append((2,coarse,tile,warps,1,False,False))
        for op,coarse,bm,w,g,sg,dot in configs:
            p=copy.deepcopy(base);geo=geom(bm,w,g,op)
            for name in (RET,SEG,DOT):
                for key,shape in getscopes(m,name):
                    if shape.get('operation')==op and shape.get('coarse')==coarse and shape.get('height')==56:
                        enabled=name==RET or name==SEG and sg or name==DOT and dot
                        v=geo if enabled else {'blocks':{},'num_warps':4,'num_stages':1}
                        setgeom(p,name,key,v)
            yield newplan(m,p,f'{family}-{op}-{coarse}-m{bm}w{w}g{g}-s{int(sg)}d{int(dot)}')
    for size in (28,14,7):
        for bm,w in ((4,2),(8,2),(8,4)):
            p=copy.deepcopy(base)
            for k,h in getscopes(m,LAY):
                if h['height']==size:
                    setgeom(p,LAY,k,dict(blocks=dict(BLOCK_MODES=bm,LINES_PER_GROUP=4,COMPACT_CHANNELS=3),num_warps=w,num_stages=1))
            yield newplan(m,p,f'legacy-{size}-m{bm}w{w}')

def screen_against(best,tags,round_name,allow,lookback):
    # Compare against a contemporaneous control screen; failed screens remain in the audit.
    tags=list(dict.fromkeys(tags))
    if not any(t!=best and t not in promoted and (best,t) not in seen for t in tags):return best
    base_run=run(best,'screen-control-'+round_name)
    if base_run is None:raise RuntimeError("champion screen failed")
    base_ms=base_run[0]['ms'];audit['coverage'].append(dict(round=round_name,baseline=base_ms,policy=allow))
    eligible=[]
    for t in dict.fromkeys(tags):
        if t==best or t in promoted or (best,t) in seen:continue
        seen.add((best,t))
        v=run(t,'screen-'+round_name+'-'+t)
        entry=dict(round=round_name,plan=t,against=best,ms=None if not v else v[0]['ms'],passed=bool(v and v[0]['same_micro']['passed']))
        audit['screens'].append(entry);save()
        if entry['passed'] and entry['ms']<base_ms*(1+allow):
            eligible.append((entry['ms'],t))
    for ms,t in sorted(eligible):
        if ms>base_ms*(1+lookback):continue
        if confirm(best,t,round_name+'-confirm-'+t):
            return t
    return best

def main():
    os.environ.setdefault('WANDB_MODE','disabled')
    # A reference model and its AdamW/RNG are created in each child; the sole source checkpoint is read-only.
    global FIXTURE
    FIXTURE=json.loads((ROOT/'private-input/fixture.json').read_text())
    assert FIXTURE['unique_train_content_sha256']==1024 and FIXTURE['shape']==[1024,3,224,224] and FIXTURE['mixup']['seed']==809
    base={}
    for m in (128,256,512,1024,64):
        if m==64 and base and (128 not in base or min(base,key=lambda x:base[x][1])!=128):continue
        t=f'{m}-'+digest({})[:14];plans[t]=dict(micro=m,records={},label='current');audit["plans"][t]=dict(micro=m,label='current',records_digest=digest({}))
        v=run(t,f'reference-{m}',discover=True,profile=True)
        if v and v[0]['same_micro']['passed']:base[m]=(t,v[0]['ms'])
    if not base:raise RuntimeError('no feasible measured reference')
    best=base[128][0] if 128 in base else min(base.values(),key=lambda x:x[1])[0]
    promoted.add(best)
    rank_winner(best,'start')
    for t,ms in sorted(base.values(),key=lambda x:x[1]):
        if t!=best and ms<=base[plans[best]['micro']][1]*1.005:
            if confirm(best,t,'physical-'+t):
                best=t;break
    all_plans=[]
    for m in base:
        all_plans.append(base[m][0])
        for gpu,b in (('NVIDIA_H200_NVL-sm90',128),('NVIDIA_GeForce_RTX_4090-sm89',128),('NVIDIA_H200_NVL-sm90',1024)):
            all_plans.append(newplan(m,donor(m,gpu,b),f'donor-{gpu}-B{b}'))
    promoted.add(best)
    best=screen_against(best,all_plans,'portfolio',.005,.005)
    # Finish this allocation only after an entire measured pass cannot improve
    # the champion, or after producing a continuation checkpoint for the next allocation.
    # An execution error is retained as unresolved, never interpreted as a flat timing.
    for generation in range(1,5):
        start=best
        candidates=list(dict.fromkeys(alternatives(best)))
        groups={}
        for t in candidates:
            label=plans[t]['label']
            group=label.split('-')[0]
            if group not in ('vback','vfwd','hback','legacy'):group='other'
            groups.setdefault(group,[]).append(t)
        for index,(family,tags) in enumerate(groups.items()):
            if family not in ('vback','vfwd','hback','legacy','other'):continue
            new=screen_against(best,tags,f'kernel-{generation}-{index}-{family}',.004,.004)
            if new!=best:best=new;break
        if best!=start:continue
        # Test the current kernel choices with the other feasible partitions.
        current=plans[best]
        targets=[]
        for m in base:
            if m==current['micro']:continue
            transplanted={}
            for n,rows in current['records'].items():
                lk={digest({k:v for k,v in json.loads(key)['shape'].items() if k!='batch'}):g for key,g in rows.items()}
                for key,shape in getscopes(m,n):
                    g=lk.get(digest({k:v for k,v in shape.items() if k!='batch'}))
                    if g is not None:setgeom(transplanted,n,key,g)
            targets.append(newplan(m,transplanted,'repartition-'+best))
        best=screen_against(best,all_plans+targets,f'repartition-{generation}',.004,.004)
        if best==start:
            audit['boundary']='screened fixed geometry/epilogue families and feasible partitions without a further confirmed win'
            break
    else:
        audit['boundary']='allocation continuation: four accepted generations or further unscreened frontier'
    rank_winner(best,'last')
    audit['winner']=best;audit['winner_records']=plans[best]['records'];audit['winner_micro']=plans[best]['micro']
    audit['capacity']={str(k):v[1] for k,v in base.items()}
    assert len({audit['runs'][f'reference-{m}']['update']['mixed_input_sha256'] for m in base})==1
    audit['unresolved_runs']=[k for k,v in audit['runs'].items() if v['status']=='execution_error']
    save()

if __name__=='__main__':
    try:main()
    finally:
        save()
