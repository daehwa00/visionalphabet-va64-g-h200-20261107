"""Two unmodified ordinary training invocations, from identical initial checkpoints in isolated roots."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zlib

import torch

root=Path(os.environ['RUN_ROOT'])
source=root/'source'
fixture=root/'private-input'
sys.path[:0]=[str(source/'scripts'),str(source/'src')]
from automatic_training_layout import digest
from effective_batch_search import valid_receipt, _plan_digest
from effective_batch_proposals import _map_proposal
from return_receipt import upload

NAME='VA-Ablation-10-dw-terminal-rmsnorm__seed501.pt'
TARGET='9a84b66a5f86df7601315abc2bca448dfc1598ce872f8a848e2f27e493e3d70f'
control=fixture/'checkpoint.pt'
command=[sys.executable,str(source/'scripts/run_va_eight_ablations.py'),'--root','run','--data-root',str(root/'data'),'--workers','0','--epochs','19']
env=dict(os.environ,WANDB_MODE='disabled',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
         LNET_LAUNCH_CACHE=str(root/'cold-cache'),TORCHINDUCTOR_CACHE_DIR=str(root/'inductor'),TRITON_CACHE_DIR=str(root/'triton-cold'),
         PYTHONPATH=str(source/'scripts')+os.pathsep+str(source/'src'))
cache=Path(env['LNET_LAUNCH_CACHE'])/'effective-updates'
diag=root/'worker-diagnostics';diag.mkdir()
load=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def loadcp(p):return torch.load(p,map_location='cpu',weights_only=True)
initial=loadcp(control);steps0={k:int(v['step']) for k,v in initial['optimizer']['state'].items()}
assert (initial['epoch'],initial['global_step'],len(steps0))==(18,18270,193)
evidence=dict(command=command,checkpoint_sha256=sha(control),initial_epoch=18,initial_step=18270,
              start_lr=[g['lr'] for g in initial['optimizer']['param_groups']],
              transport_schema='markerfix-1',contract_preflight=load(root/'contract-preflight.json'),cuda_preflight=load(root/'cuda-preflight.json'),
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parent,text=True).strip(),
              source_freeze_sha256=sha(Path(__file__).with_name('common-release.sha256')))
def save():(root/'ordinary-result.json').write_text(json.dumps(evidence,indent=2)+'\n')

def monitor_workers(newest,linked):
    for folder in cache.glob('.calibration-*'):
        jobs=list(folder.glob('*.job'))
        if not jobs:continue
        job=max(jobs,key=lambda x:x.name)
        newest.add(job.stem)
        for extension in ('.job','.log','.trace.json','.mismatch.json','.diagnostics.json'):
            src=job.with_suffix(extension);dst=diag/src.name
            if src.name in linked:continue
            try:os.link(src,dst);linked.add(src.name)
            except (FileNotFoundError,FileExistsError):pass
        traces=sorted(diag.glob('*.trace.json'),key=lambda p:p.stat().st_mtime_ns)
        for stale in traces[:-2]:stale.unlink(missing_ok=True)

def arm(name):
    cwd=root/name;dest=cwd/'run/checkpoints'/NAME;dest.parent.mkdir(parents=True)
    shutil.copyfile(control,dest);assert sha(dest)==evidence['checkpoint_sha256']
    log=cwd/'train.log';active=set();linked=set();start=time.perf_counter()
    with log.open('w') as fd:
        proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=fd,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            monitor_workers(active,linked)
            time.sleep(0.5)
    row=dict(returncode=proc.returncode,runtime_s=time.perf_counter()-start,worker_count=len(active))
    evidence[name]=row;save()
    audits=[p for p in (cwd/'run/effective-update-calibration').glob('*.json') if not p.name.endswith('.active.json')]
    if len(audits)==1:row['audit']=load(audits[0]);save()
    assert proc.returncode==0 and len(audits)==1,(name,'ordinary CLI exit or missing audit',proc.returncode)
    cp=loadcp(dest);after={k:int(v['step']) for k,v in cp['optimizer']['state'].items()}
    assert (cp['epoch'],cp['global_step'])==(19,18271) and after.keys()==steps0.keys() and all(after[k]==v+1 for k,v in steps0.items())
    row.update(epoch=cp['epoch'],global_step=cp['global_step'],moment_counters=len(after),checkpoint_sha256=sha(dest))
    save()
    return cp

def actual_contract(rows):
    out={}
    for phase,variant in [('backward',2),('coarse-backward',3)]:
        matches=[r for r in rows if r['phase']==phase and list(r['shape'])==[1024,56,56,160]]
        launches=[dict(x) for r in matches for x in r['launches']]
        out[phase]=dict(expected=variant,launches=launches,
                        passed=bool(launches) and all(x.get('structural_variant')==variant for x in launches))
    return out

def test():
    empty={k:not Path(env[k]).exists() for k in ('LNET_LAUNCH_CACHE','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
    evidence['cold_cache_absent']=empty;assert all(empty.values())
    cold=arm('cold');a=evidence['cold']['audit']
    assert a['status']=='calibrated' and evidence['cold']['worker_count']>0,('cold',a)
    candidates=list(cache.glob('*.json'));assert len(candidates)==1
    receipt=candidates[0];raw=receipt.read_bytes();mtime=receipt.stat().st_mtime_ns
    d=json.loads(raw);key=d['key'];assert key==a['key'] and valid_receipt(d,key)
    assert d['capture_schema']['counts']==dict(outputs=5,grads=193,updates=193) and d['capture_schema']['finite']
    assert d['state_before']==d['state_after'] and {'model','optimizer','gradients','rng','train_generator','mixup'}<=d['state_before'].keys()
    assert d['workload']['effective_samples']==1024 and 'H200' in d['workload']['device'] and d['search_complete']
    initial_bundle=load(source/'src/lnet/validated_effective_updates.json');assert key not in initial_bundle
    evidence.update(receipt_sha256=hashlib.sha256(raw).hexdigest(),key=key,winner=d['winner'],microbatch=d['microbatch'],
                    accumulation=1024//d['microbatch'],valid_receipt=True,state_preserved=True,
                    initial_bundle_keys=sorted(initial_bundle),fixed_input_digest=d['fixed_input_digest'])
    save()
    warm=arm('warm');a=evidence['warm']['audit']
    assert a['source']=='local' and a['key']==key and evidence['warm']['worker_count']==0
    assert receipt.read_bytes()==raw and receipt.stat().st_mtime_ns==mtime
    names=d['capture_schema']['trainable_names'];bad=[];ma=mr=0.
    for n in names:
        c=cold['model'][n].float()-initial['model'][n].float()
        w=warm['model'][n].float()-initial['model'][n].float()
        dx=w-c;ab=dx.abs().max().item();rel=(dx.norm()/c.norm().clamp_min(1e-12)).item()
        if not dx.isfinite().all() or ab>3e-5 or rel>.02:bad.append(n)
        ma,mr=max(ma,ab),max(mr,rel)
    evidence['warm_updates_vs_cold']=dict(count=len(names),failed=bad,max_abs=ma,max_relative_l2=mr)
    catalog=load(source/'src/lnet/effective_update_proposals.json')['proposals']
    source_plan,=[p for p in catalog if p['settings_digest']==TARGET]
    assess=d['assessed_microbatches']['1024'];assert assess['status']=='ok'
    expect=_map_proposal(source_plan,assess['observed'])
    norm=lambda r:_plan_digest(r,assess['defaults'])
    hits=[]
    for trial in d['trials']:
        if trial.get('microbatch')==1024 and 'records' in trial and norm(trial['records'])==norm(expect):
            hits.append(dict(name=trial['plan'],status=trial['status'],pairs=len(trial.get('pairs',[]))))
    pairproof=[dict(pair=p['pair'],order=p['order'],control_ms=p['control_ms'],candidate_ms=p['candidate_ms'],
                    control_peak_mib=p['control_peak_mib'],candidate_peak_mib=p['candidate_peak_mib'],
                    numerical_pass=p['numerical_pass'],actual=actual_contract(p.get('applied',[]))) for p in d['winning_pairs']]
    won=(d['microbatch']==1024 and d['status']=='winner' and norm(d['records'])==norm(expect))
    qualified=(won and bool(hits) and len(pairproof)==3 and max(p['candidate_ms'] for p in pairproof)<min(p['control_ms'] for p in pairproof)
               and all(p['numerical_pass'] and all(v['passed'] for v in p['actual'].values()) for p in pairproof)
               and len(names)==193 and not bad)
    evidence['proposal_provenance_1024']=d.get('proposal_provenance',{}).get('1024',{})
    evidence.update(target=source_plan['name'],target_settings_digest=TARGET,target_trials=hits,target_won=won,
                    winning_pairs=pairproof,qualified=qualified,cold_warm_reuse=True,complete=True)
    save()
    return raw

def compact_failure():
    saved=[]
    for job in sorted(diag.glob('*.job'),key=lambda x:x.stat().st_mtime_ns)[-3:]:
        d=load(job);log=job.with_suffix('.log')
        item=dict(arm=job.stem,micro=d.get('microbatch'),records=d.get('records',{}),profile=d.get('profile'),
                  worker_tail=log.read_text(errors='replace')[-4200:] if log.exists() else '')
        trace=job.with_suffix('.trace.json')
        if trace.exists():
            from tune_path_backward import retained_launches
            actual={}
            for ph in ('backward','coarse-backward'):
                for on in (True,False):
                    try:v=retained_launches(trace,(d['microbatch'],56,56,160),ph,on)
                    except (KeyError,RuntimeError,ValueError,OSError):continue
                    if v:actual[ph]=v;break
            item['actual_leaf']=actual
        for ext in ('.mismatch.json','.diagnostics.json'):
            f=job.with_suffix(ext)
            if f.exists() and f.stat().st_size<100000:item[ext]=load(f)
        saved.append(item)
    evidence['worker_diagnostics']=saved

def main():
    raw=b''
    try:
        raw=test()
    except Exception as exc:
        evidence.update(complete=False,error_type=type(exc).__name__,error=str(exc)[-4500:])
        try:compact_failure()
        except Exception as secondary:evidence['diagnostics_error']=type(secondary).__name__
        checked=[p for p in cache.glob('*.json') if p.is_file()]
        if len(checked)==1:
            value=checked[0].read_bytes()
            if valid_receipt(json.loads(value),json.loads(value).get('key')):raw=value
        save()
    evidence['raw_bytes']=len(raw)
    evidence['raw_sha256']=hashlib.sha256(raw).hexdigest() if raw else None
    save()
    header=json.dumps(evidence,separators=(',',':')).encode()
    payload=zlib.compress(b'VAMF1'+len(header).to_bytes(4,'big')+header+raw,9)
    assert len(header)<=1024**2 and len(payload)<=32*1024**2
    print('ORDINARY_MARKER_VERDICT',json.dumps({k:evidence.get(k) for k in
        ('complete','qualified','winner','microbatch','key','raw_sha256','raw_bytes','error_type')},separators=(',',':')),flush=True)
    upload(payload,'receipt')
    print('PRIVATE_RESULT_SENT',len(payload),flush=True)
    return 0 if evidence.get('qualified') else 2
if __name__=='__main__':raise SystemExit(main())
