"""Cold, warm and cold-store bundled reuse through the unmodified training CLI."""
from pathlib import Path
import base64
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
import time

import torch

root=Path(os.environ["RUN_ROOT"]);src=root/"source";fixture=root/"private-input"
sys.path[:0]=[str(src/"scripts"),str(src/"src")]
from effective_batch_search import valid_receipt
from automatic_training_layout import digest
result_path=root/"ordinary-result.json"
command=[sys.executable,str(src/"scripts/run_va_eight_ablations.py"),'--root','run','--data-root',str(root/'data'),'--workers','0','--epochs','19']
checkpoint=fixture/'checkpoint.pt'
model_name='VA-Ablation-10-dw-terminal-rmsnorm__seed501.pt'
env=dict(os.environ,WANDB_MODE='disabled',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
         LNET_LAUNCH_CACHE=str(root/'empty-normal'),TORCHINDUCTOR_CACHE_DIR=str(root/'ind-normal'),TRITON_CACHE_DIR=str(root/'tri-normal'),
         PYTHONPATH=str(src/'scripts')+':'+str(src/'src'))
def hashfile(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def loadcp(path):
    return torch.load(path, map_location='cpu',weights_only=True)
initial=loadcp(checkpoint)
assert initial['epoch']==18 and initial['global_step']==18270
initial_steps={k:int(v['step']) for k,v in initial['optimizer']['state'].items()}
assert len(initial_steps)==193
result={'command':command,'initial_sha256':hashfile(checkpoint),'initial_global_step':18270}
def save():
    result_path.write_text(json.dumps(result,indent=2))
def ordinary(name):
    work=root/name;work.mkdir()
    dest=work/'run/checkpoints'/model_name;dest.parent.mkdir(parents=True)
    shutil.copyfile(checkpoint,dest)
    assert hashfile(dest)==result['initial_sha256']
    log=work/'train.log';active=set();started=time.perf_counter()
    cache=Path(env['LNET_LAUNCH_CACHE'])/'effective-updates'
    with log.open('w') as fd:
        proc=subprocess.Popen(command,cwd=work,env=env,stdout=fd,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            for p in cache.glob('.calibration-*/*.result.json'):
                active.add(p.name)
            time.sleep(3)
    txt=log.read_text()
    if proc.returncode:
        print("ORDINARY_ERROR",name,txt[-1600:],flush=True);raise RuntimeError(f"{name} CLI exit {proc.returncode}")
    cp=loadcp(dest)
    steps={k:int(v['step']) for k,v in cp['optimizer']['state'].items()}
    assert cp['epoch']==19 and cp['global_step']==18271 and steps.keys()==initial_steps.keys()
    assert all(steps[k]==x+1 for k,x in initial_steps.items())
    paths=[p for p in (work/'run/effective-update-calibration').glob('*.json') if not p.name.endswith('.active.json')]
    assert len(paths)==1,paths
    audit=json.loads(paths[0].read_text())
    contract=json.loads((work/'run/contract.json').read_text())
    assert contract['recipe']['effective_batch_size']==1024
    row=dict(audit=audit,global_step=cp['global_step'],optimizer_counter_count=len(steps),runtime_s=time.perf_counter()-started,
             started_worker=len(active)>0,observed_workers=len(active),history_rows=len(cp['history']),checkpoint_sha256=hashfile(dest))
    result[name]=row;save()
    if audit['status'] in ('retryable-fallback','fallback'):
        print("ORDINARY_FALLBACK",name,audit.get('error'),flush=True);raise RuntimeError("calibration did not finish")
    return row

def compare_updates(names):
    cs=[loadcp(root/n/'run/checkpoints'/model_name) for n in ('cold','warm','bundled')]
    schema=set(cs[0]['model'])-set(initial['model'])
    assert not schema
    # Check finite model state and compare parameter changes separately from BN buffers.
    assert all(n in initial['model'] for n in names)
    assert len(names)==193,len(names)
    out={}
    for j,label in [(1,'warm'),(2,'bundled')]:
        bad=[];ma=mr=0.
        for n in names:
            ref=cs[0]['model'][n].float()-initial['model'][n].float()
            other=cs[j]['model'][n].float()-initial['model'][n].float()
            d=other-ref;ab=d.abs().max().item();rel=(d.norm()/ref.norm().clamp_min(1e-12)).item()
            if not d.isfinite().all() or ab>3e-5 or rel>.02:bad.append(n)
            ma,mr=max(ma,ab),max(mr,rel)
        out[label]=dict(count=len(names),failed=bad,max_abs=ma,relative_l2=mr)
    return out

def main():
    initial_empty={k:not Path(env[k]).exists() for k in ('LNET_LAUNCH_CACHE','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
    assert all(initial_empty.values()),initial_empty
    result['cold_cache_absent']=initial_empty
    result['calibration_command_is_training_command']=True
    assert not (root/'cold').exists()
    cold=ordinary('cold')
    assert cold['audit']['status']=='calibrated' and cold['started_worker']
    candidates=list((Path(env['LNET_LAUNCH_CACHE'])/'effective-updates').glob('*.json'))
    assert len(candidates)==1,candidates
    receipt=candidates[0];raw=receipt.read_bytes();mtime=receipt.stat().st_mtime_ns
    d=json.loads(raw)
    assert d['key']==cold['audit']['key'] and valid_receipt(d,d['key'])
    assert d['state_before']==d['state_after'] and {'model','optimizer','gradients','rng','train_generator','mixup'}<=d['state_before'].keys()
    assert d['search_complete'] and d['workload']['effective_samples']==1024
    result['preserved']=d['state_before']; result['receipt_sha256']=hashlib.sha256(raw).hexdigest()
    result['fixed_cli_group_digest']=d['fixed_input_digest'];result['winner']=d['winner'];result['micro']=d['microbatch']
    result['accumulation']=1024//d['microbatch'];result['status']=d['status'];save()
    catalog=src/'src/lnet/effective_update_proposals.json'
    active_catalog=catalog.read_bytes()
    verify_catalog=json.loads(active_catalog)
    assert len(verify_catalog['proposals'])>=2
    verify_catalog['proposals'].reverse()
    catalog.write_text(json.dumps(verify_catalog,sort_keys=True,indent=2)+'\n')
    reordered_sha=hashfile(catalog)
    assert reordered_sha!=hashlib.sha256(active_catalog).hexdigest()
    try:
        warm=ordinary('warm')
    finally:
        catalog.write_bytes(active_catalog)
    result['catalog_only_warm_check']=dict(cold_sha256=hashlib.sha256(active_catalog).hexdigest(),warm_sha256=reordered_sha,
                                          warm_key=warm['audit']['key'],restored=hashfile(catalog)==hashlib.sha256(active_catalog).hexdigest())
    assert warm['audit']['source']=='local' and warm['audit']['key']==d['key'] and not warm['started_worker']
    assert receipt.read_bytes()==raw and receipt.stat().st_mtime_ns==mtime
    pub=d  # Validate and replay the exact complete receipt without removing fields.
    assert valid_receipt(pub,d['key'])
    bundle=src/'src/lnet/validated_effective_updates.json'
    all_records=json.loads(bundle.read_text());all_records[d['key']]=pub
    bundle.write_text(json.dumps(all_records,indent=2,sort_keys=True)+'\n')
    env['LNET_LAUNCH_CACHE']=str(root/'empty-packaged')
    assert not Path(env['LNET_LAUNCH_CACHE']).exists()
    bundled=ordinary('bundled')
    assert bundled['audit']['source']=='bundled' and bundled['audit']['key']==d['key'] and not bundled['started_worker']
    result['postupdate_vs_cold']=compare_updates(d['capture_schema']['trainable_names'])
    from r3_catalog import inspect_receipt
    result['actual_fixed_candidate_screens']=inspect_receipt(d,root)
    result['qualified_catalog']=json.loads((root/'qualified-catalog.json').read_text())
    table=json.loads((root/'matched-table.json').read_text())
    result['fixed_matched_evidence']=[{**{k:v[k] for k in ('microbatch','accumulation','role','qualified','median_ms','img_s','median_peak_mib',
        'three_pair_gain','ranges_separated','winner_records_sha256')},'pairs':[{'order':pair['order'],'numeric':pair['numeric'],
        'arms':{n:{k:a[k] for k in ('wall_samples_ms','median_ms','images_s','peak_allocated_mib','capture_counts','capture_finite','workload_sha256')}
                for n,a in pair['arms'].items()}} for pair in v['pairs']]} for v in table['plans']]
    for compact,plan in zip(result['fixed_matched_evidence'],table['plans'],strict=True):
        for dst,pair in zip(compact['pairs'],plan['pairs'],strict=True):
            for side,arm in pair['arms'].items():
                if 'profile_outside_timer' in arm:
                    dst['arms'][side]['profile_outside_timer']=arm['profile_outside_timer']
    result['kernel_events_status']=json.loads((root/'kernel-events-status.json').read_text())
    result['source_freeze_sha256']=hashfile(Path(__file__).parent/'common-release.sha256')
    result['valid_receipt']=True
    result['source_closure']=json.loads((root/'source-manifest.json').read_text()) if (root/'source-manifest.json').exists() else {}
    (root/'winning-receipt.json').write_text(json.dumps(pub,separators=(',',':'),sort_keys=True))
    save()
    print("ORDINARY_RESULT",json.dumps({k:v for k,v in result.items() if k!='source_closure'}),flush=True)
    from return_receipt import publish
    publish(raw,result)
if __name__=='__main__':main()
