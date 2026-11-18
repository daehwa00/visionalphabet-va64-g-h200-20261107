"""Cross-device final qualification with the unmodified official RTX 4090 timer and comparator."""
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORKER = HERE / 'matched_worker.py'
COMPARE = HERE / 'matched_compare.py'
WORKER_HASH = '6009e5f8e344f82079b7f5eec86555ad2e9632f53e2ed77e63c9b1738780afe1'
COMPARE_HASH = 'a4e695bb8d67bcca8e20c2923ba7e4103eee9499aae1d56f1aaf6f38431fcc07'
CHECKPOINT = 'af07043f8407221dfe0d75c0d689d76e573119ad8708ee8df5b29cc564ed1fc6'
CONTRACT = '38e240457708c76a3648ab52c1871d8aac61a12d44e14345f9b25b0ca6040699'
PRIMITIVE = '7442c14a98afb2bec7d6e3f319103022b87e81d0c2783b570fcfd4a00f49fe64'
COUNTS = dict(outputs=5, grads=193, updates=193)
ORDERS = [('control','winner'), ('winner','control'), ('control','winner')]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def compact_arm(doc, micro, pins):
    assert doc['status']=='passed' and doc['worker_sha256']==WORKER_HASH
    assert doc['microbatch']==micro and doc['effective_batch']==1024 and doc['precision']=='bfloat16'
    assert (doc['epoch'], doc['global_step'], doc['parameters'])==(18, 18270, 2389796)
    assert (doc['checkpoint_sha256'],doc['contract_sha256'],doc['primitive_sha256'])==(CHECKPOINT,CONTRACT,PRIMITIVE)
    assert doc['fixture_sha256']==pins['fixture.json']
    assert doc['fixture_tensors_sha256']=={k:pins[k] for k in ('images.pt','labels.pt','permutation.pt')}
    assert (doc['warmup'],doc['groups'],doc['samples'])==(2,3,2)
    assert doc['capture_finite'] is True and doc['capture_counts']==COUNTS
    walls = doc['wall_samples_ms']
    assert len(walls)==3 and all(len(g)==2 and all(math.isfinite(t) and t>0 for t in g) for g in walls)
    assert doc['median_ms']==statistics.median(statistics.mean(g) for g in walls)
    assert doc['images_s']==1024000/doc['median_ms'] and math.isfinite(doc['peak_allocated_mib']) and doc['peak_allocated_mib']>0
    return {k:doc[k] for k in ('wall_samples_ms','median_ms','images_s','peak_allocated_mib','capture_counts',
                                'capture_finite','learning_rates','source_closure_sha256','workload_sha256','gpu','torch')}


def compact_gate(report):
    assert {name:len(v) for name,v in report['groups'].items()}==COUNTS
    numerical={}
    for group,values in report['groups'].items():
        failed=[n for n,v in values.items() if not v['passed']]
        numerical[group]=dict(count=len(values),failed=failed,max_abs=max(v['max_abs'] for v in values.values()),
                              max_relative_l2=max(v['relative_l2'] for v in values.values()))
    scalars=report['scalar_checks']
    assert set(scalars)=={'loss','clip_norm'}
    assert report['passed']==(all(not v['failed'] for v in numerical.values()) and all(v['passed'] for v in scalars.values()))
    return dict(passed=report['passed'],groups=numerical,scalars=scalars)


def launch_worker():
    from lnet.kernel_launch_config import workload_geometries
    records=load(Path(sys.argv[2]))
    sys.argv=[str(WORKER),*sys.argv[3:]]
    with workload_geometries(records):
        runpy.run_path(str(WORKER),run_name='__main__')


def profile_evidence(trace,records):
    from automatic_training_layout import check_compiled_plan
    from effective_batch_search import _vjp_applied
    from effective_trace_rank import rank
    actual=check_compiled_plan(trace,records)
    assert _vjp_applied(records,actual)
    rows=[]
    for row in actual:
        launches=row['launches']
        assert launches
        rows.append(dict(phase=row['phase'],shape=row['shape'],count=len(launches),
                         first=launches[0]))
    ranking=rank(trace)
    assert ranking['top'] and ranking['total_kernel_ms']>0
    return dict(trace_sha256=digest(trace),total_kernel_ms=ranking['total_kernel_ms'],
                kernel_event_count=ranking['kernel_event_count'],top=ranking['top'][:5],dispatch=rows)


def compare_plan(plan,folder,source,pins,fixture,lr):
    micro=plan['micro']
    assert micro>0 and 1024%micro==0
    folder.mkdir()
    records={'control':{},'winner':plan['winner']}
    schemas={'control':plan['defaults'],'winner':plan['winner']}
    for side in records:
        write(folder/(side+'.json'),records[side])
    result=dict(status='running',microbatch=micro,accumulation=1024//micro,
                role=plan['role'],identical_effective_plan=plan['winner']==plan['defaults'],
                control_records_sha256=digest(folder/'control.json'),winner_records_sha256=digest(folder/'winner.json'),pairs=[])
    if 'mapped_scopes' in plan:
        result['mapped_scopes']=plan['mapped_scopes']
        result['parent_scopes']=plan['parent_scopes']
    import torch
    for pair,order in enumerate(ORDERS,1):
        arms={}
        for side in order:
            prefix=folder/f'{pair}-{side}'
            dest=prefix.with_suffix('.json')
            cache=folder/(side+'-cache')
            env=dict(os.environ, LNET_LAUNCH_CACHE=str(cache/'launch'),TRITON_CACHE_DIR=str(cache/'tri'),
                     TORCHINDUCTOR_CACHE_DIR=str(cache/'ind'),PYTHONDONTWRITEBYTECODE='1',
                     PYTHONPATH=os.pathsep.join(map(str,[source/'scripts',source/'src',source,HERE])))
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker',str(folder/(side+'.json')),
                 '--checkpoint',str(fixture/'checkpoint.pt'),'--contract',os.environ['CONTRACT'],
                 '--fixture',str(fixture/'fixture.json'),'--output',str(dest),'--microbatch',str(micro),
                 '--warmup','2','--groups','3','--samples','2','--capture']
            if pair!=2: cmd.append('--profile')
            with prefix.with_suffix('.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            arms[side]=arm=compact_arm(load(dest),micro,pins)
            assert 'H200' in arm['gpu'] and arm['learning_rates']==lr
            capture=torch.load(dest.with_suffix('.capture.pt'),map_location='cpu',weights_only=True)
            assert (capture['samples'],capture['microbatches'],capture['optimizer_steps'])==(1024,1024//micro,1)
            del capture
            if pair!=2:
                arm['profile_outside_timer']=profile_evidence(dest.with_suffix('.trace.json'),schemas[side])
        assert arms['control']['source_closure_sha256']==arms['winner']['source_closure_sha256']
        assert arms['control']['workload_sha256']==arms['winner']['workload_sha256']
        gate=folder/f'{pair}-gate.json'
        proc=subprocess.run([sys.executable,str(COMPARE),str(folder/f'{pair}-control.json'),str(folder/f'{pair}-winner.json'),
                             '--output',str(gate)],check=False)
        proof=compact_gate(load(gate))
        assert proc.returncode==int(not proof['passed'])
        result['pairs'].append(dict(order=list(order),arms=arms,numeric=proof))
        write(folder/'progress.json',result)
    times={side:[p['arms'][side]['median_ms'] for p in result['pairs']] for side in records}
    result['median_ms']={s:statistics.median(v) for s,v in times.items()}
    result['img_s']={s:1024000/v for s,v in result['median_ms'].items()}
    result['median_peak_mib']={s:statistics.median(p['arms'][s]['peak_allocated_mib'] for p in result['pairs']) for s in records}
    result['numeric_passed']=all(p['numeric']['passed'] for p in result['pairs'])
    result['three_pair_gain']=all(w<c for w,c in zip(times['winner'],times['control'],strict=True))
    result['ranges_separated']=max(times['winner'])<min(times['control'])
    result['qualified']=(not result['identical_effective_plan'] and result['numeric_passed']
                         and result['three_pair_gain'] and result['ranges_separated'])
    result['status']='complete'
    write(folder/'progress.json',result)
    return result


def main():
    import torch
    root=Path(os.environ['RUN_ROOT'])
    fixture=Path(os.environ['COLD_INPUTS'])
    frozen=root/'matched-frozen'
    shutil.copytree(root/'source',frozen,ignore=shutil.ignore_patterns('__pycache__'))
    # Ordinary bundled reuse staged a measured receipt in its private copy. Start from the official bundle for both arms.
    shutil.copyfile(fixture/'packaged-4090-effective.json',frozen/'src/lnet/validated_effective_updates.json')
    manifest=load(HERE/'source.json')
    for rel,h in manifest.items(): assert digest(frozen/rel)==h,rel
    assert digest(WORKER)==WORKER_HASH and digest(COMPARE)==COMPARE_HASH
    sys.path[:0]=[str(frozen/'scripts'),str(frozen/'src')]
    fields=dict(line.split()[::-1] for line in (HERE/'inputs.sha256').read_text().splitlines())
    cp=torch.load(fixture/'checkpoint.pt',map_location='cpu',mmap=True,weights_only=True)
    assert cp['variant']=='VA-Ablation-10-dw-terminal-rmsnorm' and len(cp['optimizer']['state'])==193
    lr=[g['lr'] for g in cp['optimizer']['param_groups']]
    audit=load(root/'fixed-continuation.json')
    plans=audit['comparison_plans']
    assert plans[0]['role']=='native' and plans[0]['micro']==audit['fixed_best']['micro'] and plans[0]['winner']==audit['fixed_best']['records']
    result=dict(model=cp['variant'],checkpoint_sha256=CHECKPOINT,contract_sha256=CONTRACT,primitive_sha256=PRIMITIVE,
                source_manifest_sha256=digest(HERE/'source.json'),common_freeze_sha256=digest(HERE/'common-release.sha256'),
                worker_sha256=WORKER_HASH,comparator_sha256=COMPARE_HASH,inputs=fields,epoch=18,global_step=18270,
                description='2 warmups; 3 groups of 2 complete 1024-image updates; metrics=True; synchronized wall; restore, compile, capture and profile outside timer',
                control='frozen device bundle without trial overrides, at winner microbatch',plans=[])
    del cp
    for plan in plans:
        result['plans'].append(compare_plan(plan,root/('matched-'+plan['role']),frozen,fields,fixture,lr))
        write(root/'matched-table.json',result)


if __name__=='__main__':
    if sys.argv[1:2]==['worker']:
        launch_worker()
    else:
        main()
