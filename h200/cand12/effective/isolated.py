"""Run the common update worker in an isolated workload-geometry context."""
import hashlib
import contextlib
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import torch
import effective_batch_update as primitive
from automatic_training_layout import (
    LAYOUT_NAME,PROFILE_NAME,SEGMENT_NAME,DOT_NAME,capture_launch_scopes,
    check_compiled_plan,workload_geometries
)
from lnet.kernel_launch_config import stored_geometry
from lnet.path_backward_layout import BUNDLED_LAYOUTS
from lnet.retained_path_dispatch import BUNDLE as RET
from lnet.segmented_path_scan import BUNDLE as SEG
from lnet.output_dot_dispatch import BUNDLE as DOT

FILES={LAYOUT_NAME:BUNDLED_LAYOUTS,PROFILE_NAME:RET,SEGMENT_NAME:SEG,DOT_NAME:DOT}

def run():
    job=json.loads(Path(sys.argv[1]).read_text())
    root=Path(os.environ['RUN_ROOT'])
    result=Path(job['output'])
    args=['worker.py','--microbatch',str(job['micro']),'--checkpoint',str(root/'private-input/checkpoint.pt'),
          '--contract',str(Path('contract.json').resolve()),'--fixture',str(root/'private-input/fixture.json'),
          '--output',str(result),'--warmup',str(job.get('warmup',2)),'--groups',str(job.get('groups',3)),
          '--samples',str(job.get('samples',2)),'--capture']
    if job.get('profile'):args.append('--profile')
    raw=primitive.run_update
    validation={}
    def checked(model,runtime,opt,batch,micro,precision,*,capture=False,metrics=False):
        if not capture:return raw(model,runtime,opt,batch,micro,precision,capture=False,metrics=metrics)
        timeline=[];scales=[];losses=[];counts=[]
        original_obj=primitive.training_objective
        old_zero,old_step,old_clip=opt.zero_grad,opt.step,torch.nn.utils.clip_grad_norm_
        before=[float(s['step'].item()) for s in opt.state.values() if 'step' in s]
        reference=torch.load(root/'private-input/checkpoint.pt',map_location='cpu',weights_only=True)
        expected_steps=sorted(float(s['step']) for s in reference['optimizer']['state'].values())
        assert sorted(before)==expected_steps
        del reference
        def zero(*a,**kw):
            timeline.append('zero');return old_zero(*a,**kw)
        def step(*a,**kw):
            timeline.append('step');return old_step(*a,**kw)
        def clip(*a,**kw):
            timeline.append('clip')
            assert (a[1] if len(a)>1 else kw['max_norm'])==1.0
            return old_clip(*a,**kw)
        def objective(model,out,labels,other,lam):
            z,loss,diag=original_obj(model,out,labels,other,lam)
            counts.append(len(labels));losses.append(float(loss.detach()));timeline.append('forward')
            def hook(value):
                scales.append(float(value));timeline.append('backward')
            loss.register_hook(hook)
            return z,loss,diag
        with patch.object(opt,'zero_grad',zero),patch.object(opt,'step',step),patch.object(torch.nn.utils,'clip_grad_norm_',clip),patch.object(primitive,'training_objective',objective):
            v=raw(model,runtime,opt,batch,micro,precision,capture=True,metrics=metrics)
        after=[float(s['step'].item()) for s in opt.state.values() if 'step' in s]
        assert len(after)==len(before)==193 and all(y==x+1 for x,y in zip(before,after))
        assert timeline==['zero',*(t for _ in counts for t in ('forward','backward')),'clip','step']
        assert sum(counts)==1024 and scales==[n/1024 for n in counts]
        assert v['samples']==1024 and v['microbatches']==len(counts)==1024//micro and v['optimizer_steps']==1
        mean=sum(x*n/1024 for x,n in zip(losses,counts)); assert abs(float(v['loss'])-mean)<2e-5
        if job.get('discover'):
            cpu=batch[0].detach().cpu().contiguous()
            validation['mixed_input_sha256']=hashlib.sha256(cpu.numpy()).hexdigest()
            validation['paired_labels_sha256']=hashlib.sha256(batch[2].detach().cpu().contiguous().numpy()).hexdigest()
        validation.update(samples=1024,micro=micro,accum=len(counts),zero=1,clip=1,adamw=1,moment_counters=len(after),scales=scales,mean_loss=mean)
        return v
    with contextlib.ExitStack() as stack:
        stack.enter_context(workload_geometries(job['records'],discovery=job['label'] if job.get('discover') else ''))
        found={k:stack.enter_context(capture_launch_scopes(k)) for k in FILES}
        stack.enter_context(patch.object(primitive,'run_update',checked))
        sys.argv=args
        runpy.run_path('effective/shared_worker.py',run_name='__main__')
        if job.get('discover'):
            scopes={name:{s.key:(dict(blocks=g.blocks,num_warps=g.num_warps,num_stages=g.num_stages) if (g:=stored_geometry(name,scope=s,path=FILES[name])) else None)
                          for s in observed if s.execution_regime=='compiled'} for name,observed in found.items()}
            assert all(scopes[n] for n in FILES),list(scopes)
            result.with_suffix('.scopes.json').write_text(json.dumps(scopes))
        if job.get('profile') and job['records']:
            # The generic verifier checks non-baseline overrides. An explicit empty
            # geometry disables the corresponding optional kernel.
            defaults=json.loads((root/'measurements'/f"reference-{job['micro']}.scopes.json").read_text())
            evidence={n:{k:g for k,g in {**d,**job['records'].get(n,{})}.items() if g and g.get('blocks')} for n,d in defaults.items()}
            evidence={n:d for n,d in evidence.items() if d}
            if evidence:
                applied=check_compiled_plan(result.with_suffix('.trace.json'),evidence)
                result.with_suffix('.applied.json').write_text(json.dumps(applied))
    assert validation
    result.with_suffix('.audit.json').write_text(json.dumps(validation))

if __name__=='__main__':run()
