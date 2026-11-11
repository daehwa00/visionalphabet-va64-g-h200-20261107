import fcntl,json,os,subprocess,sys,time
from pathlib import Path
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--data-manifest',type=Path,required=True);parser.add_argument('--validation-manifest',type=Path,required=True);parser.add_argument('--microbatch',type=int,required=True);parser.add_argument('--variant',choices=['value','residual'],required=True)
args=parser.parse_args();ROOT=Path(__file__).resolve().parent;OUT=args.output_root;OUT.mkdir(parents=True,exist_ok=False)
results={}
for variant in (args.variant,):
 common=[sys.executable,str(ROOT/'train.py'),'--branch-source',variant,'--steps','3','--checkpoint-every','1','--microbatch',str(args.microbatch),'--data-manifest',str(args.data_manifest),'--validation-manifest',str(args.validation_manifest),'--cache-root',str(OUT/'cache')]
 for name,extra in [('paused',['--stop-after','2']),('resumed',['--resume',str(OUT/variant/'paused/checkpoints/latest.pt')]),('reference',[])]:
  output=OUT/variant/name;print('START',variant,name,flush=True)
  if not (output/'result.json').exists():subprocess.run(common+['--output-root',str(output),*extra],check=True)
  if name=='paused':subprocess.run([sys.executable,str(ROOT/'check_restore_exact.py'),str(output/'checkpoints/latest.pt')],check=True)
  report=json.loads((output/'result.json').read_text())
  assert report.get('optimizer_updates',0)>=2, 'No successful optimizer recovery test'
  print('END',variant,name,flush=True)
 import torch
 from checkpointing import verified_load
 a=verified_load(OUT/variant/'resumed/checkpoints/latest.pt');b=verified_load(OUT/variant/'reference/checkpoints/latest.pt')
 diff=ref=0.;max_abs=0.
 for k,x in a['model'].items():
  y=b['model'][k];d=(x.float()-y.float());diff+=float(d.double().square().sum());ref+=float(y.double().square().sum());max_abs=max(max_abs,float(d.abs().max()))
 assert (diff/ref)**.5<1e-7 and max_abs<5e-7, (diff,ref,max_abs)
 assert a['progress']['global_step']==b['progress']['global_step']==3
 assert a['progress']['tokens_seen']==b['progress']['tokens_seen']
 assert torch.equal(a['rng']['torch'],b['rng']['torch'])
 assert len(a['optimizer']['state'])==len(b['optimizer']['state'])
 for k,v in a['optimizer']['state'].items():
  for field,x in v.items():
   if isinstance(x,torch.Tensor) and field=='step':assert torch.equal(x,b['optimizer']['state'][k][field])
 results[variant]={'status':'passed','restore_before_update':'bitwise exact model/optimizer/scaler/RNG; independent CUDA runs have reduction-rounding differences','model_max_abs':max_abs,'model_relative_l2':(diff/ref)**.5,'global_step':3,'final_weights':str(OUT/variant/'resumed/checkpoints/final.pt')}
 (OUT/'GPU-RESTORE-RESULTS.json').write_text(json.dumps(results,indent=2))
 del a,b
 print('RESTORE_VERIFIED',variant,results[variant],flush=True)
