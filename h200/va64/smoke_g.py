"""Bounded synthetic G training and a fresh-process optimizer/RNG restore."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader,TensorDataset
import run_va64_retrain_imagenet1k as experiment


def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--steps',type=int,default=20); p.add_argument('--image-size',type=int,default=224); p.add_argument('--device',choices=('cpu','cuda'),default='cuda'); p.add_argument('--compile',action='store_true'); p.add_argument('--restore',action='store_true'); a=p.parse_args()
 if a.steps<1 or a.image_size<16: p.error('invalid bounded smoke size')
 torch.set_num_threads(4); torch.manual_seed(0); torch.set_float32_matmul_precision('high')
 device=torch.device(a.device)
 if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
 if a.output.exists() and not a.restore: raise FileExistsError('smoke output exists')
 a.output.mkdir(parents=True,exist_ok=True)
 model=experiment.build_model('G').to(device=device,memory_format=torch.channels_last)
 if a.compile: model.compile(mode='default',dynamic=False)
 opt=experiment.runtime._build_optimizer(model,.003,device)
 data=TensorDataset(torch.zeros(2,1),torch.zeros(2,dtype=torch.long)); loader=DataLoader(data,batch_size=2)
 bundle=experiment.runtime.LoaderBundle(loader,loader,torch.Generator().manual_seed(0),torch.Generator().manual_seed(1))
 source=experiment.source_fingerprint(); checkpoint=a.output/'checkpoint.pt'
 x=torch.randn(2,3,a.image_size,a.image_size,device=device).to(memory_format=torch.channels_last)
 target=torch.tensor([1,2],device=device)
 def step():
  opt.zero_grad(set_to_none=True)
  with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
   loss=torch.nn.functional.cross_entropy(model(x).float(),target)
  loss.backward()
  if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite loss')
  if any(not bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None): raise RuntimeError('nonfinite gradient')
  opt.step(); return float(loss.detach())
 if a.restore:
  payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
  x=torch.load(a.output/'input.pt',map_location=device,weights_only=True).to(memory_format=torch.channels_last)
  if payload['variant']!='G' or payload['source']!=source or payload['definition']!=experiment.definition_digest('G'): raise RuntimeError('incompatible smoke contract')
  model.load_state_dict(payload['model']); opt.load_state_dict(payload['optimizer']); experiment.runtime._restore_rng(payload['rng'],bundle,device)
  model.eval()
  with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'): actual=model(x).float()
  torch.testing.assert_close(actual.cpu(),payload['logits'],rtol=2e-4,atol=2e-4)
  model.train(); loss=step()
  (a.output/'restore.json').write_text(json.dumps({'status':'passed','restored_steps':payload['steps'],'next_loss':loss,'new_process':True})+'\n')
  print('G_RESTORE_PASSED',flush=True); return
 losses=[]
 for index in range(a.steps):
  if losses and (Path(__import__('os').environ.get('VA64_G_MANAGED_ROOT',str(a.output)))/'STOP').exists(): break
  losses.append(step())
  experiment.runtime._atomic_json(a.output/'step-progress.json',{'epoch':0,'global_step':index+1,'loss':losses[-1]})
  if (index+1)%5==0: print('G_SMOKE_STEP='+json.dumps({'global_step':index+1,'loss':losses[-1]}),flush=True)
 model.eval()
 with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'): logits=model(x).float().cpu()
 payload={'variant':'G','source':source,'definition':experiment.definition_digest('G'),'steps':len(losses),'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':experiment.runtime._capture_rng(bundle,device),'logits':logits}
 experiment.runtime._atomic_torch(checkpoint,payload)
 # Reconstruct x identically in the child by restoring its saved input explicitly.
 torch.save(x.cpu(),a.output/'input.pt')
 command=[sys.executable,__file__,'--output',str(a.output),'--steps',str(a.steps),'--image-size',str(a.image_size),'--device',a.device,'--restore']
 if a.compile: command.append('--compile')
 # Release GPU model allocations before launching the restore process.
 del model,opt,payload; import gc; gc.collect()
 if a.device=='cuda': torch.cuda.empty_cache()
 subprocess.run(command,check=True)
 report={'status':'passed','variant':'G','steps':len(losses),'synthetic_input':True,'image_size':a.image_size,'batch':2,'parameters':1882760,'finite_losses':losses,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'checkpoint_bytes':checkpoint.stat().st_size,'fresh_process_restore':True,'compiled':a.compile,'device':a.device}
 (a.output/'smoke.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k!='finite_losses'}),flush=True)

if __name__=='__main__': main()
