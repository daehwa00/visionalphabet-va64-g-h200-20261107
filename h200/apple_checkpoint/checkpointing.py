"""Local atomic, tensor-only APPLE checkpoint files. No network operations."""
from __future__ import annotations
import hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch

SCHEMA=2

def plain(value):return json.loads(json.dumps(value,default=str,allow_nan=False))

def digest_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def cpu_tree(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
 if isinstance(value,list):return [cpu_tree(v) for v in value]
 if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
 return value

def rng_state():
 n=np.random.get_state()
 return {'python':random.getstate(),'torch':torch.get_rng_state(),
         'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
         'numpy':(n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4]))}

def restore_rng(r):
 random.setstate(r['python']);torch.set_rng_state(r['torch'])
 if r['cuda']:
  if not torch.cuda.is_available() or len(r['cuda'])!=torch.cuda.device_count():raise ValueError('CUDA RNG device count mismatch')
  torch.cuda.set_rng_state_all(r['cuda'])
 n=r['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))

def atomic_save(path,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.name=='latest.pt' and path.exists() and path.with_suffix('.pt.json').exists():
  for old,new in [(path,path.with_name('previous.pt')),(path.with_suffix('.pt.json'),path.with_name('previous.pt.json'))]:
   link=new.with_name(new.name+'.link');link.unlink(missing_ok=True);os.link(old,link);os.replace(link,new)
 tmp=path.with_name(path.name+'.part')
 try:
  with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
  fd=os.open(path.parent,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
 finally:tmp.unlink(missing_ok=True)
 info={'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':digest_file(path),'schema':SCHEMA,'kind':payload['kind'],'global_step':payload['progress']['global_step']}
 manifest=path.with_suffix(path.suffix+'.json');m=manifest.with_name(manifest.name+'.part')
 with m.open('w') as f:f.write(json.dumps(info,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 os.replace(m,manifest)
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return info

def save_training(path,model,optimizer,scaler,*,contract,model_config,progress,best_checkpoint=None):
 if any(p.grad is not None for p in model.parameters()):raise ValueError('Save only after zero_grad(set_to_none=True), at a completed optimizer boundary')
 best=verified_load(best_checkpoint) if best_checkpoint is not None and Path(best_checkpoint).exists() else None
 if best is not None and (best['kind']!='weights' or best['contract']!=plain(contract)):raise ValueError('Best checkpoint contract mismatch')
 if progress.get('best_val') is not None and (best is None or best['progress'].get('val_loss')!=progress['best_val']):raise ValueError('Historical best missing or inconsistent')
 payload={'best':best,'schema':SCHEMA,'kind':'training','contract':plain(contract),'model_config':plain(model_config),'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),'scaler':cpu_tree(scaler.state_dict()) if scaler else None,'rng':rng_state(),'progress':plain(progress)}
 return atomic_save(path,payload)

def save_weights(path,model,*,contract,model_config,progress):
 return atomic_save(path,{'schema':SCHEMA,'kind':'weights','contract':plain(contract),'model_config':plain(model_config),'model':cpu_tree(model.state_dict()),'progress':plain(progress)})

def verified_load(path,expected_sha256=None):
 path=Path(path)
 if expected_sha256 is None:
  manifest=json.loads(path.with_suffix(path.suffix+'.json').read_text());expected_sha256=manifest['sha256']
  if path.stat().st_size!=manifest['bytes']:raise ValueError('Checkpoint size mismatch')
 if digest_file(path)!=expected_sha256:raise ValueError('Checkpoint SHA256 mismatch')
 value=torch.load(path,map_location='cpu',weights_only=True)
 if value.get('schema')!=SCHEMA or value.get('kind') not in ('weights','training'):raise ValueError('Unsupported checkpoint schema')
 return value

def restore_training(path,model,optimizer,scaler,*,contract,best_output=None):
 value=verified_load(path)
 if value['kind']!='training':raise ValueError('Weights-only file cannot resume optimizer state')
 if value['contract']!=plain(contract):raise ValueError('Resume contract mismatch: source, variant, recipe or data changed')
 model.load_state_dict(value['model'],strict=True);optimizer.load_state_dict(value['optimizer'])
 if scaler is not None:
  if value['scaler'] is None:raise ValueError('Missing scaler state')
  scaler.load_state_dict(value['scaler'])
 if value['progress'].get('best_val') is not None and value.get('best') is None:raise ValueError('Missing embedded historical best')
 if best_output is not None and value.get('best') is not None:atomic_save(best_output,value['best'])
 restore_rng(value['rng'])
 return value['progress']
