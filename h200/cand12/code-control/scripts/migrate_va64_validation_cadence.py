"""Explicit evaluation-cadence-only checkpoint migration; never mutates the original run."""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path


def same_tree(a,b):
 import torch
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
 return a==b


def verify_contracts(old,new):
 for key in ('schema','model','recipe','dataset','runtime','execution'):
  if old.get(key)!=new.get(key):raise RuntimeError('training/runtime contract changed: '+key)
 ignored={'source_root','source_sha256','output_dir','result_path','checkpoint_path'}
 if {k:v for k,v in old['task'].items() if k not in ignored}!={k:v for k,v in new['task'].items() if k not in ignored}:raise RuntimeError('training task changed')
 for key in ('conv_implementation','budget','dataset_manifest','ablation_definition','execution_kernel','execution_triton_kernel','execution_block_modes','execution_backend'):
  if old['source_sha256'].get(key)!=new['source_sha256'].get(key):raise RuntimeError('training implementation changed: '+key)
 if new.get('evaluation')!={'interval_epochs':20,'final_epoch_always':True}:raise RuntimeError('expected eval20 and final')


def main():
 p=argparse.ArgumentParser();p.add_argument('--old-directory',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--block',type=int,required=True);p.add_argument('--backend',choices=['packed','triton'],required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args();r=a.new_root.resolve();old=a.old_directory.resolve()
 sys.path[:0]=[str(r/'code/scripts'),str(r/'code/src')]
 import torch
 import run_va64_conv_scan_control as entry
 old_contracts=list((old/'contracts').glob('*__full__seed'+str(a.seed)+'__lr0p003.json'))
 if len(old_contracts)!=1:raise RuntimeError('old contract ambiguous')
 old_contract=json.loads(old_contracts[0].read_text());old_code=Path(old_contract['task']['source_root']);old_manifest_path=old_code.parent/'source-manifest.json'
 old_manifest=json.loads(old_manifest_path.read_text());new_manifest=json.loads((r/'source-manifest.json').read_text())
 if hashlib.sha256(old_manifest_path.read_bytes()).hexdigest()!=old_contract['source_sha256']['source_manifest']:raise RuntimeError('original manifest digest differs')
 allowed={'scripts/run_va64_conv_scan_control.py','scripts/imagenet1k_runtime.py'}
 for name,digest in old_manifest['files'].items():
  if hashlib.sha256((old_code/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('old source changed')
  if name not in allowed and new_manifest['files'].get(name)!=digest:raise RuntimeError('non-execution source changed: '+name)
 if set(new_manifest['files'])!=set(old_manifest['files']):raise RuntimeError('source inventory changed')
 from va64_validation_cadence_patch import runtime_source,entry_source
 for name,fn in [('imagenet1k_runtime.py',runtime_source),('run_va64_conv_scan_control.py',entry_source)]:
  if fn((old_code/'scripts'/name).read_text())!=(r/'code/scripts'/name).read_text():raise RuntimeError('non-evaluation code delta')
 out=r/f'seed_{a.seed}/runs'
 args=entry.arguments(['--code',str(r/'code'),'--source-manifest',str(r/'source-manifest.json'),'--budget',str(r/'cost-profile.json'),'--execution-block',str(a.block),'--execution-backend',a.backend,'--validation-interval','20','--seed',str(a.seed),'--data-root',old_contract['dataset']['root'],'--data-manifest',str(r/'dataset-manifest.json'),'--output-root',str(out),'--batch-size','128','--workers','8','--stop-file',str(r/f'seed_{a.seed}/STOP'),'--resume'])
 task=entry.make_task(args);new_contract=entry.runtime._contract(task,torch.device('cuda'));verify_contracts(old_contract,new_contract)
 if task.output_dir.exists() and any(task.output_dir.iterdir()):raise FileExistsError('destination must be empty')
 payload=torch.load(old/'checkpoint.pt',map_location='cpu',weights_only=True)
 entry.runtime._validate_binding(payload,old_contract)
 if payload['completed_epochs']<1:raise RuntimeError('no complete epoch checkpoint')
 model=entry.control.build_model();entry.conv_execution.install(model,a.block,a.backend);model.load_state_dict(payload['model'],strict=True)
 if sum(p.numel() for p in model.parameters())!=payload['parameters']:raise RuntimeError('parameter inventory changed')
 if a.validate_only:
  print(json.dumps({'status':'passed','mode':'read-only preflight','completed_epochs':payload['completed_epochs'],'evaluation':new_contract['evaluation'],'parameter_state_compatible':True,'source_delta_exact':True,'training_recipe_unchanged':True}),flush=True);return
 task.output_dir.mkdir(parents=True);new_hash=entry.runtime._sha256_payload(new_contract)
 def rebind(source,target):
  value=torch.load(source,map_location='cpu',weights_only=True);entry.runtime._validate_binding(value,old_contract)
  copied={**value,'contract_sha256':new_hash,'task_sha256':new_contract['task_sha256'],'source_digest_sha256':new_contract['source_digest_sha256'],'source_sha256':new_contract['source_sha256']}
  entry.runtime._validate_checkpoint(copied,new_contract,task);entry.runtime._atomic_torch(target,copied)
  reread=torch.load(target,map_location='cpu',weights_only=True)
  for key in ('model','optimizer','rng','history','global_step','completed_epochs','training_seconds'):
   if not same_tree(value[key],reread[key]):raise RuntimeError('state changed during migration: '+key)
 rebind(old/'checkpoint.pt',task.checkpoint_path)
 if (old/'best.pt').exists():rebind(old/'best.pt',task.output_dir/'best.pt')
 if (old/'best.json').exists():shutil.copyfile(old/'best.json',task.output_dir/'best.json')
 entry.runtime._atomic_json(task.output_dir/'contracts'/f'{task.task_name}.json',new_contract)
 entry.runtime._atomic_json(task.output_dir/'latest-epoch.json',payload['history'][-1])
 (task.output_dir/'epochs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in payload['history']))
 record={'status':'passed','migration':'evaluation-cadence-only; training recipe/model/data/execution unchanged','old_directory':str(old),'old_checkpoint_sha256':hashlib.sha256((old/'checkpoint.pt').read_bytes()).hexdigest(),'new_checkpoint_sha256':hashlib.sha256(task.checkpoint_path.read_bytes()).hexdigest(),'from_contract':payload['contract_sha256'],'to_contract':new_hash,'completed_epochs':payload['completed_epochs'],'global_step':payload['global_step'],'model_optimizer_rng_history_tensor_exact':True,'execution_block':a.block,'validation_interval':20,'final_epoch_always':True,'checkpoint_interval':1,'notes':'Past validation rows retained; skipped future validation rows are null.'}
 entry.runtime._atomic_json(task.output_dir/'contracts/execution-migration.json',record);print(json.dumps(record),flush=True)

if __name__=='__main__':main()
