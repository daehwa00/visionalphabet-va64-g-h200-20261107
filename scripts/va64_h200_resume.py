"""Explicit, pinned epoch-boundary migration; never relax normal resume checks."""
from __future__ import annotations
import dataclasses, hashlib, json, shutil
from pathlib import Path
import torch
import imagenet1k_runtime as runtime


def digest(path: Path) -> str:
 return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_transition(old: dict, new: dict) -> dict:
 """Only physical batching, deployment paths and audited entrypoint hashes may change."""
 if old['schema'] != new['schema'] or old['model'] != new['model']:
  raise ValueError('model/schema changed')
 for key in ('dataset','execution','runtime'):
  if old[key] != new[key]: raise ValueError(key+' changed')
 old_recipe=dict(old['recipe']); new_recipe=dict(new['recipe'])
 for recipe in (old_recipe,new_recipe):
  batch=recipe.pop('batch_size'); accum=recipe.pop('gradient_accumulation_steps')
  if batch not in (128,256) or batch*accum!=256: raise ValueError('invalid physical/global batch')
 if old_recipe != new_recipe: raise ValueError('training recipe changed')
 paths={'output_dir','result_path','checkpoint_path','source_root'}
 allowed_sources={'source_closure','h200_g_entrypoint','imagenet1k_runtime'}
 for section in ('source_sha256',):
  a=dict(old[section]);b=dict(new[section])
  if a.keys()!=b.keys(): raise ValueError('source keys changed')
  for k in a:
   if a[k]!=b[k] and k not in allowed_sources: raise ValueError('unaudited source changed: '+k)
 a=dict(old['task']);b=dict(new['task'])
 for k in paths|{'batch_size','source_sha256'}: a.pop(k,None);b.pop(k,None)
 if a!=b: raise ValueError('task identity/recipe changed')
 a=dict(old['task']['source_sha256']);b=dict(new['task']['source_sha256'])
 if a.keys()!=b.keys() or any(a[k]!=b[k] for k in a if k not in allowed_sources):
  raise ValueError('task source changed')
 return {'physical_batch_before':old['recipe']['batch_size'],'physical_batch_after':new['recipe']['batch_size'],
         'global_batch':256,'exact_trajectory':False,
         'rng':'Restore checkpoint Python/NumPy/Torch/CUDA and loader generators after model construction; benchmark runs in isolated subprocesses.',
         'data_order':'Epoch-boundary sampler generator restored; membership/order preserved. Worker assignment, augmentation RNG and Mixup groups change with physical batch.',
         'batchnorm':'Physical batch changes training batch statistics and running-stat updates; numerical trajectory is not identical.',
         'partial_global_batch':'The final 143 examples form 128+15 averaged microbatch losses or one 143-example loss; last-update example weighting differs under the unchanged runtime recipe.'}


def migrate(task, recovery: Path, plan_path: Path, device: torch.device) -> dict:
 plan=json.loads(plan_path.read_text())
 checkpoint=recovery/'checkpoint.pt'
 if digest(checkpoint)!=plan['checkpoint_sha256']: raise ValueError('pinned checkpoint hash mismatch')
 old=json.loads((recovery/plan['contract_relative_path']).read_text())
 if runtime._sha256_payload(old)!=plan['contract_sha256']: raise ValueError('pinned parent contract mismatch')
 payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
 runtime._validate_checkpoint(payload,old,task)
 if payload['completed_epochs']!=plan['completed_epochs'] or payload['global_step']!=plan['global_step']:
  raise ValueError('parent progress mismatch')
 new=runtime._contract(task,device)
 for key in ('source_closure','h200_g_entrypoint','imagenet1k_runtime'):
  if new['source_sha256'].get(key)!=plan['allowed_new_sources'][key]:
   raise ValueError('unapproved target source: '+key)
 report=validate_transition(old,new)
 if task.checkpoint_path.exists(): raise FileExistsError('migration target already exists')
 task.output_dir.mkdir(parents=True,exist_ok=True)
 # Keep original recovery payload, contract and receipt immutable outside this output.
 for name in ('best.pt','best.json','epochs.jsonl','steps.jsonl','latest-epoch.json','ablation.json','parameter-inventory.json'):
  if (recovery/name).is_file(): shutil.copyfile(recovery/name,task.output_dir/name)
 report.update(parent_checkpoint_sha256=plan['checkpoint_sha256'],parent_contract_sha256=plan['contract_sha256'],
               completed_epochs=payload['completed_epochs'],global_step=payload['global_step'],
               old_contract=old,new_contract=new)
 for key,value in {'schema':new['schema'],'contract_sha256':runtime._sha256_payload(new),
     'task_sha256':new['task_sha256'],'source_digest_sha256':new['source_digest_sha256'],
     'source_sha256':new['source_sha256']}.items(): payload[key]=value
 runtime._validate_checkpoint(payload,new,task)
 runtime._atomic_torch(task.checkpoint_path,payload)
 contracts=task.output_dir/'contracts';contracts.mkdir(exist_ok=True)
 runtime._atomic_json(contracts/(task.task_name+'.json'),new)
 runtime._atomic_json(contracts/'resume-migration.json',report)
 return report


def recover(root: Path, plan_path: Path) -> Path:
 """Download one pinned, externally verified parent; no owner secret on H200."""
 from va64_h200_transport import _agent_api, download_artifact, _safe_extract
 plan=json.loads(plan_path.read_text()); metadata=plan['artifact_metadata']
 if not metadata.get('verified'): raise ValueError('parent backup is not externally verified')
 archive=root/'parent-checkpoint.tar.gz'
 download_artifact(_agent_api(),'va64-g-seed0',metadata,archive)
 recovery=root/'recovery'
 if not _safe_extract(archive,recovery): raise ValueError('recovery path already exists or is incomplete')
 if digest(recovery/'checkpoint.pt')!=plan['checkpoint_sha256']: raise ValueError('parent checkpoint hash mismatch')
 old=json.loads((recovery/'dataset.json').read_text())
 fresh=json.loads((root/'environment-check/dataset-manifest.json').read_text())
 for value in (old,fresh): value.pop('file_list',None)
 if old!=fresh: raise ValueError('dataset/class mapping/membership changed')
 print('VA64_PARENT_RECOVERED='+json.dumps({'epoch':plan['completed_epochs'],'global_step':plan['global_step'],'checkpoint_sha256':plan['checkpoint_sha256']}),flush=True)
 return recovery


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--root',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
 args=parser.parse_args();recover(args.root,args.plan)
