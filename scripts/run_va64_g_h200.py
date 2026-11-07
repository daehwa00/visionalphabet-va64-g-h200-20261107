"""G training with the same canonical backbone initialization as B0/P0/A0."""
from __future__ import annotations
import dataclasses,hashlib,os
from pathlib import Path
import torch
import run_va64_retrain_imagenet1k as experiment

INITIAL_SHA='4fe26280065838b6b0da66b6a2ff5082ac9be55a9a661a9296bdf10cb514de94'

def initialize_common_backbone(model, path=None):
 path=path or (Path(__file__).resolve().parents[1]/'initialization/initial-B0.pt')
 if hashlib.sha256(path.read_bytes()).hexdigest()!=INITIAL_SHA: raise RuntimeError('canonical initialization hash mismatch')
 payload=torch.load(path,map_location='cpu',weights_only=True)
 if payload.get('variant')!='B' or payload.get('seed')!=0 or payload.get('trained') is not False: raise RuntimeError('not the untrained B0 initialization')
 current=model.state_dict(); old=payload['model']
 shared={name:value for name,value in old.items() if '.classifier.' not in name}
 expected={name for name in current if '.classifier.' not in name}
 if set(shared)!=expected: raise RuntimeError('canonical backbone structure mismatch')
 current.update(shared); model.load_state_dict(current,strict=True)
 return model

def main():
 args=experiment.arguments()
 if args.variant!='G' or args.seed!=0: raise ValueError('this paired entrypoint is G seed0 only')
 original_build=experiment.build_model; original_task=experiment.make_task
 def build(variant,*,prepared=True): return initialize_common_backbone(original_build(variant,prepared=prepared))
 def task(arguments):
  value=original_task(arguments)
  return dataclasses.replace(value,execution=dataclasses.replace(value.execution,gpu_mixup=True),source_sha256={**value.source_sha256,'canonical_backbone_initialization':INITIAL_SHA,'h200_g_entrypoint':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 experiment.build_model=build; experiment.make_task=task
 if os.environ.get('VA64_G_MANAGED_ROOT'):
  from va64_runtime_operations import install_training
  install_training(experiment.runtime,Path(os.environ['VA64_G_MANAGED_ROOT']))
 result=experiment.train(args)
 if os.environ.get('VA64_G_MANAGED_ROOT'):
  from va64_runtime_operations import complete_run
  complete_run(Path(os.environ['VA64_G_MANAGED_ROOT']),experiment.make_task(args).output_dir,result['status'])
if __name__=='__main__': main()
