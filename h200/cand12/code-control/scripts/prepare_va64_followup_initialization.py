"""Create matching, untrained B/P/A initialization for seeds 1 and 2 on CPU."""
from __future__ import annotations
import argparse,hashlib,json,random,sys
from pathlib import Path


def state_digest(state, *, common=False):
 digest=hashlib.sha256()
 for name,tensor in state.items():
  if common and ('.classifier.' in name or name.endswith(('.phase_x','.phase_y'))):continue
  digest.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)]).encode())
  digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
 return digest.hexdigest()


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--code',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 sys.path[:0]=[str(args.code/'src'),str(args.code/'scripts')]
 import numpy as np
 import torch
 import run_va64_retrain_imagenet1k as experiment
 from queue_va_retrain import verify_source
 torch.set_num_threads(4);source=verify_source(args.code)
 if args.output.exists():raise FileExistsError('preserve the existing canonical initialization directory')
 args.output.mkdir(parents=True)
 seed_digests=[]
 for seed in (1,2):
  folder=args.output/f'seed_{seed}';folder.mkdir();manifest={'seed':seed,'source_manifest_sha256':source,'variants':{}}
  for variant in 'BPA':
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
   model=experiment.build_model(variant);state=model.state_dict()
   if variant=='P':experiment.assert_zero_phases(model)
   path=folder/f'initial-{variant}{seed}.pt'
   torch.save({'model':state,'seed':seed,'variant':variant,'trained':False,'source_manifest_sha256':source},path)
   manifest['variants'][variant]={'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'state_sha256':state_digest(state),'common_backbone_sha256':state_digest(state,common=True)}
  common={v['common_backbone_sha256'] for v in manifest['variants'].values()}
  if len(common)!=1:raise RuntimeError('B/P/A common backbone initialization differs')
  seed_digests.append(common.pop());(folder/'canonical.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)
 if len(set(seed_digests))!=2:raise RuntimeError('different seeds produced identical common backbones')


if __name__=='__main__':main()
