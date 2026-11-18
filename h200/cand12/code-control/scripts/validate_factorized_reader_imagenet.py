"""Paired validation smoke test; no training state or production files changed."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode,v2
from torch.utils.data import DataLoader,Subset
from benchmark_factorized_reader import build_va64,build_va128
from benchmark_factorized_reader import set_mode


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--gpu-lock',type=Path,required=True);p.add_argument('--examples',type=int,default=1024);p.add_argument('--family',choices=('VA64','VA128'),default='VA64');a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 with a.gpu_lock.open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  if any(x.split(',')[0].strip()==a.gpu_uuid for x in apps.splitlines()):raise RuntimeError('GPU occupied')
  if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:raise RuntimeError('GPU binding differs')
  torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.set_float32_matmul_precision('high')
  model=(build_va128 if a.family=='VA128' else build_va64)('B',prepared=True);cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(cp['model'],strict=True);del cp;model.cuda().eval()
  tf=v2.Compose([v2.Resize(256,interpolation=InterpolationMode.BICUBIC,antialias=True),v2.CenterCrop(224),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])
  ds=ImageFolder(a.data_root/'val',transform=tf)
  if len(ds)!=50000 or len(ds.classes)!=1000:raise RuntimeError('expected full ImageNet validation set')
  indices=torch.randperm(len(ds),generator=torch.Generator().manual_seed(809))[:a.examples].tolist()
  loader=DataLoader(Subset(ds,indices),batch_size=64,num_workers=4,pin_memory=True,multiprocessing_context='spawn')
  dense_correct=fact_correct=agree=total=0;diff2=ref2=0.;maxabs=0.;start=time.perf_counter()
  with torch.inference_mode():
   for images,labels in loader:
    x=images.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last);labels=labels.cuda(non_blocking=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     set_mode(model,'dense');ref=model(x).float()
     set_mode(model,'factorized_triton');out=model(x).float()
    if not bool(torch.isfinite(ref).all() and torch.isfinite(out).all()):raise RuntimeError('nonfinite logits')
    dense_correct+=(ref.argmax(1)==labels).sum().item();fact_correct+=(out.argmax(1)==labels).sum().item();agree+=(out.argmax(1)==ref.argmax(1)).sum().item();total+=labels.numel()
    diff2+=(out-ref).square().sum().item();ref2+=ref.square().sum().item();maxabs=max(maxabs,(out-ref).abs().max().item())
  result={'family':a.family,'factorized_execution':'factorized_triton','examples':total,'sampling':'seed809 fixed random sample of the full 50000-image validation set','dense_top1_percent':100*dense_correct/total,'factorized_top1_percent':100*fact_correct/total,'top1_difference_pp':100*(fact_correct-dense_correct)/total,'prediction_agreement_percent':100*agree/total,'logits_relative_l2':(diff2/max(ref2,1e-30))**.5,'logits_max_abs':maxabs,'dtype':'BF16 autocast','seconds':time.perf_counter()-start,'caveat':'Paired 1024-image smoke test, not a full validation result or equivalence proof.'}
  a.output.write_text(json.dumps(result,indent=2)+'\n');print('READER_VALIDATION='+json.dumps(result),flush=True)


if __name__=='__main__':main()
