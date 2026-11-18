"""Isolated reader benchmark; never writes an existing training run."""
from __future__ import annotations
import argparse,copy,fcntl,json,os,subprocess,time
from contextlib import nullcontext
from pathlib import Path
import torch
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from run_va64_retrain_imagenet1k import build_model as build_va64
from run_va_retrain_imagenet1k import build_model as build_va128
import imagenet1k_runtime as runtime


def set_mode(model,mode):
 count=0
 for mod in model.modules():
  if isinstance(mod,FactorizedComplexConv2dReader):mod.execution_mode=mode;count+=1
 return count


def errors(a,b):
 a=a.detach().float();b=b.detach().float();delta=b-a
 return {'max_abs':delta.abs().max().item(),'relative_l2':(delta.norm()/a.norm().clamp_min(1e-8)).item()}


def parity(model,output,mode="factorized_triton"):
 source=next(m for m in model.modules() if isinstance(m,FactorizedComplexConv2dReader))
 result={}
 for dtype in (torch.float32,torch.bfloat16):
  torch.manual_seed(809)
  dense=copy.deepcopy(source);fact=copy.deepcopy(source);dense.execution_mode='dense';fact.execution_mode=mode
  inputs=[torch.randn(2,9,11,source.input_modes,device='cuda',requires_grad=True) for _ in range(2)]
  copies=[x.detach().clone().requires_grad_() for x in inputs]
  with torch.autocast('cuda',dtype=dtype,enabled=dtype==torch.bfloat16):a=dense(*inputs);b=fact(*copies)
  probes=[torch.randn_like(x).float() for x in a]
  sum((x.float()*p+.1*x.float().square()).mean() for x,p in zip(a,probes)).backward()
  sum((x.float()*p+.1*x.float().square()).mean() for x,p in zip(b,probes)).backward()
  value=errors(torch.cat([x.reshape(-1) for x in a]),torch.cat([x.reshape(-1) for x in b]))
  grads={n:errors(p.grad,dict(fact.named_parameters())[n].grad) for n,p in dense.named_parameters()}
  all_a=[x.grad for x in inputs]+[p.grad for p in dense.parameters()];all_b=[x.grad for x in copies]+[p.grad for p in fact.parameters()]
  gradient=errors(torch.cat([x.reshape(-1) for x in all_a]),torch.cat([x.reshape(-1) for x in all_b]))
  finite=all(torch.isfinite(x).all().item() for x in [*a,*b,*all_a,*all_b])
  result[str(dtype)]={'output':value,'gradient':gradient,'parameter_gradients':grads,'finite':finite,'passed':finite and value['relative_l2']<(2e-2 if dtype==torch.bfloat16 else 1e-4) and gradient['relative_l2']<(5e-2 if dtype==torch.bfloat16 else 2e-4) and all(g['relative_l2']<(5e-2 if dtype==torch.bfloat16 else 2e-4) or g['max_abs']<1e-6 for g in grads.values())}
 (output/'parity.json').write_text(json.dumps(result,indent=2)+'\n');print('READER_PARITY='+json.dumps(result),flush=True)
 return all(x['passed'] for x in result.values())


def measure(model,mode,args):
 set_mode(model,mode);model.train().to(memory_format=torch.channels_last)
 if args.compile:model.compile(dynamic=False,mode=args.compile_mode)
 opt=runtime._build_optimizer(model,.003,torch.device('cuda'))
 torch.manual_seed(809);x=torch.randn(args.batch,3,224,224,device='cuda').contiguous(memory_format=torch.channels_last)
 labels=torch.randint(0,1000,(args.batch,),device='cuda')
 def update():
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(x);loss=torch.nn.functional.cross_entropy(pred.float(),labels)
  loss.backward();opt.step()
  return loss
 torch.cuda.reset_peak_memory_stats()
 print('READER_WARMUP mode='+mode,flush=True);start=time.perf_counter()
 for _ in range(args.warmup):loss=update()
 torch.cuda.synchronize();warmup=time.perf_counter()-start
 warmup_peak=torch.cuda.max_memory_allocated();warmup_reserved=torch.cuda.max_memory_reserved()
 torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 for _ in range(args.steps):loss=update()
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p).all()) for p in model.parameters())
 return {'family':args.family,'mode':mode,'warmup_seconds':warmup,'measured_seconds':elapsed,'updates':args.steps,'batch':args.batch,'images_per_second':args.batch*args.steps/elapsed,'peak_allocated_bytes':max(warmup_peak,torch.cuda.max_memory_allocated()),'peak_reserved_bytes':max(warmup_reserved,torch.cuda.max_memory_reserved()),'measurement_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'memory_scope':'peak includes warmup and CUDA Graph capture','loss':loss.item(),'finite':finite,'compiled':args.compile,'compile_mode':args.compile_mode if args.compile else None,'scope':'GPU-resident synthetic images, BF16 full-model forward/backward + fused AdamW; no image decoding or augmentation'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--mode',choices=('dense','factorized','factorized_triton'),default='factorized_triton');p.add_argument('--family',choices=('VA64','VA128'),default='VA64');p.add_argument('--gpu-uuid',required=True);p.add_argument('--gpu-lock',type=Path,required=True);p.add_argument('--batch',type=int,default=128);p.add_argument('--warmup',type=int,default=30);p.add_argument('--steps',type=int,default=100);p.add_argument('--compile',action='store_true');p.add_argument('--compile-mode',choices=('default','reduce-overhead'),default='default');args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=False)
 with args.gpu_lock.open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  if any(line.split(',')[0].strip()==args.gpu_uuid for line in rows.splitlines()):raise RuntimeError('assigned GPU occupied; no work launched')
  if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:raise RuntimeError('explicit GPU binding required')
  torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=True
  model=(build_va128 if args.family=='VA128' else build_va64)('B',prepared=True);state=torch.load(args.checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(state['model'],strict=True);del state
  initial=copy.deepcopy(model.state_dict());model.cuda()
  if args.mode!='dense' and not parity(model,args.output,args.mode):raise RuntimeError('reader parity gate failed; timing not run')
  model.load_state_dict(initial);torch.cuda.synchronize();torch.cuda.empty_cache();torch.set_float32_matmul_precision('high');torch.backends.cudnn.allow_tf32=True
  result=measure(model,args.mode,args);(args.output/'measurement.json').write_text(json.dumps(result,indent=2)+'\n');print('READER_MEASUREMENT='+json.dumps(result),flush=True)


if __name__=='__main__':main()
