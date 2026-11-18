"""Diagnostic GPU training benchmark; no authoritative training checkpoints."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def trial(a):
 sys.path[:0]=[str(a.code/'scripts'),str(a.code/'src'),str(Path(__file__).parent)]
 import torch
 import run_va64_conv_scan_control as entry
 import va64_conv_scan_fast as fast
 torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
 torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
 args=entry.arguments(['--code',str(a.code),'--source-manifest',str(a.source_manifest),'--budget',str(a.cost),'--seed','0','--data-root','/home/qlab/data/ImageNet/2012','--data-manifest',str(a.data_manifest),'--output-root',str(a.output/'diagnostic-only'),'--batch-size','128','--workers','0','--stop-file',str(a.output/'STOP')])
 task=entry.make_task(args);model=entry.control.build_model()
 if a.block=='triton':fast.install(model,4,backend='triton')
 elif a.block!='baseline':fast.install(model,int(a.block))
 device=torch.device('cuda');model=model.to(device).to(memory_format=torch.channels_last)
 torch._dynamo.config.recompile_limit=64
 model.compile(mode=a.compile_mode,dynamic=False)
 optimizer=entry.runtime._build_optimizer(model,task.learning_rate,device)
 batches=[(torch.randn(128,3,224,224,device=device).to(memory_format=torch.channels_last),torch.randint(0,1000,(128,),device=device)) for _ in range(2)]
 class Loader:
  def __init__(self,updates):self.n=updates*2
  def __len__(self):return self.n
  def __iter__(self):
   for i in range(self.n):
    if a.compile_mode=='reduce-overhead':torch.compiler.cudagraph_mark_step_begin()
    yield batches[i%2]
 start=time.monotonic()
 warm,_=entry.runtime._train_epoch(model,Loader(20),optimizer,entry.runtime._make_mixup(task),task,device,0)
 torch.cuda.synchronize();warm_seconds=time.monotonic()-start
 torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 measured,steps=entry.runtime._train_epoch(model,Loader(50),optimizer,entry.runtime._make_mixup(task),task,device,20)
 torch.cuda.synchronize();seconds=time.monotonic()-start
 if measured['optimizer_steps']!=50 or not all(torch.isfinite(p).all().item() for p in model.parameters()):raise RuntimeError('nonfinite/missing updates')
 validation=entry.runtime._evaluate(model,Loader(5),task,device)
 if not torch.isfinite(torch.tensor(validation['cross_entropy'])):raise RuntimeError('nonfinite validation')
 result={'validation_test':validation,'block':a.block,'compile_mode':a.compile_mode,'status':'passed','device':torch.cuda.get_device_name(),'physical_batch':128,'global_batch':256,'accumulation':2,'warmup_updates':20,'measure_updates':50,'warmup_seconds_including_compile':warm_seconds,'seconds':seconds,'images_per_second':12800/seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'loss':measured['loss'],'data':'preallocated synthetic device batches; isolates model optimizer and GPU mixup'}
 (a.output/f'block-{a.block}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def main():
 p=argparse.ArgumentParser()
 for name in ('code','source-manifest','cost','data-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--compile-mode',default='default',choices=['default','reduce-overhead']);p.add_argument('--block');p.add_argument('--blocks',default='baseline,1,4,8');p.add_argument('--pause-pid',type=int);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 if a.block:trial(a);return
 original=a.pause_pid
 if original:
  if b'run_va64_conv_scan_control.py' not in Path(f'/proc/{original}/cmdline').read_bytes():raise RuntimeError('foreign training PID')
  os.kill(original,signal.SIGSTOP)
 def timeout(*_):raise TimeoutError('bounded benchmark deadline')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
 try:
  for block in a.blocks.split(','):
   cmd=[sys.executable,str(Path(__file__).resolve()),'--code',str(a.code),'--source-manifest',str(a.source_manifest),'--cost',str(a.cost),'--data-manifest',str(a.data_manifest),'--output',str(a.output),'--block',block,'--compile-mode',a.compile_mode]
   env=os.environ.copy();env.update(TORCHINDUCTOR_CACHE_DIR=str(a.output/f'cache-{block}/inductor'),TRITON_CACHE_DIR=str(a.output/f'cache-{block}/triton'),TORCHINDUCTOR_COMPILE_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',LNET_DISABLE_LAUNCH_AUTOTUNE='1')
   with (a.output/f'block-{block}.log').open('x') as log:
    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
   print((a.output/f'block-{block}.json').read_text(),flush=True)
 finally:
  signal.alarm(0)
  if original and Path(f'/proc/{original}').exists():os.kill(original,signal.SIGCONT)

if __name__=='__main__':main()
