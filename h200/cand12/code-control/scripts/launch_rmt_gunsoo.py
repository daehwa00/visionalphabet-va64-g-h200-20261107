"""Select measured RMT-T execution settings, verify a smoke, and train seed 0."""
from __future__ import annotations
import argparse,fcntl,json,os,shutil,subprocess,sys,time,signal
from pathlib import Path


def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--resume-calibration',action='store_true');args=parser.parse_args()
 root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
 with (root/'gpu.lock').open('a+') as slot:
  fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
  previous=None
  if (root/'pipeline-state.json').exists():
   previous=json.loads((root/'pipeline-state.json').read_text())
   if not args.resume_calibration or (root/'runs').exists() or previous.get('gpu_uuid')!=args.gpu_uuid:
    raise FileExistsError('existing pipeline requires inspection, never overwrite it')
  if shutil.disk_usage(root).free<15*1024**3:raise RuntimeError('less than 15 GiB free')
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  if any(line.split(',')[0].strip()==args.gpu_uuid for line in rows.splitlines()):raise RuntimeError('assigned GPU is busy')
  state={'started_unix':time.time(),'gpu_uuid':args.gpu_uuid,'supervisor_pid':os.getpid(),'recipe':'shared VA64 100ep/global256; official RMT-T topology/drop-path retained'}
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,PYTHONPATH=f'{root}/code/src:{root}/code/scripts:{root}/runtime-deps',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4',LNET_DISABLE_LAUNCH_AUTOTUNE='1',PYTHONUNBUFFERED='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TRITON_CACHE_DIR=str(root/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(root/'cache/inductor'))
  logs=root/'logs';logs.mkdir(exist_ok=True)
  stopped_at=[None]
  def stop(_signal=None,_frame=None):
   (root/'STOP').touch()
   if stopped_at[0] is None:stopped_at[0]=time.monotonic()
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  def run(command,stage):
   if (root/'STOP').exists():raise InterruptedError('STOP requested')
   state.update(stage=stage,command=command,updated_unix=time.time());atomic(root/'pipeline-state.json',state)
   log=logs/(stage+'.log')
   if log.exists():log=logs/(stage+'-'+str(time.time_ns())+'.log')
   with log.open('x') as stream:
    p=subprocess.Popen(command,cwd=root/'code',env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);state['child_pid']=p.pid;atomic(root/'pipeline-state.json',state)
    try:
     while True:
      try:
       code=p.wait(timeout=1);break
      except subprocess.TimeoutExpired:
       if (root/'STOP').exists() and stopped_at[0] is None:stopped_at[0]=time.monotonic()
       if stopped_at[0] is not None and time.monotonic()-stopped_at[0]>120:
        os.killpg(p.pid,signal.SIGTERM)
        try:code=p.wait(timeout=10)
        except subprocess.TimeoutExpired:
         os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
        break
    finally:
     if p.poll() is None:
      (root/'STOP').touch();os.killpg(p.pid,signal.SIGTERM)
      try:p.wait(timeout=10)
      except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
   if (root/'STOP').exists():raise InterruptedError('STOP requested')
   state['last_returncode']=code;atomic(root/'pipeline-state.json',state)
   return code
  def benchmark(name,mode,batches,workers):
   if os.environ.get("RMT_INDEX_ROOT"):name=name.replace("benchmark-","benchmark-indexed-",1)
   out=root/name
   if args.resume_calibration and (out/'selection.json').exists():
    value=json.loads((out/'selection.json').read_text());value['measurement']=next(row['measurement'] for row in value['results'] if row.get('status')=='passed' and row['physical_batch']==value['selected_batch_size']);value['benchmark_directory']=str(out)
    if value['mode']!=mode or value['workers']!=workers or value['batch_size'] not in list(map(int,batches.split(','))):raise RuntimeError('cached calibration configuration mismatch')
    return value
   command=[sys.executable,str(root/'code/scripts/benchmark_rmt_training.py'),'--rmt-source',str(root/'third_party/RMT'),'--data-root','/home/qai/datasets/imagenet1k-ext4','--output',str(out),'--batches',batches,'--workers',str(workers),'--mode',mode,'--warmup-updates','20','--measure-updates','50','--stop-file',str(root/'STOP')]
   rc=run(command,name)
   selection=out/'selection.json'
   if rc and not selection.exists():
    reports=[json.loads(p.read_text()) for p in out.glob('batch-*/result.json')]
    if reports and all(x.get('status')=='oom' for x in reports):return None
    raise RuntimeError(f'{name} failed, inspect log before training')
   value=json.loads(selection.read_text());value['measurement']=next(row['measurement'] for row in value['results'] if row.get('status')=='passed' and row['physical_batch']==value['selected_batch_size']);value['benchmark_directory']=str(out);return value
  try:
   candidates=[]
   eager_attempted=False
   try:
    selected=benchmark('benchmark-compiled','compiled','16,32,64,128',4)
   except RuntimeError as error:
    # Change execution mode, never repeat the same failing configuration.
    state['compiled_benchmark_failure']=str(error);atomic(root/'pipeline-state.json',state)
    eager_attempted=True
    selected=benchmark('benchmark-eager-fallback','eager','16,32,64,128',4)
   if selected is None and not eager_attempted:
    selected=benchmark('benchmark-eager-fallback','eager','16,32,64,128',4)
   if selected is None:raise RuntimeError('no baseline configuration fits')
   candidates.append(selected)
   if selected['compile_model']:
    eager=benchmark('benchmark-eager','eager',str(selected['batch_size']),4)
    if eager:candidates.append(eager)
   best=max(candidates,key=lambda x:x['measurement']['images_per_second'])
   workers=benchmark('benchmark-workers8',best['mode'],str(best['batch_size']),8)
   if workers:candidates.append(workers)
   best=max(candidates,key=lambda x:x['measurement']['images_per_second']);atomic(root/'resolved-execution.json',{'selected':best,'candidates':candidates})
   if best['compile_model'] and env.get('RMT_COMPILE_CACHE_BASE'):
    cache=Path(env['RMT_COMPILE_CACHE_BASE'])/f"cache-{best['batch_size']}"
    env.update(TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
   command=[sys.executable,str(root/'code/scripts/run_rmt_matched_imagenet1k.py'),'--rmt-source',str(root/'third_party/RMT'),'--model','RMT_T3','--seed','0','--data-root','/home/qai/datasets/imagenet1k-ext4','--data-manifest',str(root/'dataset-manifest.json'),'--batch-size',str(best['batch_size']),'--workers',str(best['workers']),'--stop-file',str(root/'STOP')]
   if not best['compile_model']:command+=['--no-compile']
   if run(command+['--output-root',str(root/'diagnostics'),'--smoke-updates','20'],'smoke'):raise RuntimeError('20-update smoke failed')
   rc=run(command+['--output-root',str(root/'runs')],'training')
   result=root/'runs/RMT_T3/seed_0/result.json'
   if rc or not result.is_file():raise RuntimeError('training did not finish, inspect local checkpoint/logs')
   final=json.loads(result.read_text());assert final['status']=='completed' and final['completed_epochs']==100
   state.update(stage='completed',result=str(result),finished_unix=time.time());atomic(root/'pipeline-state.json',state)
  except BaseException as error:
   state.update(stage='stopped' if (root/'STOP').exists() else 'failed',error=type(error).__name__+': '+str(error),finished_unix=time.time());atomic(root/'pipeline-state.json',state);raise


if __name__=='__main__':main()
