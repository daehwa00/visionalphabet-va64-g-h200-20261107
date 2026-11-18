"""Start the official RMT-T comparison on the free kau 4090 after one smoke."""
from __future__ import annotations
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path


def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--slot-lock',type=Path,required=True);p.add_argument('--seed',type=int,choices=(0,1,2),default=0);p.add_argument('--batch-size',type=int,choices=(16,32,64,128),default=128);p.add_argument('--workers',type=int,default=8);p.add_argument('--data-root',default='/data/ImageNet/2012');p.add_argument('--index-root',type=Path);a=p.parse_args();r=a.root.resolve()
 with a.slot_lock.open('a+') as slot,(r/'supervisor.lock').open('a+') as own:
  fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (r/'state.json').exists() or (r/'runs').exists():raise FileExistsError('inspect previous attempt before retry')
  if shutil.disk_usage(r).free<15*1024**3:raise RuntimeError('less than 15 GiB free')
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  if any(x.split(',')[0].strip()==a.gpu_uuid for x in rows.splitlines()):raise RuntimeError('GPU is occupied')
  env=os.environ.copy();env.pop('RMT_INDEX_ROOT',None);env.pop('RMT_COMPILE_CACHE_BASE',None)
  env.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,PYTHONPATH=f'{r}/code/src:{r}/code/scripts:{r}/runtime-deps',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4',PYTHONUNBUFFERED='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TRITON_CACHE_DIR=str(r/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(r/'cache/inductor'),LNET_DISABLE_LAUNCH_AUTOTUNE='1')
  if a.index_root is not None:env['RMT_INDEX_ROOT']=str(a.index_root.resolve())
  state={'owner':str(r),'gpu_uuid':a.gpu_uuid,'supervisor_pid':os.getpid(),'started_unix':time.time(),'batch_size':a.batch_size,'accumulation':256//a.batch_size,'global_batch':256,'workers':a.workers,'compile':True,'precision':'bfloat16','model':'RMT_T3','seed':a.seed,'epochs':100}
  stopped=[None]
  def stop(*_):
   (r/'STOP').touch()
   if stopped[0] is None:stopped[0]=time.monotonic()
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  command=[sys.executable,str(r/'code/scripts/run_rmt_matched_imagenet1k.py'),'--rmt-source',str(r/'third_party/RMT'),'--model','RMT_T3','--seed',str(a.seed),'--data-root',a.data_root,'--data-manifest',str(r/'dataset-manifest.json'),'--batch-size',str(a.batch_size),'--workers',str(a.workers),'--stop-file',str(r/'STOP')]
  try:
   for stage,extra in [('smoke',['--output-root',str(r/'diagnostics'),'--smoke-updates','20']),('training',['--output-root',str(r/'runs')])]:
    if (r/'STOP').exists():raise InterruptedError('STOP')
    cmd=command+extra;state.update(stage=stage,command=cmd,updated_unix=time.time());atomic(r/'state.json',state)
    with (r/'logs'/(stage+'.log')).open('x') as log:
     child=subprocess.Popen(cmd,cwd=r/'code',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['child_pid']=child.pid;atomic(r/'state.json',state)
     try:
      while True:
       try:rc=child.wait(timeout=1);break
       except subprocess.TimeoutExpired:
        if (r/'STOP').exists() and stopped[0] is None:stopped[0]=time.monotonic()
        if stopped[0] is not None and time.monotonic()-stopped[0]>120:
         os.killpg(child.pid,signal.SIGTERM)
         try:rc=child.wait(timeout=10)
         except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=10)
         break
     finally:
      if child.poll() is None:
       os.killpg(child.pid,signal.SIGTERM)
       try:child.wait(timeout=10)
       except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
    state['last_returncode']=rc
    if rc:raise RuntimeError(stage+' failed; inspect log')
   result=json.loads((r/f'runs/RMT_T3/seed_{a.seed}/result.json').read_text());assert result['status']=='completed' and result['completed_epochs']==100
   state.update(stage='completed',finished_unix=time.time());atomic(r/'state.json',state)
  except BaseException as e:
   state.update(stage='stopped' if (r/'STOP').exists() else 'failed',error=type(e).__name__+': '+str(e),finished_unix=time.time());atomic(r/'state.json',state);raise


if __name__=='__main__':main()
