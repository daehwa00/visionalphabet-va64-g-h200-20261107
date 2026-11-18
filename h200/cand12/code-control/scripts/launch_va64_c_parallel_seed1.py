"""Run C1 on a second GPU; release the paused serial supervisor after both finish."""
from __future__ import annotations
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--slot-lock',type=Path,required=True);p.add_argument('--serial-pid',type=int,required=True);a=p.parse_args();r=a.root.resolve();code=r/'code'
 sys.path.insert(0,str(code/'scripts'))
 import queue_va64_conv_control as q
 stop=r/'STOP-seed1';state={'seed':1,'gpu_uuid':a.gpu_uuid,'serial_pid':a.serial_pid,'supervisor_pid':os.getpid()}
 def save(stage,**extra):
  state.update(state=stage,updated_unix=time.time(),**extra);q.atomic(r/'parallel-seed1-state.json',state)
 signal.signal(signal.SIGTERM,lambda *_:stop.touch());signal.signal(signal.SIGINT,lambda *_:stop.touch())
 with (r/'seed1-parallel.lock').open('a+') as own,a.slot_lock.open('a+') as slot:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
  pinned=q.verify_sources(code,r/'source-manifest.json');budget=q.digest(r/'budget.json');data=q.digest(r/'dataset-manifest.json')
  def verify():
   if q.verify_sources(code,r/'source-manifest.json')!=pinned or q.digest(r/'budget.json')!=budget or q.digest(r/'dataset-manifest.json')!=data:raise RuntimeError('pinned inputs changed')
  command=[sys.executable,str(code/'scripts/run_va64_conv_control.py'),'--code',str(code),'--seed','1','--data-root','/home/qlab/data/ImageNet/2012','--data-manifest',str(r/'dataset-manifest.json'),'--source-manifest',str(r/'source-manifest.json'),'--budget',str(r/'budget.json'),'--batch-size','128','--workers','8','--stop-file',str(stop)]
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,PYTHONPATH=f'{code}/src:{code}/scripts',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4',PYTHONUNBUFFERED='1',LNET_DISABLE_LAUNCH_AUTOTUNE='1',TRITON_CACHE_DIR=str(r/'cache-seed1/triton'),TORCHINDUCTOR_CACHE_DIR=str(r/'cache-seed1/inductor'))
  try:
   for stage,output,extra in [('smoke',r/'diagnostics',['--smoke-updates','20']),('training',r/'runs',[])]:
    if stop.exists() or (r/'STOP').exists():raise InterruptedError('STOP')
    verify()
    if shutil.disk_usage(r).free<q.MIN_FREE_BYTES:raise RuntimeError('less than15GiB')
    if q.gpu_pids(a.gpu_uuid):raise RuntimeError('GPU occupied')
    directory=output/q.MODEL_KEY/'seed_1';log=r/'logs'/f'seed1-{stage}.log'
    if directory.exists() or log.exists():raise FileExistsError('existing C1 attempt; inspect before retry')
    cmd=command+['--output-root',str(output)]+extra
    with log.open('x') as stream:
     child=subprocess.Popen(cmd,cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);save(stage,command=cmd,child_pid=child.pid)
     try:rc=q.wait_child(child,stop)
     finally:
      if child.poll() is None:
       os.killpg(child.pid,signal.SIGTERM)
       try:child.wait(timeout=20)
       except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=20)
    if stop.exists():raise InterruptedError('STOP')
    if rc:raise RuntimeError(f'{stage} exited {rc}')
    if stage=='smoke':
     value=json.loads((directory/'smoke/result.json').read_text())
     if value.get('status')!='smoke_passed' or value.get('updates')!=20:raise RuntimeError('smoke failed')
    else:q.completed(directory/'result.json',seed=1,model_key=q.MODEL_KEY)
   save('seed1_completed_waiting_seed0')
   zero=r/'runs'/q.MODEL_KEY/'seed_0/result.json'
   while not zero.exists():
    if stop.exists() or (r/'STOP').exists():raise InterruptedError('STOP')
    time.sleep(30)
   q.completed(zero,seed=0,model_key=q.MODEL_KEY)
   # The serial loop now validates C0 and skips the completed C1 result.
   cmdline=Path(f'/proc/{a.serial_pid}/cmdline').read_bytes()
   if b'queue_va64_conv_control.py' not in cmdline:raise RuntimeError('serial supervisor identity changed')
   os.kill(a.serial_pid,signal.SIGCONT);save('completed_serial_supervisor_released')
  except BaseException as e:
   save('stopped' if isinstance(e,InterruptedError) else 'failed',error=type(e).__name__+': '+str(e));raise

if __name__=='__main__':main()
