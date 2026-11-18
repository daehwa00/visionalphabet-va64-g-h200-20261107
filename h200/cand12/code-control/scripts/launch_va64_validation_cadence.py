"""Switch evaluation cadence at the next completed epoch without changing training."""
from __future__ import annotations
import argparse,fcntl,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old-root',type=Path,required=True);p.add_argument('--seed',type=int,choices=(0,1),required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--slot-lock',type=Path,required=True);p.add_argument('--after-epoch',type=int,required=True);a=p.parse_args();r=a.root.resolve();own=r/f'seed_{a.seed}';own.mkdir(parents=True,exist_ok=True);(own/'logs').mkdir(exist_ok=True)
 sys.path.insert(0,str(r/'code/scripts'))
 import queue_va64_conv_scan_control as q
 stop=own/'STOP';signal.signal(signal.SIGTERM,lambda *_:stop.touch());signal.signal(signal.SIGINT,lambda *_:stop.touch())
 state={'seed':a.seed,'gpu_uuid':a.gpu_uuid,'supervisor_pid':os.getpid()}
 def save(stage,**extra):state.update(state=stage,updated_unix=time.time(),**extra);q.atomic(own/'upgrade-state.json',state)
 selection=json.loads((r/'selection.json').read_text())
 if selection.get('status')!='passed':raise RuntimeError('benchmark/correctness gates not passed')
 backend=selection['backend'];block=selection['block_modes'];old_seed=a.old_root/f'seed_{a.seed}';old_dir=old_seed/f'runs/va_k64_retrain_Cconv/seed_{a.seed}'
 pinned=q.verify_sources(r/'code',r/'source-manifest.json');cost=q.digest(r/'cost-profile.json');data=q.digest(r/'dataset-manifest.json')
 def verify():
  if q.verify_sources(r/'code',r/'source-manifest.json')!=pinned or q.digest(r/'cost-profile.json')!=cost or q.digest(r/'dataset-manifest.json')!=data:raise RuntimeError('pinned inputs changed')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,PYTHONPATH=f'{r}/code/src:{r}/code/scripts',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='4',PYTHONUNBUFFERED='1',LNET_DISABLE_LAUNCH_AUTOTUNE='1',TORCHINDUCTOR_CACHE_DIR=str(old_seed/'cache/inductor'),TRITON_CACHE_DIR=str(old_seed/'cache/triton'))
 def run(cmd,stage):
  path=own/'logs'/f'{stage}.log'
  with path.open('x') as out:
   child=subprocess.Popen(cmd,env=env,cwd=r/'code',stdout=out,stderr=subprocess.STDOUT,start_new_session=True);save(stage,child_pid=child.pid,command=cmd)
   try:rc=q.wait_child(child,stop)
   finally:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=20)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=20)
  if rc or stop.exists():raise RuntimeError(f'{stage} exited{rc}')
 with (own/'upgrade.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   save('waiting_for_epoch_boundary',after_epoch=a.after_epoch)
   def ready():
    path=old_dir/'latest-epoch.json'
    return (old_dir/'checkpoint.pt').exists() and path.exists() and json.loads(path.read_text())['epoch']>=a.after_epoch
   while not ready():
    if stop.exists():save('stopped_before_transition');return
    verify();time.sleep(3)
   old_state=json.loads((old_seed/'upgrade-state.json').read_text());old_pid=old_state['child_pid']
   if Path(f'/proc/{old_pid}/cmdline').exists() and b'run_va64_conv_scan_control.py' not in Path(f'/proc/{old_pid}/cmdline').read_bytes():raise RuntimeError('foreign training process')
   if (old_dir/'result.json').exists():q.completed(old_dir/'result.json',seed=a.seed,model_key='va_k64_retrain_Cconv');save('old_run_already_completed');return
   (old_seed/'STOP').touch();save('normal_stop_requested',old_pid=old_pid)
   deadline=time.monotonic()+120
   while Path(f'/proc/{old_pid}').exists():
    if time.monotonic()>deadline:raise TimeoutError('normal stop did not finish; no forced termination')
    time.sleep(1)
   with a.slot_lock.open('a+') as slot:
    while True:
     if stop.exists():raise InterruptedError('STOP')
     try:fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB);break
     except BlockingIOError:save('waiting_gpu_lock');time.sleep(3)
    while q.gpu_pids(a.gpu_uuid):
     if stop.exists():raise InterruptedError('STOP')
     save('waiting_gpu_release');time.sleep(3)
    verify()
    if shutil.disk_usage(r).free<q.MIN_FREE_BYTES:raise RuntimeError('less than15GiB free')
    run([sys.executable,str(r/'control/migrate_va64_validation_cadence.py'),'--old-directory',str(old_dir),'--new-root',str(r),'--seed',str(a.seed),'--block',str(block),'--backend',backend],'migration')
    base=[sys.executable,str(r/'code/scripts/run_va64_conv_scan_control.py'),'--code',str(r/'code'),'--source-manifest',str(r/'source-manifest.json'),'--budget',str(r/'cost-profile.json'),'--execution-block',str(block),'--execution-backend',backend,'--validation-interval','20','--seed',str(a.seed),'--data-root','/home/qlab/data/ImageNet/2012','--data-manifest',str(r/'dataset-manifest.json'),'--batch-size','128','--workers','8','--stop-file',str(stop)]
    verify();run(base+['--output-root',str(own/'runs'),'--resume'],'training')
    q.completed(own/f'runs/va_k64_retrain_Cconv/seed_{a.seed}/result.json',seed=a.seed,model_key='va_k64_retrain_Cconv');save('completed')
  except BaseException as e:
   save('stopped' if stop.exists() else 'failed',error=type(e).__name__+': '+str(e));raise

if __name__=='__main__':main()
