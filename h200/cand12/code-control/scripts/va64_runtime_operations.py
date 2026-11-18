"""Managed G control and optional telemetry; training checkpoints stay local."""
from __future__ import annotations
import json,os,signal,shutil,time
from pathlib import Path
from va64_control_transport import API,BASE,atomic as atomic_json

def _agent_api():
 return API(os.environ["H200_AGENT_TOKEN"],os.environ["H200_SESSION_ID"],base=os.environ.get("VA64_G_RELAY_URL",BASE))


def current(root: Path, job: str, stage: str, output: Path) -> None:
 atomic_json(root/'current.json',{'job':job,'campaign_id':root.name,'stage':stage,'output':str(output),'checkpoint_policy':os.environ.get('VA64_G_CHECKPOINT_POLICY','external')})


def wait_ready(root: Path,job: str,seconds: float=900) -> None:
 if os.environ.get('VA64_G_CHECKPOINT_POLICY')=='local_only':
  current(root,job,'bootstrap',root); return
 current(root,job,'waiting_wandb',root)
 api=_agent_api(); deadline=time.monotonic()+seconds
 while time.monotonic()<deadline:
  control=api.call('/control',attempts=1,timeout=10)
  if control.get('stop') or (root/'STOP').exists(): raise RuntimeError('stop requested before job start')
  if job in control.get('ready_jobs',[]):
   current(root,job,'bootstrap',root); return
  time.sleep(3)
 raise TimeoutError('external W&B observer did not become ready')


def backup_and_wait(root: Path,job: str,output: Path,epoch: int,step: int,row: dict | None=None) -> dict:
 from va64_h200_transport import backup_run
 from imagenet1k_runtime import TrainingInterrupted
 if (root/'STOP').exists(): raise TrainingInterrupted('stop requested before backup')
 current(root,job,'wait_backup',output)
 telemetry=output/'telemetry'; telemetry.mkdir(exist_ok=True)
 contracts=output/'contracts'; contracts.mkdir(exist_ok=True)
 for source,name in ((root/'environment-check/dataset-manifest.json','dataset.json'),(root/'environment-check/preflight.json','contracts/environment.json')):
  if source.is_file(): shutil.copyfile(source,output/name)
 atomic_json(contracts/'source.json',{'code_sha':os.environ['VA64_G_CODE_SHA'],'variant':'G','job':job})
 if (root/'console.log').is_file(): shutil.copyfile(root/'console.log',telemetry/'console.txt')
 if (root/'benchmark/selection.json').is_file():
  shutil.copyfile(root/'benchmark/selection.json',telemetry/'batch-comparison.json')
 meta={**(row or {}),'epoch':epoch,'global_step':step,'status':'checkpoint_committed'}
 atomic_json(output/'checkpoint-meta.json',meta)
 def cancel_check():
  if (root/'STOP').exists(): raise TrainingInterrupted('stop requested; cancel backup')
 def cancel_signal(*_):
  (root/'STOP').touch()
  raise TrainingInterrupted('stop signal; cancel backup without waiting')
 previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
 try:
  for sig in previous: signal.signal(sig,cancel_signal)
  receipt=backup_run(job,output,epoch,step,os.environ['VA64_G_CODE_SHA'],wait=True,cancel_check=cancel_check)
 finally:
  for sig,handler in previous.items(): signal.signal(sig,handler)
 api=_agent_api(); deadline=time.monotonic()+900
 while time.monotonic()<deadline:
  if (root/'STOP').exists(): raise TrainingInterrupted('stop requested during backup acknowledgement')
  latest=api.call('/artifact/'+job+'/latest',attempts=1,timeout=15)
  if latest and latest.get('sha')==receipt['sha'] and latest.get('verified'):
   atomic_json(output/'external-backup-receipt.json',receipt)
   return receipt
  time.sleep(3)
 raise TimeoutError('checkpoint uploaded, but external W&B/disk confirmation is missing')


def install_training(runtime,root: Path) -> None:
 os.environ['VA64_G_CHECKPOINT_POLICY']='local_only'
 original=runtime.run_task
 def managed(task,**kwargs):
  job=os.environ.get('VA64_G_JOB',f'va64-g-seed{getattr(task,"seed",0)}')
  old_callback=kwargs.get('epoch_callback')
  def epoch_callback(model,row):
   if old_callback: old_callback(model,row)
   atomic_json(task.output_dir/'checkpoint-meta.json',{**row,'status':'local_checkpoint','checkpoint_policy':'local_only'})
   current(root,job,'training',task.output_dir)
  kwargs['epoch_callback']=epoch_callback
  kwargs['stop_without_checkpoint']=True
  for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:(root/'STOP').touch())
  current(root,job,'training',task.output_dir)
  result=original(task,**kwargs)
  return result
 runtime.run_task=managed


def complete_run(root: Path, output: Path, status: str) -> None:
 # Only small local results are collected by the independent telemetry guard.
 # There is no network request, checkpoint upload, or acknowledgement wait here.
 summary={'status':status}
 for name in ('result','cost','latest-epoch'):
  path=output/(name+'.json')
  if path.is_file():
   value=json.loads(path.read_text())
   if name=='result': value={k:v for k,v in value.items() if k!='history'}
   summary[name]=value
 atomic_json(output/'final-summary.json',summary)
 print('VA64_FINAL_SUMMARY='+json.dumps(summary),flush=True)
 job=os.environ.get('VA64_G_JOB','va64-g-seed0')
 if status=='completed':
  path=root/'completed-runs.json'
  completed=json.loads(path.read_text()) if path.is_file() else {}
  completed[job]=summary
  atomic_json(path,completed)
 current(root,job,status,output)


def main():
 import argparse
 p=argparse.ArgumentParser(); p.add_argument('action',choices=('ready','backup-smoke')); p.add_argument('--root',type=Path,required=True); p.add_argument('--job',required=True,choices=('va64-g-preflight','va64-g-seed0','va64-g-seed1','va64-g-seed2')); p.add_argument('--output',type=Path); a=p.parse_args()
 if a.action=='ready': wait_ready(a.root,a.job)
 else:
  if a.output is None: p.error('--output required')
  report=json.loads((a.output/'smoke.json').read_text())
  current(a.root,a.job,'preflight',a.output)
  backup_and_wait(a.root,a.job,a.output,0,report['steps'])
  current(a.root,a.job,'completed',a.output)
if __name__=='__main__': main()
