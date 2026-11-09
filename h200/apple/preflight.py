"""Data-only H200 gate. Never starts training or downloads a full shard."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from va64_control_transport import atomic,read

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args()
 def stop(sig,frame):
  # All descendants belong to this dedicated process group, created by guard.
  signal.signal(signal.SIGTERM,signal.SIG_DFL)
  os.killpg(os.getpgrp(),signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def current(stage):atomic(a.root/'current.json',{'job':'apple70m-data-preflight','campaign_id':a.root.name,'stage':stage,'output':str(a.root/'data-probe'),'checkpoint_policy':'local_only'})
 current('waiting_wandb');deadline=time.monotonic()+900
 while 'apple70m-data-preflight' not in read(a.root/'control.json').get('ready_jobs',[]):
  if (a.root/'STOP').exists():raise SystemExit(130)
  if time.monotonic()>deadline:raise RuntimeError('W&B observer readiness timeout')
  time.sleep(1)
 current('preflight')
 gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,uuid','--format=csv,noheader'],text=True,capture_output=True,timeout=10,check=True).stdout.strip()
 if 'H200' not in gpu:raise RuntimeError('Expected an H200 allocation')
 atomic(a.root/'environment.json',{'gpu':gpu,'python':sys.version,'scratch':str(a.scratch),'output':str(a.root),'training_started':False})
 print('APPLE_ENVIRONMENT='+json.dumps(read(a.root/'environment.json')),flush=True)
 rc=subprocess.call([sys.executable,str(Path(__file__).with_name('data_probe.py')),'--output-root',str(a.root),'--cache-root',str(a.scratch),'--stop-file',str(a.root/'STOP'),'--sample-bytes',str(8*1024**2)])
 if (a.root/'STOP').exists():raise SystemExit(130)
 report=read(a.root/'data-probe/report.json')
 compact={k:v for k,v in report.items() if k not in {'storage','tokenizer','dataset','capacity_plan'}}
 compact['dataset_revision']=report.get('dataset',{}).get('revision')
 result={'status':report.get('status','failed'),'training_started':False,'data_probe':compact,'next_gate':'sampler-order and scratch-training numerical validation; training not authorized by this probe'}
 atomic(a.root/'data-probe/result.json',result)
 # Results are small; emit for relay/W&B and server-side report retention.
 print('APPLE_DATA_RESULT='+json.dumps(result),flush=True)
 current('completed' if rc==0 else 'failed')
 raise SystemExit(rc)
if __name__=='__main__':main()
