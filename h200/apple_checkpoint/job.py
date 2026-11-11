"""H200 gates followed by V-only scratch training; stop skips backup/evaluation."""
import argparse,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from va64_control_transport import atomic,read

def validate_smoke(result):
 if result.get('status')!='completed' or result.get('optimizer_updates',0)<16:
  raise RuntimeError('Real-data smoke needs sixteen successful optimizer updates')
 if not math.isfinite(result.get('branch_up_l2',float('nan'))) or result.get('branch_up_l2',0)<=0:
  raise RuntimeError('V-only branch did not learn during smoke')
 if not math.isfinite((result.get('final_validation') or {}).get('val_loss',float('nan'))):
  raise RuntimeError('Smoke validation loss is not finite')


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args()
 def stop(sig,frame):
  signal.signal(signal.SIGTERM,signal.SIG_DFL);os.killpg(os.getpgrp(),signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 stop_file=a.root/'STOP';job='apple70m-vonly-checkpoint'
 def check():
  if stop_file.exists():raise SystemExit(130)
 def current(stage,output=None,probe_step=0):
  check();atomic(a.root/'current.json',{'job':job,'campaign_id':a.root.name,'stage':stage,'output':str(output or a.root),'probe_step':probe_step,'checkpoint_policy':'local_only'})
 def run(cmd,**kw):
  check();subprocess.run([str(x) for x in cmd],check=True,**kw);check()
 current('waiting_wandb');deadline=time.monotonic()+900
 while job not in read(a.root/'control.json').get('ready_jobs',[]):
  check()
  if time.monotonic()>deadline:raise RuntimeError('W&B online readiness timeout')
  time.sleep(1)
 current('bootstrap')
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
 if len(gpu)!=1 or 'H200' not in gpu[0] or float(gpu[0].rsplit(',',1)[1])<100000:raise RuntimeError('A whole H200 is required')
 if shutil.disk_usage(a.root).free<6*1024**3:raise RuntimeError('Need 6GiB free persistent checkpoint space')
 if shutil.disk_usage(a.scratch).free<15*1024**3:raise RuntimeError('Need 15GiB free scratch space for environment/cache')
 sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 atomic(a.root/'source-manifest.json',{'commit':sha,'files':{str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.rglob('*') if f.is_file() and '__pycache__' not in f.parts}})
 bootstrap=a.scratch/'bootstrap';venv=a.scratch/'environment'
 run([sys.executable,'-m','pip','install','--disable-pip-version-check','--target',bootstrap,'uv==0.11.26'])
 env=dict(os.environ,PYTHONPATH=str(bootstrap),UV_PYTHON_INSTALL_DIR=str(a.scratch/'python'),UV_CACHE_DIR=str(a.scratch/'uv-cache'))
 run([sys.executable,'-m','uv','python','install','3.11.14'],env=env)
 run([sys.executable,'-m','uv','venv','--python','3.11.14',venv],env=env)
 py=venv/'bin/python'
 run([sys.executable,'-m','uv','pip','install','--python',py,'--index-strategy','unsafe-best-match','--default-index','https://pypi.org/simple','--index','https://download.pytorch.org/whl/cu128','-r',HERE/'resolved-requirements.txt'],env=env)
 env=dict(os.environ,PYTHONPATH=str(HERE),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HOME=str(a.scratch/'hf'),TOKENIZERS_PARALLELISM='false',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TRITON_CACHE_DIR=str(a.scratch/'triton'),TORCHINDUCTOR_CACHE_DIR=str(a.scratch/'inductor'))
 current('preflight')
 run([py,'-m','pytest','-q',HERE/'test_vonly.py',HERE/'test_train.py',HERE/'test_stream.py',HERE/'test_validation.py',HERE/'test_checkpointing.py',HERE/'test_benchmark_load.py'],env=env)
 current('data-validation')
 run([py,HERE/'validation.py','--cache-root',a.scratch/'validation','--output-root',a.root,'--stop-file',stop_file],env=env)
 candidates=[]
 for i,batch in enumerate((8,16,32,64,128)):
  out=a.root/f'benchmark-b{batch}';current('benchmark',out,i)
  cmd=[py,HERE/'train.py','--benchmark','--microbatch',batch,'--output-root',out,'--stop-file',stop_file,'--warmup',20,'--measure',100]
  completed=subprocess.run([str(x) for x in cmd],env=env);check();r=read(out/'result.json')
  if completed.returncode==42 and r.get('status')=='oom':continue
  if completed.returncode:raise RuntimeError(f'Benchmark failed for batch {batch}, exit {completed.returncode}')
  if r.get('optimizer_updates',0)<1 or not math.isfinite(r.get('tokens_per_second',0)) or r.get('tokens_per_second',0)<=0:raise RuntimeError('Invalid benchmark update/throughput')
  candidates.append({'microbatch':batch,'result':r})
 if not candidates:raise RuntimeError('All benchmark candidates OOM')
 # Filled against trainer's explicit throughput result contract.
 selected=max(candidates,key=lambda c:c['result']['tokens_per_second'])['microbatch']
 atomic(a.root/'batch-selection.json',{'candidates':candidates,'selected_microbatch':selected,'global_batch':1024,'sequence_length':2048,'selection':'measured synthetic full-model training throughput; actual-data smoke follows'})
 common=[py,HERE/'train.py','--microbatch',selected,'--data-manifest',HERE/'train-data-manifest.json','--cache-root',a.scratch/'train-cache','--validation-manifest',a.root/'validation-manifest.json','--stop-file',stop_file]
 restore=a.root/'restore-check';current('restore-check',restore)
 run([py,HERE/'restore_probe.py','--variant','value','--microbatch',selected,'--output-root',restore,'--data-manifest',HERE/'train-data-manifest.json','--validation-manifest',a.root/'validation-manifest.json'],env=env)
 checkpoint=restore/'value/resumed/checkpoints/final.pt'
 current('benchmark-smoke',a.root/'benchmark-smoke')
 run([py,HERE/'benchmark.py','--checkpoint',checkpoint,'--tokenizer-json',a.scratch/'validation/gpt-neox-20b-tokenizer.json','--output',a.root/'benchmark-smoke','--device','cuda','--batch-size',4,'--smoke-limit',2],env=env)
 smoke=a.root/'real-data-smoke';current('preflight',smoke,len(candidates))
 run(common+['--output-root',smoke,'--steps',20],env=env)
 result=read(smoke/'result.json')
 validate_smoke(result)
 train=a.root/'train';current('training',train)
 run(common+['--output-root',train,'--steps',47684],env=env)
 result=read(train/'result.json')
 if result.get('status')!='completed':raise RuntimeError('Training did not complete')
 current('final-benchmarks',train)
 summary={}
 for label in ('final','best'):
  path=a.root/'benchmarks'/label
  rc=subprocess.run([str(x) for x in [py,HERE/'benchmark.py','--checkpoint',train/f'checkpoints/{label}.pt','--tokenizer-json',a.scratch/'validation/gpt-neox-20b-tokenizer.json','--output',path,'--device','cuda','--batch-size',16]],env=env).returncode
  check();report=read(path/'summary.json');summary[label]={'exit_code':rc,'complete':report.get('formal_complete',False),'results':report.get('results',{}),'failures':report.get('failures',{})}
 result['benchmarks']=summary;atomic(train/'result.json',result)
 ok=all(x['complete'] for x in summary.values())
 current('completed' if ok else 'training_completed_eval_failed',train)
 print('APPLE_CHECKPOINT_COMPLETE='+json.dumps(result),flush=True)
 if not ok:raise SystemExit(1)
if __name__=='__main__':main()
