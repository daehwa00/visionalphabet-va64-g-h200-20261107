"""H200 gates followed by V-only scratch training; stop skips backup/evaluation."""
import argparse,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
TRAINER=HERE.parent/"apple_checkpoint"
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from va64_control_transport import atomic,read

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args()
 def stop(sig,frame):
  signal.signal(signal.SIGTERM,signal.SIG_DFL);os.killpg(os.getpgrp(),signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 stop_file=a.root/'STOP';job='apple70m-train'
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
 if shutil.disk_usage(a.root).free<15*1024**3:raise RuntimeError('Need 15GiB free persistent checkpoint space')
 if shutil.disk_usage(a.scratch).free<15*1024**3:raise RuntimeError('Need 15GiB free scratch space for environment/cache')
 sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 atomic(a.root/'source-manifest.json',{'commit':sha,'files':{str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.rglob('*') if f.is_file() and '__pycache__' not in f.parts}})
 bootstrap=a.scratch/'bootstrap';venv=a.scratch/'environment'
 run([sys.executable,'-m','pip','install','--disable-pip-version-check','--target',bootstrap,'uv==0.11.26'])
 env=dict(os.environ,PYTHONPATH=str(bootstrap),UV_PYTHON_INSTALL_DIR=str(a.scratch/'python'),UV_CACHE_DIR=str(a.scratch/'uv-cache'))
 run([sys.executable,'-m','uv','python','install','3.11.14'],env=env)
 run([sys.executable,'-m','uv','venv','--python','3.11.14',venv],env=env)
 py=venv/'bin/python'
 run([sys.executable,'-m','uv','pip','install','--python',py,'--index-strategy','unsafe-best-match','--default-index','https://pypi.org/simple','--index','https://download.pytorch.org/whl/cu128','-r',TRAINER/'resolved-requirements.txt'],env=env)
 env=dict(os.environ,PYTHONPATH=str(TRAINER),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HOME=str(a.scratch/'hf'),TOKENIZERS_PARALLELISM='false',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',TRITON_CACHE_DIR=str(a.scratch/'triton'),TORCHINDUCTOR_CACHE_DIR=str(a.scratch/'inductor'))
 current('preflight')
 run([py,'-m','pytest','-q',HERE/'test_migration.py',TRAINER/'test_checkpointing.py',TRAINER/'test_benchmark_load.py'],env=env)
 current('data-validation')
 run([py,TRAINER/'validation.py','--cache-root',a.scratch/'validation','--output-root',a.root,'--stop-file',stop_file],env=env)
 current('checkpoint-download')
 run([py,HERE/'download.py',a.root/'recovery'],env=env)
 common=['--branch-source','residual','--steps','47684','--data-manifest',TRAINER/'train-data-manifest.json','--cache-root',a.scratch/'train-cache','--validation-manifest',a.root/'validation-manifest.json','--stop-file',stop_file]
 parent=a.root/'recovery/latest.pt';receipt=read(parent.with_suffix('.pt.json'));step=receipt['global_step']
 migrated=a.root/'migration';current('migration',migrated)
 run([py,HERE/'runtime.py','migrate',*common,'--microbatch',64,'--resume',parent,'--output-root',migrated],env=env)
 reference=a.root/'reference-b8';current('reference',reference)
 run([py,HERE/'runtime.py','train',*common,'--microbatch',8,'--resume',parent,'--output-root',reference,'--stop-after',step+2],env=env)
 candidate=a.root/'candidate-b64';current('candidate',candidate)
 run([py,HERE/'runtime.py','train',*common,'--microbatch',64,'--resume',migrated/'latest.pt','--output-root',candidate,'--stop-after',step+2],env=env)
 run([py,HERE/'compare.py',reference,candidate,a.root/'comparison.json'],env=env)
 smoke=a.root/'speed-check';current('speed-check',smoke)
 run([py,HERE/'runtime.py','train',*common,'--microbatch',64,'--resume',candidate/'checkpoints/latest.pt','--output-root',smoke,'--stop-after',step+10],env=env)
 rows=[json.loads(line) for line in (smoke/'steps.jsonl').read_text().splitlines()]
 rates=[r['tokens_per_second'] for r in rows if r.get('tokens_per_second') and r.get('global_step',0)>step+4]
 import statistics
 speed=statistics.median(rates)
 if not math.isfinite(speed) or speed<220000:raise RuntimeError('H200 throughput gate failed: '+str(speed))
 atomic(a.root/'handoff-ready.json',{'parent_step':step,'resume_step':step+10,'tokens_per_second':speed,'restore_verified':True,'comparison':read(a.root/'comparison.json')})
 print('H200_HANDOFF_READY='+json.dumps(read(a.root/'handoff-ready.json')),flush=True)
 train=a.root/'train';current('training',train)
 run([py,HERE/'runtime.py','train',*common,'--microbatch',64,'--resume',smoke/'checkpoints/latest.pt','--output-root',train],env=env)
 result=read(train/'result.json')
 if result.get('status')!='completed':raise RuntimeError('Training did not complete')
 current('final-benchmarks',train)
 summary={}
 for label in ('final','best'):
  path=a.root/'benchmarks'/label
  rc=subprocess.run([str(x) for x in [py,TRAINER/'benchmark.py','--checkpoint',train/f'checkpoints/{label}.pt','--tokenizer-json',a.scratch/'validation/gpt-neox-20b-tokenizer.json','--output',path,'--device','cuda','--batch-size',16]],env=env).returncode
  check();report=read(path/'summary.json');summary[label]={'exit_code':rc,'complete':report.get('formal_complete',False),'results':report.get('results',{}),'failures':report.get('failures',{})}
 result['benchmarks']=summary;atomic(train/'result.json',result)
 ok=all(x['complete'] for x in summary.values())
 current('completed' if ok else 'training_completed_eval_failed',train)
 print('APPLE_RESUME_COMPLETE='+json.dumps(result),flush=True)
 if not ok:raise SystemExit(1)
if __name__=='__main__':main()
