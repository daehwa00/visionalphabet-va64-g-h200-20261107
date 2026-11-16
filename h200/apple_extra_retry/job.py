"""H200 gates followed by V-only scratch training; stop skips backup/evaluation."""
import argparse,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
TRAINER=HERE.parent/"apple_checkpoint"
EXTRA=HERE.parent/"apple_extra_eval"
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'scripts'))
from va64_control_transport import atomic,read

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args()
 def stop(sig,frame):
  signal.signal(signal.SIGTERM,signal.SIG_DFL);os.killpg(os.getpgrp(),signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 stop_file=a.root/'STOP';job='apple70m-extra-eval'
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
 if shutil.disk_usage(a.root).free<15*1024**3:raise RuntimeError('Need 15GiB free persistent evaluation space')
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
 run([py,'-m','pytest','-q',TRAINER/'test_benchmark_load.py',EXTRA/'test_evaluate.py'],env=env)
 current('download-models')
 run([py,HERE/'download_models.py',a.root/'models'],env=env)
 current('download-offline-data')
 run([py,HERE/'download_data.py',a.root/'datasets'],env=env)
 env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
 tokenizer=a.root/'datasets/gpt-neox-20b-tokenizer.json'
 models=json.loads((HERE/'models.json').read_text());summaries={};failures={}
 for i,model in enumerate(models):
  name=model['name'].removesuffix('.pt');out=a.root/'evaluations'/name;family=model['families'][0]
  common=[py,EXTRA/'evaluate.py','--checkpoint',a.root/'models'/model['name'],'--tokenizer',tokenizer,'--current',a.root/'current.json','--batch-size',16,'--families',family,'--dataset-root',a.root/'datasets']
  current('smoke',a.root/'smoke'/name,i)
  run(common+['--output',a.root/'smoke'/name,'--smoke-limit',2],env=env)
  if not read(a.root/'smoke'/name/'summary.json').get('diagnostic_complete'):raise RuntimeError('Offline smoke failed')
  current('evaluating',out,i)
  try:run(common+['--output',out],env=env)
  except subprocess.CalledProcessError as e:failures[name]={'exit_code':e.returncode}
  summaries[name]=read(out/'summary.json')
  if not summaries[name].get('formal_complete') or set(summaries[name].get('families',{}))!={family}:failures.setdefault(name,{'error':'Incomplete targeted retry'})
 compact={name:{'formal_complete':v.get('formal_complete',False),'families':v.get('families',{}),'failures':v.get('failures',{}),'checkpoint_sha256':v.get('protocol',{}).get('checkpoint_sha256'),'checkpoint_step':v.get('protocol',{}).get('checkpoint_step'),'seconds':v.get('seconds')} for name,v in summaries.items()}
 result={'status':'completed' if not failures else 'evaluation_failed','scope':'Only residual-final xStoryCloze and residual-best BLiMP; prior successes preserved','models':compact,'failures':failures}
 atomic(a.root/'result.json',result);current('upload-results',a.root)
 run([py,EXTRA/'upload_results.py',a.root,sha],env=env)
 result['archive']=read(a.root/'results-archive.json');atomic(a.root/'result.json',result)
 current('completed' if not failures else 'evaluation_failed',a.root)
 print('APPLE_EXTRA_RETRY_COMPLETE='+json.dumps(result),flush=True)
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
