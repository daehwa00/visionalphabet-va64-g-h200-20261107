"""Evaluate saved APPLE weights only. Never falls back to pretrained weights."""
import argparse,json,math,importlib.metadata
from pathlib import Path
import torch
from checkpointing import verified_load,digest_file
from train import build_model,source_hash

TASKS=('lambada_openai','piqa','winogrande','wsc','arc_easy','arc_challenge','sciq','logiqa')
TOKENIZER_SHA='c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624'
EVAL_VERSION='0.4.13'
# Official dataset-server conversion, fixed to its immutable commit. The main
# branch still contains a dataset script unsupported by datasets>=4. The
# conversion's sole config is named "default"; prompt/labels stay upstream.
LOGIQA_PARQUET_REVISION='fcd80241270118cd52c96dc517dc38804025f3f5'

def evaluation_spec(task):
 if task!='logiqa':return task
 import lm_eval
 from lm_eval.tasks._yaml_loader import load_yaml
 spec=load_yaml(Path(lm_eval.__file__).parent/'tasks/logiqa/logiqa.yaml',resolve_func=True)
 spec['dataset_name']='default'
 spec['dataset_kwargs']={'revision':LOGIQA_PARQUET_REVISION,'token':False}
 return spec

def validate_task_result(result,task,limit):
 """Never mark missing metrics or a partial task as a completed evaluation."""
 if not isinstance(result,dict):raise ValueError('Evaluator returned no result')
 metrics=result.get('results',{}).get(task)
 if not isinstance(metrics,dict):raise ValueError('Missing requested task result: '+task)
 for metric in (('acc,none','perplexity,none') if task=='lambada_openai' else ('acc,none',)):
  value=metrics.get(metric)
  if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Missing/nonfinite metric: '+metric)
  if metric.startswith('acc,') and not 0<=value<=1:raise ValueError('Accuracy outside [0,1]')
  if metric.startswith('perplexity,') and value<=0:raise ValueError('Perplexity must be positive')
 samples=result.get('n-samples',{}).get(task,{})
 original=samples.get('original');effective=samples.get('effective')
 if not isinstance(original,int) or original<1:raise ValueError('Missing evaluation dataset sample count')
 expected=original if limit is None else min(limit,original)
 if effective!=expected:raise ValueError(f'Incomplete task: evaluated {effective}, expected {expected}')

def evaluate_tasks(lm,manager,output,limit,evaluate):
 results={};failures={}
 for task in TASKS:
  try:
   result=evaluate(model=lm,tasks=[evaluation_spec(task)],num_fewshot=0,task_manager=manager,limit=limit,log_samples=False,bootstrap_iters=0 if limit is not None else 1000,random_seed=0,numpy_random_seed=0,torch_random_seed=0,fewshot_random_seed=0)
   validate_task_result(result,task,limit)
   (output/(task+'.json')).write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+'\n');results[task]=result['results']
  except Exception as e:
   failures[task]={'type':type(e).__name__,'message':str(e)[:1000]}
   (output/(task+'.error.json')).write_text(json.dumps(failures[task])+'\n')
 return results,failures

def load_model(path,device='cpu',require_70m=True):
 from transformers import GPTNeoXConfig
 payload=verified_load(path);source=payload['contract'].get('branch_source')
 if source not in ('value','residual'):raise ValueError('Missing or invalid APPLE branch source')
 if require_70m and payload['contract'].get('source_sha256')!=source_hash():raise ValueError('Checkpoint source SHA256 mismatch: evaluate with its original training source')
 config=GPTNeoXConfig.from_dict(payload['model_config']);config._attn_implementation='sdpa'
 if require_70m and (config.hidden_size,config.num_hidden_layers,config.num_attention_heads)!=(512,6,8):raise ValueError('Expected Pythia70M architecture')
 model=build_model(config=config,branch_source=source)
 model.load_state_dict(payload['model'],strict=True);model.to(device=device,dtype=torch.float32);model.eval()
 return model,payload

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--tokenizer-json',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',default='auto');p.add_argument('--smoke-limit',type=int);a=p.parse_args()
 if a.smoke_limit is not None and a.smoke_limit<1:raise ValueError('smoke limit must be positive')
 if digest_file(a.tokenizer_json)!=TOKENIZER_SHA:raise ValueError('Tokenizer SHA256 mismatch')
 a.output.mkdir(parents=True,exist_ok=False)
 from transformers import PreTrainedTokenizerFast
 from lm_eval import evaluator
 from lm_eval.tasks import TaskManager
 from lm_eval.models.huggingface import HFLM
 if importlib.metadata.version('lm_eval')!=EVAL_VERSION:raise ValueError('Expected lm_eval=='+EVAL_VERSION+'; install requirements-eval.txt')
 manager=TaskManager();missing=set(TASKS)-set(manager.all_tasks)
 if missing:raise ValueError('Pinned lm-eval lacks required tasks: '+','.join(sorted(missing)))
 model,payload=load_model(a.checkpoint,a.device)
 tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(a.tokenizer_json),eos_token='<|endoftext|>',bos_token='<|endoftext|>',pad_token='<|endoftext|>')
 if tokenizer.eos_token_id!=0:raise ValueError('Expected tokenizer EOS0')
 lm=HFLM(pretrained=model,tokenizer=tokenizer,backend='causal',batch_size=a.batch_size,device=a.device,dtype='float32',max_length=2048,prefix_token_id=0)
 protocol={'checkpoint_sha256':digest_file(a.checkpoint),'branch_source':payload['contract']['branch_source'],'global_step':payload['progress']['global_step'],'tasks':TASKS,'num_fewshot':0,'limit':a.smoke_limit,'diagnostic_only':a.smoke_limit is not None,'inference_dtype':'float32','max_length':2048,'tokenizer_sha256':TOKENIZER_SHA,'lm_eval_version':importlib.metadata.version('lm_eval'),'logiqa_data_revision':LOGIQA_PARQUET_REVISION,'package_versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','datasets','evaluate','accelerate','peft')},'training_contract':payload['contract']}
 results,failures=evaluate_tasks(lm,manager,a.output,a.smoke_limit,evaluator.simple_evaluate)
 complete=not failures and len(results)==len(TASKS)
 summary={'protocol':protocol,'results':results,'failures':failures,'complete':complete,'formal_complete':complete and a.smoke_limit is None}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
 if not summary['complete']:raise SystemExit(1)
 print('BENCHMARK_COMPLETE '+str(a.output/'summary.json'))
if __name__=='__main__':main()
