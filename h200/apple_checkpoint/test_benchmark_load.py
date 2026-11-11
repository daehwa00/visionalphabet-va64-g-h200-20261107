import pytest,torch
from train import build_model
from test_train import tiny_config
from checkpointing import save_weights
from benchmark import load_model,validate_task_result,evaluate_tasks,evaluation_spec,TASKS,LOGIQA_PARQUET_REVISION

@pytest.mark.parametrize('source',['value','residual'])
def test_exact_saved_variant_loaded(tmp_path,source):
 m=build_model(config=tiny_config(),branch_source=source);m.eval()
 with torch.no_grad():
  for n,p in m.named_parameters():
   if n.endswith('lrcp.up'):p.normal_(0,.02)
 x=torch.tensor([[1,2,3,4]]);expected=m(x).logits
 path=tmp_path/'final.pt';save_weights(path,m,contract={'branch_source':source},model_config=m.config.to_dict(),progress={'global_step':5})
 loaded,payload=load_model(path,require_70m=False)
 torch.testing.assert_close(expected,loaded(x).logits,rtol=0,atol=0)
 assert payload['contract']['branch_source']==source

def result_for(task,count=2):
 metrics={'acc,none':0.5}
 if task=='lambada_openai':metrics['perplexity,none']=10.
 return {'results':{task:metrics},'n-samples':{task:{'original':100,'effective':count}}}

@pytest.mark.parametrize('result',[None,{}, {'results':{}},result_for('piqa',1)])
def test_incomplete_task_is_rejected(result):
 with pytest.raises(ValueError):validate_task_result(result,'piqa',2)

def test_nonfinite_and_missing_metrics_are_rejected():
 for value in (float('nan'),float('inf'),True,None,1.1):
  result=result_for('piqa');result['results']['piqa']['acc,none']=value
  with pytest.raises(ValueError):validate_task_result(result,'piqa',2)
 result=result_for('lambada_openai');del result['results']['lambada_openai']['perplexity,none']
 with pytest.raises(ValueError):validate_task_result(result,'lambada_openai',2)

def test_failure_is_reported_and_other_tasks_continue(tmp_path):
 calls=[]
 def evaluate(**kwargs):
  spec=kwargs['tasks'][0];task=spec['task'] if isinstance(spec,dict) else spec;calls.append(task)
  assert kwargs['num_fewshot']==0 and kwargs['limit']==2
  if task=='wsc':raise ConnectionError('public dataset unavailable')
  return result_for(task)
 results,failures=evaluate_tasks(object(),object(),tmp_path,2,evaluate)
 assert tuple(calls)==TASKS
 assert set(results)==set(TASKS)-{'wsc'}
 assert failures['wsc']['type']=='ConnectionError'
 assert (tmp_path/'wsc.error.json').exists()

def test_full_evaluation_requires_all_dataset_samples():
 validate_task_result(result_for('piqa',100),'piqa',None)
 with pytest.raises(ValueError):validate_task_result(result_for('piqa',99),'piqa',None)

def test_missing_checkpoint_never_uses_pretrained(tmp_path,monkeypatch):
 from transformers import GPTNeoXForCausalLM
 monkeypatch.setattr(GPTNeoXForCausalLM,'from_pretrained',lambda *a,**k:pytest.fail('pretrained fallback forbidden'))
 with pytest.raises(FileNotFoundError):load_model(tmp_path/'missing.pt')

def test_logiqa_uses_official_parquet_without_changing_prompt():
 spec=evaluation_spec('logiqa')
 assert spec['dataset_path']=='EleutherAI/logiqa'
 assert callable(spec['doc_to_text']) and callable(spec['doc_to_target'])
 assert spec['dataset_name']=='default'
 assert spec['dataset_kwargs']=={'revision':LOGIQA_PARQUET_REVISION,'token':False}
 assert evaluation_spec('piqa')=='piqa'

@pytest.mark.parametrize('saved_hash',[None,'wrong-source'])
def test_production_load_requires_matching_training_source(tmp_path,monkeypatch,saved_hash):
 import benchmark
 monkeypatch.setattr(benchmark,'verified_load',lambda path:{'contract':{'branch_source':'value','source_sha256':saved_hash}})
 monkeypatch.setattr(benchmark,'source_hash',lambda:'current-source')
 with pytest.raises(ValueError,match='source SHA256 mismatch'):load_model(tmp_path/'final.pt')
