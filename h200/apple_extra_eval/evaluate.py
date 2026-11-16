"""Pinned, full-split additional evaluations for verified saved APPLE weights."""
import argparse,gc,hashlib,importlib.metadata,json,math,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
TRAINER=HERE.parent/'apple_checkpoint'
sys.path.insert(0,str(TRAINER))
import torch
import benchmark
from checkpointing import verified_load,digest_file
from train import atomic_json

DATASETS={
 'nyu-mll/blimp':'877fba0801ffb7cbd8c39c1ff314a46f053f6036',
 'allenai/swag':'dc48148372b3853a9c7bae7bb06c161b46d8364a',
 'Rowan/hellaswag':'218ec52e09a7e7462a5400043bb9a69a41d06b76',
 'juletxara/xstory_cloze':'00bcdde7f2ba35d38b76fb67d0f37374f4cce3da',
 'EleutherAI/wikitext_document_level':'647234772b9554e208af6c826f23b99e3cac88c8',
}

def validate(result,names,limit):
    for name in names:
        m=result.get('results',{}).get(name)
        if not m:raise ValueError('Missing task '+name)
        metric='word_perplexity,none' if name=='wikitext' else 'acc,none'
        value=m.get(metric)
        if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Nonfinite '+name)
        if name!='wikitext' and not 0<=value<=1:raise ValueError('Invalid accuracy '+name)
        n=result.get('n-samples',{}).get(name,{})
        original=n.get('original');effective=n.get('effective')
        if not isinstance(original,int) or original<1:raise ValueError('Missing sample count '+name)
        expected=original if limit is None else min(limit,original)
        if effective!=expected:raise ValueError(f'Incomplete {name}: {effective}/{expected}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--current',type=Path);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--smoke-limit',type=int);p.add_argument('--device',default='cuda');p.add_argument('--families',nargs='+',choices=['blimp','swag','hellaswag','xstorycloze','wikitext']);p.add_argument('--dataset-root',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    if importlib.metadata.version('lm_eval')!='0.4.13':raise RuntimeError('Expected lm_eval 0.4.13')
    assert digest_file(a.tokenizer)==benchmark.TOKENIZER_SHA
    from transformers import PreTrainedTokenizerFast
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    from lm_eval.tasks._yaml_loader import load_yaml
    from lm_eval.models.huggingface import HFLM
    class AuditedLM(HFLM):
        token_ppl_audit=None
        def loglikelihood_rolling(self,requests,disable_tqdm=False):
            values=super().loglikelihood_rolling(requests,disable_tqdm=disable_tqdm)
            counts=[len(self.tok_encode(req.args[0])) for req in requests]
            assert len(values)==len(counts) and sum(counts)>0
            self.token_ppl_audit={'loglikelihood_sum':sum(values),'token_count':sum(counts),'documents':len(counts),'token_perplexity':math.exp(-sum(values)/sum(counts)),'protocol':'Same document-level rolling likelihood as harness; GPT-NeoX token denominator, max context 2048, prefix EOS 0. Not cross-document concatenated sliding-window PPL.'}
            return values
    manager=TaskManager()
    blimp=sorted(t for t in manager.all_tasks if t.startswith('blimp_') and not t.startswith('blimp_nl'))
    xstory=['xstorycloze_'+language for language in ('ar','en','es','eu','hi','id','my','ru','sw','te','zh')]
    assert len(blimp)==67 and set(xstory)<=set(manager.all_tasks)
    units=[('blimp',blimp[:1] if a.smoke_limit else blimp),('swag',['swag']),('hellaswag',['hellaswag']),('xstorycloze',['xstorycloze_en'] if a.smoke_limit else xstory),('wikitext',['wikitext'])]
    if a.families:units=[(family,names) for family,names in units if family in a.families]
    assert units
    model,payload=benchmark.load_model(a.checkpoint,a.device)
    tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(a.tokenizer),eos_token='<|endoftext|>',bos_token='<|endoftext|>',pad_token='<|endoftext|>')
    lm=AuditedLM(pretrained=model,tokenizer=tokenizer,backend='causal',batch_size=a.batch_size,device=a.device,dtype='float32',max_length=2048,prefix_token_id=0)
    protocol={'checkpoint_sha256':digest_file(a.checkpoint),'variant':payload['contract']['branch_source'],'checkpoint_step':payload['progress']['global_step'],'checkpoint_validation':payload['progress'],'lm_eval':'0.4.13','num_fewshot':0,'limit':a.smoke_limit,'diagnostic_only':a.smoke_limit is not None,'dataset_revisions':DATASETS,'tokenizer_sha256':benchmark.TOKENIZER_SHA,'dtype':'float32','max_length':2048,'batch_size':a.batch_size,'wikitext_config':'wikitext-2-raw-v1','xstory_languages':xstory,'source_files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE/'evaluate.py',)}}
    protocol['requested_families']=[family for family,_ in units]
    if a.dataset_root:protocol['offline_dataset_manifest_sha256']=digest_file(a.dataset_root/'manifest.json')
    atomic_json(a.output/'protocol.json',protocol)
    summary={'formal_complete':False,'protocol':protocol,'families':{},'failures':{}}
    start=time.monotonic()
    for index,(family,names) in enumerate(units):
        if a.current:
            current=json.loads(a.current.read_text());current.update(stage='evaluating',output=str(a.output),probe_step=index,model=a.checkpoint.stem,family=family);atomic_json(a.current,current)
        specs=[]
        for name in names:
            entry=manager.task_index[name];spec=load_yaml(entry.yaml_path,resolve_func=True)
            assert spec['dataset_path'] in DATASETS
            if a.dataset_root and spec['dataset_path'] in ('nyu-mll/blimp','juletxara/xstory_cloze'):
                config=spec['dataset_name']
                files={'train':str(a.dataset_root/'blimp'/config/'train.parquet')} if family=='blimp' else {split:str(a.dataset_root/'xstorycloze'/config/(split+'.parquet')) for split in ('train','eval')}
                assert all(Path(path).is_file() for path in files.values())
                spec['dataset_path']='parquet';spec['dataset_name']=None;spec['dataset_kwargs']={'data_files':files,'token':False}
            elif spec['dataset_path']=='juletxara/xstory_cloze':
                language=spec['dataset_name']
                assert language in {'ar','en','es','eu','hi','id','my','ru','sw','te','zh'}
                base=f"https://huggingface.co/datasets/juletxara/xstory_cloze/resolve/{DATASETS['juletxara/xstory_cloze']}/{language}"
                spec['dataset_path']='parquet';spec['dataset_name']=None
                spec['dataset_kwargs']={'data_files':{'train':base+'/train/0000.parquet','eval':base+'/eval/0000.parquet'},'token':False}
            else:
                spec['dataset_kwargs']={**spec.get('dataset_kwargs',{}),'revision':DATASETS[spec['dataset_path']],'token':False}
            specs.append(spec)
        try:
            result=evaluator.simple_evaluate(model=lm,tasks=specs,num_fewshot=0,task_manager=manager,limit=a.smoke_limit,log_samples=False,bootstrap_iters=0 if a.smoke_limit else 1000,random_seed=0,numpy_random_seed=0,torch_random_seed=0,fewshot_random_seed=0)
            validate(result,names,a.smoke_limit)
            (a.output/(family+'.json')).write_text(json.dumps(result,indent=2,default=str,allow_nan=False))
            metrics={name:result['results'][name] for name in names}
            counts={name:result['n-samples'][name] for name in names}
            if family=='wikitext':
                assert lm.token_ppl_audit and lm.token_ppl_audit['documents']==counts['wikitext']['effective']
                atomic_json(a.output/'wikitext-token-ppl.json',lm.token_ppl_audit)
                compact={'word_perplexity':metrics['wikitext']['word_perplexity,none'],'token_perplexity':lm.token_ppl_audit['token_perplexity'],'documents':counts['wikitext']['effective'],'tokens':lm.token_ppl_audit['token_count']}
            else:
                compact={'accuracy':sum(m['acc,none'] for m in metrics.values())/len(metrics),'subtasks':len(metrics),'samples':sum(n['effective'] for n in counts.values())}
                if all('acc_norm,none' in m for m in metrics.values()):compact['accuracy_normalized']=sum(m['acc_norm,none'] for m in metrics.values())/len(metrics)
                if family=='xstorycloze':compact['english_accuracy']=metrics['xstorycloze_en']['acc,none'];compact['per_language_accuracy']={k:v['acc,none'] for k,v in metrics.items()}
            summary['families'][family]=compact
            print('EXTRA_EVAL_FAMILY='+json.dumps({'model':a.checkpoint.stem,'family':family,**compact}),flush=True)
        except Exception as e:
            summary['failures'][family]={'type':type(e).__name__,'message':str(e)[:1000]}
            print('EXTRA_EVAL_FAILURE='+json.dumps({'family':family,**summary['failures'][family]}),flush=True)
        summary['seconds']=time.monotonic()-start
        atomic_json(a.output/'summary.json',summary)
    summary['formal_complete']=a.smoke_limit is None and not summary['failures'] and len(summary['families'])==len(units)
    summary['diagnostic_complete']=a.smoke_limit is not None and not summary['failures'] and len(summary['families'])==len(units)
    atomic_json(a.output/'summary.json',summary)
    if summary['failures']:raise SystemExit(1)
    print('EXTRA_EVAL_COMPLETE='+json.dumps({'model':a.checkpoint.stem,'formal_complete':summary['formal_complete'],'seconds':summary['seconds']}),flush=True)
if __name__=='__main__':main()
