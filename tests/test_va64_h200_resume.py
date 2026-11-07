from pathlib import Path
import dataclasses,hashlib,json,random
import numpy as np
import pytest
import torch
from scripts import imagenet1k_runtime as runtime
from scripts import va64_h200_resume as migration


def task_at(root,batch):
 return runtime.BaselineTask(phase='full',model_key='va_k64_retrain_G',seed=0,learning_rate=.003,epochs=100,
  data_root=Path('/app/data/ImageNet-2012'),output_dir=root,result_path=root/'result.json',checkpoint_path=root/'checkpoint.pt',
  source_root=Path('/same-source'),batch_size=batch,workers=8,
  source_sha256={'source_closure':'old','h200_g_entrypoint':'old','ablation_definition':'fixed','dataset_manifest':'fixed','canonical_backbone_initialization':'fixed'})


def test_migration_preserves_model_optimizer_rng_and_rejects_recipe_change(tmp_path):
 old_task=task_at(tmp_path/'old',128);old=runtime._contract(old_task,torch.device('cpu'))
 recovery=tmp_path/'recovery';(recovery/'contracts').mkdir(parents=True)
 cp=recovery/'checkpoint.pt'
 payload={'schema':runtime.SCHEMA,'contract_sha256':runtime._sha256_payload(old),'task_sha256':old['task_sha256'],
  'source_digest_sha256':old['source_digest_sha256'],'source_sha256':old['source_sha256'],
  'completed_epochs':1,'global_step':5005,'history':[{'epoch':1,'global_step':5005}],
  'training_seconds':100.,'model':{'x':torch.randn(3)},'optimizer':{'state':{'x':torch.randn(3)}},
  'rng':{'torch':torch.get_rng_state(),'sentinel':torch.arange(10)}}
 torch.save(payload,cp);(recovery/'contracts/old.json').write_text(json.dumps(old))
 plan={'checkpoint_sha256':migration.digest(cp),'contract_sha256':runtime._sha256_payload(old),
  'contract_relative_path':'contracts/old.json','completed_epochs':1,'global_step':5005}
 pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan))
 new_task=task_at(tmp_path/'new',256)
 new_task=dataclasses.replace(new_task,source_sha256={**new_task.source_sha256,'source_closure':'new','h200_g_entrypoint':'new'})
 plan['allowed_new_sources']={k:runtime._contract(new_task,torch.device('cpu'))['source_sha256'][k] for k in ('source_closure','h200_g_entrypoint','imagenet1k_runtime')}
 pp.write_text(json.dumps(plan))
 report=migration.migrate(new_task,recovery,pp,torch.device('cpu'))
 saved=torch.load(new_task.checkpoint_path,weights_only=True)
 for section in ('model','rng'):
  for k in payload[section]: assert torch.equal(payload[section][k],saved[section][k])
 assert torch.equal(saved['optimizer']['state']['x'],payload['optimizer']['state']['x'])
 assert report['physical_batch_after']==256 and not report['exact_trajectory']
 runtime._validate_checkpoint(saved,runtime._contract(new_task,torch.device('cpu')),new_task)
 bad=runtime._contract(dataclasses.replace(new_task,learning_rate=.006),torch.device('cpu'))
 with pytest.raises(ValueError,match='recipe'):migration.validate_transition(old,bad)
 bad=runtime._contract(dataclasses.replace(new_task,source_sha256={**new_task.source_sha256,'ablation_definition':'changed'}),torch.device('cpu'))
 with pytest.raises(ValueError,match='source'):migration.validate_transition(old,bad)
 plan['checkpoint_sha256']='0'*64;pp.write_text(json.dumps(plan))
 with pytest.raises(ValueError,match='hash'):migration.migrate(dataclasses.replace(new_task,checkpoint_path=tmp_path/'other.pt'),recovery,pp,torch.device('cpu'))


def test_rng_restore_and_epoch_sampler_order_independent_of_physical_batch():
 from torch.utils.data import DataLoader,TensorDataset
 def bundle(batch):
  g=torch.Generator().manual_seed(123);v=torch.Generator().manual_seed(124)
  ds=TensorDataset(torch.arange(1024))
  return runtime.LoaderBundle(DataLoader(ds,batch_size=batch,shuffle=True,generator=g),DataLoader(ds,batch_size=batch,generator=v),g,v)
 original=bundle(128); state=runtime._capture_rng(original,torch.device('cpu'))
 expected=torch.cat([x[0] for x in original.train])
 changed=bundle(256);runtime._restore_rng(state,changed,torch.device('cpu'))
 actual=torch.cat([x[0] for x in changed.train])
 assert torch.equal(expected,actual)


def test_recover_restores_manifest_metadata_only_after_hash_validation(tmp_path,monkeypatch):
 import va64_h200_transport as transport
 root=tmp_path/'run';(root/'environment-check').mkdir(parents=True)
 dataset={'root':'/app/data/ImageNet-2012','counts':{'train':1,'val':1},'classes':['a'],'file_list':'/new/files.jsonl'}
 runtime._atomic_json(root/'environment-check/dataset-manifest.json',dataset)
 canonical={**dataset,'file_list':'/original/files.jsonl'}
 expected=hashlib.sha256(runtime._canonical_json(canonical)+b'\n').hexdigest()
 cp=b'parent-checkpoint'
 plan={'artifact_metadata':{'verified':True},'checkpoint_sha256':hashlib.sha256(cp).hexdigest(),'completed_epochs':13,'global_step':65065,'canonical_dataset_file_list':canonical['file_list'],'canonical_dataset_sha256':expected}
 pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan))
 monkeypatch.setattr(transport,'_agent_api',lambda:None)
 monkeypatch.setattr(transport,'download_artifact',lambda *args:None)
 def extract(archive,path):
  path.mkdir();(path/'checkpoint.pt').write_bytes(cp);runtime._atomic_json(path/'dataset.json',dataset);return True
 monkeypatch.setattr(transport,'_safe_extract',extract)
 recovery=migration.recover(root,pp)
 manifest=recovery/'resume-dataset-manifest.json'
 assert migration.digest(manifest)==expected
 assert json.loads(manifest.read_text())['file_list']==canonical['file_list']
