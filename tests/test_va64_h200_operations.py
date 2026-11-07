from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import va64_h200_observer as observer
from scripts import va64_runtime_operations as operations

class API:
 def __init__(self,events): self.events=events
 def call(self,path,*args):
  if path.endswith('/latest'): return {'sha':'a'*64,'bytes':10,'chunks':1,'epoch':1,'global_step':5005,'code_sha':'b'*40,'complete':True}
  if path.endswith('/ack'): self.events.append('ack'); return {}
  raise AssertionError(path)

@pytest.mark.parametrize('wandb_ok',[False,True])
def test_ack_requires_wandb_confirmation(tmp_path,monkeypatch,wandb_ok):
 events=[]; api=API(events)
 def download(_api,_job,_metadata,target): target.parent.mkdir(parents=True); target.write_bytes(b'test-archive'); events.append('download')
 monkeypatch.setattr(observer,'download_artifact',download)
 monkeypatch.setattr(observer,'validate_checkpoint_bundle',lambda _:events.append('validate'))
 class Mirror:
  def save_backup(self,*args):
   events.append('wandb')
   if not wandb_ok: raise RuntimeError('W&B unavailable')
   return {'uploaded':True}
 call=lambda:observer._backup_latest(api,tmp_path,'va64-g-seed0','b'*40,None,Mirror(),'test')
 if wandb_ok:
  assert call()=='a'*64
  assert events==['download','validate','wandb','ack']
 else:
  with pytest.raises(RuntimeError,match='W&B unavailable'): call()
  assert events==['download','validate','wandb']


def test_training_keeps_checkpoint_local_without_network(tmp_path,monkeypatch):
 monkeypatch.setenv('VA64_G_CHECKPOINT_POLICY','external')
 events=[]; output=tmp_path/'output'; output.mkdir()
 task=SimpleNamespace(output_dir=output)
 def base(task,**kwargs):
  (output/'checkpoint.pt').write_bytes(b'committed')
  kwargs['epoch_callback'](None,{'epoch':1,'global_step':5005})
  return {'status':'paused'}
 runtime=SimpleNamespace(run_task=base)
 def callback(_model,_row): events.append('derived_files')
 def backup(*args):
  pytest.fail('training must never upload checkpoints')
 monkeypatch.setattr(operations,'backup_and_wait',backup)
 monkeypatch.setattr(operations.signal,'signal',lambda *_:None)
 operations.install_training(runtime,tmp_path)
 assert runtime.run_task(task,epoch_callback=callback)['status']=='paused'
 assert events==['derived_files']
 assert __import__('json').loads((output/'checkpoint-meta.json').read_text())['status']=='local_checkpoint'


def test_backup_filename_allowlist_keeps_secrets_out():
 from scripts import va64_h200_transport as transport
 assert transport._allowed_archive_path('telemetry/console.txt')
 assert transport._allowed_archive_path('smoke.json')
 assert not transport._allowed_archive_path('owner-secrets.json')
 assert not transport._allowed_archive_path('contracts/private.key')
 with pytest.raises(ValueError): transport._allowed_archive_path('../checkpoint.pt')


def test_relay_disables_corruptible_response_compression(monkeypatch):
 import io
 from scripts import va64_h200_transport as transport
 captured=[]
 def open_response(self,request,timeout):
  captured.append({key.lower():value for key,value in request.header_items()})
  return io.BytesIO(b'{"ok":true}')
 monkeypatch.setattr(transport._RequestsSessionOpener,'open',open_response)
 api=transport.API('fixture-only-token',base='https://example.workers.dev/va64g')
 assert api.call('/health',attempts=1)=={'ok':True}
 assert captured[0]['accept-encoding']=='identity'


def test_final_results_keep_cost_and_leave_checkpoints_local(tmp_path,monkeypatch):
 import json
 from scripts import va64_h200_transport as transport
 output=tmp_path/'output'; output.mkdir()
 (output/'checkpoint.pt').write_bytes(b'last')
 (output/'best.pt').write_bytes(b'best')
 (output/'cost.json').write_text('{"status":"completed"}')
 (output/'result.json').write_text('{"status":"completed"}')
 (output/'latest-epoch.json').write_text(json.dumps({'epoch':100,'global_step':500500}))
 events=[]
 def backup(*args):
  files={name for name,_,_ in transport._source_files(output)}
  assert {'checkpoint.pt','best.pt','cost.json','result.json'}<=files
  pytest.fail('final checkpoint upload must not run')
 monkeypatch.setattr(operations,'backup_and_wait',backup)
 monkeypatch.setattr(operations,'current',lambda *args:events.append(args[2]))
 operations.complete_run(tmp_path,output,'completed')
 assert events==['completed']
 saved=json.loads((output/'final-summary.json').read_text())
 assert saved['cost']['status']=='completed'
 assert (output/'best.pt').read_bytes()==b'best'


def test_observer_waits_for_retry_after_completed_session(tmp_path, monkeypatch):
 import json
 states = [
  {'session': {'id': 'old', 'ended': True}, 'events': []},
  {'session': {'id': 'new', 'ended': False, 'code': 'b'*40}, 'events': [
   {'id': 5, 'value': {'status': {'current': {'job': 'va64-g-preflight', 'campaign_id': 'retry', 'stage': 'waiting_wandb'}}}}
  ]},
 ]
 (tmp_path/observer.STATE_NAME).write_text(json.dumps({'session_ended': True, 'cursor': 4, 'active_jobs': ['va64-g-preflight'], 'campaigns': {'va64-g-preflight': 'old'}}))
 calls=[]
 class Relay:
  token='test'
  def call(self,path,value=None):
   if path.startswith('/snapshot'): return states.pop(0)
   calls.append(value); return {}
 class Mirror:
  enabled=True
  def __init__(self,*args,**kwargs): pass
  def finish_all(self): pass
  def _run(self,campaign,job): assert campaign=='retry'; return True
  def consume(self,*args): pass
 monkeypatch.setattr(observer,'owner_api_from_secrets',lambda _:Relay())
 monkeypatch.setattr(observer,'_WandbMirror',Mirror)
 monkeypatch.setattr(observer,'_backup_latest',lambda *args:None)
 class Done(Exception): pass
 def sleep(_):
  if not states: raise Done
 monkeypatch.setattr(observer.time,'sleep',sleep)
 with pytest.raises(Done): observer.observe(tmp_path,'unused',poll_seconds=0.01)
 state=json.loads((tmp_path/observer.STATE_NAME).read_text())
 assert state['session_id']=='new'
 assert state['campaigns']=={'va64-g-preflight':'retry'}
 assert state['child_failed'] is False
 assert {'action':'ready','job':'va64-g-preflight'} in calls


def test_stop_signal_cancels_backup_without_waiting_for_upload(tmp_path,monkeypatch):
 import os,signal
 from imagenet1k_runtime import TrainingInterrupted
 import va64_h200_transport as transport
 monkeypatch.setenv('VA64_G_CODE_SHA','b'*40)
 def upload(*args,**kwargs):
  assert kwargs['wait'] and callable(kwargs['cancel_check'])
  signal.raise_signal(signal.SIGTERM)
  raise AssertionError('must not wait for backup completion')
 monkeypatch.setattr(transport,'backup_run',upload)
 previous=signal.getsignal(signal.SIGTERM)
 (tmp_path/'output').mkdir()
 with pytest.raises(TrainingInterrupted,match='cancel backup'):
  operations.backup_and_wait(tmp_path,'va64-g-seed0',tmp_path/'output',1,5005)
 assert (tmp_path/'STOP').is_file()
 assert signal.getsignal(signal.SIGTERM)==previous


def test_cancellable_backup_does_not_start_executor_thread(tmp_path,monkeypatch):
 import threading
 from scripts import va64_h200_transport as transport
 from scripts.imagenet1k_runtime import TrainingInterrupted
 archive=tmp_path/'archive.tar.gz';archive.write_bytes(b'archive')
 monkeypatch.setattr(transport,'_make_archive',lambda *args:archive)
 def upload(*args):
  assert threading.current_thread() is threading.main_thread()
  raise TrainingInterrupted('cancelled')
 monkeypatch.setattr(transport,'_upload_archive',upload)
 monkeypatch.setattr(transport._UPLOAD_POOL,'submit',lambda *args:pytest.fail('background upload must not start'))
 with pytest.raises(TrainingInterrupted):
  transport.backup_run('va64-g-seed0',tmp_path,1,5005,'b'*40,wait=True,cancel_check=lambda:None)


def test_final_backup_stop_uses_normal_stop_classification(tmp_path,monkeypatch):
 import run_va64_g_h200 as entry
 import va64_runtime_operations as ops
 from scripts.imagenet1k_runtime import TrainingInterrupted
 from imagenet1k_runtime import ExecutionOptions
 monkeypatch.setenv('VA64_G_MANAGED_ROOT',str(tmp_path))
 monkeypatch.delenv('VA64_G_RECOVERY',raising=False)
 monkeypatch.setattr(entry.experiment,'arguments',lambda:SimpleNamespace(variant='G',seed=0))
 task=SimpleNamespace(execution=ExecutionOptions(),source_sha256={},output_dir=tmp_path/'run')
 # The task wrapper requires a real dataclass for dataclasses.replace.
 from dataclasses import dataclass
 @dataclass
 class Task:
  execution:object
  source_sha256:dict
  output_dir:Path
 monkeypatch.setattr(entry.experiment,'make_task',lambda _:Task(ExecutionOptions(),{},tmp_path/'run'))
 monkeypatch.setattr(entry.experiment,'build_model',lambda *a,**k:None)
 monkeypatch.setattr(entry.experiment,'train',lambda _:{'status':'completed'})
 monkeypatch.setattr(ops,'install_training',lambda *a:None)
 def stop(*a):raise entry.experiment.runtime.TrainingInterrupted('stop during final backup')
 monkeypatch.setattr(ops,'complete_run',stop)
 with pytest.raises(SystemExit) as result:entry.main()
 assert result.value.code==130
 import json
 assert json.loads((tmp_path/'current.json').read_text())['stage']=='stopped_without_backup'


def test_local_only_start_and_final_results_need_no_network(tmp_path,monkeypatch):
 import json
 monkeypatch.setenv('VA64_G_CHECKPOINT_POLICY','local_only')
 monkeypatch.setattr(operations,'_agent_api',lambda:pytest.fail('no network gate allowed'))
 monkeypatch.setattr(operations,'backup_and_wait',lambda *args:pytest.fail('no final checkpoint backup'))
 operations.wait_ready(tmp_path,'va64-g-seed0')
 output=tmp_path/'output';output.mkdir()
 (output/'result.json').write_text(json.dumps({'status':'completed','completed_epochs':100,'final_validation':{'accuracy':.5},'history':[{'epoch':1}]}))
 (output/'cost.json').write_text(json.dumps({'training_seconds':1}))
 operations.complete_run(tmp_path,output,'completed')
 saved=json.loads((output/'final-summary.json').read_text())
 assert saved['result']['completed_epochs']==100
 assert 'history' not in saved['result']
 assert saved['cost']['training_seconds']==1
 assert json.loads((tmp_path/'current.json').read_text())['checkpoint_policy']=='local_only'
