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


def test_training_backup_runs_after_checkpoint_callback(tmp_path,monkeypatch):
 events=[]; output=tmp_path/'output'; output.mkdir()
 task=SimpleNamespace(output_dir=output)
 def base(task,**kwargs):
  (output/'checkpoint.pt').write_bytes(b'committed')
  kwargs['epoch_callback'](None,{'epoch':1,'global_step':5005})
  return {'status':'paused'}
 runtime=SimpleNamespace(run_task=base)
 def callback(_model,_row): events.append('derived_files')
 def backup(*args):
  assert (output/'checkpoint.pt').read_bytes()==b'committed'; events.append('external_backup')
 monkeypatch.setattr(operations,'backup_and_wait',backup)
 monkeypatch.setattr(operations.signal,'signal',lambda *_:None)
 operations.install_training(runtime,tmp_path)
 assert runtime.run_task(task,epoch_callback=callback)['status']=='paused'
 assert events==['derived_files','external_backup']


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


def test_final_backup_keeps_best_and_cost_before_completion(tmp_path,monkeypatch):
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
  events.append('backed_up')
 monkeypatch.setattr(operations,'backup_and_wait',backup)
 monkeypatch.setattr(operations,'current',lambda *args:events.append(args[2]))
 operations.complete_run(tmp_path,output,'completed')
 assert events==['backed_up','completed']


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
