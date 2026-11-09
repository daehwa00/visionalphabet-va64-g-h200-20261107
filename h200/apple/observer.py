"""Local W&B mirror. Authentication never enters H200 or the issue body."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from va64_control_transport import API,atomic,read
BASE='https://apple70m-h200-20261109.gpupulse-monitor.workers.dev/va64g'

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--secrets',type=Path,required=True);p.add_argument('--action',choices=['observe','status','stop','force','arm'],default='observe');a=p.parse_args()
 api=API(json.loads(a.secrets.read_text())['IN10_OWNER_TOKEN'],base=BASE)
 if a.action!='observe':
  if a.action=='status':
   v=api.call('/snapshot');print(json.dumps({'session':v.get('session'),'control':v.get('control')}))
  else:print(json.dumps(api.call('/command',{'action':a.action})))
  return
 import wandb
 a.root.mkdir(parents=True,exist_ok=True);state=read(a.root/'state.json');cursor=state.get('cursor',0);session=state.get('session');run=None;ended_session=state.get('ended_session')
 while True:
  try:
   v=api.call('/snapshot?after='+str(cursor),attempts=1);api.call('/command',{'action':'heartbeat'},attempts=1)
   s=v.get('session') or {};sid=s.get('id')
   if sid and sid!=session:
    if run:run.finish();run=None
    session=sid
   if sid and sid!=ended_session and run is None:
    run=wandb.init(entity='daehwa',project='apple70m-h200',id=hashlib.sha256(sid.encode()).hexdigest()[:16],resume='allow',name='APPLE70M-data-preflight',mode='online',dir=str(a.root),config={'phase':'data-preflight','training_started':False,'source_commit':s.get('code'),'session':sid},settings=wandb.Settings(init_timeout=60))
    if run is None or run.offline:raise RuntimeError('W&B online run required')
    print('WANDB_URL='+run.url,flush=True)
   for event in v.get('events',[]):
    payload=event['value'];payload=json.loads(payload) if isinstance(payload,str) else payload
    status=payload.get('status') or {};logs=payload.get('logs','');current=status.get('current') or {}
    with (a.root/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    with (a.root/'console.log').open('a') as f:f.write(logs)
    atomic(a.root/'latest-status.json',status)
    if run:
     progress=status.get('progress') or {};metrics={f'probe/{k}':v for k,v in progress.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
     run.log({**metrics,'observer/event_id':event['id'],'probe/ended':int(bool(status.get('ended')))},step=event['id'])
     if current.get('stage')=='waiting_wandb':api.call('/command',{'action':'ready','job':'apple70m-data-preflight'},attempts=1)
     if status.get('ended'):
      run.summary.update({'exit_code':status.get('exit_code'),'stopped':status.get('stop_requested'),'forced':status.get('forced'),'result':status.get('result',{})})
      artifact=wandb.Artifact('apple70m-preflight-'+sid[:8],type='logs');artifact.add_file(str(a.root/'console.log'));artifact.add_file(str(a.root/'latest-status.json'));run.log_artifact(artifact)
      run.finish();run=None;ended_session=sid
    cursor=event['id'];atomic(a.root/'state.json',{'cursor':cursor,'session':session,'ended_session':ended_session})
  except Exception as e:
   print('OBSERVER_RETRY='+type(e).__name__,flush=True)
  time.sleep(15)
if __name__=='__main__':main()
