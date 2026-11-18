"""Defer the normal VA64-G stop until seed 1's final result is externally visible."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path


def should_stop(status):
    completed=status.get('completed_runs',{}).get('va64-g-seed1',{})
    result=completed.get('result',{})
    if result.get('status')=='completed' and result.get('completed_epochs')==100:
        return 'seed1_completed'
    if status.get('current',{}).get('job')=='va64-g-seed2':
        return 'seed2_boundary_reached'
    return None


def monitor(api,session,state_path,cursor,poll=3):
    state={'session_id':session,'cursor':cursor,'status':'armed','cancelled_seed':2}
    def save():
        state['updated_unix']=time.time();temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    save()
    while True:
        try:
            view=api.call('/snapshot?after='+str(state['cursor']),attempts=1)
            if view.get('session',{}).get('id')!=session:
                state.update(status='blocked_session_changed');save();return
            for event in view.get('events',[]):
                value=event.get('value',{});value=json.loads(value) if isinstance(value,str) else value
                reason=should_stop(value.get('status',{}))
                if reason:
                    control=api.call('/command',{'action':'stop'},attempts=1)
                    if not control.get('stop') or control.get('force'):raise RuntimeError('normal stop not acknowledged')
                    state.update(status='normal_stop_acknowledged',reason=reason);save();return
                state['cursor']=max(state['cursor'],int(event.get('id',state['cursor'])))
            if view.get('session',{}).get('ended'):
                state.update(status='session_already_ended');save();return
            save()
        except Exception as error:
            state['last_error_type']=type(error).__name__;save()
        time.sleep(poll)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--secrets',type=Path,required=True);p.add_argument('--observer-root',type=Path,required=True);p.add_argument('--state',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.repository/'scripts'))
    from va64_control_transport import API
    owner=json.loads(a.secrets.read_text());observed=json.loads((a.observer_root/'observer-state.json').read_text())
    if a.state.exists():
        old=json.loads(a.state.read_text())
        if old['session_id']!=observed['session_id']:raise RuntimeError('existing watcher session differs')
        if old['status']!='armed':return
        cursor=old['cursor']
    else:cursor=observed['cursor']
    monitor(API(owner['IN10_OWNER_TOKEN']),observed['session_id'],a.state,cursor)

if __name__=='__main__':main()
