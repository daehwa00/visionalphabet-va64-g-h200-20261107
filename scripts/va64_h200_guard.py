"""Independent bootstrap/training watchdog; never transfers model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from va64_control_transport import API,BASE,atomic,read,retry_delay,error_details


WAITING_STAGES={'waiting_wandb','waiting_metrics','wait_backup'}


def error_label(error):
    return error_details(error)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code-sha',required=True)
    p.add_argument('--relay-url',default=BASE)
    p.add_argument('--share-token-with-child',action='store_true')
    p.add_argument('--enroll-wait-seconds',type=int,default=0)
    p.add_argument('--compact-results',action='store_true')
    p.add_argument('--metrics-optional',action='store_true')
    p.add_argument('--grace-seconds',type=int,default=900);p.add_argument('--stall-seconds',type=int,default=1200)
    p.add_argument('--poll-seconds',type=float,default=15)
    p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    if args.command[:1]==['--']:args.command=args.command[1:]
    if not args.command:raise ValueError('Missing supervised command')
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    token=secrets.token_hex(32);api=API(token,base=args.relay_url)
    enrollment={'pod':os.uname().nodename,'token_hash':hashlib.sha256(token.encode()).hexdigest(),'code_sha':args.code_sha}
    enroll_deadline=time.monotonic()+args.enroll_wait_seconds
    while True:
        try:
            response=api.call('/enroll',enrollment,attempts=1,timeout=8)
            break
        except Exception as error:
            details=error_details(error)
            retryable=(details.get('http_status') in (429,500,502,503,504)
                       or (details.get('http_status')==403 and 'source_not_allowed' in details.get('body_excerpt',''))
                       or (details.get('http_status') is None and isinstance(error,(TimeoutError,ConnectionError,URLError))))
            if not retryable or time.monotonic()>=enroll_deadline:raise
            print('VA64_ENROLL_WAIT='+json.dumps({'pod':enrollment['pod'],'code_sha':args.code_sha,'error_type':type(error).__name__}),flush=True)
            time.sleep(min(15,max(0,enroll_deadline-time.monotonic())))
    api.session=response['session_id'];atomic(root/'session.json',{'id':api.session,'code_sha':args.code_sha})
    print('VA64_SESSION='+api.session,flush=True)
    if not (root/'current.json').exists():
        atomic(root/'current.json',{'job':os.environ.get('VA64_G_JOB','va64-g-preflight'),'campaign_id':root.name,'stage':'bootstrap','output':str(root),'checkpoint_policy':os.environ.get('VA64_G_CHECKPOINT_POLICY','external')})
    stopped=False;stop_reasons=[]
    def record(kind,reason):
        entry={'time':time.time(),'kind':kind,'reason':reason}
        with (root/'guard-events.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
        print('VA64_GUARD='+json.dumps(entry),flush=True)
    def request_stop(reason):
        nonlocal stopped
        stopped=True
        if reason not in stop_reasons:stop_reasons.append(reason);record('stop',reason)
    def stop(signum,frame):
        request_stop('signal:'+signal.Signals(signum).name)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    seq=offset=printed=0;last_contact=time.monotonic();deadline=None;forced=False;last_signature=None;last_progress=time.monotonic()
    last_error=None;started=time.monotonic();drain_deadline=None
    control={};next_control=next_event=0.;control_failures=event_failures=0;last_warning=None;sent_ended=False
    gpu_metrics=None;gpu_metrics_time=0.
    child_env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
    if args.share_token_with_child:
        child_env['H200_AGENT_TOKEN']=token
        child_env['H200_SESSION_ID']=api.session
    with (root/'console.log').open('ab',buffering=0) as console:
        child=subprocess.Popen(args.command,stdout=console,stderr=subprocess.STDOUT,start_new_session=True,
                               env=child_env)
        try:
            while True:
                now=time.monotonic();current=read(root/'current.json');job=current.get('job')
                if child.poll() is not None and drain_deadline is None:
                    drain_deadline=now+60;next_event=min(next_event,now)
                if now>=next_control and drain_deadline is None:
                    try:
                        fresh=api.call('/control',attempts=1,timeout=8)
                        if not isinstance(fresh,dict):raise ValueError('Invalid control response')
                        control=fresh;last_contact=time.monotonic();atomic(root/'control.json',control)
                        control_failures=0;next_control=last_contact+args.poll_seconds
                        if isinstance(last_error,dict) and last_error.get('route')=='control':last_error=None
                    except Exception as exc:
                        control_failures+=1;last_error={'route':'control',**error_label(exc)}
                        next_control=time.monotonic()+retry_delay(exc,control_failures,args.poll_seconds)
                        record('warning',last_error)
                if control.get('stop'):request_stop('remote_stop')
                if control.get('force'):request_stop('remote_force');forced=True
                if (root/'STOP').exists() and deadline is None:request_stop('local_STOP')
                now=time.monotonic()
                seen=control.get('observer_seen',0)/1000
                warning=('control_unavailable' if now-last_contact>600 else
                         'observer_stale' if now-started>600 and time.time()-seen>600 else None)
                if warning and warning!=last_warning:record('warning',warning)
                last_warning=warning
                if warning and not args.metrics_optional: request_stop(warning)
                output=Path(current['output']) if current.get('output') else None
                progress=read(output/'step-progress.json') if output else {}
                if output and (output/'steps.jsonl').exists():
                    try:
                        with (output/'steps.jsonl').open('rb') as stream:
                            stream.seek(max(0,(output/'steps.jsonl').stat().st_size-8192)); lines=stream.read().splitlines()
                        if lines:progress=json.loads(lines[-1])
                    except (OSError,ValueError):pass
                checkpoint=read(output/'checkpoint-meta.json') if output else {}
                result=read(output/'result.json') if output else {}
                if args.compact_results:
                    result={key:value for key,value in result.items() if key in {
                        'status','model_key','seed','parameters','completed_epochs','global_step',
                        'training_seconds','final_validation','contract_sha256'}}
                signature=(job,current.get('stage'),current.get('probe_step'),progress.get('global_step'),checkpoint.get('epoch'))
                if signature!=last_signature:last_signature=signature;last_progress=now
                if current.get('stage') in WAITING_STAGES:last_progress=now
                if child.poll() is None and now-last_progress>args.stall_seconds:request_stop('progress_stalled')
                if stopped and deadline is None:
                    (root/'STOP').touch();deadline=now+args.grace_seconds
                    # Signal only the supervisor. It stops at an optimizer boundary;
                    # DataLoader workers must not receive an early SIGTERM.
                    try:os.kill(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                if forced or (deadline is not None and now>=deadline and child.poll() is None):
                    if not forced:request_stop('grace_expired')
                    forced=True
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                status={'time':time.time(),'current':current,'progress':progress,'checkpoint':checkpoint,
                        'result':result,'summary':read(output/'final-summary.json') if output else {},
                        'completed_runs':read(root/'completed-runs.json'),
                        'exit_code':child.poll(),'stop_requested':stopped,'forced':forced,'error':last_error,
                        'stop_reasons':stop_reasons,'connectivity_warning':warning,
                        'ended':child.poll() is not None}
                for name in ('memory.current','memory.max','memory.events','cpu.max','cpu.stat','cpuset.cpus.effective'):
                    try:status[name]=Path('/sys/fs/cgroup',name).read_text().strip()
                    except OSError:pass
                if now-gpu_metrics_time>=30:
                    try:
                        probe=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,clocks.sm,power.draw',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                        gpu_metrics={'sampled_at':time.time(),'csv':probe.stdout.strip(),'exit_code':probe.returncode}
                    except (OSError,subprocess.TimeoutExpired) as error:
                        gpu_metrics={'sampled_at':time.time(),'error':type(error).__name__}
                    gpu_metrics_time=now
                status['gpu_metrics']=gpu_metrics
                atomic(root/'guard-status.json',status)
                with (root/'console.log').open('rb') as stream:
                    stream.seek(printed);visible=stream.read(1024*1024)
                if visible:
                    sys.stdout.write(visible.decode('utf-8',errors='replace'));sys.stdout.flush();printed+=len(visible)
                if now>=next_event and (drain_deadline is None or drain_deadline-time.monotonic()>=8):
                    try:
                        with (root/'console.log').open('rb') as stream:stream.seek(offset);chunk=stream.read(16000)
                        # Do not let the observer finish before the last console chunk.
                        status['ended']=child.poll() is not None and offset+len(chunk)>=(root/'console.log').stat().st_size
                        api.call('/events',{'seq':seq,'logs':chunk.decode('utf-8',errors='replace'),'status':status},attempts=1,timeout=8)
                        offset+=len(chunk);seq+=1;event_failures=0;next_event=time.monotonic()+(0.25 if drain_deadline is not None else args.poll_seconds)
                        if isinstance(last_error,dict) and last_error.get('route')=='events':last_error=None
                        sent_ended|=status['ended']
                        atomic(root/'relay-cursor.json',{'session':api.session,'seq':seq,'offset':offset})
                    except Exception as exc:
                        event_failures+=1;last_error={'route':'events',**error_label(exc)}
                        next_event=time.monotonic()+retry_delay(exc,event_failures,args.poll_seconds)
                        record('warning',last_error)
                if child.poll() is not None:
                    if drain_deadline is None:drain_deadline=now+60
                    if (sent_ended and min(offset,printed)>=(root/'console.log').stat().st_size) or time.monotonic()>=drain_deadline:break
                time.sleep(min(1,args.poll_seconds))
        finally:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=child.wait(timeout=30)
            summary={'code':code,'forced':forced,'stopped':stopped,'stop_reasons':stop_reasons,'relay_offset':offset,'console_bytes':(root/'console.log').stat().st_size}
            atomic(root/'exit.json',summary);record('exit',summary)
        return 130 if stopped or forced else code


if __name__=='__main__':sys.exit(main())
