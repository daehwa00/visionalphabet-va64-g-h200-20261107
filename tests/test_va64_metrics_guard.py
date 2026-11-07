"""Exercise the live watchdog loop with a healthy child and failed telemetry."""
import json
from types import SimpleNamespace
from scripts import va64_h200_guard as guard


def test_metrics_outage_never_stops_healthy_training(tmp_path,monkeypatch):
    tick=[0.]
    def monotonic():tick[0]+=100.;return tick[0]
    class API:
        def __init__(self,*args,**kwargs):self.session=''
        def call(self,path,*args,**kwargs):
            if path=='/enroll':return {'session_id':'test'}
            raise ConnectionError('telemetry offline')
    class Child:
        pid=123456789
        polls=0
        def __init__(self,*args,**kwargs):pass
        def poll(self):
            self.polls+=1
            return 0 if self.polls>25 else None
        def wait(self,**kwargs):return 0
    signals=[]
    monkeypatch.setattr(guard,'API',API)
    monkeypatch.setattr(guard.time,'monotonic',monotonic)
    monkeypatch.setattr(guard.time,'sleep',lambda _:None)
    monkeypatch.setattr(guard.signal,'signal',lambda *args:None)
    monkeypatch.setattr(guard.os,'kill',lambda *args:signals.append(args))
    monkeypatch.setattr(guard.os,'killpg',lambda *args:None)
    monkeypatch.setattr(guard.subprocess,'Popen',Child)
    monkeypatch.setattr(guard.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='',returncode=0))
    monkeypatch.setattr(guard.sys,'argv',['guard','--root',str(tmp_path),'--code-sha','b'*40,'--metrics-optional','--stall-seconds','10000000','--','healthy-child'])
    assert guard.main()==0
    result=json.loads((tmp_path/'exit.json').read_text())
    assert result['stopped'] is False and result['forced'] is False
    assert result['stop_reasons']==[] and signals==[]
    warnings=[json.loads(x) for x in (tmp_path/'guard-events.jsonl').read_text().splitlines()]
    assert any(x['kind']=='warning' and x['reason']=='control_unavailable' for x in warnings)
