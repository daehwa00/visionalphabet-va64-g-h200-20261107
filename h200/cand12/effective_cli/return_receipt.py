"""Transfer only the complete, locally validated numeric JSON receipt to its owner."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,math,os,secrets,socket,subprocess,threading,time,zlib
from pathlib import Path
import requests

BASE="https://va-r4-marker-result-20261119.gpupulse-monitor.workers.dev/va64g"
CHUNK=8192
def publish(raw,summary):
    from effective_batch_search import valid_receipt
    d=json.loads(raw)
    assert valid_receipt(d,d['key']) and d['state_before']==d['state_after']
    original=hashlib.sha256(raw).hexdigest()
    meta=json.dumps(dict(raw_sha256=original,ordinary={k:v for k,v in summary.items() if k!="source_closure"}),separators=(',',':')).encode()
    assert len(meta)<131072
    blob=zlib.compress(b"VAER1"+len(meta).to_bytes(4,"big")+meta+raw,9)
    assert 0<len(blob)<=32*1024**2
    upload(blob,'receipt')
    print("FULL_RECEIPT_TRANSFER",json.dumps(dict(raw_sha256=original,compressed_sha256=hashlib.sha256(blob).hexdigest(),
              raw_bytes=len(raw),compressed_bytes=len(blob),key=d['key'],valid=True)),flush=True)

def upload(blob,kind):
    assert kind in ('receipt','kernel-events') and 0<len(blob)<=32*1024**2
    sig=hashlib.sha256(blob).hexdigest()
    # Both uploads authenticate as the same allocated pod. Enrollment is idempotent for this token.
    token=(Path(os.environ['RUN_ROOT'])/'private-input/.result-sender').read_text().strip()
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
    identity=dict(pod=socket.gethostname(),code_sha=code,token_hash=hashlib.sha256(token.encode()).hexdigest())
    def req(method,path,headers,obj=None):
        for i in range(6):
            try:
                r=requests.request(method,BASE+path,json=obj,headers=headers,timeout=(15,30))
                if r.status_code in (429,500,502,503,504):
                    raise requests.RequestException(f'retryable HTTP {r.status_code}')
                if not r.ok: raise RuntimeError(f'result handoff {path[:32]} HTTP {r.status_code}')
                return r.json()
            except requests.RequestException:
                if i==5: raise RuntimeError('result handoff transport interrupted') from None
                time.sleep(2**i)
    sid=req('POST','/enroll',{},identity)['session_id']
    headers={'Authorization':'Bearer '+token,'X-Session-ID':sid,'Accept-Encoding':'identity'}
    n=math.ceil(len(blob)/CHUNK)
    path='/artifact/'+kind
    req('POST',path+'/begin',headers,dict(sha=sig,bytes=len(blob),chunks=n,epoch=18,global_step=18270,code_sha=code))
    lock=threading.Lock();last=0.
    def put(i):
        nonlocal last
        with lock:
            time.sleep(max(0,last-time.monotonic()));last=time.monotonic()+.15
        req('PUT',f'{path}/{sig}/{i}',headers,{'base64':base64.b64encode(blob[i*CHUNK:(i+1)*CHUNK]).decode()})
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(put,range(n)))
    ack=req('POST',f'{path}/{sig}/finish',headers,{})
    assert ack['complete'] and ack['sha']==sig
