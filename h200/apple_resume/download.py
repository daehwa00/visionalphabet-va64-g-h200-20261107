"""Download pinned checkpoint parts over the existing authenticated relay."""
import concurrent.futures,hashlib,json,os,sys,time,threading
import requests
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
from va64_control_transport import API,atomic

class BinaryReader:
    def __init__(self,api):
        self.api=api;self.local=threading.local();self.lock=threading.Lock();self.next_request=0.
    def get(self,path,expected_bytes):
        if not hasattr(self.local,'session'):self.local.session=requests.Session()
        for attempt in range(16):
            with self.lock:
                time.sleep(max(0,self.next_request-time.monotonic()));self.next_request=time.monotonic()+.15
            try:
                response=self.local.session.get(self.api.base+path,headers={'Authorization':'Bearer '+self.api.token,'X-Session-ID':self.api.session,'Accept':'application/octet-stream'},timeout=(10,60))
                if response.status_code==429 or response.status_code>=500:
                    time.sleep(max(float(response.headers.get('Retry-After','0')),min(10,2**attempt)));continue
                response.raise_for_status()
                data=response.content
                if len(data)!=expected_bytes:raise requests.exceptions.ChunkedEncodingError('Truncated checkpoint chunk')
                return data
            except requests.RequestException:
                if attempt==15:raise RuntimeError('Checkpoint chunk download retries exhausted') from None
                self.local.session.close();self.local.session=requests.Session()
                time.sleep(min(10,2**attempt))
        raise RuntimeError('Checkpoint server retries exhausted')

def main():
    manifest=json.loads((HERE/'transfer-manifest.json').read_text())
    target=Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=True)
    api=API(os.environ['H200_AGENT_TOKEN'],session=os.environ['H200_SESSION_ID'],base=manifest['base'])
    reader=BinaryReader(api)
    temporary=target/'latest.pt.part';whole=hashlib.sha256();total=0
    with temporary.open('wb') as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for part in manifest['parts']:
            deadline=time.monotonic()+1800
            while True:
                available=api.call(f'/artifact/{part["job"]}/latest')
                if available and available.get('sha')==part['sha'] and available.get('complete'):break
                if time.monotonic()>deadline:raise TimeoutError('Checkpoint upload not complete')
                time.sleep(10)
            sha=hashlib.sha256();size=0
            for start in range(0,part['chunks'],4):
                futures=[pool.submit(reader.get,f'/artifact/{part["job"]}/{part["sha"]}/{i}',min(part['chunk_bytes'],part['bytes']-i*part['chunk_bytes'])) for i in range(start,min(start+4,part['chunks']))]
                for future in futures:
                    block=future.result();f.write(block);sha.update(block);whole.update(block);size+=len(block);total+=len(block)
                if start%32==0:print('CHECKPOINT_DOWNLOAD='+json.dumps({'bytes':total,'target':manifest['checkpoint']['bytes']}),flush=True)
            assert sha.hexdigest()==part['sha'] and size==part['bytes'],'Part integrity mismatch'
        f.flush();os.fsync(f.fileno())
    receipt=manifest['checkpoint']
    assert whole.hexdigest()==receipt['sha256'] and total==receipt['bytes'],'Checkpoint integrity mismatch'
    temporary.replace(target/'latest.pt');atomic(target/'latest.pt.json',dict(receipt,path=str(target/'latest.pt')))
    print('CHECKPOINT_DOWNLOAD_VERIFIED='+receipt['sha256'],flush=True)
if __name__=='__main__':main()
