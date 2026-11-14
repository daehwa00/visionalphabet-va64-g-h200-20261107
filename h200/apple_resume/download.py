"""Download pinned checkpoint parts over the existing authenticated relay."""
import concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
from va64_control_transport import API,atomic

def main():
    manifest=json.loads((HERE/'transfer-manifest.json').read_text())
    target=Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=True)
    api=API(os.environ['H200_AGENT_TOKEN'],session=os.environ['H200_SESSION_ID'],base=manifest['base'])
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
                futures=[pool.submit(api.call,f'/artifact/{part["job"]}/{part["sha"]}/{i}',None,None,True,60) for i in range(start,min(start+4,part['chunks']))]
                for future in futures:
                    block=future.result();f.write(block);sha.update(block);whole.update(block);size+=len(block);total+=len(block)
                if start%128==0:print('CHECKPOINT_DOWNLOAD='+json.dumps({'bytes':total,'target':manifest['checkpoint']['bytes']}),flush=True)
            assert sha.hexdigest()==part['sha'] and size==part['bytes'],'Part integrity mismatch'
        f.flush();os.fsync(f.fileno())
    receipt=manifest['checkpoint']
    assert whole.hexdigest()==receipt['sha256'] and total==receipt['bytes'],'Checkpoint integrity mismatch'
    temporary.replace(target/'latest.pt');atomic(target/'latest.pt.json',dict(receipt,path=str(target/'latest.pt')))
    print('CHECKPOINT_DOWNLOAD_VERIFIED='+receipt['sha256'],flush=True)
if __name__=='__main__':main()
