"""Stream a private release asset using an ephemeral URL from authenticated relay."""
import hashlib,json,os,sys,time
from pathlib import Path
from urllib.parse import urlparse
import requests
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
from va64_control_transport import API,atomic

def main():
    pinned=json.loads((HERE/'transfer-manifest.json').read_text())['checkpoint']
    base=json.loads((HERE/'transfer-manifest.json').read_text())['base']
    api=API(os.environ['H200_AGENT_TOKEN'],session=os.environ['H200_SESSION_ID'],base=base)
    target=Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=True);partial=target/'latest.pt.part'
    job='apple70m-vonly-checkpoint'
    for attempt in range(12):
        try:
            meta=api.call(f'/artifact/{job}/latest')
            if not meta or meta['bytes']>32768:raise RuntimeError('Private transfer link not ready')
            raw=api.call(f'/artifact/{job}/{meta["sha"]}/0',raw=True)
            assert hashlib.sha256(raw).hexdigest()==meta['sha']
            info=json.loads(raw);receipt=info['receipt']
            assert receipt['sha256']==pinned['sha256'] and receipt['bytes']==pinned['bytes']
            assert urlparse(info['url']).scheme=='https' and urlparse(info['url']).hostname=='release-assets.githubusercontent.com'
            start=partial.stat().st_size if partial.exists() else 0
            if start==pinned['bytes']:break
            headers={'Range':f'bytes={start}-'} if start else {}
            with requests.get(info['url'],headers=headers,stream=True,timeout=(15,60)) as response:
                response.raise_for_status()
                if start and response.status_code==206:
                    assert response.headers.get('Content-Range','').startswith(f'bytes {start}-')
                else:
                    assert response.status_code==200;start=0
                written=start;last_report=start
                with partial.open('ab' if start else 'wb') as f:
                    for block in response.iter_content(1024*1024):
                        if not block:continue
                        written+=len(block)
                        if written>pinned['bytes']:raise RuntimeError('Unexpected asset size')
                        f.write(block)
                        if written-last_report>=16*1024**2:
                            last_report=written
                            current=target.parent/'current.json';status=json.loads(current.read_text());status['probe_step']=written;atomic(current,status)
                            print('PRIVATE_CHECKPOINT_DOWNLOAD='+json.dumps({'bytes':written,'target':pinned['bytes']}),flush=True)
                    f.flush();os.fsync(f.fileno())
            if written!=pinned['bytes']:raise RuntimeError('Truncated private asset')
            break
        except Exception as e:
            print('PRIVATE_DOWNLOAD_RETRY='+type(e).__name__,flush=True)
            if attempt==11:raise RuntimeError('Private checkpoint transfer failed') from None
            time.sleep(min(30,2**attempt))
    assert partial.stat().st_size==pinned['bytes']
    with partial.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    assert sha==pinned['sha256'],'Checkpoint SHA mismatch'
    partial.replace(target/'latest.pt');atomic(target/'latest.pt.json',dict(pinned,path=str(target/'latest.pt')))
    print('CHECKPOINT_DOWNLOAD_VERIFIED='+sha,flush=True)
if __name__=='__main__':main()
