"""Download four pinned weights; owner credentials never reach the worker."""
import hashlib,json,os,sys,time
from pathlib import Path
from urllib.parse import urlparse
import requests
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
from va64_control_transport import API,atomic
BASE='https://apple70m-h200-20261109.gpupulse-monitor.workers.dev/va64g'
def main():
 target=Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=True)
 api=API(os.environ['H200_AGENT_TOKEN'],session=os.environ['H200_SESSION_ID'],base=BASE)
 for model in json.loads((HERE/'models.json').read_text()):
  path=target/model['name'];partial=path.with_suffix('.part')
  for attempt in range(12):
   try:
    meta=api.call('/artifact/apple70m-extra-input/latest');assert meta and meta['bytes']<32768
    raw=api.call(f'/artifact/apple70m-extra-input/{meta["sha"]}/0',raw=True);assert hashlib.sha256(raw).hexdigest()==meta['sha']
    links=json.loads(raw);item=links['models'][model['name']]
    assert item['sha256']==model['sha256'] and item['bytes']==model['bytes']
    assert urlparse(item['url']).scheme=='https' and urlparse(item['url']).hostname=='release-assets.githubusercontent.com'
    start=partial.stat().st_size if partial.exists() else 0
    if start==model['bytes']:break
    with requests.get(item['url'],headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(15,60)) as response:
     response.raise_for_status()
     if start and response.status_code==206:assert response.headers.get('Content-Range','').startswith(f'bytes {start}-')
     else:assert response.status_code==200;start=0
     total=start
     with partial.open('ab' if start else 'wb') as f:
      for block in response.iter_content(1024*1024):
       total+=len(block);assert total<=model['bytes'];f.write(block)
      f.flush();os.fsync(f.fileno())
     assert total==model['bytes']
    break
   except Exception as e:
    print('MODEL_DOWNLOAD_RETRY='+type(e).__name__,flush=True)
    if attempt==11:raise RuntimeError('Model transfer failed') from None
    time.sleep(min(30,2**attempt))
  with partial.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  assert sha==model['sha256'] and partial.stat().st_size==model['bytes']
  partial.replace(path);atomic(path.with_suffix('.pt.json'),dict(model,path=str(path),schema=2,kind='weights'))
  print('MODEL_DOWNLOAD_VERIFIED='+model['name']+' '+sha,flush=True)
if __name__=='__main__':main()
