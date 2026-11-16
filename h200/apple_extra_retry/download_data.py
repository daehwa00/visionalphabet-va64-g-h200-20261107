import hashlib,json,os,sys,time,zipfile
from pathlib import Path
from urllib.parse import urlparse
import requests,pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
from va64_control_transport import API,atomic
BASE='https://apple70m-h200-20261109.gpupulse-monitor.workers.dev/va64g'
def main():
 target=Path(sys.argv[1]);target.mkdir(parents=True,exist_ok=True)
 expected=json.loads((HERE/'data-archive.json').read_text())
 api=API(os.environ['H200_AGENT_TOKEN'],session=os.environ['H200_SESSION_ID'],base=BASE)
 for attempt in range(10):
  try:
   meta=api.call('/artifact/apple70m-extra-input/latest');raw=api.call(f'/artifact/apple70m-extra-input/{meta["sha"]}/0',raw=True)
   assert hashlib.sha256(raw).hexdigest()==meta['sha'];data=json.loads(raw)['data_archive']
   assert data['sha256']==expected['sha256'] and data['bytes']==expected['bytes']
   assert urlparse(data['url']).scheme=='https' and urlparse(data['url']).hostname=='release-assets.githubusercontent.com'
   r=requests.get(data['url'],timeout=(15,90));r.raise_for_status();blob=r.content
   assert len(blob)==expected['bytes'] and hashlib.sha256(blob).hexdigest()==expected['sha256'];break
  except Exception as e:
   print('DATA_DOWNLOAD_RETRY='+type(e).__name__,flush=True)
   if attempt==9:raise RuntimeError('Offline dataset archive transfer failed') from None
   time.sleep(min(30,2**attempt))
 archive=target/'archive.zip';archive.write_bytes(blob)
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():
   path=target/name;assert path.resolve().is_relative_to(target.resolve());path.parent.mkdir(parents=True,exist_ok=True)
   path.write_bytes(z.read(name))
 manifest=json.loads((target/'manifest.json').read_text());assert len(manifest['files'])==90
 for item in manifest['files']:
  p=target/item['path'];assert p.resolve().is_relative_to(target.resolve())
  with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  assert sha==item['sha256'] and p.stat().st_size==item['bytes']
  if 'rows' in item:assert pq.ParquetFile(p).metadata.num_rows==item['rows']
 atomic(target/'verified.json',{'archive':expected,'verified_files':90})
 print('OFFLINE_DATA_VERIFIED=90',flush=True)
if __name__=='__main__':main()
