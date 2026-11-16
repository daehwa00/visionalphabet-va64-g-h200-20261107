import hashlib,json,os,sys,time,zipfile
from pathlib import Path
import requests
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'scripts'))
from va64_control_transport import API,atomic
BASE='https://apple70m-h200-20261109.gpupulse-monitor.workers.dev/va64g'
def main():
 root=Path(sys.argv[1]);archive=root/'evaluation-results.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for path in root.rglob('*.json'):
   if 'models' not in path.relative_to(root).parts:z.write(path,path.relative_to(root))
 blob=archive.read_bytes();sha=hashlib.sha256(blob).hexdigest();chunk=256*1024
 assert len(blob)<384*1024**2
 api=API(os.environ['H200_AGENT_TOKEN'],session=os.environ['H200_SESSION_ID'],base=BASE)
 meta={'sha':sha,'bytes':len(blob),'chunks':(len(blob)+chunk-1)//chunk,'epoch':1,'global_step':47684,'code_sha':sys.argv[2]}
 api.call('/artifact/apple70m-extra-eval/begin',meta)
 with requests.Session() as s:
  for i in range(meta['chunks']):
   for attempt in range(12):
    try:
     r=s.put(BASE+f'/artifact/apple70m-extra-eval/{sha}/{i}',data=blob[i*chunk:(i+1)*chunk],headers={'Authorization':'Bearer '+api.token,'X-Session-ID':api.session,'Content-Type':'application/octet-stream'},timeout=(10,60))
     r.raise_for_status();break
    except requests.RequestException:
     if attempt==11:raise RuntimeError('Results upload failed') from None
     time.sleep(min(30,2**attempt))
   time.sleep(.2)
 api.call(f'/artifact/apple70m-extra-eval/{sha}/finish',{})
 atomic(root/'results-archive.json',meta);print('RESULTS_ARCHIVE='+json.dumps(meta),flush=True)
if __name__=='__main__':main()
