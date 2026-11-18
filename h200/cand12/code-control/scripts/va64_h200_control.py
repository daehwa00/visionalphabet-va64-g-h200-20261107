"""Owner-only control of the isolated VA64 G relay; never print credentials."""
import argparse,json,os
from pathlib import Path
from va64_h200_transport import owner_api_from_secrets

def main():
 p=argparse.ArgumentParser(); p.add_argument('action',choices=('status','stop','force','arm','release')); p.add_argument('--secrets',type=Path,required=True); p.add_argument('--relay-url'); a=p.parse_args()
 if a.relay_url: os.environ['VA64_G_RELAY_URL']=a.relay_url
 api=owner_api_from_secrets(a.secrets)
 if a.action=='status':
  result=api.call('/snapshot'); print(json.dumps({'session':result.get('session'),'control':result.get('control')}))
 else: print(json.dumps(api.call('/command',{'action':a.action})))
if __name__=='__main__': main()
