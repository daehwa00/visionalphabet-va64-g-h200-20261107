import importlib.util,json,sqlite3,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).with_name('observer.py');spec=importlib.util.spec_from_file_location('apple_observer',p);obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)
class ObserverTest(unittest.TestCase):
 def test_sql_excludes_prior_session(self):
  db=sqlite3.connect(':memory:');db.execute('CREATE TABLE events(id INTEGER, session TEXT,value TEXT)');db.executemany('INSERT INTO events VALUES(?,?,?)',[(1,'old','ended'),(2,'new','waiting')])
  source=Path(__file__).with_name('cloudflare').joinpath('index.ts').read_text();query='SELECT id,session,value FROM events WHERE id>? AND session=? ORDER BY id LIMIT 100';self.assertIn(query,source)
  self.assertEqual(db.execute(query,(0,'new')).fetchall(),[(2,'new','waiting')])
 def run_once(self,offline):
  calls=[];logs=[]
  class API:
   def __init__(self,*args,**kwargs):pass
   def call(self,path,value=None,**kwargs):
    calls.append((path,value))
    if path.startswith('/snapshot'):return {'session':{'id':'new','code':'a'*40},'events':[{'id':2,'value':{'status':{'current':{'stage':'waiting_wandb'}},'logs':''}}]}
    return {}
  run=types.SimpleNamespace(offline=offline,url='test',log=lambda *a,**k:logs.append((a,k)),finish=lambda:None)
  wb=types.SimpleNamespace(init=lambda **k:run,Settings=lambda **k:k)
  with tempfile.TemporaryDirectory() as d:
   secret=Path(d)/'secret.json';secret.write_text(json.dumps({'IN10_OWNER_TOKEN':'test'}))
   with patch.object(sys,'argv',['observer','--root',d,'--secrets',str(secret)]),patch.object(obs,'API',API),patch.dict(sys.modules,{'wandb':wb}),patch.object(obs.time,'sleep',side_effect=KeyboardInterrupt):
    with self.assertRaises(KeyboardInterrupt):obs.main()
  return calls,logs
 def test_online_ready(self):
  calls,logs=self.run_once(False);self.assertIn(('/command',{'action':'ready','job':'apple70m-data-preflight'}),calls);self.assertEqual(len(logs),1)
 def test_offline_not_ready(self):
  calls,logs=self.run_once(True);self.assertNotIn(('/command',{'action':'ready','job':'apple70m-data-preflight'}),calls);self.assertFalse(logs)
if __name__=='__main__':unittest.main()
