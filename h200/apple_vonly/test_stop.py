import http.server,json,os,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class StopTest(unittest.TestCase):
 def test_remote_stop_while_waiting_without_backup(self):
  started=time.monotonic();events=[]
  class Handler(http.server.BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_GET(self):self.reply({'stop':time.monotonic()-started>2,'observer_seen':time.time()*1000})
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
    if self.path=='/events':events.append(body)
    self.reply({'session_id':'test-session'} if self.path=='/enroll' else {'ok':True})
   def reply(self,v):
    b=json.dumps(v).encode();self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
  server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with tempfile.TemporaryDirectory() as d:
    root=Path(d)/'out';scratch=Path(d)/'cache';root.mkdir();scratch.mkdir()
    p=subprocess.run([sys.executable,str(ROOT/'scripts/va64_h200_guard.py'),'--root',str(root),'--code-sha','a'*40,'--relay-url',f'http://127.0.0.1:{server.server_port}','--poll-seconds','0.1','--grace-seconds','3','--',sys.executable,str(ROOT/'h200/apple_vonly/job.py'),'--root',str(root),'--scratch',str(scratch)],capture_output=True,text=True,timeout=20)
    self.assertEqual(p.returncode,130,p.stdout+p.stderr)
    result=json.loads((root/'exit.json').read_text());self.assertTrue(result['stopped']);self.assertFalse(result['forced']);self.assertFalse((root/'train').exists());self.assertFalse(list(scratch.iterdir()));self.assertTrue(any(e['status']['ended'] for e in events))
  finally:server.shutdown();server.server_close()
if __name__=='__main__':unittest.main()
