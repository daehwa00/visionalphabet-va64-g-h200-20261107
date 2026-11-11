import json,os,subprocess,sys
from pathlib import Path
import pytest,torch
from checkpointing import *

def setup():
 torch.manual_seed(12);m=torch.nn.Linear(4,3);o=torch.optim.AdamW(m.parameters(),lr=.001);return m,o

def step(m,o):
 o.zero_grad(set_to_none=True);x=torch.randn(2,4);loss=m(x).square().mean();loss.backward();o.step();o.zero_grad(set_to_none=True);return loss.detach()

def test_resume_exact_next_step(tmp_path):
 m,o=setup();step(m,o);p=tmp_path/'latest.pt';save_training(p,m,o,None,contract={'branch':'value'},model_config={},progress={'global_step':1,'tokens_seen':8})
 expected=step(m,o);state={k:v.clone() for k,v in m.state_dict().items()}
 m2,o2=setup();r=restore_training(p,m2,o2,None,contract={'branch':'value'});actual=step(m2,o2)
 assert r['global_step']==1;torch.testing.assert_close(expected,actual,rtol=0,atol=0)
 for k,v in state.items():torch.testing.assert_close(v,m2.state_dict()[k],rtol=0,atol=0)

def test_new_process_restore(tmp_path):
 m,o=setup();step(m,o);p=tmp_path/'latest.pt';save_training(p,m,o,None,contract={'branch':'value'},model_config={},progress={'global_step':1});expected=step(m,o)
 script="from test_checkpointing import *;m,o=setup();restore_training(sys.argv[1],m,o,None,contract={'branch':'value'});print(float(step(m,o)))"
 result=subprocess.run([sys.executable,'-c',script,str(p)],cwd=Path(__file__).parent,capture_output=True,text=True,check=True,env=dict(os.environ))
 assert float(result.stdout.strip())==float(expected)

def test_wrong_variant_rejected(tmp_path):
 m,o=setup();p=tmp_path/'latest.pt';save_training(p,m,o,None,contract={'branch':'value'},model_config={},progress={'global_step':0})
 with pytest.raises(ValueError,match='contract'):restore_training(p,m,o,None,contract={'branch':'residual'})

def test_corruption_rejected(tmp_path):
 m,_=setup();p=tmp_path/'final.pt';save_weights(p,m,contract={},model_config={},progress={'global_step':0});p.write_bytes(p.read_bytes()+b'x')
 with pytest.raises(ValueError,match='size'):verified_load(p)

def test_boundary_required(tmp_path):
 m,o=setup();m(torch.ones(1,4)).sum().backward()
 with pytest.raises(ValueError,match='boundary'):save_training(tmp_path/'bad.pt',m,o,None,contract={},model_config={},progress={'global_step':0})

def test_weights_cannot_resume(tmp_path):
 m,o=setup();p=tmp_path/'final.pt';save_weights(p,m,contract={},model_config={},progress={'global_step':0})
 with pytest.raises(ValueError,match='Weights-only'):restore_training(p,m,o,None,contract={})

def test_embedded_historical_best_survives_later_overwrite(tmp_path):
 m,o=setup();step(m,o);best=tmp_path/'best.pt';contract={'branch':'value'}
 save_weights(best,m,contract=contract,model_config={},progress={'global_step':1,'val_loss':1.})
 original=verified_load(best)['model'];step(m,o)
 latest=tmp_path/'latest.pt';save_training(latest,m,o,None,contract=contract,model_config={},progress={'global_step':2,'best_val':1.},best_checkpoint=best)
 step(m,o);save_weights(best,m,contract=contract,model_config={},progress={'global_step':3,'val_loss':.5})
 m2,o2=setup();target=tmp_path/'resumed/best.pt'
 restore_training(latest,m2,o2,None,contract=contract,best_output=target)
 recovered=verified_load(target)
 assert recovered['progress']['global_step']==1
 for k,v in original.items():assert torch.equal(v,recovered['model'][k])

def test_failed_save_retains_previous_generation(tmp_path,monkeypatch):
 m,o=setup();p=tmp_path/'latest.pt';save_training(p,m,o,None,contract={},model_config={},progress={'global_step':0})
 def fail(payload,handle):handle.write(b'partial');raise OSError('simulated disk error')
 monkeypatch.setattr(torch,'save',fail)
 with pytest.raises(OSError):save_training(p,m,o,None,contract={},model_config={},progress={'global_step':1})
 assert verified_load(p)['progress']['global_step']==0
 assert verified_load(tmp_path/'previous.pt')['progress']['global_step']==0
