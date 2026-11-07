from __future__ import annotations
import copy
import pytest
import torch
from lnet.pac_feature_pool_readout import signed_feature_pool, feature_pool_product_scan
from lnet.pac_product_scan_reference import product_scan_full4_reference
from lnet.pac_real2d_math import discrete_pole_real2d
from scripts import run_va64_retrain_imagenet1k as runner

@pytest.fixture(autouse=True)
def threads():
 old=torch.get_num_threads(); torch.set_num_threads(2)
 yield
 torch.set_num_threads(old)

def inputs(device='cpu',dtype=torch.float32):
 torch.manual_seed(19)
 poles=[]
 for phase in (.4,-.7):
  damping=torch.full((1,1,1,16),.2,device=device)
  poles.append(tuple(t.detach().requires_grad_() for t in discrete_pole_real2d(damping,torch.full_like(damping,phase),1.0)))
 source=tuple(torch.randn(2,5,7,16,device=device,dtype=dtype,requires_grad=True) for _ in range(2))
 return (*poles,*source)

def test_signed_gap_order_and_cancellation():
 real=torch.arange(8.).reshape(1,2,1,4,1).requires_grad_(); imag=-real
 out=signed_feature_pool(real,imag)
 torch.testing.assert_close(out,torch.tensor([[2.,-2.,3.,-3.,4.,-4.,5.,-5.]]))
 out.sum().backward(); assert torch.isfinite(real.grad).all()
 z=torch.tensor([1.,-1.]).reshape(1,2,1,1,1).expand(1,2,1,4,1)
 assert signed_feature_pool(z,z).eq(0).all()
 assert signed_feature_pool(torch.zeros_like(z),torch.zeros_like(z)).eq(0).all()

@pytest.mark.parametrize('normalization',['pointwise','global'])
@pytest.mark.parametrize('memory',['retain','recompute'])
def test_g_scan_reference_and_gradients(normalization,memory):
 px,py,real,imag=inputs(); flat=(*px,*py,real,imag)
 cloned=tuple(t.detach().clone().requires_grad_() for t in flat)
 expected=signed_feature_pool(*product_scan_full4_reference(cloned[:4],cloned[4:8],cloned[8:],gain_normalization=normalization))
 actual=feature_pool_product_scan(px,py,(real,imag),gain_normalization=normalization,memory_policy=memory)
 torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
 weights=torch.randn_like(actual)
 a=torch.autograd.grad((actual*weights).sum(),flat)
 b=torch.autograd.grad((expected*weights).sum(),cloned)
 for left,right in zip(a,b,strict=True): torch.testing.assert_close(left,right,rtol=2e-4,atol=1e-5)

def test_g_model_backbone_and_checkpoint(tmp_path):
 torch.manual_seed(0); baseline=runner.build_model('B',prepared=False)
 torch.manual_seed(0); model=runner.build_model('G',prepared=False)
 assert sum(p.numel() for p in model.parameters())==1882760
 assert model.model.classifier.q4_dim==640
 for name,value in baseline.state_dict().items():
  if '.classifier.' not in name: assert torch.equal(value,model.state_dict()[name]),name
 optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
 image=torch.randn(2,3,32,32)
 loss=model(image).float().square().mean(); loss.backward(); optimizer.step()
 assert torch.isfinite(loss)
 path=tmp_path/'g.pt'; torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},path)
 restored=runner.build_model('G',prepared=False); state=torch.load(path,weights_only=True)
 restored.load_state_dict(state['model']); opt=torch.optim.AdamW(restored.parameters()); opt.load_state_dict(state['optimizer'])
 model.eval(); restored.eval()
 torch.testing.assert_close(model(image),restored(image),rtol=0,atol=0)
 with pytest.raises(RuntimeError): baseline.load_state_dict(state['model'])

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_g_cuda_parity(dtype):
 px,py,real,imag=inputs('cuda',dtype)
 flat=(*px,*py,real,imag)
 cpu=tuple(t.detach().cpu().float().requires_grad_() for t in flat)
 expected=signed_feature_pool(*product_scan_full4_reference(cpu[:4],cpu[4:8],cpu[8:]))
 out=feature_pool_product_scan(px,py,(real,imag))
 torch.testing.assert_close(out.cpu(),expected,rtol=.03 if dtype==torch.bfloat16 else 2e-4,atol=.01 if dtype==torch.bfloat16 else 2e-5)
 grads=torch.autograd.grad(out.square().sum(),flat)
 refs=torch.autograd.grad(expected.square().sum(),cpu)
 for g,r in zip(grads,refs,strict=True):
  assert torch.isfinite(g).all()
  torch.testing.assert_close(g.float().cpu(),r,rtol=.06 if dtype==torch.bfloat16 else 1e-3,atol=.08 if dtype==torch.bfloat16 else 3e-4)

def test_g_canonical_backbone_preserves_new_head(tmp_path,monkeypatch):
 import hashlib
 from scripts import run_va64_g_h200 as entry
 torch.manual_seed(0); original=runner.build_model('B',prepared=False)
 torch.manual_seed(17); model=runner.build_model('G',prepared=False)
 head={k:v.clone() for k,v in model.state_dict().items() if '.classifier.' in k}
 path=tmp_path/'initial.pt'; torch.save({'trained':False,'variant':'B','seed':0,'model':original.state_dict()},path)
 monkeypatch.setattr(entry,'INITIAL_SHA',hashlib.sha256(path.read_bytes()).hexdigest())
 entry.initialize_common_backbone(model,path)
 for name,value in model.state_dict().items():
  assert torch.equal(value,head[name] if name in head else original.state_dict()[name]),name


def test_g_contract_rejects_other_mode_and_source(tmp_path):
 import argparse,dataclasses,json
 manifest=tmp_path/'data.json'; manifest.write_text(json.dumps({'root':str(tmp_path),'counts':{'train':1281167,'val':50000}}))
 args=argparse.Namespace(variant='G',seed=0,data_root=tmp_path,data_manifest=manifest,output_root=tmp_path/'out',batch_size=128,workers=0,resume=False,gpu_mixup=True)
 task=runner.make_task(args); contract=runner.runtime._contract(task,torch.device('cpu'))
 payload={'schema':runner.runtime.SCHEMA,'contract_sha256':runner.runtime._sha256_payload(contract),'task_sha256':contract['task_sha256'],'source_digest_sha256':contract['source_digest_sha256']}
 runner.runtime._validate_binding(payload,contract)
 args.variant='B'; other=runner.runtime._contract(runner.make_task(args),torch.device('cpu'))
 with pytest.raises(RuntimeError,match='incompatible'): runner.runtime._validate_binding(payload,other)
 changed=runner.runtime._contract(dataclasses.replace(task,source_sha256={'changed':'f'*64}),torch.device('cpu'))
 with pytest.raises(RuntimeError,match='incompatible'): runner.runtime._validate_binding(payload,changed)
