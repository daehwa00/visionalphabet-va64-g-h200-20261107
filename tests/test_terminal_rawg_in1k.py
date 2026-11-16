import argparse,json
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
import run_terminal_rawg_imagenet1k as experiment


def test_in1k_head_and_backbone():
 torch.set_num_threads(2);torch.manual_seed(0)
 reference=experiment.rawg._build(experiment.rawg.VARIANT,experiment.rawg.baseline.ComplexScanConfig(output_dim=100,stem_strides=(2,2))).eval()
 torch.manual_seed(0);model=experiment.build_model(prepared=False).eval()
 assert sum(p.numel() for p in model.parameters())==3910504
 assert model._training_optimization_inventory=={'readers':12}
 for key,value in reference.state_dict().items():
  if '.classifier.' not in key:torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
 with torch.no_grad():
  x=torch.randn(2,3,32,32)
  torch.testing.assert_close(model.forward_features(x),reference.forward_features(x),rtol=0,atol=0)
  assert model(x).shape==(2,1000)
 assert torch.count_nonzero(model.classifier.affine.linear.weight[:,512:])==0


@pytest.mark.parametrize('batch,accumulation',[(128,8),(256,4)])
def test_protocol(tmp_path,batch,accumulation):
 manifest=tmp_path/'data.json';manifest.write_text(json.dumps({'root':str(tmp_path.resolve()),'counts':{'train':1281167,'val':50000}}))
 args=argparse.Namespace(data_root=tmp_path,data_manifest=manifest,output_root=tmp_path/'output',batch_size=batch,workers=8,resume=False)
 task=experiment.make_task(args)
 assert task.seed==0 and task.epochs==100 and task.learning_rate==0.003
 assert task.effective_batch_size==1024 and task.gradient_accumulation_steps==accumulation
 assert task.num_classes==1000 and task.execution.gpu_mixup


def test_accumulation_1024_gradient_and_tail(tmp_path):
 torch.set_num_threads(2);torch.manual_seed(1)
 x=torch.randn(1280,4);y=torch.arange(1280)%1000
 model=torch.nn.Linear(4,1000);expected=torch.autograd.grad(torch.nn.functional.cross_entropy(model(x[:1024]),y[:1024]),tuple(model.parameters()))
 opt=torch.optim.SGD(model.parameters(),lr=.003);observed=[]
 opt.register_step_pre_hook(lambda *_:observed.append([p.grad.clone() for p in model.parameters()]))
 task=experiment.runtime.BaselineTask(phase='full',model_key='test',seed=0,learning_rate=.003,epochs=100,data_root=tmp_path,output_dir=tmp_path,result_path=tmp_path/'r',checkpoint_path=tmp_path/'c',batch_size=256,effective_batch_size=1024,precision='float32')
 mixup=lambda a,b:(a,torch.nn.functional.one_hot(b,1000).float())
 report,steps=experiment.runtime._train_epoch(model,DataLoader(TensorDataset(x,y),batch_size=256),opt,mixup,task,torch.device('cpu'),0)
 assert steps==2 and len(observed)==2
 for a,b in zip(observed[0],expected):torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)
 assert report['loss']>0


def test_owner_accepts_bound_rawg_checkpoint(tmp_path):
 import io,tarfile
 from va64_h200_observer import validate_checkpoint_bundle
 torch.set_num_threads(2);model=experiment.build_model(prepared=False)
 task=experiment.runtime.BaselineTask(phase='full',model_key=experiment.MODEL_KEY,seed=0,learning_rate=.003,epochs=100,data_root=tmp_path,output_dir=tmp_path,result_path=tmp_path/'r',checkpoint_path=tmp_path/'c',effective_batch_size=1024)
 contract=experiment.runtime._contract(task,torch.device('cpu'))
 payload={'model':model.state_dict(),'optimizer':{},'rng':{},'schema':experiment.runtime.SCHEMA,'contract_sha256':experiment.runtime._sha256_payload(contract),'task_sha256':contract['task_sha256'],'source_digest_sha256':contract['source_digest_sha256']}
 checkpoint=io.BytesIO();torch.save(payload,checkpoint)
 archive=tmp_path/'test.tar.gz'
 with tarfile.open(archive,'w:gz') as stream:
  for name,data in [('checkpoint.pt',checkpoint.getvalue()),('contracts/run.json',json.dumps(contract).encode())]:
   info=tarfile.TarInfo(name);info.size=len(data);stream.addfile(info,io.BytesIO(data))
 validate_checkpoint_bundle(archive)
