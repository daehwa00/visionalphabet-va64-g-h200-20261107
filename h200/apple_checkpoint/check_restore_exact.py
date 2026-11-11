import sys,torch,json
from train import build_model,optimizer_groups,source_hash
from checkpointing import verified_load,restore_training
from transformers import GPTNeoXConfig
p=sys.argv[1];v=verified_load(p);assert v['contract']['source_sha256']==source_hash()
m=build_model(config=GPTNeoXConfig.from_dict(v['model_config']),branch_source=v['contract']['branch_source']).cuda()
o=torch.optim.AdamW(optimizer_groups(m),betas=(.9,.95),eps=1e-8,fused=True);s=torch.amp.GradScaler('cuda',init_scale=4096,growth_interval=1000)
restore_training(p,m,o,s,contract=v['contract'])
for k,x in m.state_dict().items():assert torch.equal(x.cpu(),v['model'][k]),k
for k,state in o.state_dict()['state'].items():
 for field,x in state.items():
  y=v['optimizer']['state'][k][field]
  if isinstance(x,torch.Tensor):assert torch.equal(x.cpu(),y),(k,field)
  else:assert x==y
assert s.state_dict()==v['scaler']
assert torch.equal(torch.get_rng_state(),v['rng']['torch'])
for x,y in zip(torch.cuda.get_rng_state_all(),v['rng']['cuda']):assert torch.equal(x,y)
print(json.dumps({'status':'passed','restored_model_optimizer_scaler_rng':'bitwise_exact','branch_source':v['contract']['branch_source'],'step':v['progress']['global_step']}))
