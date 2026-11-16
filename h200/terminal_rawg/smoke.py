"""Actual 224px BF16 model, effective batch1024, optimizer and checkpoint smoke."""
import argparse,copy,json,time
from pathlib import Path
import torch
import run_terminal_rawg_imagenet1k as experiment

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,choices=(128,256),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
 model=experiment.build_model().cuda().to(memory_format=torch.channels_last);model.train()
 assert model._training_optimization_inventory=={'readers':12}
 opt=experiment.runtime._build_optimizer(model,0.003,torch.device('cuda'))
 model.compile(mode='default',dynamic=False)
 x=torch.randn(a.batch_size,3,224,224,device='cuda').to(memory_format=torch.channels_last);y=torch.arange(a.batch_size,device='cuda')%1000
 accumulation=1024//a.batch_size;before=model.backbone.same_resolution_blocks['56'].reader.point_weight.detach().clone();start=time.perf_counter()
 torch.cuda.reset_peak_memory_stats()
 for step in range(2):
  opt.zero_grad(set_to_none=True)
  for micro in range(accumulation):
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x);loss=torch.nn.functional.cross_entropy(out,y,label_smoothing=0.1)/accumulation
   assert out.shape==(a.batch_size,1000) and torch.isfinite(loss)
   loss.backward()
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  opt.step();print('RAWG_SMOKE_STEP='+str(step+1),flush=True)
 assert not torch.equal(before,model.backbone.same_resolution_blocks['56'].reader.point_weight)
 assert model.classifier.affine.linear.weight[:,512:].abs().sum()>0
 model.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):assert torch.isfinite(model(x)).all()
 a.output.mkdir(parents=True,exist_ok=True)
 saved={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'optimizer':copy.deepcopy(opt.state_dict()),'rng':torch.cuda.get_rng_state_all()}
 torch.save(saved,a.output/'smoke.pt');loaded=torch.load(a.output/'smoke.pt',map_location='cpu',weights_only=True)
 fresh=experiment.build_model(prepared=False);fresh.load_state_dict(loaded['model'],strict=True)
 for key,value in fresh.state_dict().items():assert torch.equal(value,loaded['model'][key])
 fresh_opt=experiment.runtime._build_optimizer(fresh,0.003,torch.device('cpu'));fresh_opt.load_state_dict(loaded['optimizer'])
 assert max(float(v['step']) for v in fresh_opt.state.values())==2
 result={'status':'passed','physical_batch':a.batch_size,'effective_batch':1024,'accumulation':accumulation,'optimizer_steps':2,'parameters':experiment.PARAMETERS,'finite_gradients':True,'state_dict_roundtrip_exact':True,'optimizer_restored':True,'reader_optimizations':model._training_optimization_inventory,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'elapsed_seconds':time.perf_counter()-start}
 (a.output/'smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
 try:main()
 except torch.cuda.OutOfMemoryError:
  print('RAWG_SMOKE_CUDA_OOM',flush=True);raise SystemExit(42)
