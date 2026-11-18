import argparse, json, time, statistics, hashlib, platform, copy
from pathlib import Path
import torch
import run_va128_real_memory_imagenet100 as runner

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
p.add_argument('--batch',type=int,default=1)
p.add_argument('--mode',default='reduce-overhead')
p.add_argument('--profile',action='store_true')
p.add_argument("--prepare",choices=["none","bake","reader"],default="none")
p.add_argument("--no-spatial-autotune",action="store_true")
p.add_argument("--images")
p.add_argument("--fuse-spatial-rms",action="store_true")
p.add_argument("--fuse-projection",action="store_true")
a=p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(809)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=True
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
assert c['variant']==runner.VARIANT and c['epoch']==100 and c['seed']==501
m=runner._build(runner.VARIANT,runner.runtime.model_config()).eval()
m.load_state_dict(c['model'],strict=True)
assert m.classifier.affine.linear.out_features==100
m=m.cuda().to(memory_format=torch.channels_last)
x=torch.randn(a.batch,3,224,224,device='cuda').contiguous(memory_format=torch.channels_last)
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 ref=m(x)[0].clone()
 prepare_start=time.perf_counter()
 if a.prepare!="none":
  m=copy.deepcopy(m)
  with torch.autocast("cuda",enabled=False):
   m.backbone.prepare_for_inference_()
 if a.prepare=="reader":
  from lnet.real_memory_inference import prepare_real_memory_inference
  m=prepare_real_memory_inference(m,autotune_spatial=not a.no_spatial_autotune,fuse_spatial_rms=a.fuse_spatial_rms,fuse_projection=a.fuse_projection)
 torch.cuda.synchronize()
 prepare_seconds=time.perf_counter()-prepare_start
 fn=torch.compile(m,mode=a.mode,dynamic=False) if a.mode!='eager' else m
 start=time.perf_counter()
 for _ in range(10):
  torch.compiler.cudagraph_mark_step_begin(); y=fn(x)[0]
 torch.cuda.synchronize()
 prep=time.perf_counter()-start
 delta=y.float()-ref.float()
 check={'shape':list(y.shape),'finite':bool(y.isfinite().all()),'max_abs':delta.abs().max().item(),'relative_l2':(delta.norm()/ref.float().norm()).item()}
 assert check['finite'] and check['shape']==[a.batch,100] and check['max_abs']<=.05 and check['relative_l2']<=.005,check
 setup_peak=torch.cuda.max_memory_allocated()/2**20
 torch.cuda.reset_peak_memory_stats()
 wall=[]; gpu=[]; submit=[]; wait_tail=[]
 for _ in range(7):
  s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
  torch.cuda.synchronize(); t=time.perf_counter(); s.record()
  for j in range(100):
   torch.compiler.cudagraph_mark_step_begin(); y=fn(x)[0]
  e.record(); submitted=time.perf_counter(); torch.cuda.synchronize()
  submit.append((submitted-t)*10); wait_tail.append((time.perf_counter()-submitted)*10)
  wall.append((time.perf_counter()-t)*10); gpu.append(s.elapsed_time(e)/100)
 torch.save(y.clone(),out.with_suffix(".output.pt"))
 result={"prepare":a.prepare,"fuse_spatial_rms":a.fuse_spatial_rms,"fuse_projection":a.fuse_projection,"spatial_autotune":not a.no_spatial_autotune,"copy_preparation_seconds":prepare_seconds,'batch':a.batch,'mode':a.mode,'wall_ms':wall,'event_ms':gpu,'median_ms':statistics.median(wall),'images_s':a.batch*1000/statistics.median(wall),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'preparation_seconds':prep,'check':check,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'epoch':c['epoch'],'parameters':sum(v.numel() for v in m.parameters()),'tf32':False,'precision':'bf16 autocast','memory_format':'channels_last','steady_peak_includes':'PyTorch active allocations after warmup; CUDA Graph private pool reservations reported separately','timing_excludes':'preparation,compile,capture,profiler'}
 result.update(setup_peak_allocated_mib=setup_peak,reserved_mib=torch.cuda.memory_reserved()/2**20,host_submit_ms=submit,sync_tail_ms=wait_tail)
 out.write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
 if a.images:
  images=torch.load(a.images,weights_only=True)
  outputs=[]
  for chunk in images.split(a.batch):
   torch.compiler.cudagraph_mark_step_begin()
   outputs.append(fn(chunk.cuda().contiguous(memory_format=torch.channels_last))[0].clone().cpu())
  torch.save(torch.cat(outputs),out.with_suffix(".images.pt"))
 if a.profile:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
   for _ in range(5):
    torch.compiler.cudagraph_mark_step_begin(); fn(x)
   torch.cuda.synchronize()
  prof.export_chrome_trace(str(out.with_suffix('.trace.json')))
  out.with_suffix('.profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=60))
