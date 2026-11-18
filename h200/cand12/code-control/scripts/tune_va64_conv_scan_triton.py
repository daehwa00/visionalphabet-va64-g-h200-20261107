"""Bounded diagnostic weight-gradient tile comparison; no model changes."""
import argparse,os,signal,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pause-pid',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
import torch
import va64_conv_scan_triton as op
torch.manual_seed(4);torch.set_num_threads(4)
if b'run_va64_conv_scan_control.py' not in Path(f'/proc/{a.pause_pid}/cmdline').read_bytes():raise RuntimeError('foreign PID')
os.kill(a.pause_pid,signal.SIGSTOP)
try:
 r=torch.randn(128,56,56,80,device='cuda',dtype=torch.bfloat16,requires_grad=True);i=torch.randn_like(r,requires_grad=True)
 wr=torch.randn(320,1,3,3,device='cuda',dtype=torch.bfloat16,requires_grad=True);wi=torch.randn_like(wr,requires_grad=True)
 gr=torch.randn(128,56,56,4,80,device='cuda',dtype=torch.bfloat16);gi=torch.randn_like(gr)
 rows=[];reference=None
 for bs,bm in [(128,16),(256,16),(128,32),(256,32)]:
  op.WEIGHT_BS=bs;op.WEIGHT_BM=bm
  def step():
   y=op.convolution(r,i,wr,wi)
   return torch.autograd.grad(y,(r,i,wr,wi),(gr,gi))
  for _ in range(5):out=step()
  if reference is None:reference=[t.clone() for t in out[2:]]
  errors=[float((x.float()-y.float()).norm()/x.float().norm()) for x,y in zip(reference,out[2:])]
  if max(errors)>.01:raise RuntimeError('tile gradient discrepancy')
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
  for _ in range(20):step()
  end.record();end.synchronize();ms=start.elapsed_time(end)/20
  row={'weight_bs':bs,'weight_bm':bm,'forward_backward_ms':ms,'weight_gradient_relative_errors':errors};rows.append(row);print(json.dumps(row),flush=True)
 a.output.write_text(json.dumps({'status':'passed','results':rows},indent=2)+'\n')
finally:os.kill(a.pause_pid,signal.SIGCONT)
