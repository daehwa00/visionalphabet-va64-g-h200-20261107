"""GPU correctness gate for the fused kernel, independent of speed."""
import argparse,copy,json,os,signal,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--pause-pid',type=int);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.code/'scripts'),str(a.code/'src'),str(Path(__file__).parent)]
import torch
from va64_conv_scan_control import FourPathConvolution
from va64_conv_scan_triton import convolution
torch.manual_seed(7);torch.backends.cudnn.allow_tf32=False;torch.set_num_threads(4)
if a.pause_pid:
 if b'run_va64_conv_scan_control.py' not in Path(f'/proc/{a.pause_pid}/cmdline').read_bytes():raise RuntimeError('foreign PID')
 os.kill(a.pause_pid,signal.SIGSTOP)
try:
 results=[]
 for dtype in (torch.float32,torch.bfloat16):
  for modes,h,w in [(8,9,11),(80,7,7)]:
   reference=FourPathConvolution(modes).cuda();other=copy.deepcopy(reference)
   r=torch.randn(2,h,w,modes,device='cuda',dtype=dtype,requires_grad=True);i=torch.randn_like(r,requires_grad=True)
   r2=r.detach().clone().requires_grad_();i2=i.detach().clone().requires_grad_()
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
    expected=reference(r,i);actual=convolution(r2,i2,other.kernel_real.weight,other.kernel_imag.weight)
   gr=torch.randn_like(expected[0]).float();gi=torch.randn_like(expected[1]).float()
   (expected[0].float()*gr+expected[1].float()*gi).sum().backward();(actual[0].float()*gr+actual[1].float()*gi).sum().backward()
   pairs=list(zip(expected,actual))+[(r.grad,r2.grad),(i.grad,i2.grad),(reference.kernel_real.weight.grad,other.kernel_real.weight.grad),(reference.kernel_imag.weight.grad,other.kernel_imag.weight.grad)]
   errors=[]
   for x,y in pairs:
    assert x.shape==y.shape and torch.isfinite(y).all()
    error=float((x.float()-y.float()).norm()/x.float().norm().clamp_min(1e-10));errors.append(error)
    if dtype==torch.float32:torch.testing.assert_close(x,y,atol=2e-4,rtol=3e-3)
    elif error>.025:raise RuntimeError('BF16 relative discrepancy exceeds2.5percent')
   results.append({'dtype':str(dtype),'shape':list(r.shape),'relative_l2_errors_outputs_inputs_weights':errors})
 a.output.write_text(json.dumps({'status':'passed','results':results},indent=2)+'\n');print(a.output.read_text(),flush=True)
finally:
 if a.pause_pid and Path(f'/proc/{a.pause_pid}').exists():os.kill(a.pause_pid,signal.SIGCONT)
