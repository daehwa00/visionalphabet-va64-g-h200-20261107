"""Count supported forward FLOPs and exact reader convolution MAC formulas on CPU."""
import argparse,json,time
from pathlib import Path
import torch
from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from benchmark_factorized_reader import set_mode,build_va64,build_va128


def main():
 p=argparse.ArgumentParser();p.add_argument('--family',choices=('VA64','VA128'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(0);m=(build_va128 if a.family=='VA128' else build_va64)('B',prepared=False).eval();shapes=[];handles=[]
 for name,reader in m.named_modules():
  if isinstance(reader,FactorizedComplexConv2dReader):
   def hook(mod,args,name=name):shapes.append({'name':name,'shape':list(args[0].shape),'input_modes':mod.input_modes,'output_modes':mod.output_modes,'rank':mod.rank,'kernel':mod.kernel_size})
   handles.append(reader.register_forward_pre_hook(hook))
 x=torch.randn(1,3,224,224);result={'family':a.family,'parameters':sum(p.numel() for p in m.parameters())}
 for mode in ('dense','factorized'):
  set_mode(m,mode);start=time.perf_counter()
  with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:y=m(x)
  result[mode]={'counted_flops':sum(e.flops for e in prof.key_averages()),'seconds_with_profiler':time.perf_counter()-start,'output_finite':bool(torch.isfinite(y).all()),'operations':[{'name':e.key,'flops':e.flops} for e in prof.key_averages() if e.flops]}
  if mode=='dense':ref=y;original=shapes.copy();[h.remove() for h in handles]
  else:result['fp32_logits_max_abs_difference']=float((ref-y).abs().max())
 dense=sum(4*s['shape'][0]*s['shape'][1]*s['shape'][2]*s['input_modes']*s['output_modes']*s['kernel']**2 for s in original)
 fact=sum(4*s['shape'][0]*s['shape'][1]*s['shape'][2]*s['output_modes']*s['rank']*(s['input_modes']+s['kernel']**2) for s in original)
 result.update(readers=original,reader_dense_macs=dense,reader_factorized_macs=fact,reader_mac_reduction=1-fact/dense,profiler_counted_flop_reduction=1-result['factorized']['counted_flops']/result['dense']['counted_flops'],counting_note='Supported forward operators counted by PyTorch profiler; excludes uncounted scan/norm work, backward, copies and kernel-launch cost.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('readers','dense','factorized')}),flush=True);print('FLOPS',result['dense']['counted_flops'],result['factorized']['counted_flops'])


if __name__=='__main__':main()
