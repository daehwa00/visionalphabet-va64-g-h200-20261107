"""Report full forward arithmetic of the fixed, mode-preserving Cconv; no tuning."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline-source-root',type=Path,default=ROOT/'artifacts/h200-va64-g-20261107/repository/src');a=p.parse_args()
 sys.path.insert(0,str(a.baseline_source_root.resolve()))
 import lnet
 if Path(lnet.__file__).resolve().parent!=a.baseline_source_root.resolve()/'lnet':raise RuntimeError('wrong baseline source')
 import va64_conv_scan_control as c
 from profile_va64_conv_control import ConvArithmeticRules,fold_parametrizations
 from profile_rmt_common_flops import DEFAULT_AUDITOR,load_module,profile
 torch.set_num_threads(4);torch.manual_seed(0)
 auditor=load_module(DEFAULT_AUDITOR,'conv_scan_frozen_arithmetic')
 model=c.build_model(prepared=False).cpu().float().eval();sample=torch.randn(1,3,224,224)
 shapes={}
 def record(name):
  return lambda module,args:shapes.update({name:{'name':name,'input_shape':list(args[0].shape),'modes':module.modes}})
 handles=[m.register_forward_pre_hook(record(n)) for n,m in model.named_modules() if isinstance(m,c.FourPathConvolution)]
 folded=fold_parametrizations(model,sample)
 for h in handles:h.remove()
 if len(shapes)!=12:raise RuntimeError('not all12scan replacements executed')
 result=profile(model,ConvArithmeticRules(auditor))
 if not result['valid']:raise RuntimeError('uncovered arithmetic')
 result.update(schema='visionalphabet.va64_conv_scan_cost.v2',kernel_size=3,definition=c.definition(),cost_matched=False,architecture_selection='fixed mode-preserving complex depthwise3x3; no validation-based tuning',scan_sites=list(shapes.values()),folded_parameters=folded,baseline_flops=c.BASELINE_FLOPS,ratio_to_B=result['flops_per_image']/c.BASELINE_FLOPS,source_sha256={'scripts/va64_conv_scan_control.py':hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),'scripts/profile_va64_conv_scan_control.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'auditor':hashlib.sha256(DEFAULT_AUDITOR.read_bytes()).hexdigest()})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('valid','params','flops_per_image','ratio_to_B','unsupported_operators')}))
if __name__=='__main__':main()
