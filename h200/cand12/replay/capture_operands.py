"""Capture exactly one pre-launch state of the real, compiled final14 training step."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner

root=Path(os.environ["CAPTURE_DIR"])
run=CachingAutotuner.run
seen=0
class Done(Exception):
    pass

def digest(x):
    return hashlib.sha256(x.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()

def intercept(self,*args,**kw):
    global seen
    if (self.inductor_meta.get("kernel_name","").startswith("_product_scan_coarse4_associative_backward_kernel")
        and len(args)>=36 and tuple(args[4].shape)==tuple(args[13].shape)==(128,56,56,160)
        and self.triton_meta["constants"].get("LINES_PER_GROUP")==4):
        seen+=1
        if seen==7:
            assert list(args[32:])==[56,7168,160,8,160,1792,1], args[32:]
            torch.cuda.synchronize()
            groups={};views=[];values={};unique={};sha={}
            for i,t in enumerate(args[:32]):
                assert isinstance(t,torch.Tensor)
                st=t.untyped_storage()
                g=groups.setdefault((st._cdata,t.dtype),len(groups))
                meta=dict(group=g,offset=t.storage_offset(),shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype).split('.')[-1],ptr_mod_512=t.data_ptr()%512)
                views.append(meta)
                if i==28:continue  # completely overwritten before being read
                k=(g,t.storage_offset(),tuple(t.shape),tuple(t.stride()))
                if k not in unique:
                    unique[k]=i
                    v=t.detach().to('cpu').contiguous().clone()
                    values[i]=v;sha[str(i)]=digest(v)
                meta['value']=unique[k]
            assert all(torch.count_nonzero(values[views[i]['value']])==0 for i in [19,20,21,22,24,25,26,27])
            path=Path(self.filename)
            shutil.copyfile(path,root/"compiled_launch.py")
            result=run(self,*args,**kw)
            torch.cuda.synchronize()
            expected={i:args[i].detach().flatten()[::(97 if i==28 else 1)].cpu().clone() for i in (19,20,21,22,24,25,26,27,28)}
            meta=dict(views=views,sha256=sha,scalars=args[32:],name=self.inductor_meta["kernel_name"],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),constants=self.triton_meta["constants"],target_invocation=seen)
            torch.save(dict(values=values,expected=expected), root/"operands.pt")
            (root/"capture.json").write_text(json.dumps(meta,indent=2))
            raise Done
    return run(self,*args,**kw)
CachingAutotuner.run=intercept
try:
    runpy.run_path("benchmark_dw_terminal_rmsnorm_training.py",run_name="__main__")
except Done:
    print("CAPTURE_COMPLETE",flush=True)
else:
    raise RuntimeError(f"No capture: saw {seen} target calls")
