import json
from pathlib import Path
import statistics
import sys
import torch


def compare(actual, expected):
    a = torch.load(actual, map_location="cpu", weights_only=True)
    b = torch.load(expected, map_location="cpu", weights_only=True)
    out = {}
    for group in ("logits","grads","updates"):
        aa,bb=({group:a[group]}, {group:b[group]}) if group=="logits" else (a[group], b[group])
        assert aa.keys()==bb.keys()
        max_abs=max_rel=0
        failed=[]
        for k,x in aa.items():
            y=bb[k].float()
            assert x.shape==y.shape and x.isfinite().all() and y.isfinite().all()
            delta=x.float()-y
            ab=delta.abs().max().item()
            rel=(delta.norm()/y.norm().clamp_min(1e-12)).item()
            limit=.03*y.abs().max().item()+1e-6 if group=="grads" else 3e-5 if group=="updates" else .05
            if ab>limit or rel>(.005 if group=="logits" else .02):
                failed.append(k)
            max_abs,max_rel=max(max_abs,ab),max(max_rel,rel)
        out[group]=dict(count=len(aa),max_abs=max_abs,max_rel=max_rel,failed_count=len(failed),first_failed=failed[:3])
    return out


def summarize(path):
    r=json.loads(path.read_text())
    assert r["checkpoint_sha256"]=="af07043f8407221dfe0d75c0d689d76e573119ad8708ee8df5b29cc564ed1fc6" and r["epoch"]==18 and r["global_step"]==18270
    assert r["input_sha256"]=="710485fc2e507008692881465a9bab9fd13d6531b10eba68239959cfc67ee1fa"
    assert r["status"]=="passed" and not r["missing_gradients"] and r["validation_finite"] and r["final_finite"]
    assert (r["batch"], r["warmup"], r["steps_per_group"], r["repeats"])==(128,5,10,5) and r["reset_each_step"]
    sample={k:r[k] for k in ("median_ms","event_ms","wall_ms","images_s","peak_allocated_mib","cpu_affinity","first_loss","final_loss")}
    tracefile=path.with_suffix(".trace.json")
    if tracefile.exists():
        es=json.loads(tracefile.read_text())["traceEvents"]
        cpu={e["args"]["External id"]:e["args"] for e in es if e.get("cat")=="cpu_op" and "External id" in e.get("args",{})}
        gpu=[e for e in es if e.get("cat")=="kernel"]
        hot=[]
        for e in gpu:
            if "_product_scan_coarse4_associative_backward_kernel" not in e["name"]:continue
            args=e["args"]; dispatch=cpu[args["External id"]]
            if dispatch["Input Dims"][4]==dispatch["Input Dims"][13]==[128,56,56,160]:
                hot.append(dict(ms=e["dur"]/1000,grid=args["grid"],block=args["block"],regs=args["registers per thread"],shared=args["shared memory"],tile=dispatch["kernel_kwargs"],warps=dispatch["num_warps"]))
        assert len(hot)==1
        sample.update(hot=hot[0],sum_kernel_ms=sum(e["dur"] for e in gpu)/1000)
    return sample


if __name__=="__main__":
    if sys.argv[1]=="compare":
        d=compare(*sys.argv[2:4])
        print("NUMERIC",sys.argv[4],json.dumps(d),flush=True)
        if "--gate" in sys.argv and any(v["failed_count"] for v in d.values()):sys.exit(43)
    else:
        for name in sys.argv[1:]:
            p=Path(name)
            print("SAMPLE",p.stem,json.dumps(summarize(p)),flush=True)
