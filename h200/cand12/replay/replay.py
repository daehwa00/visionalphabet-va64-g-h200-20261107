"""Reconstruct captured tensor views and replay production and diagnostic kernels."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys

import torch
from torch._inductor.runtime.hints import DeviceProperties

OUTS=(19,20,21,22,24,25,26,27,28)
ATOMIC=OUTS[:-1]
RETAINED_ORDER=(*range(10),15,16,17,13,14,19,20,21,22,24,25,26,27,28)
MODES=("legacy","skip_path_params","skip_pole_params","skip_state_output","reconstruct_only","mlp_only",
       "special_assoc","special_reconstruct","special_adjoint","special_double")

def sha(x):
    return hashlib.sha256(x.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()

def restore(meta,bundle):
    # Allocate one backing store per original storage. Copy only defined inputs.
    slots={}
    for v in meta['views']:
        g=v['group'];n=v['offset']+1+sum((s-1)*t for s,t in zip(v['shape'],v['stride'])) if all(v['shape']) else 0
        slots[g]=max(slots.get(g,0),n)
    buffers={g:torch.empty(n,dtype=getattr(torch,next(v['dtype'] for v in meta['views'] if v['group']==g)),device='cuda') for g,n in slots.items()}
    args=[]
    for i,v in enumerate(meta['views']):
        t=buffers[v['group']].as_strided(v['shape'],v['stride'],v['offset'])
        assert t.data_ptr()%512==v['ptr_mod_512']
        if v.get('value')==i:
            val=bundle['values'][i]
            assert sha(val)==meta["sha256"][str(i)]
            t.copy_(val)
            assert sha(t.cpu())==meta["sha256"][str(i)]
        args.append(t)
    return args

def edit(text,nodes):
    ls=text.splitlines(True)
    for first,last,new in sorted(nodes,reverse=True):
        ls[first-1:last]=[new]
    return ''.join(ls)

def generate(base,mode):
    tree=ast.parse(base)
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    edits=[]
    if mode.startswith("special_"):
        sites=[n for n in ast.walk(kernel) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="_prefix_pair"]
        sites.sort(key=lambda n:n.lineno)
        assert len(sites)==4
        for index,n in enumerate(sites):
            flag=n.value.args[-1]
            assert ast.unparse(flag)=="DOUBLING"
            on=mode=="special_double" or (mode=="special_reconstruct" and index<2) or (mode=="special_adjoint" and index>=2)
            edits.append((flag.lineno,flag.end_lineno," "*flag.col_offset+str(on)+",\n"))
    if mode.startswith("skip_"):
        counts=0
        for n in ast.walk(kernel):
            if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)):continue
            callee=ast.unparse(n.value.func);ptr=ast.unparse(n.value.args[0])
            remove=(mode=="skip_path_params" and callee=="tl.atomic_add" and ptr.startswith("grad_path_") or
                    mode=="skip_pole_params" and callee=="tl.atomic_add" and ptr.startswith(("grad_gamma_","grad_decay_")) or
                    mode=="skip_state_output" and callee=="tl.store")
            if remove:
                edits.append((n.lineno,n.end_lineno," "*n.col_offset+"pass\n"));counts+=1
        assert counts>=4
    if mode in ("reconstruct_only","mlp_only"):
        # Controlled probes, not training candidates. Return early after exporting
        # four live channels. Both probes write the same four BF16 output planes.
        marker="    decay_norm = ar * ar + ai * ai" if mode=="reconstruct_only" else "        # Unpack all 8 channels"
        line=next(i for i,l in enumerate(base.splitlines(),1) if l.startswith(marker))
        indent="    " if mode=="reconstruct_only" else "        "
        code="span = tl.cast(batch_size, tl.int64) * height * width * modes\n"
        if mode=="reconstruct_only":
            # Consume both polarities of both complex paths so neither scan dies.
            vals=["par+nar","pai+nai","pbr+nbr","pbi+nbi"]
        else:
            vals=[f"_unpack_path_gradient(grad_packed_path, path_coordinate, {i}, BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES, COARSE_PATHS, True) + _unpack_path_gradient(grad_packed_path, path_coordinate, {i+2}, BLOCK_HEIGHT, BLOCK_LINES, BLOCK_MODES, COARSE_PATHS, True)" for i in (0,1,4,5)]
        code+="".join(f"tl.store(grad_source_real_a + {i} * span + offset, {v}, mask=active)\n" for i,v in enumerate(vals))
        code+="return\n"
        edits.append((line,kernel.end_lineno,"".join(indent+l for l in code.splitlines(True))))
    text=edit(base,edits)
    props=DeviceProperties.create(torch.device("cuda",0))
    text,count=re.subn(r"'device': DeviceProperties\([^\n)]*\)", "'device': "+repr(props),text,count=1)
    assert count==1
    return kernel.name,text

def error(a,b):
    af=a.flatten();bf=b.flatten();sq=torch.zeros(3,device='cuda',dtype=torch.float64);dmax=0.;refmax=0.
    for i in range(0,af.numel(),4*1024*1024):
        x=af[i:i+4194304].float();y=bf[i:i+4194304].float();diff=x-y
        dmax=max(dmax,float(diff.abs().max()));refmax=max(refmax,float(y.abs().max()))
        sq+=torch.stack((diff.square().sum(),y.square().sum(),(~torch.isfinite(diff)).float().sum())).double()
    dl,rl,nonfinite=sq.tolist()
    rel=(dl/max(rl,1e-24))**.5
    return dict(max_abs=dmax,relative_l2=rel,finite=nonfinite==0,passed=nonfinite==0 and rel<=.02 and dmax<=.03*refmax+1e-6)

def compiled_stats(fn,outdir,tag):
    ln=fn.launchers[0];out=dict(regs=ln.n_regs,spill_slots=ln.n_spills,shared=ln.shared)
    cache=Path(os.environ["TRITON_CACHE_DIR"])/ln.cache_hash
    bins=list(cache.glob("*.cubin"))
    if bins:
        cubin=bins[0];out["binary_sha256"]=hashlib.sha256(cubin.read_bytes()).hexdigest()
        cu=shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
        if Path(cu).exists():
            res=subprocess.run([cu,"--dump-resource-usage",str(cubin)],capture_output=True,text=True)
            asm=subprocess.run([cu,"--dump-sass",str(cubin)],capture_output=True,text=True)
            if not asm.returncode:
                ops=Counter(re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z0-9]+)",asm.stdout))
                out.update(opcodes=dict(ops),static_resource=res.stdout.strip()[-400:])
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("fixture",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--only",choices=MODES);p.add_argument("--profile-one",action="store_true");a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    meta=json.loads((a.fixture/"capture.json").read_text())
    orig=(a.fixture/"compiled_launch.py").read_text()
    assert hashlib.sha256(orig.encode()).hexdigest()==meta["source_sha256"]
    spec=(a.fixture/"compiled_retained.py").read_text()
    bundle=torch.load(a.fixture/"operands.pt",weights_only=True,mmap=True,map_location='cpu')
    args=restore(meta,bundle)
    del bundle['values']
    stream=torch.cuda.current_stream().cuda_stream
    scalars=tuple(meta['scalars'])
    runs={};checks={};resources={};ref={}
    modes=(a.only,) if a.only else MODES
    def reset():
        for i in ATOMIC:args[i].zero_()
    def launch(m):
        fn,indices=runs[m];fn.run(*(args[j] for j in indices),*scalars,stream=stream)
    for m in modes:
        special=m.startswith("special_")
        name,text=generate(spec if special else orig,m)
        path=a.out/(m+".py");path.write_text(text)
        module_spec=importlib.util.spec_from_file_location("replay_"+m,path)
        mod=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=mod;module_spec.loader.exec_module(mod)
        fn=getattr(mod,name);runs[m]=(fn,RETAINED_ORDER if special else range(32))
        try:
            reset();args[28].fill_(float('nan'));launch(m);torch.cuda.synchronize()
            resources[m]=compiled_stats(fn,a.out,m)
            if m=="legacy":
                for i in OUTS:ref[i]=args[i].clone()
                hist={i:error(args[i].flatten()[::(97 if i==28 else 1)],bundle["expected"][i].cuda()) for i in OUTS}
                checks["captured_vs_replayed"]=hist
                assert all(x["finite"] for x in hist.values()),hist
            elif special and ref:
                checks[m]={i:error(args[i],ref[i]) for i in OUTS}
            print("COMPILED",m,resources[m]["regs"],resources[m]["shared"],flush=True)
        except Exception as exc:
            if m=="legacy":raise
            print("DIAGNOSTIC_UNAVAILABLE",m,repr(exc)[-450:],flush=True)
            del runs[m]
    if a.profile_one:
        torch.cuda.cudart().cudaProfilerStart()
        reset();launch(a.only or "legacy");torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop();return
    variants=list(runs)
    for _ in range(4):
        for m in variants:reset();launch(m)
    torch.cuda.synchronize()
    samples={m:[] for m in variants};telemetry=[]
    for i in range(9):
        order=variants.copy();random.Random(149+i).shuffle(order)
        for m in order:
            pairs=[]
            for _ in range(5):
                reset()
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();launch(m);end.record();pairs.append((begin,end))
            torch.cuda.synchronize()
            samples[m].append(statistics.median(x.elapsed_time(y) for x,y in pairs))
        q=subprocess.run(["nvidia-smi","--query-gpu=clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu","--format=csv,noheader,nounits"],text=True,capture_output=True)
        telemetry.append(q.stdout.strip())
    # Cold-cache control: same flush buffer on both devices, outside kernel events.
    zap=torch.empty(64*1024*1024,device="cuda",dtype=torch.float32)
    cold={}
    for m in variants:
        times=[]
        for _ in range(7):
            reset();zap.fill_(1)
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record();launch(m);end.record();end.synchronize();times.append(begin.elapsed_time(end))
        cold[m]=times
    result=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,sha256=meta['sha256'],scalars=scalars,samples=samples,cold=cold,checks=checks,resources=resources,telemetry=telemetry)
    (a.out/"result.json").write_text(json.dumps(result,indent=2))
    # Keep the scheduler's bounded output focused on results.
    for m in variants:
        print("REPLAY_ROW",m,json.dumps(dict(ms=statistics.median(samples[m]),samples=samples[m],cold=statistics.median(cold[m]),resource=resources[m],checks=checks.get(m,{}))))
    print("REPLAY_HASH",hashlib.sha256(json.dumps(meta["sha256"],sort_keys=True).encode()).hexdigest())
    print("REPLAY_CAPTURE_CHECK",json.dumps(checks.get("captured_vs_replayed")))
    print("REPLAY_TELEMETRY",json.dumps(telemetry))

if __name__=="__main__":main()
