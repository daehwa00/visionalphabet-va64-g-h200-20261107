"""Observe exact H200 scopes, then validate deployment from an empty local tuning store."""
import argparse
import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import sys

from lnet.kernel_launch_config import capture_launch_scopes, device_key, store_geometry, stored_geometry
from lnet.path_backward_layout import BUNDLED_LAYOUTS, LAYOUT_NAME, candidate_geometry
from tune_path_backward import load_scope, launch_contracts


def bundle(root, layout, pairs):
    gpu = device_key()
    d=json.loads(root.read_text());records=d["records"].setdefault(gpu,{}).setdefault(layout,{})
    new={}
    for scope,geom in pairs:
        store_geometry(layout,geom,scope=scope)
        assert stored_geometry(layout,scope=scope)==geom
        new[scope.key]=dict(blocks=geom.blocks,num_stages=geom.num_stages,num_warps=geom.num_warps)
    for k,v in new.items():
        assert k not in records or records[k]==v
        records[k]=v
    root.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n")
    return {gpu:{layout:new}}


def profiled(retained,trace):
    es=json.loads(trace.read_text())["traceEvents"]
    cpu={e["args"]["External id"]:e["args"] for e in es if e.get("cat")=="cpu_op" and "kernel_file" in e.get("args",{})}
    out=[]
    for e in es:
        if e.get("cat")!="kernel":continue
        n=e["name"]
        op={"_path_forward_kernel":0,"_path_backward_kernel":1,"_horizontal_path_backward_kernel":2}.get(n)
        if op is None:continue
        a=cpu[e["args"]["External id"]];tree=ast.parse(Path(a["kernel_file"]).read_text())
        meta=next(x.value for x in ast.walk(tree) if isinstance(x,ast.keyword) and x.arg=='triton_meta')
        constants=next(ast.literal_eval(v) for k,v in zip(meta.keys,meta.values) if getattr(k,'value','')=='constants')
        coarse=int(constants.get("COARSE",False))
        geom=retained[(op,coarse)]
        tile={k:int(v) for k,v in (l.split('=') for l in a["kernel_kwargs"].split(','))}
        assert tile==geom.blocks and a["num_warps"]==geom.num_warps
        out.append(dict(op=op,coarse=coarse,blocks=tile,warps=geom.num_warps,grid=e["args"]["grid"],regs=e["args"]["registers per thread"],shared=e["args"]["shared memory"],ms=e["dur"]/1000))
    assert len(out)==len(retained) and set((v["op"],v["coarse"]) for v in out)==set(retained),out
    return out


def main():
    root=Path(os.environ["RUN_ROOT"]);arm=os.environ["ARM"]
    phase=os.environ.get("TRIAL_PHASE","run")
    profile=LAYOUT_NAME
    if arm!="control":
        from lnet.retained_path_dispatch import BUNDLE, PROFILE_NAME, step_geometry
        profile=PROFILE_NAME
    file=root/f"observed-{arm}.json"
    if phase=="observe":
        with capture_launch_scopes(profile) as scopes:
            runpy.run_path('benchmark_matched.py',run_name='__main__')
        data=[asdict(s) for s in scopes if s.execution_regime=="compiled" and dict(s.shape_items).get("batch")==128]
        assert data
        file.write_text(json.dumps(data,indent=2))
        return
    geoms={}
    if phase=="package":
        if arm=="control":
            observed=[load_scope(s) for s in json.loads(file.read_text())]
            ps=[]
            for h,spec in [(14,(4,8,4,3)),(28,(4,8,4,3)),(56,(4,4,2,3))]:
                found=[s for s in observed if tuple(dict(s.shape_items)[k] for k in ("height","width","modes"))==(h,h,160)]
                assert len(found)==1
                ps.append((found[0],candidate_geometry(*spec)))
            result=bundle(BUNDLED_LAYOUTS,profile,ps)
        else:
            template=json.loads(BUNDLE.read_text())["records"]["NVIDIA_GeForce_RTX_4090-sm89"][PROFILE_NAME]
            observed=[load_scope(s) for s in json.loads(file.read_text())]
            ps=[]
            for key,v in template.items():
                shape=json.loads(key)["shape"]
                if shape['batch']!=128:continue
                found=[s for s in observed if dict(s.shape_items)==shape]
                assert len(found)==1,(shape,observed)
                s=found[0];b=v['blocks']
                geom=step_geometry(shape["operation"],b['GROUP'],b['BM'],v['num_warps'],b.get('DOUBLING',0))
                ps.append((s,geom))
            assert len(ps)==5
            result=bundle(BUNDLE,profile,ps)
        print("H200_RECORD",arm,json.dumps(result,separators=(',',':')),flush=True)
        return
    assert not (Path(os.environ["LNET_LAUNCH_CACHE"])/"launch_geometry.json").exists()
    runpy.run_path(os.environ.get("MATCHED_ENTRY","benchmark_matched.py"),run_name="__main__")
    if "--profile" in sys.argv:
        path=Path(sys.argv[sys.argv.index("--output")+1]).with_suffix(".trace.json")
        if arm=="control":
            for h,bm,w in ((14,8,4),(28,8,4),(56,4,2)):
                ls=launch_contracts(path,(128,h,h,160))
                assert all((v['group'],v['compact'],v['lines'],v['modes'],v['warps'])==(4,3,1,bm,w) for v in ls)
                if h==56: print("TRAIN_LAUNCHES",path.stem,json.dumps(ls))
        else:
            entries=json.loads(BUNDLE.read_text())['records'][device_key()][PROFILE_NAME]
            for key,v in entries.items():
                sh=json.loads(key)['shape'];b=v['blocks']
                if sh['batch']==128:
                    geoms[(sh['operation'],sh['coarse'])]=step_geometry(sh['operation'],b['GROUP'],b['BM'],v['num_warps'],b.get('DOUBLING',0))
            print("TRAIN_LAUNCHES",path.stem,json.dumps(profiled(geoms,path)))


if __name__=="__main__":main()
