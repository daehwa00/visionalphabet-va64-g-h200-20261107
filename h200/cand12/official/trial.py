"""Verify the official outdot and tile-8 coarse opt-ins independently on H200."""
import ast
from dataclasses import asdict,replace
import json,os
from pathlib import Path
import runpy,sys
from lnet.kernel_launch_config import capture_launch_scopes,device_key,stored_geometry
from lnet.retained_path_dispatch import PROFILE_NAME,BUNDLE as RB,step_geometry
from lnet.segmented_path_scan import SEGMENT_NAME,BUNDLE as SB,segment_revision
from tune_path_backward import load_scope

def table(path,name):return json.loads(path.read_text())["records"].get(device_key(),{}).get(name,{})
def put(path,name,key,geom):
    payload=json.loads(path.read_text());t=payload["records"].setdefault(device_key(),{}).setdefault(name,{})
    t[key]=geom;path.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
def parent(coarse):
    candidates=[(load_scope({**{k:v for k,v in j.items() if k!="shape"},"shape_items":sorted(j["shape"].items())}),v) for x,v in table(RB,PROFILE_NAME).items()
                if (j:=json.loads(x))["shape"]["batch"]==128 and j["shape"]["operation"]==1 and j["shape"]["coarse"]==int(coarse)]
    assert len(candidates)==1
    return candidates[0]

def report(trace,arm,old):
    ev=json.loads(trace.read_text())["traceEvents"]
    cpu={e["args"]["External id"]:e["args"] for e in ev if e.get("cat")=="cpu_op" and "kernel_file" in e.get("args",{})}
    kernels=[]
    for e in ev:
        if e.get("cat")!="kernel" or e["name"] not in ("_path_backward_kernel","_bucket_backward_kernel","_path_forward_kernel","_horizontal_path_backward_kernel"):continue
        a=cpu[e['args']['External id']];p=Path(a['kernel_file']);src=p.read_text();tree=ast.parse(src)
        meta=next(k.value for k in ast.walk(tree) if isinstance(k,ast.keyword) and k.arg=='triton_meta')
        constants=next(ast.literal_eval(v) for k,v in zip(meta.keys,meta.values) if getattr(k,'value',None)=='constants')
        coarse=bool(constants.get('COARSE',False))
        op=0 if e['name']=='_path_forward_kernel' else 2 if e['name']=='_horizontal_path_backward_kernel' else 1
        tile={k:int(v) for k,v in (q.split('=') for q in a['kernel_kwargs'].split(','))}
        bm,warps=(8,4) if op==0 else (32,4) if op==2 else (8,4) if coarse and arm in ('coarse','both','donor') else (4,2)
        assert tile['BM']==bm and a['num_warps']==warps and tile['GROUP']==(1 if op==2 else 4)
        row=dict(op=op,coarse=int(coarse),tile=tile,warps=warps,grid=e["args"]["grid"],regs=e["args"]["registers per thread"],shared=e["args"]["shared memory"],ms=e["dur"]/1000)
        if op==1:
            fns={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
            def calls(fn,tag):
                return any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==tag for n in ast.walk(fns[fn]))
            segmented=calls('_prefix_pair','_segment_four')
            public_dot=calls('_path_backward_kernel','_output_weight_bias_vjp') if '_path_backward_kernel' in fns else False
            if coarse:
                assert segmented==(arm in ('coarse','both','donor')) and not public_dot
            else:
                if arm in ('direct','both','donor'):assert segmented and public_dot
                else:
                    assert not public_dot
                    if old in ('control','outdot','tree','native'):assert segmented
                    if old=='outdot':assert 'wo_grad' in src
                    if old=='tree':assert 'tail_r' in src
                    if old=='native':assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.associative_scan' for n in ast.walk(fns['_segment_four']))
                    if old in ('rows2','rows8'):assert calls('_prefix_pair','_row_prefix')
                    if old=='buckets':assert e['name']=='_bucket_backward_kernel'
            row.update(segmented=segmented,official_outdot=public_dot,old_impl=old if not coarse and arm in ('control','coarse') else None)
        kernels.append(row)
    assert len(kernels)==5 and {(v['op'],v['coarse']) for v in kernels}=={(0,0),(1,0),(1,1),(2,0),(2,1)}, kernels
    return kernels

def main():
    arm=os.environ['ARM'];phase=os.environ['PHASE'];root=Path(os.environ['RUN_ROOT']);old=os.environ['BASE_VARIANT']
    if phase=='observe':
        from lnet.output_dot_dispatch import DOT_NAME,dot_revision
        with capture_launch_scopes(DOT_NAME) as seen:runpy.run_path("benchmark_matched.py",run_name="__main__")
        direct,_=parent(0)
        target=replace(direct,kernel_revision=dot_revision(direct.kernel_revision))
        assert target in seen
        (root/'dot-observed.json').write_text(json.dumps(asdict(target)))
        print("OBSERVED_DOT",target.key);return
    if phase=='package':
        proof={};gpu=device_key()
        if arm in ('direct','both','donor'):
            from lnet.output_dot_dispatch import DOT_NAME,BUNDLE,dot_revision
            s=load_scope(json.loads((root/'dot-observed.json').read_text()))
            par,g=parent(0)
            assert s==replace(par,kernel_revision=dot_revision(par.kernel_revision)) and g['blocks']==dict(BM=4,DOUBLING=1,GROUP=4) and g['num_warps']==2
            assert seg_check(par,g)
            put(BUNDLE,DOT_NAME,s.key,g);proof[DOT_NAME]={s.key:g}
        if arm in ('coarse','both','donor'):
            par,old_geo=parent(1)
            assert old_geo['blocks']==dict(BM=4,GROUP=4,DOUBLING=1) and old_geo['num_warps']==2
            n=step_geometry(1,4,8,4,1)
            g=dict(blocks=n.blocks,num_warps=n.num_warps,num_stages=n.num_stages)
            cs=replace(par,kernel_revision=segment_revision(par.kernel_revision))
            put(RB,PROFILE_NAME,par.key,g);put(SB,SEGMENT_NAME,cs.key,g)
            proof[PROFILE_NAME]={par.key:g};proof[SEGMENT_NAME]={cs.key:g}
        if arm=='donor':
            # A separate, non-promotable all-geometry comparison matches the complete
            # 4090 deployment, including the retained choices at 14 and 28.
            from lnet.path_backward_layout import BUNDLED_LAYOUTS,LAYOUT_NAME
            data=json.loads(BUNDLED_LAYOUTS.read_text());t=data['records'][gpu][LAYOUT_NAME]
            donor=json.loads(Path('official/4090-layouts.json').read_text())['records']['NVIDIA_GeForce_RTX_4090-sm89'][LAYOUT_NAME]
            for key in list(t):
                sh=json.loads(key)['shape']
                if sh['batch']==128 and sh['height'] in (14,28,56):
                    matches=[v for k,v in donor.items() if json.loads(k)['shape']==sh]
                    assert len(matches)==1
                    t[key]=matches[0]
            BUNDLED_LAYOUTS.write_text(json.dumps(data,indent=2,sort_keys=True))
        print("RECORD",arm,json.dumps({gpu:proof},separators=(',',':')));return
    assert not (Path(os.environ["LNET_LAUNCH_CACHE"])/"launch_geometry.json").exists()
    runpy.run_path(os.environ.get("MATCHED_ENTRY","benchmark_matched.py"),run_name="__main__")
    if "--profile" in sys.argv:
        p=Path(sys.argv[sys.argv.index("--output")+1]).with_suffix(".trace.json")
        print("TRAIN_LAUNCHES",p.stem,json.dumps(report(p,arm,old)))
        if arm=='donor':
            from tune_path_backward import launch_contracts
            for h,bm in [(14,8),(28,4)]:
                rows=launch_contracts(p,(128,h,h,160))
                assert rows and all((v["modes"],v["warps"],v["group"],v["compact"])==(bm,2,4,3) for v in rows)
            print("DONOR_LOWER_VERIFIED",p.stem)
def seg_check(par,g):
    s=replace(par,kernel_revision=segment_revision(par.kernel_revision))
    return table(SB,SEGMENT_NAME).get(s.key)==g
if __name__=='__main__':main()
