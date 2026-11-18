"""Observe and verify the exact #713 H200 parent and alternate four-row JIT in real training."""
import ast
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import runpy
import sys

from lnet.kernel_launch_config import capture_launch_scopes, device_key, stored_geometry
from lnet.segmented_path_scan import BUNDLE, SEGMENT_NAME, segment_revision
from lnet.retained_path_dispatch import BUNDLE as BASE_BUNDLE, PROFILE_NAME
from tune_path_backward import load_scope
from replay.train_trial import profiled

ARMS={"control":set(),"direct":{0},"coarse":{1},"both":{0,1}}


def derive(seen):
    result={}
    gpu=device_key()
    table=json.loads(BASE_BUNDLE.read_text())["records"][gpu][PROFILE_NAME]
    for key,geometry in table.items():
        parent=json.loads(key)
        sh=parent["shape"]
        if (sh["batch"],sh["height"],sh["width"],sh["modes"],sh["operation"])!=(128,56,56,160,1):
            continue
        parent["kernel_revision"]=segment_revision(parent["kernel_revision"])
        match=[s for s in seen if json.loads(s.key)==parent]
        assert len(match)==1,(parent,seen)
        scope=match[0]
        assert geometry["blocks"]=={"BM":4,"DOUBLING":1,"GROUP":4} and geometry["num_warps"]==2
        result[sh["coarse"]]=(scope.key,geometry)
    assert set(result)=={0,1}
    return result


def validate_launches(trace,selected):
    events=json.loads(trace.read_text())["traceEvents"]
    calls={e["args"]["External id"]:e["args"] for e in events if e.get("cat")=="cpu_op" and "kernel_file" in e.get("args",{})}
    gpu=device_key()
    base=json.loads(BASE_BUNDLE.read_text())["records"][gpu][PROFILE_NAME]
    geoms={};implementations={}
    from lnet.retained_path_dispatch import step_geometry
    for key,v in base.items():
        shape=json.loads(key)["shape"];b=v["blocks"]
        if shape["batch"]==128:
            op=shape["operation"];geoms[(op,shape["coarse"])]=step_geometry(op,b["GROUP"],b["BM"],v["num_warps"],b.get("DOUBLING",0))
    for e in events:
        if e.get("cat")!="kernel" or e["name"]!="_path_backward_kernel":continue
        call=calls[e["args"]["External id"]]
        tree=ast.parse(Path(call["kernel_file"]).read_text())
        meta=next(k.value for k in ast.walk(tree) if isinstance(k,ast.keyword) and k.arg=="triton_meta")
        constants=next(ast.literal_eval(v) for k,v in zip(meta.keys,meta.values) if getattr(k,"value",None)=="constants")
        prefix=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_prefix_pair")
        flag=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="_segment_four" for n in ast.walk(prefix))
        coarse=int(constants["COARSE"])
        assert coarse not in implementations
        implementations[coarse]=flag
    assert set(implementations)=={0,1}
    assert {k for k,v in implementations.items() if v}==selected,implementations
    rows=profiled(geoms,trace)
    for row in rows:
        if row["op"]==1:row["four_rows"]=implementations[row["coarse"]]
    return rows


def main():
    root=Path(os.environ["RUN_ROOT"]);arm=os.environ["ARM"];phase=os.environ["PHASE"]
    observed=root/"observed.json"
    if phase=="observe":
        with capture_launch_scopes(SEGMENT_NAME) as scopes:
            runpy.run_path("benchmark_matched.py",run_name="__main__")
        seen=[s for s in scopes if dict(s.shape_items).get("batch")==128 and s.execution_regime=="compiled"]
        mapping=derive(seen)
        observed.write_text(json.dumps([asdict(s) for s in seen],indent=2))
        print("OBSERVED",[(k,json.loads(s)["kernel_revision"]) for k,(s,g) in mapping.items()])
        return
    mapping=derive([load_scope(s) for s in json.loads(observed.read_text())])
    assert not (Path(os.environ["LNET_LAUNCH_CACHE"])/"launch_geometry.json").exists()
    gpu=device_key();wanted={mapping[k][0]:mapping[k][1] for k in ARMS[arm]}
    if phase=="package":
        payload=json.loads(BUNDLE.read_text())
        assert gpu not in payload["records"] or not payload["records"][gpu].get(SEGMENT_NAME), "unexpected initial H200 segment record"
        if wanted:payload["records"].setdefault(gpu,{})[SEGMENT_NAME]=wanted
        BUNDLE.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")
        print("CANDIDATE_RECORD",arm,json.dumps({gpu:{SEGMENT_NAME:wanted}},separators=(',',':')))
        return
    for key,val in mapping.values():
        scope=next(s for s in map(load_scope,json.loads(observed.read_text())) if s.key==key)
        actual=stored_geometry(SEGMENT_NAME,scope=scope,path=BUNDLE)
        if key in wanted:assert actual is not None and actual.blocks==val['blocks'] and actual.num_warps==val['num_warps']
        else:assert actual is None
    runpy.run_path(os.environ.get("MATCHED_ENTRY","benchmark_matched.py"),run_name="__main__")
    if "--profile" in sys.argv:
        path=Path(sys.argv[sys.argv.index("--output")+1]).with_suffix(".trace.json")
        print("TRAIN_LAUNCHES",path.stem,json.dumps(validate_launches(path,ARMS[arm])))

if __name__=="__main__":main()
