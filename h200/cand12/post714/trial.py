"""Run either the pinned generator source or the portable candidate with observed direct H200 scope."""
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

from lnet.kernel_launch_config import capture_launch_scopes, device_key
from lnet.retained_path_dispatch import PROFILE_NAME, BUNDLE as RETAINED, step_geometry, step_revision
from lnet.segmented_path_scan import SEGMENT_NAME, BUNDLE as SEGMENT, segment_revision
from tune_path_backward import load_scope

ARMS={'control','outdot','tree','native','rows2','rows8','buckets'}


def save_bundle(path,name,verified,base,scope,variant=None):
    current=json.loads(path.read_text())
    table=current['records'].setdefault(device_key(),{}).setdefault(name,{})
    key=scope.key
    block=dict(base)
    block["blocks"]=dict(base["blocks"])
    if variant is not None:block["blocks"]["VARIANT"]=variant
    for old in list(table):
        if json.loads(old)["shape"]==dict(scope.shape_items) and old!=key:del table[old]
    table[key]=block
    path.write_text(json.dumps(current,indent=2,sort_keys=True)+'\n')
    verified.setdefault(name,{})[key]=block


def layout_evidence(trace,arm):
    ev=json.loads(trace.read_text())['traceEvents']
    ops={x['args']['External id']:x['args'] for x in ev if x.get('cat')=='cpu_op' and 'kernel_file' in x.get('args',{})}
    rows=[]
    for e in ev:
        if e.get('cat')!='kernel' or e['name'] not in ('_path_backward_kernel','_bucket_backward_kernel','_path_forward_kernel','_horizontal_path_backward_kernel'):continue
        a=ops[e['args']['External id']];text=Path(a['kernel_file']).read_text();tree=ast.parse(text)
        meta=next(n.value for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='triton_meta')
        cons=next(ast.literal_eval(v) for k,v in zip(meta.keys,meta.values) if getattr(k,'value',None)=='constants')
        coarse=bool(cons.get('COARSE',False))
        tile={k:int(v) for k,v in (i.split('=') for i in a['kernel_kwargs'].split(','))}
        op=0 if e['name']=='_path_forward_kernel' else 2 if e['name']=='_horizontal_path_backward_kernel' else 1
        assert (tile['BM'],tile['GROUP'],a['num_warps'])==((8,4,4) if op==0 else (32,1,4) if op==2 else (4,4,2))
        if op!=2:assert tile['DOUBLING']==int(op==1)
        impl=None
        if op==1:
            fns={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
            def referenced(fn,target):
                return any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==target for n in ast.walk(fns[fn]))
            if coarse:
                # A generated direct-only outdot may be present in source inside a
                # compile-time false branch. The actual coarse path keeps original reductions.
                assert not referenced('_prefix_pair','_segment_four')
                assert e['name']=='_path_backward_kernel'
                if 'wo_grad =' in text or 'wo_grad=' in text:
                    assert 'if not COARSE:' in text
                impl='retained'
            else:
                assert coarse is False
                if arm=='buckets':
                    assert e['name']=='_bucket_backward_kernel'
                    impl='buckets'
                elif arm in ('rows2','rows8'):
                    assert '_row_prefix' in text
                    target=next(n for n in ast.walk(fns['_prefix_pair']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_row_prefix')
                    assert ast.literal_eval(target.args[-2])==int(arm[-1])
                    impl=arm
                else:
                    assert referenced('_prefix_pair','_segment_four')
                    four=fns['_segment_four']
                    native=any(isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.associative_scan' for n in ast.walk(four))
                    local_names={n.id for n in ast.walk(fns['_local_prefix']) if isinstance(n,ast.Name)}
                    is_outdot=any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='wo_grad' for t in n.targets) for n in ast.walk(fns['_path_backward_kernel']))
                    impl='outdot' if is_outdot else 'native' if native else 'tree' if 'tail_r' in local_names else 'control'
                    assert impl==arm,(impl,arm)
        rows.append(dict(op=op,coarse=int(coarse),impl=impl,tile=tile,warps=a['num_warps'],grid=e['args']['grid'],regs=e['args']['registers per thread'],shared=e['args']['shared memory'],ms=e['dur']/1000))
    assert len(rows)==5 and {(r['op'],r['coarse']) for r in rows}=={(0,0),(1,0),(1,1),(2,0),(2,1)},rows
    return rows


def main():
    gpu=device_key();root=Path(os.environ['RUN_ROOT']);arm=os.environ['ARM'];phase=os.environ['PHASE']
    source=Path(RETAINED).parents[2]
    canonical=arm in ('tree','native','outdot') and os.environ.get('PORTABLE')!='1'
    if not canonical:
        from lnet import prefix_trials as pt
    family=PROFILE_NAME if canonical else pt.NAME
    suffix='-portable' if os.environ.get('PORTABLE')=='1' else ''
    f=root/(arm+suffix+'-observed.json')
    if phase=='observe':
        with capture_launch_scopes(family) as found:
            runpy.run_path('benchmark_matched.py',run_name='__main__')
        found=[s for s in found if s.execution_regime=='compiled' and dict(s.shape_items).get('batch')==128]
        assert found,(family,arm)
        f.write_text(json.dumps([asdict(s) for s in found],indent=2))
        return
    if phase=='package':
        assert arm!='control'
        observed=[load_scope(s) for s in json.loads(f.read_text())]
        parents=json.loads(RETAINED.read_text())['records'][gpu][PROFILE_NAME]
        segs=json.loads(SEGMENT.read_text())['records'][gpu][SEGMENT_NAME]
        assert len([s for s in segs if json.loads(s)['shape']['batch']==128])==1
        proof={}
        if canonical:
            # Keep every measured geometry; derive actual parent revisions from this
            # source. In particular outdot changes the backward JIT revision for both
            # shapes. Coarse gets only its parent record, never a four-row record.
            for old,g in parents.copy().items():
                shape=json.loads(old)['shape']
                if shape['batch']!=128:continue
                matches=[s for s in observed if dict(s.shape_items)==shape]
                assert len(matches)==1,(shape,observed)
                parent=matches[0]
                assert parent.kernel_revision==step_revision(shape['operation'])
                save_bundle(RETAINED,PROFILE_NAME,proof,g,parent)
                if shape['operation']==1 and not shape['coarse']:
                    from dataclasses import replace
                    seg=replace(parent,kernel_revision=segment_revision(parent.kernel_revision))
                    save_bundle(SEGMENT,SEGMENT_NAME,proof,g,seg)
        else:
            assert len(observed)==1
            sc=observed[0]
            parentkey,g=next((k,v) for k,v in parents.items() if json.loads(k)['shape']==dict(sc.shape_items))
            assert sc.kernel_revision==pt.revision(json.loads(parentkey)['kernel_revision'])
            save_bundle(pt.BUNDLE,pt.NAME,proof,g,sc,pt.VARIANTS[arm])
        print('CANDIDATE_RECORD',arm+suffix,json.dumps({gpu:proof},separators=(',',':')))
        paths=['direct_path_scan_kernels.py','segmented_path_scan.py','retained_path_dispatch.py']
        print('SOURCE_FINGERPRINT',arm+suffix,json.dumps({n:hashlib.sha256((source/'src'/'lnet'/n).read_bytes()).hexdigest() for n in paths}))
        return
    assert not (Path(os.environ['LNET_LAUNCH_CACHE'])/'launch_geometry.json').exists()
    runpy.run_path(os.environ.get('MATCHED_ENTRY','benchmark_matched.py'),run_name='__main__')
    if '--profile' in sys.argv:
        trace=Path(sys.argv[sys.argv.index('--output')+1]).with_suffix('.trace.json')
        print('TRAIN_LAUNCHES',trace.stem,json.dumps(layout_evidence(trace,arm)))
        events=json.loads(trace.read_text())['traceEvents']
        top=sorted((e for e in events if e.get('cat')=='kernel'),key=lambda e:e['dur'],reverse=True)[:7]
        print('TOP_KERNELS',trace.stem,json.dumps([[e['name'],e['dur']/1000,e['args'].get('grid')] for e in top]))

if __name__=='__main__':main()
