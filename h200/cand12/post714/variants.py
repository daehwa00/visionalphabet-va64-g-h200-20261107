"""Isolated candidate trees for matched 59.27-ms experiments."""
from pathlib import Path
import argparse,json,shutil,sys
p=argparse.ArgumentParser()
p.add_argument("name");p.add_argument("--summary",choices=("tree","native","transposed","packed"));p.add_argument("--mlp",choices=("outer","outdot","inbias"));p.add_argument("--base",default="control")
p.add_argument("--role",choices=("direct","coarse","both"),default="direct")
a=p.parse_args();d=Path(a.name);shutil.copytree(a.base,d,ignore=shutil.ignore_patterns("__pycache__",".pytest_cache"))
seg=d/"src/lnet/segmented_path_scan.py";kernel=d/"src/lnet/direct_path_scan_kernels.py"
if a.summary:
 s=seg.read_text()
 if a.summary=="tree":
    s=s.replace("    r2, i2 = ar * r1 - ai * i1 + r2, ar * i1 + ai * r1 + i2\n    r3, i3 = ar * r2 - ai * i2 + r3, ar * i2 + ai * r2 + i3", """    tail_r,tail_i=ar*r2-ai*i2+r3, ar*i2+ai*r2+i3
    r2,i2=ar*r1-ai*i1+r2, ar*i1+ai*r1+i2
    a2r,a2i=ar*ar-ai*ai,ar*ai+ai*ar
    r3,i3=a2r*r1-a2i*i1+tail_r,a2r*i1+a2i*r1+tail_i""")
 else:
    i=s.index("    for k in tl.static_range(4):");j=s.index("    neighbor = group + 1",i)
    if a.summary in ("native","transposed"):
        s=s.replace("from .direct_path_scan_kernels import _path_backward_kernel as _original_kernel","from .direct_path_scan_kernels import _path_backward_kernel as _original_kernel\nfrom .triton_product_scan_kernels import _compose_complex_pair")
        i=s.index("    for k in tl.static_range(4):");j=s.index("    neighbor = group + 1",i)
        body="""    pr=tl.broadcast_to(pr,(16,modes));pi=tl.broadcast_to(pi,(16,modes))
    scan=tl.associative_scan((pr,pi,xr,xi,zr,zi),0,_compose_complex_pair,reverse=REVERSE)
    xr,xi,zr,zi=scan[2],scan[3],scan[4],scan[5]
"""
        if a.summary=="transposed":
            body=body.replace("(pr,pi,xr,xi,zr,zi),0","(tl.trans(pr),tl.trans(pi),tl.trans(xr),tl.trans(xi),tl.trans(zr),tl.trans(zi)),1").replace("=scan[2],scan[3],scan[4],scan[5]","=tl.trans(scan[2]),tl.trans(scan[3]),tl.trans(scan[4]),tl.trans(scan[5])")
    else:
        body="""    vr=tl.join(xr,zr);vi=tl.join(xi,zi)
    for k in tl.static_range(4):
        offset=1<<k
        neighbor=group+offset if REVERSE else group-offset
        idx=tl.broadcast_to((neighbor&15)[:,:,None],(16,modes,2))
        valid=neighbor<16 if REVERSE else neighbor>=0
        ur,ui=tl.gather(vr,idx,0),tl.gather(vi,idx,0)
        vr,vi=(tl.where(valid[:,:,None],pr[:,:,None]*ur-pi[:,:,None]*ui+vr,vr),
               tl.where(valid[:,:,None],pr[:,:,None]*ui+pi[:,:,None]*ur+vi,vi))
        pr,pi=pr*pr-pi*pi,pr*pi+pi*pr
    xr,zr=tl.split(vr)
    xi,zi=tl.split(vi)
"""
    s=s[:i]+body+s[j:]
 seg.write_text(s)
if a.mlp:
 s=kernel.read_text();i=s.index("def _path_backward_kernel");pre,b=s[:i],s[i:]
 guard={"direct":"not COARSE","coarse":"COARSE","both":"True"}[a.role]
 import textwrap
 if a.mlp=="outer":
    i=b.index("    path_input_weight_gradient = tl.dot");j=b.index("    tl.atomic_add(\n        grad_path_input_bias",i)
    new="""    for component in tl.static_range(8):
        feature=tl.sum(tl.where(path_coordinate[None,None,:]==component,packed_path.to(tl.float32),0.0),axis=2)
        grad=tl.sum(grad_preactivation.to(tl.float32)*feature[:,:,None],axis=1)
        dst=(mode_vector[:,None]*packed_path_hidden+hidden_coordinate[None,:])*8+component
        tl.atomic_add(grad_path_input_weight+dst,grad,mask=output_mask,sem="relaxed")
"""
 elif a.mlp=="outdot":
    i=b.index("    tl.atomic_add(\n        grad_path_output_weight")
    j=b.index("    weight_real =",i)
    new="""    # VJP of the two output channels, plus their bias in a padded hidden lane.
    gout=tl.where(path_coordinate[None,:,None]==0, active_grad_real[:,None,:],
                  tl.where(path_coordinate[None,:,None]==1,active_grad_imag[:,None,:],0.0)).to(tl.bfloat16)
    hw=tl.where(hidden_coordinate[None,None,:]==8,1.0,hidden).to(tl.bfloat16)
    wo_grad=tl.dot(gout,hw)
    dst=mode_vector[:,None,None]*2*packed_path_hidden+path_coordinate[None,:,None]*packed_path_hidden+hidden_coordinate[None,None,:]
    tl.atomic_add(grad_path_output_weight+dst,wo_grad,(path_coordinate[None,:,None]<2)&(hidden_coordinate[None,None,:]<packed_path_hidden)&(mode_vector[:,None,None]<modes),sem="relaxed")
    bo_grad=tl.sum(tl.where(hidden_coordinate[None,None,:]==8,wo_grad,0.0),axis=2)
    tl.atomic_add(grad_path_output_bias+2*mode_vector[:,None]+path_coordinate[None,:],bo_grad,(path_coordinate[None,:]<2)&(mode_vector[:,None]<modes),sem="relaxed")
"""
 else:
    i=b.index("    path_input_weight_gradient = tl.dot");j=b.index("    grad_packed_path =",i)
    block=b[i:j];new=block.replace("tl.permute(grad_preactivation, (0, 2, 1)), packed_path","tl.permute(grad_preactivation, (0, 2, 1)), tl.where(path_coordinate[None,None,:]==8,1.0,packed_path).to(tl.bfloat16)").replace("tl.sum(grad_preactivation.to(tl.float32), axis=1)", "tl.sum(tl.where(path_coordinate[None,None,:]==8,path_input_weight_gradient,0.0),axis=2)")
 b=b[:i]+"    if "+guard+":\n"+textwrap.indent(new,"    ")+"    else:\n"+textwrap.indent(b[i:j],"    ")+b[j:]
 kernel.write_text(pre+b)
sys.path.insert(0,str((d/"src").resolve()))
from lnet.retained_path_dispatch import step_revision,PROFILE_NAME
from lnet.segmented_path_scan import segment_revision,SEGMENT_NAME
gpu="NVIDIA_GeForce_RTX_4090-sm89"
for fname,family,extra in (("validated_retained_steps",PROFILE_NAME,False),("validated_segmented_scans",SEGMENT_NAME,True)):
 f=d/"src/lnet"/(fname+".json");j=json.loads(f.read_text());r={}
 for key,g in j["records"][gpu][family].items():
    info=json.loads(key);v=step_revision(info["shape"]["operation"]);info["kernel_revision"]=segment_revision(v) if extra else v
    r[json.dumps(info,sort_keys=True,separators=(",",":"))]=g
 j["records"][gpu][family]=r;f.write_text(json.dumps(j,sort_keys=True,indent=2)+"\n")
