"""Independent exact-scope alternatives to a verified four-row direct backward."""
import hashlib
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import FunctionType

import triton
from ..kernel_launch_config import geometry_store_signature, register_kernel_specializer, validated_geometry
from ..path_backward_layout import _LayoutGraphPass
from ..retained_path_dispatch import PROFILE_NAME, VERTICAL_BACKWARD
from ..segmented_path_scan import _original_kernel, segment_revision, select_segment
from . import rows2, rows8, tree, native
from .. import segmented_path_scan as standard
from .buckets import _bucket_backward_kernel
from . import outdot

NAME="direct_vertical_prefix_trial"
BUNDLE=Path(__file__).with_name("validated.json")
def rebound(kernel, **overrides):
    original=kernel.fn
    fn=FunctionType(original.__code__,{**original.__globals__,**overrides},
                    original.__name__,original.__defaults__,original.__closure__)
    fn.__annotations__=original.__annotations__.copy()
    fn.__kwdefaults__=original.__kwdefaults__
    fn.__module__=__name__
    return triton.jit(fn)
tree_segment=rebound(standard._segment_four,_local_prefix=tree._local_prefix)
tree_prefix=rebound(standard._prefix_pair,_segment_four=tree_segment)
native_prefix=rebound(standard._prefix_pair,_segment_four=native._segment_four)
KERNELS={
    1:rebound(_original_kernel,_prefix_pair=rows2._prefix_pair),
    2:rebound(_original_kernel,_prefix_pair=rows8._prefix_pair),
    3:rebound(_original_kernel,_prefix_pair=tree_prefix),
    4:_bucket_backward_kernel,
    5:rebound(_original_kernel,_prefix_pair=native_prefix),
    6:outdot.specialize(rebound(_original_kernel,_prefix_pair=standard._prefix_pair),_original_kernel),
}
VARIANTS={"rows2":1,"rows8":2,"tree":3,"buckets":4,"native":5,"outdot":6}
@cache
def revision(parent):
    h=hashlib.sha256(segment_revision(parent).encode())
    for k in KERNELS.values():h.update(k.cache_key.encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()
def select(scope,base):
    previous=select_segment(scope,base)
    sh=dict(scope.shape_items)
    if previous is None or sh.get("operation")!=VERTICAL_BACKWARD or sh.get("coarse")!=0:
        return previous
    key=replace(scope,kernel_revision=revision(scope.kernel_revision))
    cfg=validated_geometry(NAME,key,BUNDLE)
    if cfg is not None:
        variant=cfg.blocks.get("VARIANT")
        if variant in KERNELS and cfg==base.with_overrides(blocks={"VARIANT":variant}):
            if variant!=4 or sh["batch"]%4==0:
                return KERNELS[variant]
    return previous
def graph_pass(previous):
    register_kernel_specializer(PROFILE_NAME,select)
    signature=geometry_store_signature(NAME,BUNDLE,omit_empty=True)
    return _LayoutGraphPass(revision(signature),previous) if signature else previous
