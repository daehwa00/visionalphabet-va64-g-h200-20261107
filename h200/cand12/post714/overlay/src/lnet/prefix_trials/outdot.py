"""Apply the shared generator's direct-only VJP to a fresh private JIT clone."""
from textwrap import indent

OUTPUT_GRADIENT = '# VJP of the two output channels, plus their bias in a padded hidden lane.\ngout=tl.where(path_coordinate[None,:,None]==0, active_grad_real[:,None,:],\n              tl.where(path_coordinate[None,:,None]==1,active_grad_imag[:,None,:],0.0)).to(tl.bfloat16)\nhw=tl.where(hidden_coordinate[None,None,:]==8,1.0,hidden).to(tl.bfloat16)\nwo_grad=tl.dot(gout,hw)\ndst=mode_vector[:,None,None]*2*packed_path_hidden+path_coordinate[None,:,None]*packed_path_hidden+hidden_coordinate[None,None,:]\ntl.atomic_add(grad_path_output_weight+dst,wo_grad,(path_coordinate[None,:,None]<2)&(hidden_coordinate[None,None,:]<packed_path_hidden)&(mode_vector[:,None,None]<modes),sem="relaxed")\nbo_grad=tl.sum(tl.where(hidden_coordinate[None,None,:]==8,wo_grad,0.0),axis=2)\ntl.atomic_add(grad_path_output_bias+2*mode_vector[:,None]+path_coordinate[None,:],bo_grad,(path_coordinate[None,:]<2)&(mode_vector[:,None]<modes),sem="relaxed")\n'


def specialize(fresh, original):
    # This clone has not been passed to Triton or Inductor. No caller ever
    # mutates the canonical kernel, its dependencies, or its computed cache key.
    assert fresh is not original and fresh.fn is not original.fn
    text = fresh.src
    begin = "    tl.atomic_add(\n        grad_path_output_weight + output_offset,"
    end = "    weight_real = ("
    assert text.count(begin) == text.count(end) == 1
    i, j = text.index(begin), text.index(end)
    body = "    if not COARSE:\n" + indent(OUTPUT_GRADIENT, "        ")
    body += "    else:\n" + indent(text[i:j], "    ")
    fresh._unsafe_update_src(text[:i] + body + text[j:])
    return fresh
