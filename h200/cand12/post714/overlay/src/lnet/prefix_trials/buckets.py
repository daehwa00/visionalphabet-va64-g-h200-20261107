"""Distribute path-parameter atomics over existing batch rows before the usual global sum."""
import triton
import triton.language as tl

from ..segmented_path_scan import _path_backward_kernel


@triton.jit
def _bucket_backward_kernel(
    decay_real, decay_imag, gamma_real, gamma_imag,
    source_real_a, source_imag_a, source_real_b, source_imag_b,
    variance_x, variance_y, path_input_weight, path_input_bias, path_output_weight,
    grad_collapsed_real, grad_collapsed_imag,
    grad_path_input_weight, grad_path_input_bias, grad_path_output_weight, grad_path_output_bias,
    grad_decay_real, grad_decay_imag, grad_gamma_real, grad_gamma_imag, grad_source_real_a,
    height:tl.constexpr, width:int, line_count:int, modes:int, packed_path_hidden:int,
    epsilon:tl.constexpr, BM:tl.constexpr, GROUP:tl.constexpr, COARSE:tl.constexpr, DOUBLING:tl.constexpr,
):
    # Direct-only selector requires a batch count divisible by four. Every
    # destination batch row is summed by try_backward before escaping the op.
    line=tl.program_id(1)*GROUP+tl.program_id(0)%GROUP
    batch=line//width
    x=line-batch*width
    delta=((batch^(x & 3))-batch)*modes
    _path_backward_kernel(
        decay_real,decay_imag,gamma_real,gamma_imag,
        source_real_a,source_imag_a,source_real_b,source_imag_b,
        variance_x,variance_y,path_input_weight,path_input_bias,path_output_weight,
        grad_collapsed_real,grad_collapsed_imag,
        grad_path_input_weight+delta*packed_path_hidden*8,
        grad_path_input_bias+delta*packed_path_hidden,
        grad_path_output_weight+delta*packed_path_hidden*2,grad_path_output_bias+delta*2,
        grad_decay_real,grad_decay_imag,grad_gamma_real,grad_gamma_imag,grad_source_real_a,
        height,width,line_count,modes,packed_path_hidden,epsilon,BM,GROUP,COARSE,DOUBLING,
    )
