"""Execution-only packing of unchanged mode-separable complex 3x3 convolution.

Parameter names/shapes and the mathematical operator are unchanged. Groups of
modes use explicitly zero off-diagonal blocks to enable efficient real kernels.
BF16 accumulation order differs from four independently rounded convolutions.
"""
from __future__ import annotations
from types import MethodType
import torch
from torch import nn,Tensor
from torch.nn import functional as F


def packed_forward(self, real:Tensor, imag:Tensor):
    modes=self.modes;block=self.packed_modes
    if real.shape!=imag.shape or real.ndim!=4 or real.shape[-1]!=modes:
        raise ValueError('matching NHWM fields required')
    batch,height,width,_=real.shape
    values=torch.stack((real,imag),dim=-1).reshape(batch,height,width,2*modes).permute(0,3,1,2)
    wr=self.kernel_real.weight.reshape(modes,4,3,3)
    wi=self.kernel_imag.weight.reshape(modes,4,3,3)
    # (mode,path,out-real/imag,in-real/imag,kH,kW)
    pair=torch.stack((torch.stack((wr,-wi),dim=2),torch.stack((wi,wr),dim=2)),dim=2)
    base=pair.reshape(modes//block,block,8,2,3,3)
    identity=torch.eye(block,device=base.device,dtype=base.dtype)
    weight=(base[:,:,:,None,:,:,:]*identity[None,:,None,:,None,None,None]).reshape(8*modes,2*block,3,3)
    output=F.conv2d(values,weight,padding=1,groups=modes//block)
    output=output.reshape(batch,modes,4,2,height,width).permute(0,4,5,2,1,3)
    return output[...,0],output[...,1]


def install(model:nn.Module,block_modes:int,backend:str="packed"):
    count=0
    for module in model.modules():
        if module.__class__.__name__=='FourPathConvolution' and hasattr(module,'kernel_real'):
            if block_modes<=0 or module.modes%block_modes:raise ValueError('block must divide mode count')
            module.packed_modes=block_modes
            module.forward=MethodType(triton_forward if backend=="triton" else packed_forward,module)
            count+=1
    if count!=12:raise RuntimeError(f'expected12 scan operators, found{count}')
    return model


def triton_forward(self,real,imag):
    if real.shape!=imag.shape or real.ndim!=4 or real.shape[-1]!=self.modes:raise ValueError('matchingNHWMfields required')
    if not real.is_cuda:return packed_forward(self,real,imag)
    from va64_conv_scan_triton import convolution
    return convolution(real,imag,self.kernel_real.weight,self.kernel_imag.weight)
