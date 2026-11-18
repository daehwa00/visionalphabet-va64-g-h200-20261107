"""Fused mode-separable complex depthwise3x3; same parameters/operator as Cconv."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.library import triton_op,wrap_triton
import triton
import triton.language as tl

WEIGHT_BS=128
WEIGHT_BM=32


@triton.jit
def _forward(R,I,WR,WI,OR,OI,N:tl.constexpr,H:tl.constexpr,W:tl.constexpr,M:tl.constexpr,B:tl.constexpr):
 q=tl.program_id(0)*B+tl.arange(0,B);valid=q<N*H*W*M
 m=q%M;p=q//M;x=p%W;y=(p//W)%H;n=p//(H*W)
 for d in tl.static_range(4):
  ar=tl.full((B,),0,tl.float32);ai=tl.full((B,),0,tl.float32)
  for ky in tl.static_range(3):
   for kx in tl.static_range(3):
    yy=y+ky-1;xx=x+kx-1;mask=valid&(yy>=0)&(yy<H)&(xx>=0)&(xx<W)
    idx=((n*H+yy)*W+xx)*M+m;k=(m*4+d)*9+ky*3+kx
    r=tl.load(R+idx,mask,0).to(tl.float32);i=tl.load(I+idx,mask,0).to(tl.float32)
    wr=tl.load(WR+k,valid,0).to(tl.float32);wi=tl.load(WI+k,valid,0).to(tl.float32)
    ar+=r*wr-i*wi;ai+=r*wi+i*wr
  out=(p*4+d)*M+m;tl.store(OR+out,ar,valid);tl.store(OI+out,ai,valid)


@triton.jit
def _input_grad(GR,GI,WR,WI,DR,DI,N:tl.constexpr,H:tl.constexpr,W:tl.constexpr,M:tl.constexpr,B:tl.constexpr):
 q=tl.program_id(0)*B+tl.arange(0,B);valid=q<N*H*W*M
 m=q%M;p=q//M;x=p%W;y=(p//W)%H;n=p//(H*W)
 ar=tl.full((B,),0,tl.float32);ai=tl.full((B,),0,tl.float32)
 for ky in tl.static_range(3):
  for kx in tl.static_range(3):
   yy=y+1-ky;xx=x+1-kx;mask=valid&(yy>=0)&(yy<H)&(xx>=0)&(xx<W)
   pp=(n*H+yy)*W+xx
   for d in tl.static_range(4):
    idx=(pp*4+d)*M+m;k=(m*4+d)*9+ky*3+kx
    gr=tl.load(GR+idx,mask,0).to(tl.float32);gi=tl.load(GI+idx,mask,0).to(tl.float32)
    wr=tl.load(WR+k,valid,0).to(tl.float32);wi=tl.load(WI+k,valid,0).to(tl.float32)
    ar+=gr*wr+gi*wi;ai+=-gr*wi+gi*wr
 tl.store(DR+q,ar,valid);tl.store(DI+q,ai,valid)


@triton.jit
def _weight_partials(R,I,GR,GI,PR,PI,N:tl.constexpr,H:tl.constexpr,W:tl.constexpr,M:tl.constexpr,BS:tl.constexpr,BM:tl.constexpr):
 split=tl.program_id(0);p=split*BS+tl.arange(0,BS)
 m=tl.program_id(1)*BM+tl.arange(0,BM)
 x=p%W;y=(p//W)%H;n=p//(H*W);pmask=(p[:,None]<N*H*W)&(m[None,:]<M)
 for ky in tl.static_range(3):
  for kx in tl.static_range(3):
   yy=y+ky-1;xx=x+kx-1;mask=pmask&(yy[:,None]>=0)&(yy[:,None]<H)&(xx[:,None]>=0)&(xx[:,None]<W)
   inp=((n[:,None]*H+yy[:,None])*W+xx[:,None])*M+m[None,:]
   r=tl.load(R+inp,mask,0).to(tl.float32);i=tl.load(I+inp,mask,0).to(tl.float32)
   for d in tl.static_range(4):
    idx=(p[:,None]*4+d)*M+m[None,:]
    gr=tl.load(GR+idx,pmask,0).to(tl.float32);gi=tl.load(GI+idx,pmask,0).to(tl.float32)
    dw=tl.sum(r*gr+i*gi,axis=0);di=tl.sum(-i*gr+r*gi,axis=0)
    out=((split*4+d)*9+ky*3+kx)*M+m
    tl.store(PR+out,dw,m<M);tl.store(PI+out,di,m<M)


@triton_op('va64_conv_opt::forward',mutates_args={})
def _op(r:Tensor,i:Tensor,wr:Tensor,wi:Tensor)->tuple[Tensor,Tensor]:
 n,h,w,m=r.shape
 out_r=torch.empty((n,h,w,4,m),device=r.device,dtype=r.dtype);out_i=torch.empty_like(out_r)
 wrap_triton(_forward)[(triton.cdiv(r.numel(),128),)](r,i,wr,wi,out_r,out_i,n,h,w,m,128,num_warps=4)
 return out_r,out_i


@triton_op('va64_conv_opt::backward',mutates_args={})
def _backward_op(r:Tensor,i:Tensor,wr:Tensor,wi:Tensor,gr:Tensor,gi:Tensor)->tuple[Tensor,Tensor,Tensor,Tensor]:
 n,h,w,m=r.shape
 dr=torch.empty_like(r);di=torch.empty_like(i)
 wrap_triton(_input_grad)[(triton.cdiv(r.numel(),128),)](gr,gi,wr,wi,dr,di,n,h,w,m,128,num_warps=4)
 splits=triton.cdiv(n*h*w,WEIGHT_BS)
 pr=torch.empty((splits,4,9,m),device=r.device,dtype=torch.float32);pi=torch.empty_like(pr)
 wrap_triton(_weight_partials)[(splits,triton.cdiv(m,WEIGHT_BM))](r,i,gr,gi,pr,pi,n,h,w,m,WEIGHT_BS,WEIGHT_BM,num_warps=8)
 dwr=pr.sum(0).permute(2,0,1).reshape_as(wr).to(wr.dtype)
 dwi=pi.sum(0).permute(2,0,1).reshape_as(wi).to(wi.dtype)
 return dr,di,dwr,dwi


def _setup(ctx,inputs,output):ctx.save_for_backward(*inputs)
def _backward(ctx,gr,gi):
 r,i,wr,wi=ctx.saved_tensors
 if gr is None:gr=torch.zeros((*r.shape[:3],4,r.shape[-1]),device=r.device,dtype=r.dtype)
 if gi is None:gi=torch.zeros_like(gr)
 return _backward_op(r,i,wr,wi,gr.contiguous(),gi.contiguous())
_op.register_autograd(_backward,setup_context=_setup)


def convolution(r,i,wr,wi):
 dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else r.dtype
 return _op(r.to(dtype).contiguous(),i.to(dtype).contiguous(),wr.to(dtype).contiguous(),wi.to(dtype).contiguous())
