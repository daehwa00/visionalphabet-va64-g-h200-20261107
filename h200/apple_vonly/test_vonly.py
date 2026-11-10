import unittest,torch
from lrcp.source_controls import LRCPSourceControl
class VOnlyTest(unittest.TestCase):
 def check(self,device,dtype):
  torch.manual_seed(7)
  m=LRCPSourceControl(2,4,2,.7,down_init_std=.2,up_init_std=.2,source='value').to(device=device,dtype=dtype)
  for stats in (False,True):
   m.stats_enabled=stats;m.use_fused_training=True
   a=torch.randn(2,3,2,4,device=device,dtype=dtype,requires_grad=True);v=torch.randn_like(a,requires_grad=True)
   out=m.forward_bshd(a,v)
   ref=a+torch.einsum('bshr,hrd->bshd',torch.einsum('bshd,hdr->bshr',v,m.down),m.up)*.7
   torch.testing.assert_close(out,ref,rtol=.02 if dtype in (torch.float16,torch.bfloat16) else 1e-6,atol=.02 if dtype in (torch.float16,torch.bfloat16) else 1e-7)
   g=torch.autograd.grad(out.sum(),(a,v,m.down,m.up),retain_graph=True);r=torch.autograd.grad(ref.sum(),(a,v,m.down,m.up))
   for x,y in zip(g,r):torch.testing.assert_close(x,y,rtol=.02 if dtype in (torch.float16,torch.bfloat16) else 1e-6,atol=.02 if dtype in (torch.float16,torch.bfloat16) else 1e-7)
   self.assertTrue(torch.equal(g[0],torch.ones_like(a)))
 def test_cpu(self):self.check('cpu',torch.float64)
 @unittest.skipUnless(torch.cuda.is_available(),'no GPU')
 def test_cuda(self):
  for dtype in (torch.float16,torch.bfloat16):self.check('cuda',dtype)
if __name__=='__main__':unittest.main()
