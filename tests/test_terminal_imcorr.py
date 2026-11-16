import pytest
import torch
from lnet.terminal_imaginary_correlation import complex_vertical_correlation,complex_correlation_descriptor
from lnet.pac_product_scan_pipeline import run_product_scan_pipeline_packed
import run_va_a_imcorr as runner


def inputs(device='cpu',dtype=torch.float32):
    torch.manual_seed(4)
    def pole():
        return tuple(x.to(device).requires_grad_() for x in [torch.full((1,1,1,4),.7),torch.full((1,1,1,4),.2),torch.full((1,1,1,4),.4),torch.full((1,1,1,4),.1)])
    return pole(),pole(),torch.randn(2,3,5,8,device=device,dtype=dtype,requires_grad=True)


@pytest.mark.parametrize('reverse',[False,True])
def test_complex_oracle_and_gradients(reverse):
    _,p,x=inputs();r,i=x.chunk(2,-1)
    parts=complex_vertical_correlation(p,(r,i),reverse=reverse)
    actual=torch.complex(*parts)
    a=torch.complex(p[0],p[1]).reshape(1,1,4);g=torch.complex(p[2],p[3]).reshape(1,1,4);z=torch.complex(r,i)
    if reverse:z=z.flip(1);a=a.conj();g=g.conj()
    s=torch.zeros_like(z[:,0]);terms=[]
    for h in z.unbind(1):
        u=g*h;s=a*s+u;terms.append(2*s*u.conj()-u.abs().square())
    expected=torch.stack(terms,1).mean((1,2))
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    ga=torch.autograd.grad(actual.abs().square().sum(),(*p,x),retain_graph=True)
    gb=torch.autograd.grad(expected.abs().square().sum(),(*p,x))
    for left,right in zip(ga,gb):torch.testing.assert_close(left,right,atol=3e-6,rtol=3e-5)
    assert actual.abs().max()>0


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_descriptor_oracle_and_backward(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    px,py,x=inputs(device,dtype)
    combined=complex_correlation_descriptor(px,py,x,gain_normalization='pointwise')
    from lnet.pac_product_scan_reference import bidirectional_product_scan_reference
    horizontal=bidirectional_product_scan_reference(px,x.chunk(2,-1))
    pairs=[complex_vertical_correlation(py,h,reverse=rev) for rev in (False,True) for h in (horizontal[:2],horizontal[2:])]
    expected=torch.cat([p[0] for p in pairs]+[p[1] for p in pairs],-1)
    torch.testing.assert_close(combined,expected,rtol=2e-4,atol=2e-5)
    gradients=torch.autograd.grad(combined.square().sum(),(*px,*py,x))
    assert combined.shape==(2,32)
    assert all(torch.isfinite(g).all() for g in gradients)


def test_model_scratch_backbone_and_inventory():
    torch.set_num_threads(2);torch.manual_seed(501)
    old=runner.anchor._build(runner.anchor.VARIANT,runner.baseline.ComplexScanConfig(output_dim=100,stem_strides=(2,2))).eval()
    torch.manual_seed(501)
    new=runner._build(runner.VARIANT,runner.baseline.ComplexScanConfig(output_dim=100,stem_strides=(2,2))).eval()
    assert sum(p.numel() for p in new.parameters())==2988004
    assert new.classifier.q4_dim==1024
    assert torch.count_nonzero(new.classifier.affine.linear.weight[:,512:])==0
    with torch.no_grad():
        x=torch.randn(2,3,32,32)
        a=old(x)[0];b=new(x)[0]
        assert torch.isfinite(b).all()
        assert not torch.allclose(a,b)
    for key,value in old.backbone.state_dict().items():
        if not key.startswith("classifier."):
            torch.testing.assert_close(value,new.backbone.state_dict()[key],rtol=0,atol=0)


def test_raw_g_has_no_power_normalization():
    px,py,x=inputs()
    a=complex_correlation_descriptor(px,py,x)
    b=complex_correlation_descriptor(px,py,2*x)
    torch.testing.assert_close(b,4*a,rtol=3e-5,atol=3e-6)
    assert torch.count_nonzero(complex_correlation_descriptor(px,py,torch.zeros_like(x)))==0


def test_intermediate_features_unchanged_and_optimized():
    from lnet.real_memory_training import enable_real_memory_training_
    torch.set_num_threads(2);torch.manual_seed(501)
    old=runner.anchor._build(runner.anchor.VARIANT,runner.baseline.ComplexScanConfig(output_dim=100,stem_strides=(2,2))).eval()
    enable_real_memory_training_(old)
    torch.manual_seed(501)
    new=runner._build(runner.VARIANT,runner.baseline.ComplexScanConfig(output_dim=100,stem_strides=(2,2))).eval()
    assert new._training_optimization_inventory=={'readers':12}
    x=torch.randn(2,3,32,32)
    with torch.no_grad():
        torch.testing.assert_close(old.forward_features(x),new.forward_features(x),rtol=0,atol=0)
        torch.testing.assert_close(new.forward_features(x),new.backbone.packed_features(x)[-1],rtol=0,atol=0)
