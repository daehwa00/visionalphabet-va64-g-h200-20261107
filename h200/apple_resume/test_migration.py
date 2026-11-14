import copy
import pytest
import torch
from runtime import convert,digest

def sample():
    contract={'branch_source':'residual','global_batch':1024,'microbatch':8,'gradient_accumulation_steps':128,'source_sha256':'source','data_manifest_sha256':'data','seed':1234,'sequence_length':2048}
    best={'kind':'weights','contract':copy.deepcopy(contract),'model':{'w':torch.tensor([1.])},'progress':{'val_loss':2.9}}
    saved={'schema':2,'kind':'training','contract':contract,'model':{'w':torch.tensor([2.])},'optimizer':{'moment':torch.tensor([.5])},'scaler':{'scale':1024.},'rng':{'cuda':[torch.tensor([1],dtype=torch.uint8)]},'progress':{'global_step':27000,'tokens_seen':27000*1024*2048,'best_val':2.9},'best':best}
    target=dict(contract,microbatch=64,gradient_accumulation_steps=16)
    return saved,target

def test_only_execution_changes_and_all_state_preserved():
    saved,target=sample();before=copy.deepcopy(saved)
    result=convert(saved,target,{'sha256':'parent'})
    for key in ('model','optimizer','scaler','rng','progress'):
        assert digest(saved[key])==digest(result[key])
    assert result['contract']==target and result['best']['contract']==target
    assert digest(result['best']['model'])==digest(saved['best']['model'])
    assert digest(saved)==digest(before)

@pytest.mark.parametrize('field,value',[('source_sha256','other'),('data_manifest_sha256','other'),('seed',2),('sequence_length',1024)])
def test_reject_unrelated_contract_change(field,value):
    saved,target=sample();target[field]=value
    with pytest.raises(ValueError):convert(saved,target,{})

@pytest.mark.parametrize('bad',['variant','batch','cursor','rng','best'])
def test_reject_invalid_state(bad):
    saved,target=sample()
    if bad=='variant':target['branch_source']='value'
    if bad=='batch':target['gradient_accumulation_steps']=32
    if bad=='cursor':saved['progress']['tokens_seen']+=1
    if bad=='rng':saved['rng']['cuda']=[]
    if bad=='best':saved['best']['progress']['val_loss']=3.
    with pytest.raises(AssertionError):convert(saved,target,{})
