import math
import pytest
from job import validate_smoke

def good():return {'status':'completed','optimizer_updates':20,'branch_up_l2':.1,'final_validation':{'val_loss':10.}}

def test_successful_smoke():validate_smoke(good())

@pytest.mark.parametrize('change',[{'optimizer_updates':0},{'optimizer_updates':15},{'branch_up_l2':0.},{'branch_up_l2':float('nan')},{'final_validation':{'val_loss':float('inf')}},{'status':'oom'}])
def test_smoke_rejects_invalid_updates(change):
 r=good();r.update(change)
 with pytest.raises(RuntimeError):validate_smoke(r)
