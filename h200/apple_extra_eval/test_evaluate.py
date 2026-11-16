import math
import pytest
from evaluate import validate

def result(n=10,effective=10,metric='acc,none',value=.5,name='swag'):
 return {'results':{name:{metric:value}},'n-samples':{name:{'original':n,'effective':effective}}}

def test_full_count_required():
 validate(result(),['swag'],None)
 with pytest.raises(ValueError,match='Incomplete'):validate(result(effective=2),['swag'],None)

def test_diagnostic_limit_is_explicit():
 validate(result(effective=2),['swag'],2)
 with pytest.raises(ValueError):validate(result(),['swag'],2)

def test_missing_and_invalid_metric_rejected():
 with pytest.raises(ValueError):validate(result(),['hellaswag'],None)
 for value in [float('nan'),float('inf'),-1,2]:
  with pytest.raises(ValueError):validate(result(value=value),['swag'],None)

def test_wikitext_uses_word_ppl():
 validate(result(name='wikitext',metric='word_perplexity,none',value=25.),['wikitext'],None)
 with pytest.raises(ValueError):validate(result(name='wikitext'),['wikitext'],None)
