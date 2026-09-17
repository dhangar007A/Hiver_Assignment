import pytest
from src.baseline_simple import classify_simple
from src.baseline_trivial import classify_trivial

def test_baseline_simple():
    res = classify_simple("my internet is down")
    assert res['intent'] == 'service_outage'
    
    res2 = classify_simple("where is my refund")
    assert res2['intent'] == 'billing_issue'

def test_baseline_trivial():
    res = classify_trivial("anything", "dummy_path_that_doesnt_exist.csv")
    assert res['intent'] == 'account_access' # fallback
    assert res['confidence'] == 1.0
