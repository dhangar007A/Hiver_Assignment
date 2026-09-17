import pytest
from src.escalation import decide_escalation, decide_escalation_simple

def test_escalation_simple_baseline():
    res = decide_escalation_simple("I will sue you!")
    assert res['should_escalate'] is True
    
    res2 = decide_escalation_simple("How do I log in?")
    assert res2['should_escalate'] is False

def test_escalation_agent():
    intent_mock = {"intent": "billing_issue", "confidence": 0.4, "rationale": "mock"}
    sim_mock = 0.8
    # Should escalate due to low confidence (default threshold 0.7)
    res = decide_escalation("How do I log in?", intent_mock, sim_mock)
    assert res['should_escalate'] is True
    assert "Low intent confidence" in res['reason']
