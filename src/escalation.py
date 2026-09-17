import yaml
import re

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def decide_escalation(message, intent_result, retrieval_similarity, config_path="config/config.yaml"):
    config = load_config(config_path)
    thresholds = config.get('escalation', {'confidence_threshold': 0.7, 'similarity_threshold': 0.5})
    
    reasons = []
    
    # Trigger 1: Low intent confidence
    if intent_result['confidence'] < thresholds['confidence_threshold']:
        reasons.append(f"Low intent confidence ({intent_result['confidence']:.2f})")
        
    # Trigger 2: Low retrieval similarity
    if retrieval_similarity < thresholds['similarity_threshold']:
        reasons.append(f"Low retrieval similarity ({retrieval_similarity:.2f})")
        
    # Trigger 3: High-risk intent
    # Assuming 'complaint' or 'legal' might be intents, but let's just check intent name
    # We didn't define a specific high-risk intent in taxonomy, but let's mock one
    high_risk_intents = ['legal_threat', 'safety_issue']
    if intent_result['intent'] in high_risk_intents:
        reasons.append(f"High-risk intent detected ({intent_result['intent']})")
        
    # Trigger 4: Negative sentiment / toxicity keywords
    # A simple keyword check for demo purposes
    angry_keywords = [r'\bsue\b', r'\blawyer\b', r'\bmanager\b', r'\bcancel\b', r'\bworst\b', r'\bterrible\b']
    message_lower = message.lower()
    for pattern in angry_keywords:
        if re.search(pattern, message_lower):
            reasons.append("High-risk or angry keyword detected")
            break
            
    # Combine into decision
    should_escalate = len(reasons) > 0
    reason_string = " | ".join(reasons) if should_escalate else "Handled autonomously; no escalation triggers met."
    
    return {
        "should_escalate": should_escalate,
        "reason": reason_string
    }

def decide_escalation_simple(message):
    """Simple baseline counterpart: pure keyword-trigger escalation."""
    angry_keywords = [r'\bsue\b', r'\blawyer\b', r'\bmanager\b', r'\bcancel\b']
    message_lower = message.lower()
    
    for pattern in angry_keywords:
        if re.search(pattern, message_lower):
            return {
                "should_escalate": True,
                "reason": f"Matched baseline angry keyword rule: {pattern}"
            }
            
    return {
        "should_escalate": False,
        "reason": "No baseline escalation triggers met."
    }

if __name__ == "__main__":
    msg = "This is terrible, I want to talk to a manager now!"
    intent_mock = {"intent": "billing_issue", "confidence": 0.8, "rationale": "mock"}
    sim_mock = 0.6
    
    result = decide_escalation(msg, intent_mock, sim_mock)
    print(f"Full Agent Escalation: {result}")
    
    simple = decide_escalation_simple(msg)
    print(f"Simple Baseline Escalation: {simple}")
