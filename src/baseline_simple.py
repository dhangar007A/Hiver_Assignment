import re

def classify_simple(message):
    """
    Simple baseline: keyword/regex rules mapping surface tokens to intents.
    First-match-wins with a default fallback.
    """
    message = message.lower()
    
    rules = {
        "account_access": [r'log\s*in', r'password', r'account', r'lock'],
        "billing_issue": [r'charg', r'bill', r'pay', r'card', r'refund'],
        "service_outage": [r'down', r'outage', r'crash', r'not work', r'internet'],
        "order_status": [r'track', r'ship', r'deliver', r'where is my'],
    }
    
    for intent, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, message):
                return {
                    "intent": intent,
                    "confidence": 0.8,
                    "rationale": f"Matched simple keyword rule: '{pattern}'"
                }
                
    return {
        "intent": "general_inquiry", # Fallback
        "confidence": 0.5,
        "rationale": "No keyword rules matched. Used fallback."
    }

if __name__ == "__main__":
    msg = "Why was my card billed twice?"
    result = classify_simple(msg)
    print(f"Simple Baseline Result: {result}")
