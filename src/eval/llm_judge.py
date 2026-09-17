import json
from src.llm_client import LLMClient

def evaluate_reply(customer_msg, agent_reply, reference_note, provider="mock"):
    schema = {
        "type": "object",
        "properties": {
            "groundedness": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Consistent with retrieved precedent, no invented facts."},
            "helpfulness": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Addresses the customer's real issue."},
            "tone": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Polite and professional brand voice."},
            "actionability": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Clear next step for the customer."},
            "safety": {"type": "integer", "minimum": 1, "maximum": 5, "description": "No inappropriate promises or PII leakage."},
            "overall": {"type": "number", "minimum": 1, "maximum": 5},
            "rationale": {"type": "string"}
        },
        "required": ["groundedness", "helpfulness", "tone", "actionability", "safety", "overall", "rationale"]
    }
    
    system = "You are an expert customer support evaluator. Score the agent's reply on a 1-5 scale for the provided dimensions."
    user = f"Customer Message: {customer_msg}\nAgent Reply: {agent_reply}\nReference Notes: {reference_note}\n\nEvaluate the reply."
    
    client = LLMClient(provider=provider)
    result = client.complete(system=system, user=user, json_schema=schema, temperature=0.0)
    
    if isinstance(result, str):
        # Fallback
        result = {
            "groundedness": 3, "helpfulness": 3, "tone": 3, "actionability": 3, "safety": 3,
            "overall": 3.0, "rationale": "Fallback mock evaluation."
        }
    return result
