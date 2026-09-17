import yaml
import json
from src.llm_client import LLMClient

def load_taxonomy(taxonomy_path="config/taxonomy.yaml"):
    with open(taxonomy_path, "r") as f:
        return yaml.safe_load(f)["intents"]

def build_system_prompt(taxonomy):
    prompt = "You are a customer support intent classifier. Categorize the incoming customer message into exactly one of the following intents:\n\n"
    for intent in taxonomy:
        prompt += f"- **{intent['name']}**: {intent['definition']}\n"
        prompt += f"  Examples: {', '.join(intent['examples'])}\n"
        prompt += f"  Not included: {intent['not_included']}\n\n"
    return prompt

def classify_intent(message, taxonomy_path="config/taxonomy.yaml", provider="mock"):
    taxonomy = load_taxonomy(taxonomy_path)
    system_prompt = build_system_prompt(taxonomy)
    
    schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [t["name"] for t in taxonomy]
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0"
            },
            "rationale": {
                "type": "string",
                "description": "Brief explanation of why this intent was chosen."
            }
        },
        "required": ["intent", "confidence", "rationale"]
    }
    
    client = LLMClient(provider=provider)
    result = client.complete(
        system=system_prompt,
        user=f"Classify the following message:\n\n{message}",
        json_schema=schema,
        temperature=0.0
    )
    
    return result

if __name__ == "__main__":
    msg = "I can't seem to login to my account, the password reset is broken."
    print(f"Message: {msg}")
    result = classify_intent(msg, provider="mock")
    print(json.dumps(result, indent=2))
