import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from src.llm_client import LLMClient

class ReplyGenerator:
    def __init__(self, index_dir="data/index", provider="mock"):
        self.provider = provider
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            self.index = faiss.read_index(f"{index_dir}/resolved_issues.index")
            with open(f"{index_dir}/metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load index ({e}). RAG will not work.")
            self.index = None
            self.metadata = []

    def retrieve(self, message, k=3):
        if self.index is None or len(self.metadata) == 0:
            return [], 0.0
            
        query_vector = self.model.encode([message]).astype('float32')
        distances, indices = self.index.search(query_vector, min(k, len(self.metadata)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        # return results and a similarity score (inverse of L2 distance, normalized vaguely)
        avg_dist = np.mean(distances[0]) if len(distances[0]) > 0 else 100.0
        similarity_score = max(0.0, 1.0 - (avg_dist / 10.0)) 
        
        return results, similarity_score

    def generate(self, message, intent):
        retrieved_examples, similarity = self.retrieve(message)
        
        system_prompt = """You are a helpful customer support agent for the brand. 
Draft a reply to the customer's message based ONLY on the provided past resolved examples.
CRITICAL INSTRUCTION: Do NOT fabricate order numbers, refund amounts, policy specifics, or promises not evidenced in the retrieved context.
Maintain the brand's polite and helpful tone.
"""
        
        context_str = "Past resolved examples (for style and approach reference only):\n"
        for i, ex in enumerate(retrieved_examples):
            context_str += f"Example {i+1}:\nCustomer: {ex['customer_text']}\nAgent: {ex['brand_text']}\n\n"
            
        user_prompt = f"{context_str}\nCurrent Customer Message: {message}\nPredicted Intent: {intent}\n\nDraft the agent's reply."
        
        schema = {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": "The drafted reply to the customer."
                },
                "grounding_used": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of example indices (1, 2, etc.) that informed the reply. Empty if none."
                }
            },
            "required": ["reply", "grounding_used"]
        }
        
        client = LLMClient(provider=self.provider)
        result = client.complete(
            system=system_prompt,
            user=user_prompt,
            json_schema=schema,
            temperature=0.0
        )
        
        if isinstance(result, str):
            # Fallback if json parsing failed in client
            result = {"reply": result, "grounding_used": []}
            
        result["retrieval_similarity"] = float(similarity)
        return result

def generate_simple_baseline(intent):
    """Simple baseline counterpart: per-intent fill-in-the-blank templates."""
    templates = {
        "account_access": "I'm sorry you're having trouble logging in. Please reset your password here: [LINK]",
        "billing_issue": "We can help with your billing issue. Please DM us your account number.",
        "service_outage": "We are currently experiencing an outage and are working to resolve it. Thanks for your patience.",
        "order_status": "You can track your order here: [TRACKING_LINK]",
        "general_inquiry": "Thanks for your question! Please visit our FAQ for more details: [FAQ_LINK]"
    }
    reply = templates.get(intent, "Thanks for reaching out. A representative will be with you shortly.")
    return {"reply": reply, "grounding_used": []}

if __name__ == "__main__":
    generator = ReplyGenerator(provider="mock")
    msg = "Where is my refund for the returned item?"
    res = generator.generate(msg, "billing_issue")
    print(f"RAG Result: {json.dumps(res, indent=2)}")
    print(f"Simple Baseline: {generate_simple_baseline('billing_issue')}")
