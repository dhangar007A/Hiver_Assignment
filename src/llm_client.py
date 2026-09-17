import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()

class LLMClient:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or os.getenv("MODEL_PROVIDER", "openai")
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.provider == "nvidia":
            import openai
            self.client = openai.OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY")
            )
        elif self.provider == "mock":
            self.client = None
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_cache_key(self, system: str, user: str, json_schema: dict = None) -> str:
        cache_data = json.dumps({"system": system, "user": user, "schema": json_schema}, sort_keys=True)
        return hashlib.md5(cache_data.encode()).hexdigest()

    def complete(self, system: str, user: str, json_schema: dict = None, temperature: float = 0.0) -> dict | str:
        cache_key = self._get_cache_key(system, user, json_schema)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache first
        if cache_file.exists():
            with open(cache_file, "r") as f:
                cached_resp = json.load(f)["response"]
                # If we expect a JSON dict but the cache contains a string, ignore cache!
                if json_schema and isinstance(cached_resp, str):
                    pass
                else:
                    return cached_resp

        print(f"Calling LLM ({self.provider} - {self.model})...")
        response = None
        
        if self.provider == "mock":
            # Very dumb mock logic
            if json_schema:
                # Assuming the schema requires an intent and confidence
                response = {"intent": "account_access", "confidence": 0.9, "rationale": "Mock response"}
            else:
                response = "Mock generated reply based on context."
        elif self.provider == "nvidia":
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": 0.95,
                "max_tokens": 1024,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
            }
            
            if json_schema:
                # ensure we prompt it to return json matching schema
                messages[0]["content"] += f"\nReturn a JSON object that strictly conforms to this schema:\n{json.dumps(json_schema)}"

            max_retries = 20
            for attempt in range(max_retries):
                try:
                    completion = self.client.chat.completions.create(**kwargs)
                    content = completion.choices[0].message.content
                    break
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        if attempt < max_retries - 1:
                            sleep_time = min(2 ** attempt, 60)
                            print(f"Rate limited. Retrying in {sleep_time}s...")
                            time.sleep(sleep_time)
                        else:
                            raise e
                    else:
                        raise e
            
            if json_schema:
                try:
                    start = content.find('{')
                    end = content.rfind('}')
                    if start != -1 and end != -1:
                        clean_content = content[start:end+1]
                    else:
                        clean_content = content.strip()
                    response = json.loads(clean_content)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse JSON from response. Returning fallback dict. Response: {content}")
                    response = {
                        "error": "parse_failed", 
                        "intent": "general_inquiry", 
                        "confidence": 0.0, 
                        "rationale": "Parsing failed", 
                        "should_escalate": False, 
                        "reason": "Parsing failed",
                        "reply": "I am sorry, but I am unable to assist with this request at the moment due to a system error."
                    }
            else:
                response = content

        # Save to cache
        with open(cache_file, "w") as f:
            json.dump({"response": response}, f)

        return response
