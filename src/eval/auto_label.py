import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.llm_client import LLMClient

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_taxonomy():
    with open("config/taxonomy.yaml", "r") as f:
        return yaml.safe_load(f)

def process_row(idx, row, intent_names):
    client = LLMClient() # each thread gets its own client just in case
    text = row['customer_text']
    system_prompt = f"""You are an expert customer support quality assurance analyst. 
Analyze the following customer message and determine:
1. true_intent: Must be exactly one of {intent_names}
2. should_escalate: true or false (true if they are very angry, threatening legal action, or explicitly demanding a human/supervisor)
3. reference_note: A short 1 sentence summary of what the customer is asking.

Respond ONLY with a valid JSON object."""

    schema = {
        "type": "object",
        "properties": {
            "true_intent": {"type": "string", "enum": intent_names},
            "should_escalate": {"type": "boolean"},
            "reference_note": {"type": "string"}
        },
        "required": ["true_intent", "should_escalate", "reference_note"]
    }
    
    try:
        response = client.complete(system_prompt, text, json_schema=schema)
        if isinstance(response, dict):
            return idx, response.get('true_intent', 'general_inquiry'), response.get('should_escalate', False), response.get('reference_note', 'Auto-labeled note')
        else:
            return idx, 'general_inquiry', False, 'Failed to parse JSON'
    except Exception as e:
        return idx, 'general_inquiry', False, f"Error: {e}"

def auto_label_golden_set():
    config = load_config()
    taxonomy = load_taxonomy()
    intent_names = [i['name'] for i in taxonomy['intents']]
    
    golden_path_out = config['data']['golden_path']
    df = pd.read_csv("data/golden/golden_set.csv")
    
    # Pre-cast columns to object to avoid dtype warnings
    df['true_intent'] = df.get('true_intent', pd.Series(dtype='object')).astype(object)
    df['should_escalate'] = df.get('should_escalate', pd.Series(dtype='object')).astype(object)
    df['reference_note'] = df.get('reference_note', pd.Series(dtype='object')).astype(object)
    
    print(f"Auto-labeling {len(df)} rows concurrently...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, idx, row, intent_names): idx for idx, row in df.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, intent, escalate, note = future.result()
            df.at[idx, 'true_intent'] = intent
            df.at[idx, 'should_escalate'] = escalate
            df.at[idx, 'reference_note'] = note
            
    df.to_csv(golden_path_out, index=False)
    print(f"Auto-labeling complete! Saved to {golden_path_out}")

if __name__ == "__main__":
    auto_label_golden_set()
