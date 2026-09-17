import pandas as pd
import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_golden_set(holdout_path, golden_path, target_size=150):
    print(f"Loading holdout data from {holdout_path}...")
    try:
        df = pd.read_csv(holdout_path)
    except FileNotFoundError:
        print("Holdout data not found. Please run preprocess.py first.")
        # Create a mock golden set if running without real data
        df = pd.DataFrame({
            'customer_text': [
                "I can't log in.",
                "Where is my package?",
                "This app is terrible, I'm going to sue you!"
            ],
            'brand_text': [
                "Reset your password.",
                "Check tracking link.",
                "Please calm down."
            ],
            'resolved': [True, True, False]
        })
    
    # In a real scenario, we would stratify by something, but here we'll just random sample
    sample_size = min(target_size, len(df))
    if sample_size > 0:
        sampled = df.sample(n=sample_size, random_state=42).copy()
    else:
        sampled = df.copy()
        
    # Add empty columns for the candidate (human) to fill in
    sampled['true_intent'] = ""
    sampled['should_escalate'] = ""
    sampled['reference_note'] = ""
    
    # If we are using mock data, let's fill in a few rows so eval can run
    if len(sampled) <= 5 and 'resolved' in sampled.columns: 
        print("Using mock data, filling in mock labels for testing.")
        # Ensure we only assign up to the length of the sample
        mock_intents = ["account_access", "order_status", "general_inquiry", "billing_issue", "service_outage"]
        mock_escalates = [False, False, True, False, False]
        mock_notes = ["Provide reset link", "Provide tracking info", "Legal threat", "Check bill", "Service is down"]
        
        n = len(sampled)
        sampled['true_intent'] = mock_intents[:n]
        sampled['should_escalate'] = mock_escalates[:n]
        sampled['reference_note'] = mock_notes[:n]

    out_dir = Path(golden_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sampled.to_csv(golden_path, index=False)
    print(f"Golden set scaffold created at {golden_path} with {len(sampled)} rows.")
    print("ACTION REQUIRED: You must manually label the 'true_intent', 'should_escalate', and 'reference_note' columns in this CSV.")

if __name__ == "__main__":
    config = load_config()
    # Assuming holdout is saved alongside processed data
    holdout_path = Path(config['data']['processed_path']).parent / "holdout.csv"
    build_golden_set(holdout_path, config['data']['golden_path'])
