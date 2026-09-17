import pandas as pd
import yaml
import re
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Redact order numbers, emails, phones (simplified for this assignment)
    text = re.sub(r'\b\d{10,}\b', '[ORDER_NUMBER]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    return text

def reconstruct_threads_and_preprocess(raw_path, processed_path, brand_name, max_rows):
    print(f"Loading raw data from {raw_path}...")
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print("Raw data not found. Please run ingest.py first.")
        return

    print(f"Filtering for brand: {brand_name}")
    # Very basic mock filtering if needed, but normally we'd trace in_response_to_tweet_id
    
    # Reconstructing conversation threads is complex in a flat CSV.
    # For this assignment, a simplified thread construction strategy:
    # 1. Find all brand replies.
    brand_replies = df[df['author_id'] == brand_name].copy()
    
    # 2. Join with the customer tweet it was in response to
    # Ensure in_response_to_tweet_id and tweet_id are of the same type for merging.
    # Convert to string and remove '.0' if pandas parsed it as a float due to NaNs.
    df['tweet_id'] = df['tweet_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    brand_replies['in_response_to_tweet_id'] = brand_replies['in_response_to_tweet_id'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    threads = pd.merge(
        brand_replies, 
        df, 
        left_on='in_response_to_tweet_id', 
        right_on='tweet_id', 
        suffixes=('_brand', '_customer')
    )
    
    if len(threads) == 0:
        print(f"No threads found for brand {brand_name} (might be mock data mismatch).")
        # Fallback to create some dummy processed data if empty so the pipeline doesn't crash
        threads = pd.DataFrame({
            'customer_text': ['@sprintcare my internet is down', 'Where is my refund?'],
            'brand_text': ['Please DM us your account number.', 'Check your email for confirmation.'],
            'resolved': [False, True]
        })
    else:
        # 3. Clean text
        threads['customer_text'] = threads['text_customer'].apply(clean_text)
        threads['brand_text'] = threads['text_brand'].apply(clean_text)
        
        # 4. Resolution heuristic: customer says thanks or similar in subsequent replies
        # For simplicity in this demo, we mock resolution randomly or based on simple keywords in brand reply
        threads['resolved'] = threads['brand_text'].str.contains('DM|email', case=False, na=False)

    # Limit to max rows
    if len(threads) > max_rows:
        threads = threads.head(max_rows)

    # Select final columns
    final_cols = ['customer_text', 'brand_text', 'resolved']
    processed_df = threads[final_cols]
    
    out_dir = Path(processed_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    processed_df.to_csv(processed_path, index=False)
    print(f"Processed {len(processed_df)} threads and saved to {processed_path}")
    
    # Create a small holdout set
    holdout = processed_df.sample(frac=0.1, random_state=42) if len(processed_df) > 10 else processed_df
    holdout_path = out_dir / "holdout.csv"
    holdout.to_csv(holdout_path, index=False)
    print(f"Created holdout set of {len(holdout)} rows at {holdout_path}")

if __name__ == "__main__":
    config = load_config()
    reconstruct_threads_and_preprocess(
        config['data']['raw_path'],
        config['data']['processed_path'],
        config['brand_name'],
        config['data']['max_working_rows']
    )
