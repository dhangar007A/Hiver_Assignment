import pandas as pd
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def analyze_brands(raw_path):
    """Analyzes the raw dataset to identify top brands by volume and potential resolution rate."""
    print(f"Loading data from {raw_path}...")
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"File {raw_path} not found. Please run ingest.py first.")
        return

    # Identify brand accounts (where inbound is False)
    brand_replies = df[df['inbound'] == False]
    brand_volumes = brand_replies['author_id'].value_counts()
    
    print("\nTop 10 brands by response volume:")
    print(brand_volumes.head(10))

    # Basic analysis for a specific brand (e.g., how many unique threads)
    # This is a simplified version; real analysis would trace response chains
    
    print("\nRecommendation: 'sprintcare', 'AmazonHelp', or 'AppleSupport' are usually good candidates due to high volume.")
    
if __name__ == "__main__":
    config = load_config()
    raw_path = config['data']['raw_path']
    analyze_brands(raw_path)
