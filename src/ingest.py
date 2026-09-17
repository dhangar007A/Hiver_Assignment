import os
import zipfile
import pandas as pd
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_and_extract_kaggle_dataset(raw_path):
    # This requires Kaggle API credentials to be set up (~/.kaggle/kaggle.json or env vars)
    dataset_name = "thoughtvector/customer-support-on-twitter"
    data_dir = Path(raw_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / "customer-support-on-twitter.zip"
    
    if not Path(raw_path).exists():
        print(f"Downloading dataset {dataset_name} from Kaggle...")
        # Note: In a real environment, kaggle API must be configured
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset_name, path=str(data_dir), unzip=True)
            print("Download and extraction complete.")
        except Exception as e:
            print(f"Failed to download from Kaggle: {e}")
            print("Creating a mock dataset for testing purposes...")
            create_mock_dataset(raw_path)
    else:
        print(f"Dataset already exists at {raw_path}")

def create_mock_dataset(raw_path):
    """Creates a small mock dataset for testing when Kaggle API is unavailable."""
    mock_data = [
        {"tweet_id": 1, "author_id": "customer1", "inbound": True, "created_at": "2017-10-31 22:10:47", "text": "@sprintcare my internet is down again", "response_tweet_id": "2", "in_response_to_tweet_id": pd.NA},
        {"tweet_id": 2, "author_id": "sprintcare", "inbound": False, "created_at": "2017-10-31 22:12:00", "text": "@customer1 I'm sorry to hear that. Can you DM us your account number?", "response_tweet_id": "3", "in_response_to_tweet_id": "1"},
        {"tweet_id": 3, "author_id": "customer1", "inbound": True, "created_at": "2017-10-31 22:15:00", "text": "@sprintcare sure, sent.", "response_tweet_id": pd.NA, "in_response_to_tweet_id": "2"},
        # Add a resolved thread for AmazonHelp
        {"tweet_id": 4, "author_id": "customer2", "inbound": True, "created_at": "2017-11-01 10:00:00", "text": "@AmazonHelp Where is my refund?", "response_tweet_id": "5", "in_response_to_tweet_id": pd.NA},
        {"tweet_id": 5, "author_id": "AmazonHelp", "inbound": False, "created_at": "2017-11-01 10:05:00", "text": "@customer2 Please check your email for the refund confirmation.", "response_tweet_id": "6", "in_response_to_tweet_id": "4"},
        {"tweet_id": 6, "author_id": "customer2", "inbound": True, "created_at": "2017-11-01 10:10:00", "text": "@AmazonHelp Thanks, got it!", "response_tweet_id": pd.NA, "in_response_to_tweet_id": "5"},
    ]
    df = pd.DataFrame(mock_data)
    df.to_csv(raw_path, index=False)
    print(f"Mock dataset created at {raw_path}")

if __name__ == "__main__":
    config = load_config()
    raw_path = config['data']['raw_path']
    download_and_extract_kaggle_dataset(raw_path)
