import pandas as pd
import numpy as np
import faiss
import pickle
import yaml
from sentence_transformers import SentenceTransformer
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_index(processed_path, index_dir="data/index"):
    print(f"Loading data from {processed_path}...")
    try:
        df = pd.read_csv(processed_path)
    except FileNotFoundError:
        print("Processed data not found. Please run preprocess.py first.")
        return

    # Filter to only resolved threads for the index
    resolved_df = df[df['resolved'] == True].copy()
    if len(resolved_df) == 0:
        print("No resolved threads found in the dataset to build the index.")
        # Fallback to create a dummy index for pipeline testing
        resolved_df = pd.DataFrame({
            'customer_text': ['Where is my refund?'],
            'brand_text': ['Please check your email for the refund confirmation.']
        })
        
    texts = resolved_df['customer_text'].tolist()
    
    print("Embedding customer messages for the index...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    print("Building FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    Path(index_dir).mkdir(parents=True, exist_ok=True)
    
    # Save index
    faiss.write_index(index, f"{index_dir}/resolved_issues.index")
    
    # Save metadata (the actual threads to retrieve)
    metadata = resolved_df[['customer_text', 'brand_text']].to_dict('records')
    with open(f"{index_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    print(f"Index built and saved to {index_dir} with {len(metadata)} items.")

if __name__ == "__main__":
    config = load_config()
    build_index(config['data']['processed_path'])
