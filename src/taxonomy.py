import pandas as pd
import yaml
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def derive_taxonomy(processed_path, sample_size=300):
    """
    Derives intents bottom-up using sentence embeddings and clustering.
    In practice, this script helps the candidate identify clusters, 
    which they then manually review and name to create config/taxonomy.yaml.
    """
    print(f"Loading data from {processed_path}...")
    try:
        df = pd.read_csv(processed_path)
    except FileNotFoundError:
        print("Processed data not found. Please run preprocess.py first.")
        return

    # Take a sample for clustering (excluding the holdout which isn't loaded here anyway)
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    texts = sample_df['customer_text'].tolist()

    if not texts:
        print("No texts to cluster.")
        return

    print("Embedding customer messages...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Clustering into candidate intents (k=6)...")
    num_clusters = min(6, len(texts))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    kmeans.fit(embeddings)
    
    sample_df['cluster'] = kmeans.labels_
    
    print("\n--- Cluster Samples ---")
    for i in range(num_clusters):
        print(f"\nCluster {i}:")
        cluster_texts = sample_df[sample_df['cluster'] == i]['customer_text'].head(5).tolist()
        for text in cluster_texts:
            print(f"  - {text}")
            
    print("\nNext step: Manually review these clusters and create `config/taxonomy.yaml` with 6-10 human-readable intents.")

if __name__ == "__main__":
    config = load_config()
    derive_taxonomy(config['data']['processed_path'])
