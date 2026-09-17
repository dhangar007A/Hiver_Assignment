import pandas as pd
import yaml

def load_taxonomy(taxonomy_path="config/taxonomy.yaml"):
    with open(taxonomy_path, "r") as f:
        return yaml.safe_load(f)["intents"]

def classify_trivial(message, training_data_path="data/golden/golden_set.csv"):
    """
    Trivial baseline: always predict the single most frequent intent class 
    from the training data. Ignores the message entirely.
    """
    # For demonstration, we hardcode the most frequent class if the file doesn't exist
    try:
        df = pd.read_csv(training_data_path)
        most_frequent = df['true_intent'].mode()[0]
    except (FileNotFoundError, KeyError):
        most_frequent = "account_access" # Default fallback
        
    return {
        "intent": most_frequent,
        "confidence": 1.0,
        "rationale": "Trivial baseline always predicts the most frequent class."
    }

if __name__ == "__main__":
    msg = "My internet is down."
    result = classify_trivial(msg)
    print(f"Trivial Baseline Result: {result}")
