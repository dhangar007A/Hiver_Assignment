import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

def calculate_agreement(human_scores_path, llm_scores_path):
    """
    Calculates agreement between human-provided scores and LLM-judge scores.
    The candidate would create a CSV with their own manual 1-5 ratings for a subset.
    """
    try:
        human_df = pd.read_csv(human_scores_path)
        llm_df = pd.read_csv(llm_scores_path)
    except FileNotFoundError:
        print("Scores not found. Please provide human scores and run LLM judge first.")
        return {}

    # Merge on a common ID or text
    merged = pd.merge(human_df, llm_df, on='id', suffixes=('_human', '_llm'))
    
    dimensions = ['groundedness', 'helpfulness', 'tone', 'actionability', 'safety']
    agreement_results = {}
    
    for dim in dimensions:
        h_scores = merged[f'{dim}_human']
        l_scores = merged[f'{dim}_llm']
        
        # Exact match
        exact_match = accuracy_score(h_scores, l_scores)
        
        # Within 1 point match
        within_1 = (abs(h_scores - l_scores) <= 1).mean()
        
        # Correlation
        corr, _ = spearmanr(h_scores, l_scores)
        
        agreement_results[dim] = {
            "exact_match": float(exact_match),
            "within_1_point": float(within_1),
            "spearman_corr": float(corr) if not pd.isna(corr) else 0.0
        }
        
    return agreement_results

if __name__ == "__main__":
    # Mocking for standalone run
    print("Run this script pointing to real score CSVs to compute agreement.")
