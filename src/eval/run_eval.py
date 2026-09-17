import pandas as pd
import yaml
import json
from pathlib import Path
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.classify_intent import classify_intent
from src.baseline_trivial import classify_trivial
from src.baseline_simple import classify_simple
from src.generate_reply import ReplyGenerator, generate_simple_baseline
from src.escalation import decide_escalation, decide_escalation_simple
from src.eval.metrics import calculate_intent_metrics, calculate_escalation_metrics
from src.eval.llm_judge import evaluate_reply

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_row(idx, row, provider, golden_path):
    generator = ReplyGenerator(provider=provider)
    msg = row['customer_text']
    true_intent = row['true_intent']
    true_escalate = row['should_escalate']
    ref_note = row['reference_note']
    
    # 1. Trivial Baseline
    trivial_intent = classify_trivial(msg, golden_path)
    trivial_escalate = {"should_escalate": False, "reason": "Trivial never escalates"}
    
    # 2. Simple Baseline
    simple_intent = classify_simple(msg)
    simple_reply = generate_simple_baseline(simple_intent['intent'])
    simple_escalate = decide_escalation_simple(msg)
    
    # 3. Agent (Full system)
    agent_intent = classify_intent(msg, provider=provider)
    agent_reply = generator.generate(msg, agent_intent['intent'])
    agent_escalate = decide_escalation(msg, agent_intent, agent_reply.get('retrieval_similarity', 0.0))
    
    # Run LLM Judge on replies
    simple_eval = evaluate_reply(msg, simple_reply['reply'], ref_note, provider=provider)
    agent_eval = evaluate_reply(msg, agent_reply['reply'], ref_note, provider=provider)
    
    return {
        "msg": msg,
        "true_intent": true_intent,
        "true_escalate": true_escalate,
        "trivial_intent_pred": trivial_intent['intent'],
        "trivial_escalate_pred": trivial_escalate['should_escalate'],
        "simple_intent_pred": simple_intent['intent'],
        "simple_escalate_pred": simple_escalate['should_escalate'],
        "simple_reply_eval": simple_eval,
        "agent_intent_pred": agent_intent['intent'],
        "agent_escalate_pred": agent_escalate['should_escalate'],
        "agent_reply_eval": agent_eval,
        "agent_reply_text": agent_reply['reply']
    }

def run_evaluation(config):
    golden_path = config['data']['golden_path']
    try:
        df = pd.read_csv(golden_path)
    except FileNotFoundError:
        print("Golden set not found.")
        return
        
    provider = config['model']['provider']
    
    results = []
    print(f"Running evaluation on {len(df)} rows concurrently...")
    
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_row, idx, row, provider, golden_path) for idx, row in df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
            
    # Aggregate Metrics
    res_df = pd.DataFrame(results)
    
    metrics = {
        "intent_accuracy": {
            "trivial": calculate_intent_metrics(res_df['true_intent'], res_df['trivial_intent_pred']),
            "simple": calculate_intent_metrics(res_df['true_intent'], res_df['simple_intent_pred']),
            "agent": calculate_intent_metrics(res_df['true_intent'], res_df['agent_intent_pred']),
        },
        "escalation": {
            "trivial": calculate_escalation_metrics(res_df['true_escalate'], res_df['trivial_escalate_pred']),
            "simple": calculate_escalation_metrics(res_df['true_escalate'], res_df['simple_escalate_pred']),
            "agent": calculate_escalation_metrics(res_df['true_escalate'], res_df['agent_escalate_pred']),
        }
    }
    
    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    res_df.to_csv(out_dir / "detailed_results.csv", index=False)
    print("Evaluation complete. Results saved to results/")

if __name__ == "__main__":
    run_evaluation(load_config())
