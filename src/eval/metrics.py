from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd

def calculate_intent_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist()
    }

def calculate_escalation_metrics(y_true, y_pred):
    # Convert bools or strings to strict bools
    y_true_bool = [str(x).lower() == 'true' for x in y_true]
    y_pred_bool = [str(x).lower() == 'true' for x in y_pred]
    
    precision = precision_score(y_true_bool, y_pred_bool, zero_division=0)
    recall = recall_score(y_true_bool, y_pred_bool, zero_division=0)
    f1 = f1_score(y_true_bool, y_pred_bool, zero_division=0)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
