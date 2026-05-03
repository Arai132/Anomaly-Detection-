import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    result = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "n_anomalies_detected": int(y_pred.sum()),
        "n_anomalies_actual": int(y_true.sum()),
    }
    if scores is not None:
        try:
            result["roc_auc"] = round(roc_auc_score(y_true, scores), 4)
            result["avg_precision"] = round(average_precision_score(y_true, scores), 4)
        except ValueError:
            pass
    return result


def compare_detectors(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(results).T.sort_values("f1", ascending=False)
