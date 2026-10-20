"""
Evaluation metrics calculation for ABSA:
Accuracy, Macro-F1, Weighted-F1, Per-class F1, Precision, Recall, and Confusion Matrix.
"""

from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    target_names: List[str] = None,
) -> Dict:
    """
    Computes complete classification metrics:
    - Accuracy
    - Macro-F1
    - Weighted-F1
    - Per-class precision, recall, f1, support
    - Confusion matrix
    """
    if target_names is None:
        target_names = [ID2LABEL[i] for i in range(3)]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(target_names))), zero_division=0
    )

    per_class = {}
    for idx, name in enumerate(target_names):
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(y_true)),
    }


def get_classification_report_text(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    target_names: List[str] = None,
) -> str:
    """Generates standard sklearn text classification report."""
    if target_names is None:
        target_names = [ID2LABEL[i] for i in range(3)]
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(target_names))),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
