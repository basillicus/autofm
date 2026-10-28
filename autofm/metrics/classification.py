"""Classification metrics for the autofm FM pipeline.

Fault forecasting is heavily class-imbalanced (most windows are Normal), so
plain accuracy is misleading. These functions compute the metrics that matter
for an operator-facing fault predictor and return plain, JSON-serializable
dicts.

Two families:

* **Multi-class** (the fault-type task over ``CLASS_NAMES``): macro-F1,
  weighted-F1, balanced accuracy, per-class precision/recall/F1/support, and
  the confusion matrix.
* **Binary fault detection** (``Normal`` vs. *any* fault): precision, recall,
  F1, ROC-AUC and PR-AUC (average precision). PR-AUC is the headline under
  imbalance - it answers "when we raise a fault alert, how trustworthy is it,
  and how many real faults do we catch".

If predicted probabilities are supplied, threshold-free ranking metrics
(ROC-AUC / PR-AUC) and a Brier score are added; without them, only the
threshold metrics are computed so nothing hard-fails.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Class id 0 is Normal (see autofm.data.labels.CLASS_NAMES); everything else is
# some kind of fault.
NORMAL_CLASS_ID = 0


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Per-class and aggregate multi-class metrics for the fault-type task."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    if len(y_true) == 0:
        # Empty split (e.g. smoke scale): return a well-formed zero bundle.
        empty_cm = [[0] * len(labels) for _ in labels]
        return {
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "balanced_accuracy": 0.0,
            "per_class": {class_names[i]: {"precision": 0.0, "recall": 0.0,
                                           "f1": 0.0, "support": 0}
                          for i in labels},
            "confusion_matrix": empty_cm,
        }

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    per_class = {
        class_names[i]: {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in labels
    }
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         labels=labels, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted",
                                            labels=labels, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def binary_fault_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    *,
    normal_id: int = NORMAL_CLASS_ID,
    n_classes: int | None = None,
) -> dict:
    """Fault-vs-Normal detection metrics (the operational "did we catch it").

    Args:
        y_true: true multi-class labels.
        y_pred: predicted multi-class labels.
        proba:  optional (N, K) class probabilities; when given, a fault score
                ``1 - p[normal]`` drives threshold-free ROC-AUC / PR-AUC and the
                Brier score.
        normal_id: the Normal class id (everything else counts as a fault).
        n_classes: expected number of columns in ``proba``; when given, a
                proba whose width differs is rejected (skips AUCs) so a
                mis-shaped array is never scored on the wrong column.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "positive_rate_true": 0.0, "positive_rate_pred": 0.0}

    true_fault = (y_true != normal_id).astype(int)
    pred_fault = (y_pred != normal_id).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        true_fault, pred_fault, average="binary", zero_division=0,
    )
    out: dict = {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),       # fraction of real faults caught
        "f1": round(float(f1), 4),
        "positive_rate_true": round(float(true_fault.mean()), 4),
        "positive_rate_pred": round(float(pred_fault.mean()), 4),
    }

    # Ranking metrics need a continuous score, both classes present, and a
    # correctly-shaped proba.
    proba_ok = False
    if proba is not None:
        proba = np.asarray(proba)
        proba_ok = (
            proba.ndim == 2
            and proba.shape[0] == len(y_true)
            and normal_id < proba.shape[1]
            and (n_classes is None or proba.shape[1] == n_classes)
        )
    if proba_ok and true_fault.min() != true_fault.max():
        fault_score = 1.0 - proba[:, normal_id]
        out["roc_auc"] = round(float(roc_auc_score(true_fault, fault_score)), 4)
        out["pr_auc"] = round(float(average_precision_score(true_fault, fault_score)), 4)
        out["brier"] = round(float(brier_score_loss(true_fault, fault_score)), 4)

    return out


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    proba: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> dict:
    """Full metric bundle: multi-class + binary fault detection (+ uncertainty)."""
    metrics = {
        "multiclass": multiclass_metrics(y_true, y_pred, class_names),
        "binary_fault": binary_fault_metrics(
            y_true, y_pred, proba, n_classes=len(class_names)),
        "n_samples": int(len(y_true)),
    }
    if uncertainty is not None and len(uncertainty):
        metrics["mean_uncertainty"] = round(float(np.mean(uncertainty)), 4)
    return metrics
