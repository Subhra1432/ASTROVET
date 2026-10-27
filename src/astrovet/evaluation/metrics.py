"""
Evaluation metrics: PR-AUC, ROC-AUC, Recall@Precision, Precision@K, and Calibration.

Scientific safeguards:
- Metrics reported with raw sample counts, positive/negative class balance.
- Precision@K models realistic human reviewer capacity.
- Evaluated strictly on the untouched test partition.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from astrovet.models.calibration import compute_calibration_metrics


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_targets: Tuple[int, ...] = (10, 25, 50),
    fixed_precisions: Tuple[float, ...] = (0.80, 0.90, 0.95),
) -> Dict[str, Any]:
    """
    Calculate full evaluation suite on a held-out test partition.
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.clip(np.array(y_prob, dtype=float), 0.0, 1.0)

    n_samples = len(y_t)
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))

    if n_pos == 0 or n_neg == 0:
        return {
            "error": "Single class present in evaluation set; AUC cannot be computed.",
            "total_samples": n_samples,
            "positive_count": n_pos,
            "negative_count": n_neg,
        }

    # 1. Ranking metrics
    roc_auc = float(roc_auc_score(y_t, y_p))
    pr_auc = float(average_precision_score(y_t, y_p))

    # 2. Precision-Recall curve & Recall at fixed precision
    precisions, recalls, thresholds = precision_recall_curve(y_t, y_p)
    recall_at_precision = {}
    for target_p in fixed_precisions:
        # Find highest recall where precision >= target_p
        valid_idx = np.where(precisions >= target_p)[0]
        if len(valid_idx) > 0:
            best_rec = float(np.max(recalls[valid_idx]))
        else:
            best_rec = 0.0
        recall_at_precision[f"recall_at_p{int(target_p*100)}"] = round(best_rec, 4)

    # 3. Precision@K (reviewer capacity)
    sort_order = np.argsort(y_p)[::-1]
    sorted_y_t = y_t[sort_order]
    precision_at_k = {}
    for k in k_targets:
        if k <= n_samples:
            p_k = float(np.mean(sorted_y_t[:k]))
            precision_at_k[f"precision_at_{k}"] = round(p_k, 4)

    # 4. Calibration metrics
    cal_metrics = compute_calibration_metrics(y_t, y_p)

    return {
        "sample_counts": {
            "total": n_samples,
            "positive": n_pos,
            "negative": n_neg,
            "prevalence": round(n_pos / max(1, n_samples), 4),
        },
        "ranking": {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "recall_at_fixed_precision": recall_at_precision,
            "precision_at_k": precision_at_k,
        },
        "calibration": {
            "brier_score": cal_metrics["brier_score"],
            "expected_calibration_error": cal_metrics["expected_calibration_error"],
        },
    }
