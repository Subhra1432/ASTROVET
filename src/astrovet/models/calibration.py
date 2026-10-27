"""
Probability calibration and reliability estimation.

Scientific safeguards:
- Fitted strictly on the group-disjoint calibration partition.
- Computes Expected Calibration Error (ECE) and Brier Score.
- Never evaluates calibration on training data.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """
    Logistic probability calibrator (Platt scaling).
    """

    def __init__(self) -> None:
        self.calibrator = LogisticRegression(C=1.0)
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        """Fit calibration on held-out calibration set."""
        # Logit transform
        eps = 1e-6
        clipped = np.clip(raw_probs, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        self.calibrator.fit(logits, y_true)
        self.is_fitted = True
        return self

    def calibrate(self, raw_prob: float) -> float:
        """Calibrate a single probability value."""
        if not self.is_fitted:
            return raw_prob
        eps = 1e-6
        clipped = np.clip(raw_prob, eps, 1.0 - eps)
        logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return float(self.calibrator.predict_proba(logit)[0, 1])

    def calibrate_batch(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return raw_probs
        eps = 1e-6
        clipped = np.clip(raw_probs, eps, 1.0 - eps)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculate Brier score and Expected Calibration Error (ECE).
    """
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.clip(np.array(y_prob, dtype=np.float64), 0.0, 1.0)

    # 1. Brier score: mean squared error of probability predictions
    brier = float(np.mean((y_p - y_t)**2))

    # 2. Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_weights = []

    ece = 0.0
    n_total = len(y_t)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_p >= low) & (y_p < high) if i < n_bins - 1 else (y_p >= low) & (y_p <= high)
        n_in_bin = int(np.sum(mask))
        if n_in_bin > 0:
            bin_acc = float(np.mean(y_t[mask]))
            bin_conf = float(np.mean(y_p[mask]))
            weight = n_in_bin / n_total
            ece += weight * np.abs(bin_acc - bin_conf)

            bin_centers.append(float(0.5 * (low + high)))
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_weights.append(weight)

    return {
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(float(ece), 4),
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_weights": bin_weights,
    }
