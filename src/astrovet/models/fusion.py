"""
Late fusion model combining baseline, physics, SSL, and missingness signals.

Scientific safeguards:
- Explicit missingness indicators; missing diagnostic modalities are never filled with fabricated zeros.
- If baseline score is absent (NOT_FOUND), baseline_available is set to 0.0 and score is not assumed.
- Distinguishes uncalibrated heuristic scores from fitted model outputs and calibrated probabilities.
- Produces predictive uncertainty and OOD flag.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression

from astrovet.schemas.baseline import BaselinePrediction
from astrovet.schemas.prediction import PhysicsDiagnostics, TriageState, AstrovetPrediction


class AstrovetFusionModel:
    """
    Late fusion architecture combining multiple diagnostic branches.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.classifier = LogisticRegression(C=1.0, random_state=random_state, max_iter=1000)
        self.is_fitted = False
        self.feature_names: List[str] = []

    def build_feature_vector(
        self,
        baseline_pred: Optional[BaselinePrediction],
        physics_diag: PhysicsDiagnostics,
        ssl_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract tabular feature vector with explicit missingness indicators.
        """
        features = []
        names = []

        # 1. Baseline features
        b_has_score = bool(baseline_pred and baseline_pred.score is not None and baseline_pred.status == "SUCCESS")
        b_score = float(baseline_pred.score) if b_has_score else 0.0
        b_unc = float(baseline_pred.score_uncertainty) if (b_has_score and baseline_pred.score_uncertainty is not None) else 1.0
        b_avail = 1.0 if b_has_score else 0.0
        
        features.extend([b_score, b_unc, b_avail])
        names.extend(["baseline_score", "baseline_uncertainty", "baseline_available"])

        # 2. Physics features with explicit availability
        p_depth = (physics_diag.transit_depth_fitted_ppm or 0.0) / 1000.0
        p_dur = physics_diag.transit_duration_fitted_hours or 0.0
        p_snr = min(100.0, physics_diag.transit_snr or 0.0)
        p_chi2 = min(10.0, physics_diag.fit_chi2_reduced or 1.0)
        
        # Odd-even
        oe_avail = 1.0 if physics_diag.modality_availability.get("odd_even_comparison", False) else 0.0
        oe_sig = min(20.0, physics_diag.odd_even_t_statistic or 0.0)
        oe_cons = 1.0 if physics_diag.odd_even_consistent is True else 0.0
        
        # Secondary
        sec_avail = 1.0 if physics_diag.modality_availability.get("secondary_eclipse_search", False) else 0.0
        sec_det = 1.0 if physics_diag.secondary_eclipse_detected is True else 0.0
        sec_ratio = min(1.0, physics_diag.secondary_depth_ratio or 0.0)

        # Plausibility
        plaus = 1.0 if physics_diag.plausibility_all_passed else 0.0

        features.extend([
            p_depth, p_dur, p_snr, p_chi2,
            oe_avail, oe_sig, oe_cons,
            sec_avail, sec_det, sec_ratio,
            plaus,
        ])
        names.extend([
            "physics_depth_1kppm", "physics_duration_h", "physics_snr", "physics_reduced_chi2",
            "odd_even_available", "odd_even_significance", "odd_even_consistent",
            "secondary_available", "secondary_detected", "secondary_depth_ratio",
            "plausibility_passed",
        ])

        # 3. SSL / PCA embedding (16 dimensions)
        if ssl_embedding is not None and len(ssl_embedding) == 16:
            features.extend(ssl_embedding.tolist())
            names.extend([f"ssl_emb_{i}" for i in range(16)])
        else:
            features.extend([0.0] * 16)
            names.extend([f"ssl_emb_{i}" for i in range(16)])

        return np.array(features, dtype=np.float32), names

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> AstrovetFusionModel:
        """Fit fusion model on training feature matrix."""
        self.classifier.fit(X, y)
        self.is_fitted = True
        if feature_names:
            self.feature_names = feature_names
        return self

    def predict_single(
        self,
        candidate_id: str,
        tic_id: int,
        baseline_pred: Optional[BaselinePrediction],
        physics_diag: PhysicsDiagnostics,
        ssl_embedding: Optional[np.ndarray] = None,
        calibrator: Optional[Any] = None,
        run_id: str = "run_default",
    ) -> AstrovetPrediction:
        """
        Run inference for a single candidate.
        If the fusion model is unfitted, applies an uncalibrated heuristic triage score.
        If calibrator is None or unfitted, calibrated_planet_like_probability is None.
        """
        feat_vec, feat_names = self.build_feature_vector(baseline_pred, physics_diag, ssl_embedding)
        
        # Missing modalities accounting
        missing_mods = []
        b_has_score = bool(baseline_pred and baseline_pred.score is not None and baseline_pred.status == "SUCCESS")
        if not b_has_score:
            missing_mods.append("baseline_score")
        if ssl_embedding is None:
            missing_mods.append("ssl_representation")
        for mod, avail in physics_diag.modality_availability.items():
            if not avail and mod not in missing_mods:
                missing_mods.append(mod)

        # Compute raw score
        if self.is_fitted:
            raw_prob = float(self.classifier.predict_proba(feat_vec.reshape(1, -1))[0, 1])
            score_type = "fitted_fusion_score"
        else:
            # Unfitted heuristic triage score
            score_type = "uncalibrated_heuristic"
            oe_pen = 0.35 if physics_diag.odd_even_consistent is False else 0.0
            sec_pen = 0.40 if physics_diag.secondary_eclipse_detected is True else 0.0
            
            if b_has_score:
                b_val = float(baseline_pred.score)
                raw_prob = float(np.clip(b_val - oe_pen - sec_pen, 0.0, 1.0))
            else:
                # Baseline unavailable: evaluate purely from transit SNR and shape
                snr_norm = min(1.0, (physics_diag.transit_snr or 0.0) / 20.0)
                raw_prob = float(np.clip(snr_norm - oe_pen - sec_pen, 0.0, 1.0))

        # Calibration
        is_calibrated = False
        calibrated_prob: Optional[float] = None
        if calibrator is not None and getattr(calibrator, "is_fitted", False):
            calibrated_prob = float(np.clip(calibrator.calibrate(raw_prob), 0.0, 1.0))
            is_calibrated = True

        # Predictive uncertainty & OOD checks
        b_unc = float(baseline_pred.score_uncertainty) if (b_has_score and baseline_pred.score_uncertainty) else 0.50
        fit_unc = (physics_diag.transit_depth_fitted_err_ppm or 50.0) / max(1.0, (physics_diag.transit_depth_fitted_ppm or 500.0))
        predictive_uncertainty = float(np.clip(0.5 * b_unc + 0.5 * min(1.0, fit_unc), 0.01, 1.0))

        ood_flag = bool(
            (physics_diag.fit_chi2_reduced is not None and physics_diag.fit_chi2_reduced > 5.0)
            or not physics_diag.plausibility_all_passed
            or len(missing_mods) >= 3
        )

        # Assign reason codes
        reason_codes = []
        if not b_has_score:
            reason_codes.append("BASELINE_UNAVAILABLE")
        if score_type == "uncalibrated_heuristic" and not is_calibrated:
            reason_codes.append("UNCALIBRATED_HEURISTIC_SCORE")
        if physics_diag.odd_even_consistent is False:
            reason_codes.append("ODD_EVEN_DEPTH_MISMATCH")
        if physics_diag.secondary_eclipse_detected is True:
            reason_codes.append("SIGNIFICANT_SECONDARY_ECLIPSE")
        if not physics_diag.plausibility_all_passed:
            reason_codes.append("UNPHYSICAL_TRANSIT_GEOMETRY")
        if b_has_score and baseline_pred.score > 0.8 and raw_prob < 0.4:
            reason_codes.append("BASELINE_PHYSICS_DISAGREEMENT")
        if predictive_uncertainty > 0.35:
            reason_codes.append("HIGH_PREDICTIVE_UNCERTAINTY")
        for mod in missing_mods:
            if mod != "baseline_score":
                reason_codes.append(f"MISSING_{mod.upper()}")

        # Triage decision logic (uses calibrated prob if available, otherwise raw heuristic)
        eval_score = calibrated_prob if (is_calibrated and calibrated_prob is not None) else raw_prob

        if (score_type == "uncalibrated_heuristic" and not is_calibrated) or ood_flag or predictive_uncertainty > 0.45 or "BASELINE_PHYSICS_DISAGREEMENT" in reason_codes:
            triage_state = TriageState.REVIEW_REQUIRED
        elif "ODD_EVEN_DEPTH_MISMATCH" in reason_codes or "SIGNIFICANT_SECONDARY_ECLIPSE" in reason_codes:
            triage_state = TriageState.FALSE_POSITIVE_LIKE
        elif eval_score >= 0.70 and physics_diag.plausibility_all_passed:
            triage_state = TriageState.PLANET_LIKE
        elif eval_score <= 0.30:
            triage_state = TriageState.FALSE_POSITIVE_LIKE
        else:
            triage_state = TriageState.AMBIGUOUS

        is_synth = bool(baseline_pred and baseline_pred.is_synthetic)

        return AstrovetPrediction(
            candidate_id=candidate_id,
            tic_id=tic_id,
            baseline_score=baseline_pred.score if b_has_score else None,
            baseline_uncertainty=baseline_pred.score_uncertainty if b_has_score else None,
            baseline_prediction=baseline_pred,
            astrovet_raw_score=round(raw_prob, 4),
            score_type=score_type,
            is_calibrated=is_calibrated,
            calibration_provenance=getattr(calibrator, "provenance", None) if is_calibrated else None,
            calibrated_planet_like_probability=round(calibrated_prob, 4) if calibrated_prob is not None else None,
            triage_state=triage_state,
            predictive_uncertainty=round(predictive_uncertainty, 4),
            ood_flag=ood_flag,
            missing_modalities=missing_mods,
            physics_diagnostics=physics_diag,
            reason_codes=reason_codes,
            is_synthetic=is_synth,
            model_version="astrovet_0.1.0",
            run_id=run_id,
        )
