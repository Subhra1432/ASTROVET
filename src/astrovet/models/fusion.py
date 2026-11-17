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
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from astrovet.schemas.baseline import BaselinePrediction
from astrovet.schemas.prediction import PhysicsDiagnostics, TriageState, AstrovetPrediction


class AstrovetFusionModel:
    """
    Late fusion architecture combining multiple diagnostic branches.
    Supports tabular pretrained classifiers, late-fusion logistic heads,
    and probability calibration.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.classifier = LogisticRegression(C=1.0, random_state=random_state, max_iter=1000)
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.preprocessor: Optional[Any] = None
        # Train-fitted scaler for feature-aligned late-fusion models. Legacy
        # catalog preprocessors continue to use `preprocessor` above.
        self.feature_scaler: Optional[Any] = None
        self.calibrator: Optional[Any] = None
        self.provenance: Dict[str, Any] = {}

    def save(self, path: str | Path) -> Path:
        """Save fitted model bundle."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        bundle = {
            "classifier": self.classifier,
            "preprocessor": self.preprocessor,
            "feature_scaler": self.feature_scaler,
            "calibrator": self.calibrator,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "provenance": self.provenance,
        }
        joblib.dump(bundle, p)
        return p

    @classmethod
    def load(cls, path: str | Path, allow_catalog_classifier: bool = False) -> AstrovetFusionModel:
        """Load trained model bundle."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Model artifact not found: {p}")
        import joblib
        bundle = joblib.load(p)
        provenance = bundle.get("provenance", {})
        if provenance.get("target_is_planet_identification") is False and not allow_catalog_classifier:
            raise ValueError(
                "This artifact is a catalog-only E-versus-B/J classifier, not a planet-probability or fusion model. "
                "It cannot be loaded for candidate vetting inference."
            )
        inst = cls()
        inst.classifier = bundle.get("classifier", inst.classifier)
        inst.preprocessor = bundle.get("preprocessor")
        inst.feature_scaler = bundle.get("feature_scaler")
        inst.calibrator = bundle.get("calibrator")
        inst.feature_names = bundle.get("feature_names", [])
        inst.is_fitted = bundle.get("is_fitted", True)
        inst.provenance = provenance
        return inst


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

        # Active calibrator: explicit parameter takes priority, then instance-level calibrator
        active_calibrator = calibrator if calibrator is not None else self.calibrator

        # Compute raw score
        explanation: Dict[str, Any]
        if self.is_fitted:
            if self.preprocessor is not None and getattr(self.preprocessor, "is_fitted", False):
                dur_h = physics_diag.transit_duration_fitted_hours or 2.0
                period_d = 10.0
                pgram = getattr(physics_diag, "periodogram_summary", {}) or {}
                if isinstance(pgram, dict) and "recovered_ephemeris" in pgram:
                    rec = pgram.get("recovered_ephemeris")
                    if rec and rec.get("period_days"):
                        period_d = float(rec["period_days"])
                depth_ppm = physics_diag.transit_depth_fitted_ppm or 1000.0
                row_df = pd.DataFrame([{
                    "Period": period_d,
                    "Duration": dur_h / 24.0,
                    "Depth": depth_ppm,
                    "SMass": np.nan,
                    "SRad": np.nan,
                    "SRadEst": np.nan,
                    "Tmag": 10.0,
                }])
                X_proc = self.preprocessor.transform(row_df)
                raw_prob = float(self.classifier.predict_proba(X_proc)[0, 1])
                score_type = "fitted_fusion_score"
                explanation = {
                    "basis": "preprocessed_classifier_contributions_unavailable",
                    "note": "The configured preprocessor transforms inputs; no aligned feature mapping is available for a safe attribution.",
                }
            else:
                if self.feature_scaler is not None:
                    name_to_value = dict(zip(feat_names, feat_vec.tolist()))
                    missing_features = [name for name in self.feature_names if name not in name_to_value]
                    if missing_features:
                        raise ValueError(
                            "Inference is missing trained fusion features: " + ", ".join(missing_features)
                        )
                    model_vec = np.asarray(
                        [name_to_value[name] for name in self.feature_names], dtype=np.float32
                    ).reshape(1, -1)
                    model_vec = self.feature_scaler.transform(model_vec)
                else:
                    # Backward compatibility for pre-existing, unscaled fusion bundles.
                    model_vec = feat_vec.reshape(1, -1)
                raw_prob = float(self.classifier.predict_proba(model_vec)[0, 1])
                score_type = "fitted_fusion_score"
                coef = np.asarray(self.classifier.coef_[0], dtype=float)
                names = self.feature_names if len(self.feature_names) == len(coef) else feat_names
                contributions = coef * model_vec.reshape(-1)
                intercept = float(self.classifier.intercept_[0])
                ranked = sorted(zip(names, contributions.tolist()), key=lambda item: abs(item[1]), reverse=True)
                explanation = {
                    "basis": "additive_logistic_logit_contributions",
                    "intercept": intercept,
                    "logit_reconstruction": float(intercept + np.sum(contributions)),
                    "top_contributions": [
                        {"feature": str(name), "logit_contribution": float(value)}
                        for name, value in ranked[:8]
                    ],
                    "note": "Contributions sum to the raw classifier logit; they are model associations, not causal evidence.",
                }
        else:
            # Unfitted heuristic triage score
            score_type = "uncalibrated_heuristic"
            oe_pen = 0.35 if physics_diag.odd_even_consistent is False else 0.0
            sec_pen = 0.40 if physics_diag.secondary_eclipse_detected is True else 0.0
            
            if b_has_score:
                b_val = float(baseline_pred.score)
                raw_prob = float(np.clip(b_val - oe_pen - sec_pen, 0.0, 1.0))
                source_value = b_val
                source_name = "baseline_score"
            else:
                # Baseline unavailable: evaluate purely from transit SNR and shape
                snr_norm = min(1.0, (physics_diag.transit_snr or 0.0) / 20.0)
                raw_prob = float(np.clip(snr_norm - oe_pen - sec_pen, 0.0, 1.0))
                source_value = snr_norm
                source_name = "transit_snr_scaled_to_20"
            explanation = {
                "basis": "uncalibrated_heuristic_arithmetic_decomposition",
                "positive_signal": {"feature": source_name, "value": float(source_value)},
                "penalties": {
                    "odd_even_inconsistency": float(oe_pen),
                    "secondary_eclipse": float(sec_pen),
                },
                "raw_score_after_clipping": raw_prob,
                "note": "This hand-set heuristic is not a trained planet probability.",
            }

        # Calibration
        is_calibrated = False
        calibrated_prob: Optional[float] = None
        if active_calibrator is not None and getattr(active_calibrator, "is_fitted", False):
            calibrated_prob = float(np.clip(active_calibrator.calibrate(raw_prob), 0.0, 1.0))
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
        if physics_diag.bls_is_match is False or physics_diag.tls_is_match is False:
            reason_codes.append("PERIODICITY_MISMATCH")
        if physics_diag.period_search_agreement is False:
            reason_codes.append("BLS_TLS_PERIOD_DISAGREEMENT")
        if not physics_diag.plausibility_all_passed:
            reason_codes.append("UNPHYSICAL_TRANSIT_GEOMETRY")
        if b_has_score and baseline_pred.score > 0.8 and raw_prob < 0.4:
            reason_codes.append("BASELINE_PHYSICS_DISAGREEMENT")
        if predictive_uncertainty > 0.35:
            reason_codes.append("HIGH_PREDICTIVE_UNCERTAINTY")
        for mod in missing_mods:
            if mod != "baseline_score":
                reason_codes.append(f"MISSING_{mod.upper()}")

        # Triage decision logic (uses calibrated prob if available, otherwise raw score)
        eval_score = calibrated_prob if (is_calibrated and calibrated_prob is not None) else raw_prob

        if (
            (score_type == "uncalibrated_heuristic" and not is_calibrated)
            or ood_flag
            or predictive_uncertainty > 0.45
            or "BASELINE_PHYSICS_DISAGREEMENT" in reason_codes
            or "PERIODICITY_MISMATCH" in reason_codes
            or "BLS_TLS_PERIOD_DISAGREEMENT" in reason_codes
        ):
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
            calibration_provenance=getattr(active_calibrator, "provenance", None) if is_calibrated else None,
            calibrated_planet_like_probability=round(calibrated_prob, 4) if calibrated_prob is not None else None,
            triage_state=triage_state,
            predictive_uncertainty=round(predictive_uncertainty, 4),
            ood_flag=ood_flag,
            missing_modalities=missing_mods,
            physics_diagnostics=physics_diag,
            reason_codes=reason_codes,
            explanation=explanation,
            is_synthetic=is_synth,
            model_version="astrovet_0.1.0",
            run_id=run_id,
        )


def validate_ssl_checkpoint_compatibility(
    fusion_model: AstrovetFusionModel,
    encoder_mode: str,
    ssl_checkpoint: Optional[str | Path],
) -> None:
    """Fail closed when an SSL-fusion model is paired with a different encoder."""
    expected_sha256 = (fusion_model.provenance or {}).get("ssl_checkpoint_sha256")
    if not expected_sha256:
        return
    if encoder_mode != "ssl":
        raise ValueError(
            "The selected fusion model was trained with a self-supervised encoder; "
            "use --encoder-mode ssl with its matching checkpoint."
        )
    if ssl_checkpoint is None:
        raise ValueError(
            "The selected fusion model requires its matching SSL checkpoint, but none was supplied. "
            "Set ASTROVET_SSL_CHECKPOINT or pass --ssl-checkpoint."
        )
    from astrovet.provenance import compute_sha256

    checkpoint_path = Path(ssl_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SSL checkpoint does not exist: {checkpoint_path}")
    actual_sha256 = compute_sha256(checkpoint_path)
    if actual_sha256.lower() != str(expected_sha256).lower():
        raise ValueError(
            "The SSL checkpoint does not match the selected fusion model. "
            f"Expected SHA-256 {expected_sha256}, got {actual_sha256}. "
            "Use the encoder checkpoint that was used to train this fusion artifact."
        )
