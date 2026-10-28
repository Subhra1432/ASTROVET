import unittest
import numpy as np

from astrovet.schemas.baseline import BaselinePrediction
from astrovet.schemas.prediction import PhysicsDiagnostics, TriageState
from astrovet.schemas.candidate import CandidateRecord, Ephemeris
from astrovet.models.fusion import AstrovetFusionModel
from astrovet.models.calibration import PlattCalibrator
from astrovet.explain.report import generate_candidate_markdown_report


class TestFusionAndReport(unittest.TestCase):
    def test_fusion_prediction_and_triage(self):
        fusion = AstrovetFusionModel()

        # Planet-like candidate
        b_pred = BaselinePrediction(
            candidate_id="1001-1-S1",
            score=0.92,
            score_uncertainty=0.03,
            status="SUCCESS",
            execution_mode="CATALOG",
        )
        p_diag = PhysicsDiagnostics(
            transit_depth_fitted_ppm=1200.0,
            transit_duration_fitted_hours=2.1,
            transit_snr=15.0,
            fit_chi2_reduced=1.1,
            odd_even_consistent=True,
            secondary_eclipse_detected=False,
            plausibility_all_passed=True,
            modality_availability={"transit_model_fit": True, "odd_even_comparison": True},
        )

        # 1. Uncalibrated heuristic path
        pred_uncal = fusion.predict_single(
            candidate_id="1001-1-S1",
            tic_id=1001,
            baseline_pred=b_pred,
            physics_diag=p_diag,
            ssl_embedding=np.zeros(16, dtype=np.float32),
            calibrator=None,
            run_id="test_run_uncal",
        )

        self.assertEqual(pred_uncal.triage_state, TriageState.REVIEW_REQUIRED)
        self.assertFalse(pred_uncal.is_calibrated)
        self.assertIsNone(pred_uncal.calibrated_planet_like_probability)
        self.assertGreaterEqual(pred_uncal.astrovet_raw_score, 0.70)
        self.assertIn("UNCALIBRATED_HEURISTIC_SCORE", pred_uncal.reason_codes)
        self.assertFalse(pred_uncal.ood_flag)

        # 2. Calibrated path with fitted calibrator
        calibrator = PlattCalibrator()
        # Mock fit on calibration partition
        raw_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.85, 0.15])
        y_true = np.array([0, 0, 1, 1, 1, 0])
        calibrator.fit(raw_probs, y_true)

        pred_cal = fusion.predict_single(
            candidate_id="1001-1-S1",
            tic_id=1001,
            baseline_pred=b_pred,
            physics_diag=p_diag,
            ssl_embedding=np.zeros(16, dtype=np.float32),
            calibrator=calibrator,
            run_id="test_run_cal",
        )

        self.assertTrue(pred_cal.is_calibrated)
        self.assertIsNotNone(pred_cal.calibrated_planet_like_probability)
        self.assertGreaterEqual(pred_cal.calibrated_planet_like_probability, 0.60)

        # 3. Missing baseline candidate test
        b_missing = BaselinePrediction(
            candidate_id="1002-1-S1",
            score=None,
            status="NOT_FOUND",
            execution_mode="CATALOG",
        )
        pred_missing = fusion.predict_single(
            candidate_id="1002-1-S1",
            tic_id=1002,
            baseline_pred=b_missing,
            physics_diag=p_diag,
            ssl_embedding=np.zeros(16, dtype=np.float32),
            calibrator=None,
            run_id="test_run_missing",
        )
        self.assertIsNone(pred_missing.baseline_score)
        self.assertIn("BASELINE_UNAVAILABLE", pred_missing.reason_codes)

        # 4. Markdown report generation
        cand = CandidateRecord(
            candidate_id="1001-1-S1",
            tic_id=1001,
            planet_num=1,
            sector_run="1",
            ephemeris=Ephemeris(
                period_days=3.0,
                epoch_bjd_tdb=1900.5,
                duration_hours=2.1,
                transit_depth_ppm=1200.0,
            ),
        )
        report_text = generate_candidate_markdown_report(cand, pred_uncal)
        self.assertIn("ASTROVET Candidate Vetting Report", report_text)
        self.assertIn("REVIEW_REQUIRED", report_text)
        self.assertIn("Scientific Status", report_text)
        self.assertIn("ASTROVET Score (Uncalibrated Heuristic)", report_text)
        self.assertIn("Calibration was not applied", report_text)
        self.assertNotIn("Calibrated Planet-Like Probability", report_text)
        self.assertNotIn("Disjoint Calibration Set", report_text)

        # 5. Calibrated report generation (without recorded provenance)
        report_cal_text = generate_candidate_markdown_report(cand, pred_cal)
        self.assertIn("Calibrated Planet-Like Probability", report_cal_text)
        self.assertNotIn("Disjoint Calibration Set", report_cal_text)
        self.assertIn("Calibration provenance not recorded", report_cal_text)

        # 6. Calibrated report with explicit disjoint calibration set provenance
        pred_cal.calibration_provenance = "Disjoint Calibration Set (Host-Disjoint Split v1)"
        report_cal_prov_text = generate_candidate_markdown_report(cand, pred_cal)
        self.assertIn("Disjoint Calibration Set (Host-Disjoint Split v1)", report_cal_prov_text)


if __name__ == "__main__":
    unittest.main()
