import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import joblib

from astrovet.models.training import (
    load_and_audit_dataset,
    partition_host_disjoint,
    TabularFeaturePreprocessor,
    run_training_pipeline,
    EXCLUDED_LEAKAGE_COLUMNS,
)
from astrovet.models.fusion import AstrovetFusionModel
from astrovet.schemas.prediction import PhysicsDiagnostics, TriageState


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Create a synthetic mini catalog mimicking tces_with_labels_v3.csv structure
        # (used exclusively for unit test assertion of pipeline logic)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_catalog.csv"

        lines = [
            "# Astronet-Triage-v2 TCE test catalog",
            "# Comment line 2",
            "TIC ID,Epoch,Period,Duration,Depth,SMass,SRad,SRadEst,Tmag,Year,MinT,MaxT,Split,Consensus Label,L1,L2,L3,L4,L5,L6,L7,L8,Notes",
            ",BTJD,days,days,ppm,M_sun,R_sun,R_sun,,,BTJD,BTJD,,,,,,,,,,,",
        ]

        # 120 host stars with 1 or 2 candidates each
        rng = np.random.default_rng(42)
        labels = ["E", "B", "J", "S", "N"]
        for i in range(1, 121):
            tic = 1000 + i
            num_cands = 2 if i % 10 == 0 else 1
            for c in range(num_cands):
                lbl = labels[i % len(labels)]
                period = float(rng.uniform(0.5, 30.0))
                duration = float(rng.uniform(0.02, 0.3))
                depth = float(rng.uniform(200.0, 10000.0))
                smass = float(rng.uniform(0.8, 1.5)) if i % 2 == 0 else ""
                srad = float(rng.uniform(0.8, 2.0)) if i % 3 != 0 else ""
                sradest = float(rng.uniform(0.8, 2.0))
                tmag = float(rng.uniform(8.0, 14.0))
                lines.append(
                    f"{tic},1500.0,{period:.4f},{duration:.4f},{depth:.1f},{smass},{srad},{sradest:.4f},{tmag:.2f},1,1400.0,1600.0,train,{lbl},,,,,,,,,"
                )

        self.csv_path.write_text("\n".join(lines), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_and_audit_dataset(self):
        df_valid, audit = load_and_audit_dataset(self.csv_path)
        self.assertGreater(len(df_valid), 0)
        # Verify unresolved labels (S, N) are strictly excluded
        self.assertNotIn("S", df_valid["Consensus Label"].values)
        self.assertNotIn("N", df_valid["Consensus Label"].values)
        # Verify target is binary 0 or 1
        self.assertTrue(set(df_valid["target"].unique()).issubset({0, 1}))
        # Verify audit metadata
        self.assertEqual(audit["label_mapping"]["positive_eclipsing_signal"], ["E"])
        self.assertEqual(audit["label_mapping"]["negative_contact_binary_or_junk"], ["B", "J"])

    def test_leakage_safe_partitioning(self):
        df_valid, _ = load_and_audit_dataset(self.csv_path)
        splits, meta = partition_host_disjoint(df_valid, random_seed=42)

        train_tics = set(splits["train"]["TIC ID"].unique())
        val_tics = set(splits["val"]["TIC ID"].unique())
        cal_tics = set(splits["cal"]["TIC ID"].unique())
        test_tics = set(splits["test"]["TIC ID"].unique())

        # Assert zero overlap across all 6 pairs
        self.assertEqual(len(train_tics.intersection(val_tics)), 0)
        self.assertEqual(len(train_tics.intersection(cal_tics)), 0)
        self.assertEqual(len(train_tics.intersection(test_tics)), 0)
        self.assertEqual(len(val_tics.intersection(cal_tics)), 0)
        self.assertEqual(len(val_tics.intersection(test_tics)), 0)
        self.assertEqual(len(cal_tics.intersection(test_tics)), 0)
        self.assertTrue(meta["leakage_zero_overlap_verified"])

    def test_train_only_preprocessor(self):
        df_valid, _ = load_and_audit_dataset(self.csv_path)
        splits, _ = partition_host_disjoint(df_valid, random_seed=42)

        preprocessor = TabularFeaturePreprocessor()
        preprocessor.fit(splits["train"])
        self.assertTrue(preprocessor.is_fitted)

        # Check that no leakage column is in feature_names
        for col in EXCLUDED_LEAKAGE_COLUMNS:
            self.assertNotIn(col, preprocessor.feature_names)

        X_train = preprocessor.transform(splits["train"])
        X_test = preprocessor.transform(splits["test"])

        self.assertEqual(X_train.shape[1], len(preprocessor.feature_names))
        self.assertEqual(X_test.shape[1], len(preprocessor.feature_names))
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_test).any())

    def test_end_to_end_training_pipeline(self):
        out_art_dir = Path(self.temp_dir.name) / "artifacts"
        rep = run_training_pipeline(
            dataset_path=self.csv_path,
            artifacts_dir=out_art_dir,
            random_seed=42,
            c_candidates=[0.1, 1.0],
        )

        self.assertEqual(rep["status"], "COMPLETED")
        self.assertIn("ranking", rep["test_evaluation"])
        self.assertIn("roc_auc", rep["test_evaluation"]["ranking"])
        self.assertIn("pr_auc", rep["test_evaluation"]["ranking"])
        self.assertIn("calibrated_brier_score", rep["test_evaluation"]["calibration"])

        # Check saved files exist
        model_file = Path(rep["model_artifact"])
        self.assertTrue(model_file.is_file())
        manifest_file = Path(rep["split_manifest_file"])
        self.assertTrue(manifest_file.is_file())

        # Verify scientific guardrail prevents misusing catalog-only model for vetting
        with self.assertRaises(ValueError):
            AstrovetFusionModel.load(model_file, allow_catalog_classifier=False)

        # Test model loading into AstrovetFusionModel with explicit opt-in
        loaded_fusion = AstrovetFusionModel.load(model_file, allow_catalog_classifier=True)
        self.assertTrue(loaded_fusion.is_fitted)
        self.assertIsNotNone(loaded_fusion.preprocessor)
        self.assertIsNotNone(loaded_fusion.calibrator)

        # Test single prediction with loaded model
        p_diag = PhysicsDiagnostics(
            transit_depth_fitted_ppm=850.0,
            transit_duration_fitted_hours=2.4,
            transit_snr=18.0,
            fit_chi2_reduced=1.05,
            odd_even_consistent=True,
            secondary_eclipse_detected=False,
            plausibility_all_passed=True,
            modality_availability={"transit_model_fit": True},
        )
        pred = loaded_fusion.predict_single(
            candidate_id="1001-1-S1",
            tic_id=1001,
            baseline_pred=None,
            physics_diag=p_diag,
            ssl_embedding=None,
        )
        self.assertEqual(pred.score_type, "fitted_fusion_score")
        self.assertTrue(pred.is_calibrated)
        self.assertIsNotNone(pred.calibrated_planet_like_probability)
        self.assertGreaterEqual(pred.calibrated_planet_like_probability, 0.0)
        self.assertLessEqual(pred.calibrated_planet_like_probability, 1.0)


if __name__ == "__main__":
    unittest.main()