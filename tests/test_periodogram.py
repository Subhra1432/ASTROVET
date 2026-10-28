"""
Unit and integration tests for ASTROVET Box Least Squares (BLS) and Transit Least Squares (TLS)
periodicity search, ephemeris recovery, and diagnostic integration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import numpy as np

from astrovet.views.periodogram import (
    validate_and_prepare_timeseries,
    compare_period_with_ephemeris,
    compute_bls_periodogram,
    compute_tls_periodogram,
    run_periodicity_search,
)
from astrovet.explain.report import vet_and_generate_report


class TestPeriodogramAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        # Create a synthetic periodic transit time-series for deterministic testing
        rng = np.random.default_rng(42)
        self.period = 2.5  # days
        self.t0 = 100.25  # days
        self.duration_hours = 2.0  # hours (0.0833 days)
        self.depth = 0.003  # 3000 ppm
        self.time = np.arange(100.0, 115.0, 2.0 / (24.0 * 60.0))  # 15 days of 2-minute cadences
        self.flux = np.ones_like(self.time) + rng.normal(0, 0.0003, size=len(self.time))
        self.flux_err = np.full_like(self.time, 0.0003)

        # Inject box-like dips
        phase = ((self.time - self.t0 + 0.5 * self.period) % self.period) - 0.5 * self.period
        in_transit = np.abs(phase) < (self.duration_hours / 24.0 / 2.0)
        self.flux[in_transit] -= self.depth

    def test_validate_and_prepare_timeseries_clean(self) -> None:
        t, f, fe, t_span, err = validate_and_prepare_timeseries(self.time, self.flux, self.flux_err)
        self.assertIsNone(err)
        self.assertEqual(len(t), len(self.time))
        self.assertAlmostEqual(t_span, 15.0, delta=0.1)
        # Median flux should be normalized to 1.0
        self.assertAlmostEqual(float(np.median(f)), 1.0, places=4)

    def test_validate_and_prepare_timeseries_rejects_insufficient(self) -> None:
        t, f, fe, t_span, err = validate_and_prepare_timeseries(
            self.time[:20], self.flux[:20], self.flux_err[:20], min_cadences=50
        )
        self.assertIsNotNone(err)
        self.assertIn("Insufficient cadences", err)

    def test_validate_and_prepare_timeseries_handles_nans_and_zeros(self) -> None:
        dirty_t = self.time.copy()
        dirty_f = self.flux.copy()
        dirty_fe = self.flux_err.copy()
        dirty_f[10:20] = np.nan
        dirty_f[30:35] = -1.0
        dirty_fe[40:50] = 0.0

        t, f, fe, t_span, err = validate_and_prepare_timeseries(dirty_t, dirty_f, dirty_fe)
        self.assertIsNone(err)
        self.assertTrue(np.all(np.isfinite(t)))
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertTrue(np.all(np.isfinite(fe)))
        self.assertTrue(np.all(fe > 0.0))

    def test_compare_period_fundamental_match(self) -> None:
        comp = compare_period_with_ephemeris(
            recovered_period=2.501,
            recovered_epoch=100.25,
            recovered_duration_hours=2.0,
            candidate_period=2.5,
            candidate_epoch=100.25,
            candidate_duration_hours=2.0,
        )
        self.assertTrue(comp["compared_to_ephemeris"])
        self.assertTrue(comp["is_period_match"])
        self.assertFalse(comp["is_harmonic"])
        self.assertIn("Fundamental", comp["harmonic_relation"])
        self.assertLess(comp["period_diff_percent"], 0.2)

    def test_compare_period_subharmonic_detection(self) -> None:
        comp = compare_period_with_ephemeris(
            recovered_period=1.2505,
            recovered_epoch=100.25,
            recovered_duration_hours=2.0,
            candidate_period=2.5,
            candidate_epoch=100.25,
            candidate_duration_hours=2.0,
        )
        self.assertTrue(comp["compared_to_ephemeris"])
        self.assertTrue(comp["is_period_match"])
        self.assertTrue(comp["is_harmonic"])
        self.assertIn("0.5x", comp["harmonic_relation"])

    def test_compare_period_harmonic_detection(self) -> None:
        comp = compare_period_with_ephemeris(
            recovered_period=5.001,
            recovered_epoch=100.25,
            recovered_duration_hours=2.0,
            candidate_period=2.5,
            candidate_epoch=100.25,
            candidate_duration_hours=2.0,
        )
        self.assertTrue(comp["compared_to_ephemeris"])
        self.assertTrue(comp["is_period_match"])
        self.assertTrue(comp["is_harmonic"])
        self.assertIn("2x", comp["harmonic_relation"])

    def test_compare_period_mismatch(self) -> None:
        comp = compare_period_with_ephemeris(
            recovered_period=3.75,
            recovered_epoch=100.25,
            recovered_duration_hours=2.0,
            candidate_period=2.5,
            candidate_epoch=100.25,
            candidate_duration_hours=2.0,
        )
        self.assertTrue(comp["compared_to_ephemeris"])
        self.assertFalse(comp["is_period_match"])
        self.assertFalse(comp["is_harmonic"])

    def test_compute_bls_periodogram_recovery(self) -> None:
        res = compute_bls_periodogram(
            time=self.time,
            flux=self.flux,
            flux_err=self.flux_err,
            min_period_days=1.0,
            max_period_days=5.0,
            candidate_period=self.period,
            candidate_epoch=self.t0,
            candidate_duration=self.duration_hours,
        )
        self.assertTrue(res["is_available"])
        self.assertEqual(res["algorithm"], "Box Least Squares (BLS)")
        self.assertAlmostEqual(res["best_period_days"], self.period, delta=0.05)
        self.assertGreater(res["peak_snr"], 5.0)
        self.assertTrue(res["compared_to_ephemeris"])
        self.assertTrue(res["is_period_match"])
        self.assertEqual(res["time_convention"], "BJD - 2457000.0 (BTJD)")
        self.assertIn("provenance", res)

    def test_compute_tls_periodogram_recovery(self) -> None:
        # Test on 4-day subset for fast unit testing
        t_sub = self.time[self.time < 108.0]
        f_sub = self.flux[self.time < 108.0]
        fe_sub = self.flux_err[self.time < 108.0]

        res = compute_tls_periodogram(
            time=t_sub,
            flux=f_sub,
            flux_err=fe_sub,
            min_period_days=1.5,
            max_period_days=3.5,
            candidate_period=self.period,
            candidate_epoch=self.t0,
            candidate_duration=self.duration_hours,
        )
        self.assertTrue(res["is_available"])
        self.assertEqual(res["algorithm"], "Transit Least Squares (TLS)")
        self.assertAlmostEqual(res["best_period_days"], self.period, delta=0.08)
        self.assertGreater(res["sde"], 5.0)
        self.assertGreater(res["transit_count"], 0)
        self.assertTrue(res["compared_to_ephemeris"])
        self.assertTrue(res["is_period_match"])
        self.assertIn("model_folded_phase", res)
        self.assertIn("model_folded_flux", res)

    def test_run_periodicity_search_cross_agreement(self) -> None:
        t_sub = self.time[self.time < 108.0]
        f_sub = self.flux[self.time < 108.0]
        fe_sub = self.flux_err[self.time < 108.0]

        res = run_periodicity_search(
            time=t_sub,
            flux=f_sub,
            flux_err=fe_sub,
            min_period_days=1.5,
            max_period_days=3.5,
        )
        self.assertTrue(res["bls"]["is_available"])
        self.assertTrue(res["tls"]["is_available"])
        self.assertTrue(res["period_agreement"])
        self.assertIsNotNone(res["recovered_ephemeris"])
        rec = res["recovered_ephemeris"]
        self.assertAlmostEqual(rec["period_days"], self.period, delta=0.08)
        self.assertIn(rec["source_algorithm"], ["TLS", "BLS"])

    def test_vet_and_generate_report_integrates_periodogram(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            # Use synthetic fixture with allow_synthetic=True for hermetic unit testing
            rep = vet_and_generate_report(
                candidate_id="999999999-1-S23",
                output_dir=str(out_dir),
                fits_file=None,
                allow_synthetic=True,
                search_periods=True,
                period=2.5,
                epoch=1928.0,
                duration=2.0,
            )
            self.assertIn("periodogram_results", rep)
            pgram = rep["periodogram_results"]
            self.assertTrue(pgram["bls"]["is_available"])
            self.assertTrue(pgram["tls"]["is_available"])

            # Verify report file has Section 4
            report_text = rep["report_file"].read_text(encoding="utf-8")
            self.assertIn("## 4. Periodicity Search & Verification (BLS & TLS)", report_text)
            self.assertIn("Box Least Squares (BLS)", report_text)
            self.assertIn("Transit Least Squares (TLS)", report_text)

            # Verify 6-panel diagnostic plot was created
            self.assertTrue(rep["plot_file"].is_file())
            self.assertGreater(rep["plot_file"].stat().st_size, 5000)


if __name__ == "__main__":
    unittest.main()
