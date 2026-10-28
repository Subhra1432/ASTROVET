import unittest
import numpy as np

from astrovet.views.folding import FoldedView, TransitEventStamp
from astrovet.physics.transit_model import fit_transit_model, approximate_limb_darkened_transit
from astrovet.physics.odd_even import compute_odd_even_diagnostic
from astrovet.physics.secondary import compute_secondary_diagnostic
from astrovet.physics.diagnostics import run_physics_diagnostics


class TestPhysicsDiagnostics(unittest.TestCase):
    def test_transit_model_fit(self):
        # Generate synthetic transit signal
        phase_h = np.linspace(-3.0, 3.0, 200)
        true_depth = 0.0015 # 1500 ppm
        true_dur = 2.0 # hours
        model = approximate_limb_darkened_transit(phase_h, true_depth, true_dur)
        noise = np.random.normal(0, 0.0001, len(phase_h))
        flux = model + noise
        flux_err = np.full_like(flux, 0.0001)

        fit_res = fit_transit_model(phase_h, flux, flux_err, initial_duration_hours=2.0)
        self.assertTrue(fit_res.is_converged)
        self.assertAlmostEqual(fit_res.fitted_depth_ppm, 1500.0, delta=250.0)
        self.assertAlmostEqual(fit_res.fitted_duration_hours, 2.0, delta=0.5)
        self.assertGreater(fit_res.transit_snr, 5.0)
        self.assertFalse(fit_res.is_validated_physical_inversion)

    def test_odd_even_consistency_metric(self):
        # Create mock stamps where odd and even have identical depths
        rng = np.random.default_rng(42)
        stamps = []
        for i in range(6):
            t_center = 1000.0 + i * 2.0
            t = np.linspace(t_center - 0.2, t_center + 0.2, 50)
            in_t = np.abs(t - t_center) < 0.04
            f = np.ones_like(t)
            f[in_t] -= 0.0015
            f += rng.normal(0, 0.00005, len(t))
            fe = np.full_like(f, 0.0001)
            stamps.append(TransitEventStamp(
                event_index=i,
                expected_transit_time=t_center,
                time=t,
                flux=f,
                flux_err=fe,
                is_odd=bool(i % 2 != 0),
            ))

        time_all = np.concatenate([s.time for s in stamps])
        flux_all = np.concatenate([s.flux for s in stamps])
        fe_all = np.concatenate([s.flux_err for s in stamps])

        diag = compute_odd_even_diagnostic(
            stamps=stamps,
            time=time_all,
            flux=flux_all,
            flux_err=fe_all,
            period_days=2.0,
            epoch_btjd=1000.0,
            duration_hours=1.92,
        )
        self.assertTrue(diag.is_available)
        self.assertTrue(diag.is_consistent)
        self.assertLess(diag.significance_sigma, 3.0)

    def test_secondary_eclipse_detection(self):
        # Full orbit with deep secondary eclipse (simulating eclipsing binary)
        phase = np.linspace(-0.5, 0.5, 500)
        flux = np.ones_like(phase)
        # Secondary at phase +/- 0.5 with duration 2 hours
        dur_phase = (2.0 / 24.0) / 2.0
        sec_mask = np.minimum(np.abs(phase - 0.5), np.abs(phase + 0.5)) <= dur_phase
        flux[sec_mask] -= 0.0010  # 1000 ppm secondary

        full_view = FoldedView(
            phase=phase,
            flux=flux,
            flux_err=np.full_like(flux, 0.00005),
            bin_counts=np.full(len(phase), 50),
            phase_in_hours=phase * 24.0,
            valid_mask=np.ones(len(phase), dtype=bool),
        )

        sec_diag = compute_secondary_diagnostic(
            full_orbit_view=full_view,
            primary_depth_ppm=2000.0,
            duration_hours=2.0,
            period_days=2.0,
        )
        self.assertTrue(sec_diag.is_available)
        self.assertTrue(sec_diag.secondary_detected)


if __name__ == "__main__":
    unittest.main()
