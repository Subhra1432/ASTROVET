"""
Physics diagnostic integration and physical plausibility verification.

Scientific safeguards:
- Plausibility checks: period > 0, duration < period, feasible transit geometry.
- Explicit modality availability tracking; missing diagnostics are left as None/untested,
  NEVER silently replaced with fabricated zeros.
- Diagnostics are treated as uncertain scientific evidence, not infallible rules.
- Transit fit is documented explicitly as an approximate screening diagnostic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from astrovet.schemas.prediction import PhysicsDiagnostics
from astrovet.views.folding import FoldedView, TransitEventStamp
from astrovet.physics.transit_model import fit_transit_model, ApproximateTransitFitResult
from astrovet.physics.odd_even import compute_odd_even_diagnostic
from astrovet.physics.secondary import compute_secondary_diagnostic
from astrovet.physics.consistency import compute_event_consistency_diagnostic


def run_physics_diagnostics(
    primary_zoom: FoldedView,
    full_orbit: FoldedView,
    stamps: List[TransitEventStamp],
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_days: float,
    epoch_btjd: float,
    duration_hours: float,
    reported_depth_ppm: Optional[float] = None,
    periodogram_results: Optional[Dict[str, Any]] = None,
) -> Tuple[PhysicsDiagnostics, ApproximateTransitFitResult]:
    """
    Run complete physics diagnostics battery and assemble PhysicsDiagnostics contract.
    """
    # 1. Physical plausibility checks
    is_period_positive = bool(period_days > 0.0)
    is_duration_feasible = bool(0.1 <= duration_hours < (period_days * 24.0))
    is_depth_feasible = bool(reported_depth_ppm is None or (5.0 <= reported_depth_ppm <= 150000.0))

    # 2. Approximate transit model fit
    fit_res = fit_transit_model(
        phase_hours=primary_zoom.phase_in_hours,
        flux=primary_zoom.flux,
        flux_err=primary_zoom.flux_err,
        initial_duration_hours=duration_hours,
        initial_depth_ppm=reported_depth_ppm,
    )

    effective_depth_ppm = fit_res.fitted_depth_ppm if fit_res.fitted_depth_ppm > 0 else (reported_depth_ppm or 1000.0)

    # 3. Odd vs Even diagnostics (cadence & event level)
    odd_even_diag = compute_odd_even_diagnostic(
        stamps=stamps,
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=period_days,
        epoch_btjd=epoch_btjd,
        duration_hours=duration_hours,
    )

    # 4. Secondary eclipse diagnostics
    sec_diag = compute_secondary_diagnostic(full_orbit, effective_depth_ppm, duration_hours, period_days)

    # 5. Transit-to-transit consistency
    cons_diag = compute_event_consistency_diagnostic(stamps, duration_hours)

    # Plausibility summary
    plausibility_all_passed = bool(
        is_period_positive and is_duration_feasible and is_depth_feasible and fit_res.is_converged
    )

    bls = (periodogram_results or {}).get("bls", {})
    tls = (periodogram_results or {}).get("tls", {})

    modality_availability = {
        "transit_model_fit": bool(fit_res.is_converged),
        "odd_even_comparison": bool(odd_even_diag.is_available),
        "secondary_eclipse_search": bool(sec_diag.is_available),
        "cross_event_consistency": bool(cons_diag.is_available),
        "bls_periodogram": bool(bls.get("is_available", False)),
        "tls_periodogram": bool(tls.get("is_available", False)),
    }

    diag = PhysicsDiagnostics(
        transit_depth_fitted_ppm=fit_res.fitted_depth_ppm if fit_res.is_converged else None,
        transit_depth_fitted_err_ppm=fit_res.fitted_depth_err_ppm if fit_res.is_converged else None,
        transit_duration_fitted_hours=fit_res.fitted_duration_hours if fit_res.is_converged else None,
        transit_snr=fit_res.transit_snr if fit_res.is_converged else None,
        fit_chi2_reduced=fit_res.reduced_chi2 if fit_res.is_converged else None,
        
        odd_depth_ppm=odd_even_diag.odd_depth_ppm if odd_even_diag.is_available else None,
        even_depth_ppm=odd_even_diag.even_depth_ppm if odd_even_diag.is_available else None,
        odd_even_depth_diff_ppm=odd_even_diag.depth_diff_ppm if odd_even_diag.is_available else None,
        odd_even_t_statistic=odd_even_diag.significance_sigma if odd_even_diag.is_available else None,
        odd_even_p_value=odd_even_diag.p_value if odd_even_diag.is_available else None,
        odd_even_consistent=odd_even_diag.is_consistent if odd_even_diag.is_available else None,
        
        secondary_depth_ppm=sec_diag.secondary_depth_ppm if sec_diag.is_available else None,
        secondary_snr=sec_diag.secondary_snr if sec_diag.is_available else None,
        secondary_depth_ratio=sec_diag.secondary_depth_ratio if sec_diag.is_available else None,
        secondary_eclipse_detected=sec_diag.secondary_detected if sec_diag.is_available else None,
        
        transit_count_observed=cons_diag.observed_transit_count if cons_diag.is_available else len(stamps),
        transit_depth_scatter_ppm=cons_diag.mean_event_depth_ppm * cons_diag.depth_scatter_fraction if cons_diag.is_available else None,
        transit_timing_scatter_minutes=cons_diag.timing_scatter_minutes if cons_diag.is_available else None,
        
        is_period_positive=is_period_positive,
        is_duration_feasible=is_duration_feasible,
        is_depth_feasible=is_depth_feasible,
        plausibility_all_passed=plausibility_all_passed,

        # Periodicity verification
        bls_best_period_days=bls.get("best_period_days") if bls.get("is_available") else None,
        bls_snr=bls.get("peak_snr") if bls.get("is_available") else None,
        bls_period_diff_pct=bls.get("period_diff_percent") if bls.get("is_available") else None,
        bls_is_match=bls.get("is_period_match") if bls.get("is_available") else None,

        tls_best_period_days=tls.get("best_period_days") if tls.get("is_available") else None,
        tls_sde=tls.get("sde") if tls.get("is_available") else None,
        tls_snr=tls.get("snr") if tls.get("is_available") else None,
        tls_odd_even_mismatch=tls.get("odd_even_mismatch") if tls.get("is_available") else None,
        tls_period_diff_pct=tls.get("period_diff_percent") if tls.get("is_available") else None,
        tls_is_match=tls.get("is_period_match") if tls.get("is_available") else None,

        period_search_agreement=(periodogram_results or {}).get("period_agreement"),
        periodogram_summary=periodogram_results or {},
        modality_availability=modality_availability,
    )

    return diag, fit_res
