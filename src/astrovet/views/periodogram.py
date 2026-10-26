"""
Periodogram analysis for periodicity verification, false alarm checks, and ephemeris recovery.

Scientific safeguards:
- Box Least Squares (BLS; Kovacs et al. 2002) for standard box-like transit screening.
- Transit Least Squares (TLS; Hippke & Heller 2019) using realistic limb-darkened transit shapes.
- Robust timeseries preparation: finite value checks, non-zero uncertainty imputation, strict time sorting.
- Explicit ephemeris comparison (fundamental and harmonic: 0.5x, 2x, 0.333x, 3x) against provided TCE values.
- Recovers candidate parameters when ephemeris is missing without fabricating example values.
- Periodicity verification is a screening aid, NEVER proof of exoplanetary status.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from astropy.timeseries import BoxLeastSquares


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        fval = float(val)
        return fval if np.isfinite(fval) else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        fval = float(val)
        return int(fval) if np.isfinite(fval) else default
    except (ValueError, TypeError):
        return default


def _safe_array_list(val: Any) -> list:
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return []


def validate_and_prepare_timeseries(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_cadences: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[str]]:
    """
    Validate, sanitize, and sort time series data for periodogram analysis.
    Returns:
        (clean_time, clean_flux, clean_flux_err, time_span_days, error_message)
    """
    if len(time) == 0 or len(flux) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0.0,
            "Light curve arrays are empty.",
        )

    # 1. Finite mask
    finite_mask = np.isfinite(time) & np.isfinite(flux) & (flux > 0.0)
    if not np.any(finite_mask):
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0.0,
            "No finite, positive flux values in light curve.",
        )

    t = np.array(time[finite_mask], dtype=np.float64)
    f = np.array(flux[finite_mask], dtype=np.float64)

    # 2. Uncertainty handling (ensure strictly positive non-zero dy)
    if flux_err is not None:
        fe_raw = np.array(flux_err[finite_mask], dtype=np.float64)
        valid_err = np.isfinite(fe_raw) & (fe_raw > 0.0)
        if np.any(valid_err):
            median_err = float(np.median(fe_raw[valid_err]))
            fe = np.where(valid_err, fe_raw, median_err if median_err > 0 else 1e-4)
        else:
            sample_std = float(np.std(f))
            fe = np.full_like(f, max(1e-5, sample_std if sample_std > 0 else 1e-3))
    else:
        sample_std = float(np.std(f))
        fe = np.full_like(f, max(1e-5, sample_std if sample_std > 0 else 1e-3))

    # 3. Time sorting
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    f = f[sort_idx]
    fe = fe[sort_idx]

    # 4. Flux normalization (median normalization to 1.0 for BLS/TLS templates)
    median_flux = float(np.median(f))
    if median_flux > 0.0:
        f = f / median_flux
        fe = fe / median_flux

    # 5. Cadence count check
    if len(t) < min_cadences:
        return (
            t, f, fe, 0.0,
            f"Insufficient cadences for reliable periodogram analysis ({len(t)} < {min_cadences}).",
        )

    time_span_days = float(t[-1] - t[0])
    if time_span_days <= 0.0:
        return (
            t, f, fe, 0.0,
            f"Non-positive observational time baseline ({time_span_days:.4f} days).",
        )

    return t, f, fe, time_span_days, None


def compare_period_with_ephemeris(
    recovered_period: float,
    recovered_epoch: float,
    recovered_duration_hours: float,
    candidate_period: float,
    candidate_epoch: float,
    candidate_duration_hours: float,
    tolerance_pct: float = 1.5,
) -> Dict[str, Any]:
    """
    Compare periodogram-recovered parameters with candidate catalog ephemeris.
    Detects exact fundamental matches as well as 0.5x, 2x, 0.333x, 3x harmonics.
    """
    if candidate_period <= 0.0 or recovered_period <= 0.0:
        return {
            "compared_to_ephemeris": False,
            "reason": "Invalid non-positive period values.",
        }

    period_ratio = recovered_period / candidate_period
    period_diff_pct = abs(recovered_period - candidate_period) / candidate_period * 100.0

    # 1. Fundamental match check
    is_fundamental_match = bool(period_diff_pct < tolerance_pct)

    # 2. Harmonic relations check (0.5x, 2x, 0.333x, 3x)
    harmonic_candidates = [
        (0.5, "0.5x (Half-period / Subharmonic)"),
        (2.0, "2x (Double-period / Harmonics)"),
        (1.0 / 3.0, "0.333x (Third-period)"),
        (3.0, "3x (Triple-period)"),
    ]

    is_harmonic = False
    harmonic_relation = "1x (Fundamental)" if is_fundamental_match else "None"
    closest_diff_pct = period_diff_pct

    if not is_fundamental_match:
        for ratio, label in harmonic_candidates:
            diff_pct = abs(period_ratio - ratio) / ratio * 100.0
            if diff_pct < tolerance_pct:
                is_harmonic = True
                harmonic_relation = label
                closest_diff_pct = diff_pct
                break

    is_period_match = bool(is_fundamental_match or is_harmonic)

    # Epoch difference modulo period (in hours)
    effective_p = candidate_period if not is_harmonic else (candidate_period * period_ratio)
    phase_offset = ((recovered_epoch - candidate_epoch + 0.5 * effective_p) % effective_p) - 0.5 * effective_p
    epoch_diff_hours = abs(phase_offset) * 24.0

    # Duration comparison
    duration_diff_pct = None
    if candidate_duration_hours > 0.0 and recovered_duration_hours > 0.0:
        duration_diff_pct = abs(recovered_duration_hours - candidate_duration_hours) / candidate_duration_hours * 100.0

    return {
        "compared_to_ephemeris": True,
        "candidate_period_days": round(candidate_period, 5),
        "recovered_period_days": round(recovered_period, 5),
        "period_diff_percent": round(period_diff_pct, 3),
        "is_period_match": is_period_match,
        "is_harmonic": is_harmonic,
        "harmonic_relation": harmonic_relation,
        "epoch_diff_hours": round(epoch_diff_hours, 3),
        "candidate_duration_hours": round(candidate_duration_hours, 3),
        "recovered_duration_hours": round(recovered_duration_hours, 3),
        "duration_diff_percent": round(duration_diff_pct, 2) if duration_diff_pct is not None else None,
    }


def compute_bls_periodogram(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period_days: float = 0.5,
    max_period_days: Optional[float] = None,
    duration_grid_hours: Optional[Tuple[float, ...]] = None,
    candidate_period: Optional[float] = None,
    candidate_epoch: Optional[float] = None,
    candidate_duration: Optional[float] = None,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """
    Run Box Least Squares (BLS) transit periodogram on clean light-curve data.
    """
    t, f, fe, t_span, err = validate_and_prepare_timeseries(time, flux, flux_err, min_cadences=50)
    if err:
        return {
            "is_available": False,
            "algorithm": "Box Least Squares (BLS)",
            "reason": err,
        }

    # Constrain search range
    upper_period = max_period_days if max_period_days is not None else min(30.0, max(1.0, t_span / 2.0))
    lower_period = max(0.2, min_period_days)
    if upper_period <= lower_period:
        upper_period = lower_period + 1.0

    # Setup duration grid
    if duration_grid_hours is None:
        durations_days = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0]) / 24.0
    else:
        durations_days = np.array(duration_grid_hours) / 24.0
    # Keep only durations smaller than half the minimum period
    durations_days = durations_days[durations_days < (lower_period * 0.5)]
    if len(durations_days) == 0:
        durations_days = np.array([lower_period * 0.05, lower_period * 0.1])

    try:
        bls = BoxLeastSquares(t, f, dy=fe)
        # Use frequency factor auto-grid
        period_grid = np.linspace(lower_period, upper_period, 2000)
        periodogram = bls.power(period_grid, durations_days)

        best_idx = int(np.argmax(periodogram.power))
        best_period = float(periodogram.period[best_idx])
        best_power = float(periodogram.power[best_idx])
        best_duration_hours = float(periodogram.duration[best_idx] * 24.0)
        best_t0 = float(periodogram.transit_time[best_idx])
        best_depth_ppm = float(periodogram.depth[best_idx] * 1e6) if hasattr(periodogram, "depth") else 0.0

        # Compute SNR relative to background power
        finite_power = periodogram.power[np.isfinite(periodogram.power)]
        power_median = float(np.median(finite_power)) if len(finite_power) > 0 else 0.0
        power_std = float(np.std(finite_power)) if len(finite_power) > 0 else 1.0
        snr = float((best_power - power_median) / max(1e-6, power_std))

        # Sample period/power grid for plotting / JSON transport
        step = max(1, len(period_grid) // max_samples)
        sample_periods = period_grid[::step].tolist()
        sample_powers = periodogram.power[::step].tolist()

        res: Dict[str, Any] = {
            "is_available": True,
            "algorithm": "Box Least Squares (BLS)",
            "best_period_days": round(best_period, 5),
            "best_power": round(best_power, 6),
            "best_duration_hours": round(best_duration_hours, 3),
            "best_t0_btjd": round(best_t0, 4),
            "best_depth_ppm": round(best_depth_ppm, 1),
            "peak_snr": round(snr, 2),
            "min_period_days": round(lower_period, 3),
            "max_period_days": round(upper_period, 3),
            "cadence_count": len(t),
            "time_span_days": round(t_span, 3),
            "period_grid": sample_periods,
            "power_grid": sample_powers,
            "time_convention": "BJD - 2457000.0 (BTJD)",
            "provenance": {
                "engine": "astropy.timeseries.BoxLeastSquares",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        }

        # Compare with candidate ephemeris if available
        if candidate_period and candidate_epoch and candidate_duration:
            comp = compare_period_with_ephemeris(
                recovered_period=best_period,
                recovered_epoch=best_t0,
                recovered_duration_hours=best_duration_hours,
                candidate_period=candidate_period,
                candidate_epoch=candidate_epoch,
                candidate_duration_hours=candidate_duration,
            )
            res.update(comp)

        return res

    except Exception as e:
        return {
            "is_available": False,
            "algorithm": "Box Least Squares (BLS)",
            "reason": f"BLS computation failed: {e}",
        }


def compute_tls_periodogram(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period_days: float = 0.5,
    max_period_days: Optional[float] = None,
    candidate_period: Optional[float] = None,
    candidate_epoch: Optional[float] = None,
    candidate_duration: Optional[float] = None,
    oversampling_factor: int = 3,
    duration_grid_step: float = 1.1,
    max_samples: int = 500,
) -> Dict[str, Any]:
    """
    Run Transit Least Squares (TLS; Hippke & Heller 2019) on clean light-curve data.
    Uses realistic limb-darkened transit template models.
    """
    t, f, fe, t_span, err = validate_and_prepare_timeseries(time, flux, flux_err, min_cadences=50)
    if err:
        return {
            "is_available": False,
            "algorithm": "Transit Least Squares (TLS)",
            "reason": err,
        }

    try:
        from transitleastsquares import transitleastsquares
    except ImportError as exc:
        return {
            "is_available": False,
            "algorithm": "Transit Least Squares (TLS)",
            "reason": f"transitleastsquares library is not installed: {exc}",
        }

    # Constrain search range
    upper_period = max_period_days if max_period_days is not None else min(30.0, max(1.0, t_span / 2.0))
    lower_period = max(0.2, min_period_days)
    if upper_period <= lower_period:
        upper_period = lower_period + 1.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = transitleastsquares(t, f, dy=fe)
            tls_results = model.power(
                period_min=lower_period,
                period_max=upper_period,
                oversampling_factor=oversampling_factor,
                duration_grid_step=duration_grid_step,
                show_progress_bar=False,
            )

        best_period = _safe_float(getattr(tls_results, "period", None), 0.0)
        best_t0 = _safe_float(getattr(tls_results, "T0", None), 0.0)
        dur = _safe_float(getattr(tls_results, "duration", None), 0.0)
        best_duration_hours = dur * 24.0 if dur is not None else 0.0
        depth_val = _safe_float(getattr(tls_results, "depth", None), 1.0)
        best_depth_ppm = float((1.0 - depth_val) * 1e6) if depth_val is not None else 0.0
        sde = _safe_float(getattr(tls_results, "SDE", None), 0.0)
        snr = _safe_float(getattr(tls_results, "snr", None))
        odd_even_mismatch = _safe_float(getattr(tls_results, "odd_even_mismatch", None))
        transit_count = _safe_int(getattr(tls_results, "transit_count", None), 0)
        distinct_transit_count = _safe_int(getattr(tls_results, "distinct_transit_count", None), 0)
        empty_transit_count = _safe_int(getattr(tls_results, "empty_transit_count", None), 0)

        # Sample period / power spectrum
        raw_periods = np.array(tls_results.periods) if hasattr(tls_results, "periods") and isinstance(tls_results.periods, (np.ndarray, list)) else np.array([])
        raw_power = np.array(tls_results.power) if hasattr(tls_results, "power") and isinstance(tls_results.power, (np.ndarray, list)) else np.array([])
        if len(raw_periods) > 0 and len(raw_power) > 0:
            step = max(1, len(raw_periods) // max_samples)
            sample_periods = raw_periods[::step].tolist()
            sample_powers = raw_power[::step].tolist()
        else:
            sample_periods, sample_powers = [], []

        # Model folded curve for plotting
        model_phase = _safe_array_list(getattr(tls_results, "model_folded_phase", None))
        model_flux = _safe_array_list(getattr(tls_results, "model_folded_model", None))
        folded_phase = _safe_array_list(getattr(tls_results, "folded_phase", None))
        folded_y = _safe_array_list(getattr(tls_results, "folded_y", None))

        res: Dict[str, Any] = {
            "is_available": True,
            "algorithm": "Transit Least Squares (TLS)",
            "best_period_days": round(best_period, 5),
            "best_t0_btjd": round(best_t0, 4),
            "best_duration_hours": round(best_duration_hours, 3),
            "best_depth_ppm": round(best_depth_ppm, 1),
            "sde": round(sde, 2),
            "snr": round(snr, 2) if snr is not None else None,
            "odd_even_mismatch": round(odd_even_mismatch, 3) if odd_even_mismatch is not None else None,
            "transit_count": transit_count,
            "distinct_transit_count": distinct_transit_count,
            "empty_transit_count": empty_transit_count,
            "min_period_days": round(lower_period, 3),
            "max_period_days": round(upper_period, 3),
            "cadence_count": len(t),
            "time_span_days": round(t_span, 3),
            "period_grid": sample_periods,
            "power_grid": sample_powers,
            "model_folded_phase": model_phase[:500],
            "model_folded_flux": model_flux[:500],
            "folded_phase": folded_phase[::max(1, len(folded_phase) // 500)],
            "folded_y": folded_y[::max(1, len(folded_y) // 500)],
            "time_convention": "BJD - 2457000.0 (BTJD)",
            "provenance": {
                "engine": "transitleastsquares",
                "version": "2.0.2",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        }

        # Compare with candidate ephemeris if available
        if candidate_period and candidate_epoch and candidate_duration:
            comp = compare_period_with_ephemeris(
                recovered_period=best_period,
                recovered_epoch=best_t0,
                recovered_duration_hours=best_duration_hours,
                candidate_period=candidate_period,
                candidate_epoch=candidate_epoch,
                candidate_duration_hours=candidate_duration,
            )
            res.update(comp)

        return res

    except Exception as e:
        return {
            "is_available": False,
            "algorithm": "Transit Least Squares (TLS)",
            "reason": f"TLS computation failed: {e}",
        }


def run_periodicity_search(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    min_period_days: float = 0.5,
    max_period_days: Optional[float] = None,
    candidate_period: Optional[float] = None,
    candidate_epoch: Optional[float] = None,
    candidate_duration: Optional[float] = None,
    run_bls: bool = True,
    run_tls: bool = True,
) -> Dict[str, Any]:
    """
    Execute unified periodicity screening using both BLS and TLS algorithms.
    Cross-checks results and verifies consistency with candidate ephemeris.
    """
    bls_res = (
        compute_bls_periodogram(
            time=time,
            flux=flux,
            flux_err=flux_err,
            min_period_days=min_period_days,
            max_period_days=max_period_days,
            candidate_period=candidate_period,
            candidate_epoch=candidate_epoch,
            candidate_duration=candidate_duration,
        )
        if run_bls
        else {"is_available": False, "reason": "BLS execution disabled."}
    )

    tls_res = (
        compute_tls_periodogram(
            time=time,
            flux=flux,
            flux_err=flux_err,
            min_period_days=min_period_days,
            max_period_days=max_period_days,
            candidate_period=candidate_period,
            candidate_epoch=candidate_epoch,
            candidate_duration=candidate_duration,
        )
        if run_tls
        else {"is_available": False, "reason": "TLS execution disabled."}
    )

    # Determine cross-algorithm agreement
    period_agreement = False
    agreement_diff_pct = None
    if bls_res.get("is_available") and tls_res.get("is_available"):
        p_bls = bls_res["best_period_days"]
        p_tls = tls_res["best_period_days"]
        agreement_diff_pct = abs(p_bls - p_tls) / max(1e-4, p_tls) * 100.0
        # Agreement within 1.5% or harmonic
        period_agreement = bool(
            agreement_diff_pct < 1.5
            or abs((p_bls / p_tls) - 0.5) < 0.015
            or abs((p_bls / p_tls) - 2.0) < 0.015
        )

    # Recommend recovered ephemeris when candidate ephemeris was absent
    recovered_ephemeris = None
    if tls_res.get("is_available"):
        recovered_ephemeris = {
            "period_days": tls_res["best_period_days"],
            "epoch_btjd": tls_res["best_t0_btjd"],
            "duration_hours": tls_res["best_duration_hours"],
            "depth_ppm": tls_res["best_depth_ppm"],
            "source_algorithm": "TLS",
            "detection_metric": f"SDE={tls_res.get('sde', 0.0):.1f}",
        }
    elif bls_res.get("is_available"):
        recovered_ephemeris = {
            "period_days": bls_res["best_period_days"],
            "epoch_btjd": bls_res["best_t0_btjd"],
            "duration_hours": bls_res["best_duration_hours"],
            "depth_ppm": bls_res["best_depth_ppm"],
            "source_algorithm": "BLS",
            "detection_metric": f"SNR={bls_res.get('peak_snr', 0.0):.1f}",
        }

    return {
        "bls": bls_res,
        "tls": tls_res,
        "period_agreement": period_agreement,
        "period_agreement_diff_percent": round(agreement_diff_pct, 3) if agreement_diff_pct is not None else None,
        "recovered_ephemeris": recovered_ephemeris,
    }
