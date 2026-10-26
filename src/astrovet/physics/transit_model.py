"""
Approximate Transit Diagnostic Model and Residual Estimator.

Scientific status & limitations (Crucial scientific safeguard):
----------------------------------------------------------------
This module implements a fast, parameterized geometric transit template with quadratic
limb-darkening intensity weighting for rapid candidate screening and residual calculation.
It is an APPROXIMATE DIAGNOSTIC TEMPLATE designed for candidate prioritization and triage,
NOT a validated physical parameter inversion (such as full numerical Mandel & Agol (2002)
integration provided by heavy packages like BATMAN or PyTransit).

Parameters derived from this fit (e.g. fitted depth, duration, SNR, chi2) serve as
relative consistency diagnostics and residual baselines, and should NOT be cited as
rigorous physical orbital solutions or planet confirmations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import curve_fit


@dataclass
class ApproximateTransitFitResult:
    """Results of fitting the approximate transit diagnostic template."""
    fitted_depth_ppm: float
    fitted_depth_err_ppm: float
    fitted_duration_hours: float
    fitted_t0_offset_hours: float
    transit_snr: float
    chi2: float
    reduced_chi2: float
    model_flux: np.ndarray
    residuals: np.ndarray
    is_converged: bool
    model_type: str = "approximate_limb_darkened_diagnostic_template"
    is_validated_physical_inversion: bool = False


# Backward compatibility alias
TransitFitResult = ApproximateTransitFitResult


def approximate_limb_darkened_transit(
    phase_hours: np.ndarray,
    depth: float,
    duration_hours: float,
    t0_offset_hours: float = 0.0,
    limb_u1: float = 0.35,
    limb_u2: float = 0.20,
) -> np.ndarray:
    """
    Approximate geometric transit profile with quadratic limb-darkening weighting.
    Serves as an empirical template for residual estimation and shape consistency.
    """
    t = phase_hours - t0_offset_hours
    half_dur = max(1e-4, duration_hours / 2.0)
    
    # Ingress/egress duration approximation (~15% of half-duration)
    tau = 0.15 * half_dur
    t_flat = max(0.0, half_dur - tau)
    
    abs_t = np.abs(t)
    model = np.ones_like(t)

    # In-transit flat bottom region with limb darkening curvature
    flat_mask = abs_t <= t_flat
    ingress_mask = (abs_t > t_flat) & (abs_t < half_dur)

    # Limb darkening profile across transit floor
    if half_dur > 0:
        r_coord = np.clip(abs_t[flat_mask] / half_dur, 0.0, 1.0)
        mu = np.sqrt(np.maximum(1e-6, 1.0 - r_coord**2))
        intensity = 1.0 - limb_u1 * (1.0 - mu) - limb_u2 * (1.0 - mu)**2
        norm_intensity = intensity / max(1e-6, 1.0 - limb_u1 / 3.0 - limb_u2 / 6.0)
        model[flat_mask] = 1.0 - depth * norm_intensity

    # Smooth ingress/egress transition
    if np.any(ingress_mask):
        frac = (half_dur - abs_t[ingress_mask]) / tau
        ingress_depth = depth * 0.5 * (1.0 - np.cos(np.pi * frac))
        model[ingress_mask] = 1.0 - ingress_depth

    return model


def fit_transit_model(
    phase_hours: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    initial_duration_hours: float,
    initial_depth_ppm: Optional[float] = None,
) -> ApproximateTransitFitResult:
    """
    Fit the approximate transit diagnostic template to phase-folded data.
    Explicitly labeled as an approximate screening diagnostic.
    """
    valid_mask = np.isfinite(phase_hours) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)
    p_valid = phase_hours[valid_mask]
    f_valid = flux[valid_mask]
    fe_valid = flux_err[valid_mask]

    n_points = len(p_valid)
    if n_points < 10:
        return ApproximateTransitFitResult(
            fitted_depth_ppm=0.0,
            fitted_depth_err_ppm=0.0,
            fitted_duration_hours=initial_duration_hours,
            fitted_t0_offset_hours=0.0,
            transit_snr=0.0,
            chi2=0.0,
            reduced_chi2=1.0,
            model_flux=np.ones_like(phase_hours),
            residuals=flux - 1.0,
            is_converged=False,
            is_validated_physical_inversion=False,
        )

    near_center = np.abs(p_valid) < (initial_duration_hours / 3.0)
    if initial_depth_ppm is not None and initial_depth_ppm > 0:
        init_depth_frac = initial_depth_ppm * 1e-6
    elif np.sum(near_center) >= 3:
        init_depth_frac = max(1e-5, float(1.0 - np.median(f_valid[near_center])))
    else:
        init_depth_frac = 0.001

    p0 = [init_depth_frac, max(0.5, initial_duration_hours), 0.0]
    bounds = (
        [1e-6, 0.2, -initial_duration_hours * 0.5],
        [0.20, initial_duration_hours * 3.0, initial_duration_hours * 0.5],
    )

    try:
        popt, pcov = curve_fit(
            lambda t, d, dur, t0: approximate_limb_darkened_transit(t, d, dur, t0),
            p_valid,
            f_valid,
            p0=p0,
            bounds=bounds,
            sigma=fe_valid,
            maxfev=2000,
        )
        fitted_depth_frac = float(popt[0])
        fitted_duration = float(popt[1])
        fitted_t0 = float(popt[2])
        depth_err_frac = float(np.sqrt(np.diag(pcov))[0]) if pcov is not None else init_depth_frac * 0.1
        is_converged = True
    except Exception:
        fitted_depth_frac = init_depth_frac
        fitted_duration = initial_duration_hours
        fitted_t0 = 0.0
        depth_err_frac = init_depth_frac * 0.5
        is_converged = False

    full_model = approximate_limb_darkened_transit(phase_hours, fitted_depth_frac, fitted_duration, fitted_t0)
    residuals = flux - full_model

    val_res = residuals[valid_mask]
    val_fe = flux_err[valid_mask]
    chi2 = float(np.sum((val_res / val_fe)**2))
    dof = max(1, len(val_res) - 3)
    reduced_chi2 = float(chi2 / dof)
    transit_snr = float(fitted_depth_frac / max(1e-7, depth_err_frac))

    return ApproximateTransitFitResult(
        fitted_depth_ppm=round(fitted_depth_frac * 1e6, 2),
        fitted_depth_err_ppm=round(depth_err_frac * 1e6, 2),
        fitted_duration_hours=round(fitted_duration, 3),
        fitted_t0_offset_hours=round(fitted_t0, 4),
        transit_snr=round(transit_snr, 2),
        chi2=round(chi2, 2),
        reduced_chi2=round(reduced_chi2, 3),
        model_flux=full_model,
        residuals=residuals,
        is_converged=is_converged,
        is_validated_physical_inversion=False,
    )
