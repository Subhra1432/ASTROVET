"""
Secondary eclipse diagnostics for occulting stellar companions.

Scientific safeguard:
A planet candidate should not show a secondary eclipse deeper than physically feasible
for thermal planetary emission or reflected light (typically << 100 ppm for small planets).
A prominent secondary eclipse at phase ~0.5 is a classic indicator of an eclipsing binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from astrovet.views.folding import FoldedView


@dataclass
class SecondaryDiagnostic:
    secondary_depth_ppm: float
    secondary_snr: float
    secondary_depth_ratio: float
    secondary_detected: bool
    is_available: bool
    message: str


def compute_secondary_diagnostic(
    full_orbit_view: FoldedView,
    primary_depth_ppm: float,
    duration_hours: float,
    period_days: float,
) -> SecondaryDiagnostic:
    """
    Search for a secondary eclipse around orbital phase 0.5 (or -0.5).
    """
    if len(full_orbit_view.phase) < 20 or primary_depth_ppm <= 0:
        return SecondaryDiagnostic(
            secondary_depth_ppm=0.0,
            secondary_snr=0.0,
            secondary_depth_ratio=0.0,
            secondary_detected=False,
            is_available=False,
            message="Insufficient full-orbit data or zero primary depth.",
        )

    # Secondary window: +/- 1.5 * duration around phase 0.5 (wrapping at +/- 0.5)
    dur_phase = (duration_hours / 24.0) / period_days
    phase_dist_from_half = np.minimum(
        np.abs(full_orbit_view.phase - 0.5),
        np.abs(full_orbit_view.phase + 0.5)
    )
    
    sec_mask = phase_dist_from_half <= dur_phase
    out_mask = (np.abs(full_orbit_view.phase) > dur_phase * 2.0) & (~sec_mask)

    if np.sum(sec_mask) < 2 or np.sum(out_mask) < 5:
        return SecondaryDiagnostic(
            secondary_depth_ppm=0.0,
            secondary_snr=0.0,
            secondary_depth_ratio=0.0,
            secondary_detected=False,
            is_available=False,
            message="Insufficient cadences in secondary eclipse phase window.",
        )

    baseline = float(np.nanmedian(full_orbit_view.flux[out_mask]))
    baseline_noise = float(np.nanstd(full_orbit_view.flux[out_mask]))
    sec_flux = float(np.nanmedian(full_orbit_view.flux[sec_mask]))

    sec_depth_frac = max(0.0, baseline - sec_flux)
    sec_depth_ppm = sec_depth_frac * 1e6
    noise_floor = max(1e-7, baseline_noise / np.sqrt(np.sum(sec_mask)))
    sec_snr = float(sec_depth_frac / noise_floor)
    
    depth_ratio = float(sec_depth_ppm / max(1.0, primary_depth_ppm))

    # A secondary is significantly detected if SNR > 3.0 and depth ratio > 0.08
    is_detected = bool(sec_snr >= 3.0 and depth_ratio >= 0.08 and sec_depth_ppm > 80.0)

    msg = (
        f"Significant secondary eclipse detected (depth={sec_depth_ppm:.1f} ppm, "
        f"SNR={sec_snr:.1f}, ratio={depth_ratio:.2%}). Likely eclipsing binary."
        if is_detected
        else "No significant secondary eclipse detected."
    )

    return SecondaryDiagnostic(
        secondary_depth_ppm=round(sec_depth_ppm, 2),
        secondary_snr=round(sec_snr, 2),
        secondary_depth_ratio=round(depth_ratio, 4),
        secondary_detected=is_detected,
        is_available=True,
        message=msg,
    )
