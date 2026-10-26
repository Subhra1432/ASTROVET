"""
Quality bitmask filtering policies for TESS SPOC light curves.

Scientific safeguards:
- Apply quality filtering with a named, auditable bitmask policy.
- Log retained and rejected cadence counts and breakdown by flag bit.
- Preserve time base and gap coordinates; never silently delete bad cadences without accounting.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

# Standard TESS SPOC quality flag definitions
SPOC_FLAG_DESCRIPTIONS = {
    1: "Attitude tweak",
    2: "Safe mode",
    4: "Spacecraft desaturation (momentum dump)",
    8: "Coarse point",
    16: "Earth point",
    32: "Zero crossing",
    64: "Reaction wheel desaturation",
    128: "Cosmic ray in collateral data",
    256: "Manual exclusion",
    512: "Discontinuity corrected",
    1024: "Impulsive outlier",
    2048: "Cosmic ray in optimal aperture",
    4096: "Stray light threshold exceeded",
}

# Standard recommended SPOC quality bitmask: flags that compromise photometric precision
DEFAULT_SPOC_QUALITY_BITMASK = 1 | 2 | 4 | 8 | 16 | 32 | 128 | 256 | 2048 | 4096  # 6575


def filter_quality_cadences(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
    bitmask: int = DEFAULT_SPOC_QUALITY_BITMASK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Filter light-curve cadences according to the specified bitmask policy.
    Returns:
        (clean_time, clean_flux, clean_flux_err, quality_report)
    """
    total_cadences = len(time)
    
    # Finite checks (finite time and flux)
    finite_mask = np.isfinite(time) & np.isfinite(flux) & (flux > 0.0)
    
    # Bitmask checks
    quality_mask = (quality & bitmask) == 0
    
    retained_mask = finite_mask & quality_mask
    retained_count = int(np.sum(retained_mask))
    rejected_count = total_cadences - retained_count
    
    # Detailed breakdown of flagged bits
    flag_breakdown = {}
    for bit, desc in SPOC_FLAG_DESCRIPTIONS.items():
        count = int(np.sum((quality & bit) > 0))
        if count > 0:
            flag_breakdown[f"bit_{bit}_{desc}"] = count

    non_finite_count = int(total_cadences - np.sum(finite_mask))

    quality_report = {
        "bitmask_used": bitmask,
        "total_cadences": total_cadences,
        "retained_cadences": retained_count,
        "rejected_cadences": rejected_count,
        "retained_fraction": round(retained_count / max(1, total_cadences), 4),
        "non_finite_cadences": non_finite_count,
        "flag_breakdown": flag_breakdown,
    }

    return (
        time[retained_mask],
        flux[retained_mask],
        flux_err[retained_mask],
        quality_report,
    )
