"""
Scientific reason codes and evidence mapping for human reviewers.

Scientific safeguard:
Explanations describe observed model and diagnostic behaviour, NOT causal physical proof.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from astrovet.schemas.prediction import PhysicsDiagnostics, BaselinePrediction


REASON_CODE_DESCRIPTIONS = {
    "ODD_EVEN_DEPTH_MISMATCH": (
        "Odd and even transit depths differ significantly (>3 sigma), "
        "a classic hallmark of an eclipsing binary with alternating primary and secondary eclipses."
    ),
    "SIGNIFICANT_SECONDARY_ECLIPSE": (
        "A statistically significant secondary eclipse is detected around orbital phase 0.5, "
        "indicating an occulting companion or binary system."
    ),
    "UNPHYSICAL_TRANSIT_GEOMETRY": (
        "Transit duration or depth is incompatible with physical orbital mechanics "
        "(e.g., duration >= orbital period or unfeasible impact parameter)."
    ),
    "TRANSIT_FIT_HIGH_RESIDUALS": (
        "Transit fit exhibits abnormally high residuals (reduced chi-squared > 3.0), "
        "suggesting stellar variability, flare, or instrumental systematic rather than clean transit."
    ),
    "BASELINE_PHYSICS_DISAGREEMENT": (
        "High disagreement between reference baseline (ExoMiner++) and physical consistency diagnostics."
    ),
    "HIGH_PREDICTIVE_UNCERTAINTY": (
        "Total predictive uncertainty exceeds confidence threshold; human inspection required."
    ),
    "UNCALIBRATED_HEURISTIC_SCORE": (
        "No fitted fusion model or validated calibration was supplied. The score is an experimental heuristic, so the candidate remains in human review."
    ),
    "HIGH_EVENT_SCATTER": (
        "Individual transits exhibit erratic depth or timing variations across observing sectors."
    ),
    "PERIODICITY_MISMATCH": (
        "Periodogram search (BLS/TLS) recovered a dominant periodicity differing significantly (>1.5%) "
        "from the candidate catalog ephemeris, suggesting possible misidentification or alias."
    ),
    "PERIODICITY_HARMONIC_DETECTED": (
        "Periodogram search detected an integer multiple or submultiple (harmonic/subharmonic) of the reported period, "
        "often indicative of equal-depth eclipsing binaries (period doubled) or aliases."
    ),
    "TLS_ODD_EVEN_MISMATCH": (
        "Transit Least Squares (TLS) template search detected an odd/even transit depth mismatch, "
        "pointing to an alternating occultation signature."
    ),
    "PERIODICITY_NOT_DETECTED": (
        "Independent periodogram search (BLS/TLS) did not recover a significant transit signature "
        "at the reported ephemeris."
    ),
}


def get_reason_explanation(reason_code: str) -> str:
    """Return plain-English scientific explanation for a given reason code."""
    if reason_code in REASON_CODE_DESCRIPTIONS:
        return REASON_CODE_DESCRIPTIONS[reason_code]
    elif reason_code.startswith("MISSING_MODALITIES"):
        return "One or more expected diagnostic modalities were unavailable in the input data products."
    return f"Diagnostic observation: {reason_code}"
