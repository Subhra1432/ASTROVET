"""
ASTROVET prediction contracts and triage states.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from astrovet.schemas.baseline import BaselinePrediction


class TriageState(str, Enum):
    """
    Candidate triage classification.
    High model score is a prioritization recommendation for human inspection,
    NEVER a planet confirmation or statistical validation.
    """
    PLANET_LIKE = "PLANET_LIKE"
    FALSE_POSITIVE_LIKE = "FALSE_POSITIVE_LIKE"
    AMBIGUOUS = "AMBIGUOUS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class PhysicsDiagnostics(BaseModel):
    """
    Physics-informed diagnostics with explicit availability and uncertainty.
    Never replaces missing diagnostics with fabricated zeros.
    """
    transit_depth_fitted_ppm: Optional[float] = None
    transit_depth_fitted_err_ppm: Optional[float] = None
    transit_duration_fitted_hours: Optional[float] = None
    transit_snr: Optional[float] = None
    fit_chi2_reduced: Optional[float] = None
    
    # Odd vs Even transit diagnostics (identifies eclipsing binaries)
    odd_depth_ppm: Optional[float] = None
    even_depth_ppm: Optional[float] = None
    odd_even_depth_diff_ppm: Optional[float] = None
    odd_even_t_statistic: Optional[float] = None
    odd_even_p_value: Optional[float] = None
    odd_even_consistent: Optional[bool] = None
    
    # Secondary eclipse diagnostics (identifies secondary occultations)
    secondary_depth_ppm: Optional[float] = None
    secondary_snr: Optional[float] = None
    secondary_depth_ratio: Optional[float] = None
    secondary_eclipse_detected: Optional[bool] = None
    
    # Event-to-event consistency
    transit_count_observed: Optional[int] = None
    transit_depth_scatter_ppm: Optional[float] = None
    transit_timing_scatter_minutes: Optional[float] = None
    
    # Physical plausibility validation
    is_period_positive: bool = True
    is_duration_feasible: bool = True
    is_depth_feasible: bool = True
    plausibility_all_passed: bool = True
    
    # Periodicity verification diagnostics (BLS & TLS)
    bls_best_period_days: Optional[float] = None
    bls_snr: Optional[float] = None
    bls_period_diff_pct: Optional[float] = None
    bls_is_match: Optional[bool] = None
    
    tls_best_period_days: Optional[float] = None
    tls_sde: Optional[float] = None
    tls_snr: Optional[float] = None
    tls_odd_even_mismatch: Optional[float] = None
    tls_period_diff_pct: Optional[float] = None
    tls_is_match: Optional[bool] = None
    
    period_search_agreement: Optional[bool] = None
    periodogram_summary: Dict[str, Any] = Field(default_factory=dict)

    # Explicit availability tracking
    modality_availability: Dict[str, bool] = Field(default_factory=dict)


class AstrovetPrediction(BaseModel):
    """
    Complete ASTROVET vetting and prioritization report contract.
    """
    candidate_id: str = Field(..., description="Canonical candidate identifier")
    tic_id: int = Field(..., description="Host star TIC ID")
    
    # Upstream baseline reference
    baseline_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    baseline_uncertainty: Optional[float] = Field(None, ge=0.0)
    baseline_prediction: Optional[BaselinePrediction] = None
    
    # ASTROVET Hybrid Scores
    astrovet_raw_score: float = Field(..., ge=0.0, le=1.0, description="Raw heuristic or model fusion score")
    score_type: str = Field(default="uncalibrated_heuristic", description="Score derivation: uncalibrated_heuristic or fitted_fusion_score")
    is_calibrated: bool = Field(default=False, description="Whether probability calibration was applied from a fitted calibrator")
    calibration_provenance: Optional[str] = Field(None, description="Provenance of calibration dataset if calibrated")
    calibrated_planet_like_probability: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Calibrated probability if fitted calibrator was supplied, else None"
    )
    triage_state: TriageState = Field(..., description="Actionable triage status")
    
    # Uncertainty & OOD Flags
    predictive_uncertainty: float = Field(..., ge=0.0, description="Total predictive uncertainty estimate")
    ood_flag: bool = Field(default=False, description="Out-Of-Distribution detection flag")
    missing_modalities: List[str] = Field(default_factory=list, description="List of unavailable data modalities")
    
    # Evidence & Explanations
    physics_diagnostics: PhysicsDiagnostics = Field(default_factory=PhysicsDiagnostics)
    periodogram_diagnostics: Optional[Dict[str, Any]] = Field(default=None, description="BLS and TLS periodicity search outputs")
    reason_codes: List[str] = Field(default_factory=list, description="Machine and human reviewable reason codes")
    explanation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-candidate score decomposition; explicitly distinguishes learned contributions from heuristic terms",
    )
    
    # Provenance
    is_synthetic: bool = Field(default=False, description="Flag indicating synthetic demonstration/test fixture")
    model_version: str = Field(default="astrovet_0.1.0")
    run_id: str = Field(..., description="Unique execution run identifier")
