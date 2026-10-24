"""
ExoMiner++ baseline prediction contracts.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class BaselinePrediction(BaseModel):
    """
    Standardized output from the official NASA ExoMiner++ baseline adapter.
    Scientific rule: Baseline predictions are never treated as ground truth labels.
    If a candidate is not found in the baseline catalog, score is None (not a neutral 0.50).
    """
    candidate_id: str = Field(..., description="Canonical candidate identifier")
    upstream_model: str = Field(default="ExoMiner++_v1.0.0", description="Upstream model version / tag")
    
    # Probabilistic outputs (None if candidate was not scored / not found)
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall ExoMiner score (planet confidence)")
    score_pc: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score for Planet Candidate class")
    score_afp: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score for Astrophysical False Positive")
    score_ntp: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score for Non-Transiting Phenomenon (false alarm)")
    score_uncertainty: Optional[float] = Field(None, ge=0.0, description="Standard deviation across ensemble models")
    
    # Execution and provenance
    status: str = Field(default="SUCCESS", description="Execution status: SUCCESS, NOT_FOUND, ERROR")
    execution_mode: str = Field(default="CATALOG", description="Source: CATALOG, SYNTHETIC_FIXTURE")
    is_synthetic: bool = Field(default=False, description="Flag indicating mock/synthetic fixture data")
    
    # Overlap safeguards
    checkpoint_overlap_audited: bool = Field(default=False, description="Whether checkpoint training set was audited")
    checkpoint_overlap_status: str = Field(
        default="UNKNOWN",
        description="Overlap status: KNOWN_DISJOINT, POTENTIAL_OVERLAP, KNOWN_TRAIN, UNKNOWN"
    )
    
    raw_output: Dict[str, Any] = Field(default_factory=dict, description="Raw outputs from upstream pipeline or catalog")
