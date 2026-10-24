"""
Candidate records and ephemeris data contracts.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator


class Disposition(str, Enum):
    """
    Standardized astronomical disposition statuses.
    Scientific rule: Unresolved records must NOT silently become negative examples.
    """
    CONFIRMED_PLANET = "CONFIRMED_PLANET"
    PLANET_CANDIDATE = "PLANET_CANDIDATE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_ALARM = "FALSE_ALARM"
    AMBIGUOUS_UNRESOLVED = "AMBIGUOUS_UNRESOLVED"


class Ephemeris(BaseModel):
    """Transit ephemeris with explicit units and time conventions."""
    period_days: float = Field(..., gt=0.0, description="Orbital period in days")
    period_err_days: Optional[float] = Field(None, ge=0.0, description="Uncertainty in period (days)")
    epoch_bjd_tdb: float = Field(..., description="Transit epoch in BJD_TDB or BTJD (BTJD = BJD - 2457000.0)")
    epoch_err_days: Optional[float] = Field(None, ge=0.0, description="Uncertainty in epoch (days)")
    duration_hours: float = Field(..., gt=0.0, description="Transit duration in hours")
    duration_err_hours: Optional[float] = Field(None, ge=0.0, description="Uncertainty in duration (hours)")
    time_convention: str = Field(default="BTJD", description="Time standard (BTJD / BJD_TDB)")
    transit_depth_ppm: Optional[float] = Field(None, ge=0.0, description="Transit depth in parts per million")


class CandidateRecord(BaseModel):
    """
    Canonical Candidate Record per ASTROVET Blueprint Section 8.
    Never uses TIC ID alone as primary event key.
    """
    candidate_id: str = Field(..., description="Unique ASTROVET identifier (e.g. 390651552-1-S23)")
    tic_id: int = Field(..., gt=0, description="Target star TIC identifier")
    planet_num: int = Field(..., gt=0, description="TCE planet number for this star")
    sector_run: str = Field(..., description="Sector or multi-sector range (e.g. '23' or '1-67')")
    source_event_id: Optional[str] = Field(None, description="Upstream TCE UID or TOI string")
    toi_id: Optional[str] = Field(None, description="Matched TOI ID if available")
    mission: str = Field(default="TESS", description="Observing mission")
    pipeline: str = Field(default="SPOC", description="Upstream reduction pipeline")
    cadence_seconds: float = Field(default=120.0, gt=0.0, description="Nominal observation cadence in seconds")
    ephemeris: Ephemeris = Field(..., description="Ephemeris parameters")
    
    # Stellar & system context
    stellar_radius_solar: Optional[float] = Field(None, gt=0.0)
    stellar_mass_solar: Optional[float] = Field(None, gt=0.0)
    stellar_teff_k: Optional[float] = Field(None, gt=0.0)
    gaia_ruwe: Optional[float] = Field(None, ge=0.0)
    
    # Ground-truth source labels (kept strictly separate from predictions)
    label_raw: Optional[str] = Field(None, description="Raw label from source catalog")
    label_normalized: Optional[Disposition] = Field(None, description="Normalized ASTROVET disposition")
    label_source: Optional[str] = Field(None, description="Catalog release and access date of the label")
    
    # Provenance
    product_manifest_id: Optional[str] = Field(None, description="Checksum set ID of source files")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_duration_less_than_period(self) -> CandidateRecord:
        period_hours = self.ephemeris.period_days * 24.0
        if self.ephemeris.duration_hours >= period_hours:
            raise ValueError(
                f"Physical impossibility: Transit duration ({self.ephemeris.duration_hours}h) "
                f"must be strictly less than orbital period ({period_hours}h)."
            )
        return self
