"""
Candidate identity resolution, normalization, and validation.

Scientific safeguard:
TIC ID alone is a stellar target identifier, NOT a unique transit event key.
A single star can host multiple exoplanets, multiple candidate signals, be observed across
multiple sectors, and have multiple SPOC pipeline runs. ASTROVET enforces compound event keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CandidateIdentity:
    """Canonical compound identifier for a transit event."""
    candidate_id: str
    tic_id: int
    planet_num: int
    sector_run: str
    source_event_id: Optional[str] = None
    toi_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tic_id, int) or self.tic_id <= 0:
            raise ValueError(f"tic_id must be a positive integer, got {self.tic_id}")
        if not isinstance(self.planet_num, int) or self.planet_num <= 0:
            raise ValueError(f"planet_num must be a positive integer, got {self.planet_num}")
        if not self.sector_run or not isinstance(self.sector_run, str):
            raise ValueError(f"sector_run must be a non-empty string, got {self.sector_run}")
        if not self.candidate_id or not isinstance(self.candidate_id, str):
            raise ValueError(f"candidate_id must be a non-empty string, got {self.candidate_id}")


def make_canonical_candidate_id(tic_id: int, planet_num: int, sector_run: str) -> str:
    """
    Construct canonical candidate_id matching SPOC / ExoMiner standard:
    <tic_id>-<planet_num>-S<sector_run> (e.g. 390651552-1-S23).
    """
    clean_sec = sector_run.strip()
    if clean_sec.upper().startswith("S"):
        clean_sec = clean_sec[1:]
    return f"{int(tic_id)}-{int(planet_num)}-S{clean_sec}"


def parse_candidate_string(identifier: str) -> Tuple[int, int, str]:
    """
    Parse various common string representations into (tic_id, planet_num, sector_run).
    Supports:
    - SPOC TCE UID: '390651552-1-S23' or '390651552-1-S1-3'
    - Space/underscore: 'TIC390651552_1_S23', 'TIC 390651552 1 S23'
    - Tuple-like string: '390651552, 1, 23'
    """
    raw = identifier.strip()

    # Pattern 1: standard SPOC UID 123456789-1-S23 or 123456789-1-S1-67
    m1 = re.match(r"^TIC\s*0*([0-9]+)[-_ ]+([0-9]+)[-_ ]+S?([0-9]+(?:-[0-9]+)?)$", raw, re.IGNORECASE)
    if m1:
        return int(m1.group(1)), int(m1.group(2)), m1.group(3)

    m2 = re.match(r"^0*([0-9]+)[-_ ]+([0-9]+)[-_ ]+S?([0-9]+(?:-[0-9]+)?)$", raw, re.IGNORECASE)
    if m2:
        return int(m2.group(1)), int(m2.group(2)), m2.group(3)

    # Rejection of naked TIC ID
    m_tic_only = re.match(r"^(?:TIC\s*)?0*([0-9]+)$", raw, re.IGNORECASE)
    if m_tic_only:
        raise ValueError(
            f"Provided identifier '{identifier}' is a raw stellar TIC ID without planet number "
            f"or sector run. TIC ID alone cannot uniquely identify a transit candidate event."
        )

    raise ValueError(f"Unable to parse candidate identifier format: '{identifier}'")


def resolve_candidate_identity(
    candidate_id: Optional[str] = None,
    tic_id: Optional[int] = None,
    planet_num: Optional[int] = None,
    sector_run: Optional[str] = None,
    source_event_id: Optional[str] = None,
    toi_id: Optional[str] = None,
) -> CandidateIdentity:
    """
    Validate and return a canonical CandidateIdentity.
    Guarantees no ambiguity and creates canonical candidate_id.
    """
    if candidate_id:
        parsed_tic, parsed_plnt, parsed_sec = parse_candidate_string(candidate_id)
        final_tic = tic_id or parsed_tic
        final_plnt = planet_num or parsed_plnt
        final_sec = sector_run or parsed_sec
    else:
        if tic_id is None or planet_num is None or sector_run is None:
            raise ValueError("Must provide either a parseable candidate_id or (tic_id, planet_num, sector_run)")
        final_tic = tic_id
        final_plnt = planet_num
        final_sec = str(sector_run)

    canonical_id = make_canonical_candidate_id(final_tic, final_plnt, final_sec)
    return CandidateIdentity(
        candidate_id=canonical_id,
        tic_id=final_tic,
        planet_num=final_plnt,
        sector_run=final_sec,
        source_event_id=source_event_id,
        toi_id=toi_id,
    )
