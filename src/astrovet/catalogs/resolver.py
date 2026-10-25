"""
Catalog candidate resolution, parsing, and data validation.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import pandas as pd

from astrovet.ids import resolve_candidate_identity
from astrovet.schemas.candidate import CandidateRecord, Ephemeris, Disposition


def parse_disposition(raw_label: Optional[str]) -> Disposition:
    """Map raw catalog strings to standard ASTROVET Disposition enum."""
    if raw_label is None or pd.isna(raw_label):
        return Disposition.AMBIGUOUS_UNRESOLVED
    
    clean = str(raw_label).strip().upper()
    if clean in ["CP", "CONFIRMED", "CONFIRMED_PLANET", "KNOWN_PLANET", "KP"]:
        return Disposition.CONFIRMED_PLANET
    elif clean in ["PC", "CANDIDATE", "PLANET_CANDIDATE"]:
        return Disposition.PLANET_CANDIDATE
    elif clean in ["FP", "FALSE_POSITIVE", "AFP", "EB"]:
        return Disposition.FALSE_POSITIVE
    elif clean in ["FA", "FALSE_ALARM", "NTP", "NOISE", "INSTRUMENTAL"]:
        return Disposition.FALSE_ALARM
    else:
        return Disposition.AMBIGUOUS_UNRESOLVED


def dataframe_to_candidate_records(
    df: pd.DataFrame,
    source_name: str = "TESS_SPOC_CATALOG",
) -> List[CandidateRecord]:
    """
    Convert a catalog DataFrame (e.g. ExoMiner PC catalog or SPOC TCE table)
    into a list of validated CandidateRecord instances.
    """
    records: List[CandidateRecord] = []
    
    # Identify column mappings flexibly
    col_map = {}
    for col in df.columns:
        norm = col.strip().lower()
        if norm in ["tce id", "uid", "tce_id"]:
            col_map["tce_id"] = col
        elif norm in ["tic id", "target_id", "tic", "tid"]:
            col_map["tic_id"] = col
        elif norm in ["sector run", "sector_run", "sector"]:
            col_map["sector_run"] = col
        elif norm in ["planet number", "tce_plnt_num", "planet_num", "pn"]:
            col_map["planet_num"] = col
        elif norm in ["orbital period [day]", "tce_period", "period", "period_days"]:
            col_map["period"] = col
        elif norm in ["transit epoch [btjd]", "tce_time0bk", "epoch", "time0"]:
            col_map["epoch"] = col
        elif norm in ["transit duration [hour]", "tce_duration", "duration", "duration_hours"]:
            col_map["duration"] = col
        elif norm in ["transit depth [ppm]", "tce_depth", "depth", "depth_ppm"]:
            col_map["depth"] = col
        elif norm in ["gaia ruwe", "ruwe"]:
            col_map["ruwe"] = col
        elif norm in ["label", "disposition", "disp", "tfopwg_disp"]:
            col_map["label"] = col

    for idx, row in df.iterrows():
        # Candidate Identity
        raw_tce_id = str(row[col_map["tce_id"]]) if "tce_id" in col_map else None
        raw_tic = int(row[col_map["tic_id"]]) if "tic_id" in col_map else None
        raw_plnt = int(row[col_map["planet_num"]]) if "planet_num" in col_map else 1
        raw_sec = str(row[col_map["sector_run"]]) if "sector_run" in col_map else "1"

        try:
            cid = resolve_candidate_identity(
                candidate_id=raw_tce_id,
                tic_id=raw_tic,
                planet_num=raw_plnt,
                sector_run=raw_sec,
            )
        except Exception:
            continue

        # Ephemeris
        period = float(row[col_map["period"]]) if "period" in col_map and not pd.isna(row[col_map["period"]]) else None
        epoch = float(row[col_map["epoch"]]) if "epoch" in col_map and not pd.isna(row[col_map["epoch"]]) else None
        duration = float(row[col_map["duration"]]) if "duration" in col_map and not pd.isna(row[col_map["duration"]]) else None
        depth = float(row[col_map["depth"]]) if "depth" in col_map and not pd.isna(row[col_map["depth"]]) else None

        if period is None or epoch is None or duration is None:
            continue
        if duration >= period * 24.0:
            continue

        ephem = Ephemeris(
            period_days=period,
            epoch_bjd_tdb=epoch,
            duration_hours=duration,
            time_convention="BTJD",
            transit_depth_ppm=depth,
        )

        raw_label = str(row[col_map["label"]]) if "label" in col_map and not pd.isna(row[col_map["label"]]) else None
        normalized_label = parse_disposition(raw_label) if raw_label else None

        ruwe = float(row[col_map["ruwe"]]) if "ruwe" in col_map and not pd.isna(row[col_map["ruwe"]]) else None

        cand_rec = CandidateRecord(
            candidate_id=cid.candidate_id,
            tic_id=cid.tic_id,
            planet_num=cid.planet_num,
            sector_run=cid.sector_run,
            source_event_id=raw_tce_id,
            ephemeris=ephem,
            gaia_ruwe=ruwe,
            label_raw=raw_label,
            label_normalized=normalized_label,
            label_source=source_name,
        )
        records.append(cand_rec)

    return records
