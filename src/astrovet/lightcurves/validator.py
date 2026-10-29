"""
Light-curve upload validation, secure storage, and preprocessing audit.

Scientific safeguards:
- Preserve uploaded original files unchanged.
- Restrict storage to an isolated local directory with sanitized unique names.
- Enforce strict file type (FITS / FITS.GZ) and size limits.
- Validate FITS headers, mandatory columns, finite cadences, and minimum usable thresholds.
- Compute auditable SHA-256 and cadence attrition metrics before/after every filtering step.
- Never aggressively flatten or erase transit signals.
"""

from __future__ import annotations

import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from astropy.io import fits

from astrovet.lightcurves.reader import read_tess_fits_lightcurve, RawLightCurveData
from astrovet.lightcurves.quality import filter_quality_cadences, DEFAULT_SPOC_QUALITY_BITMASK
from astrovet.lightcurves.normalization import segment_and_normalize_lightcurve
from astrovet.provenance import compute_sha256


MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MINIMUM_USABLE_CADENCES = 50


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal or filesystem injection."""
    # Strip any directory paths
    base = Path(filename).name
    # Keep only alphanumeric, dots, hyphens, and underscores
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
    # Remove leading dots or multiple consecutive dots
    sanitized = re.sub(r"^\.+", "", sanitized)
    sanitized = re.sub(r"\.{2,}", ".", sanitized)
    if not sanitized:
        sanitized = "uploaded_lightcurve.fits"
    return sanitized[:128]


def validate_upload_bytes(
    data: bytes,
    filename: str,
    max_size: int = MAX_UPLOAD_SIZE_BYTES,
) -> None:
    """Validate file size, extension, and binary magic bytes."""
    if not data or len(data) == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    if len(data) > max_size:
        raise ValueError(
            f"Uploaded file exceeds maximum allowed size ({len(data):,} bytes > {max_size:,} bytes limit)."
        )

    clean_name = filename.lower().strip()
    if not (clean_name.endswith(".fits") or clean_name.endswith(".fits.gz")):
        raise ValueError(
            f"Invalid file extension for '{filename}'. Only .fits and .fits.gz light-curve files are accepted."
        )

    # Magic byte verification
    if clean_name.endswith(".fits.gz"):
        # Gzip magic number: 0x1f, 0x8b
        if len(data) < 2 or data[0] != 0x1F or data[1] != 0x8B:
            raise ValueError(f"File '{filename}' has .fits.gz extension but lacks valid gzip magic header.")
    else:
        # Standard uncompressed FITS magic header: starts with 'SIMPLE  ='
        if len(data) < 80 or not data[:8].startswith(b"SIMPLE"):
            raise ValueError(f"File '{filename}' lacks standard FITS magic header ('SIMPLE  =').")


def store_uploaded_lightcurve(
    data: bytes,
    filename: str,
    upload_dir: str | Path = "data/uploads",
) -> Tuple[Path, str]:
    """
    Validate, hash, and store uploaded light curve unchanged in the designated upload directory.
    Returns (stored_path, sha256_hex).
    """
    validate_upload_bytes(data, filename)

    sha256_hash = hashlib.sha256(data).hexdigest()
    sanitized_name = sanitize_filename(filename)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_name = f"{sha256_hash[:12]}_{ts}_{sanitized_name}"

    out_dir = Path(upload_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stored_path = out_dir / unique_name

    # Write original bytes unchanged
    stored_path.write_bytes(data)

    return stored_path, sha256_hash


def audit_lightcurve_fits(
    fits_path: str | Path,
    quality_bitmask: int = DEFAULT_SPOC_QUALITY_BITMASK,
) -> Dict[str, Any]:
    """
    Perform a complete scientific audit on an uploaded FITS light curve:
    - Column and metadata extraction
    - Cadence attrition accounting (raw -> finite -> quality-filtered)
    - Quality flag bit distribution
    - Segment and gap structure
    - Identification of scientific warnings
    """
    p = Path(fits_path)
    if not p.is_file():
        raise FileNotFoundError(f"FITS file not found: {p}")

    file_size = p.stat().st_size
    sha256_val = compute_sha256(p)

    # Read using astronomical reader
    try:
        raw_lc = read_tess_fits_lightcurve(p)
    except Exception as e:
        raise ValueError(f"Malformed FITS structure in '{p.name}': {e}")

    total_raw = len(raw_lc.time)
    if total_raw == 0:
        raise ValueError(f"FITS file '{p.name}' contains 0 time cadences.")

    # Apply quality and finite cadence filtering
    clean_t, clean_f, clean_fe, q_rep = filter_quality_cadences(
        raw_lc.time,
        raw_lc.flux,
        raw_lc.flux_err,
        raw_lc.quality,
        bitmask=quality_bitmask,
    )

    usable_count = len(clean_t)
    if usable_count < MINIMUM_USABLE_CADENCES:
        raise ValueError(
            f"FITS light curve '{p.name}' contains only {usable_count} usable cadences after filtering "
            f"(minimum {MINIMUM_USABLE_CADENCES} required for transit search and vetting)."
        )

    # Segment and normalization audit
    norm_lc = segment_and_normalize_lightcurve(clean_t, clean_f, clean_fe)

    warnings: List[str] = []
    if q_rep["retained_fraction"] < 0.50:
        warnings.append(
            f"High cadence rejection rate: {(1.0 - q_rep['retained_fraction'])*100:.1f}% of cadences "
            f"flagged as non-finite or low-quality."
        )
    if raw_lc.tic_id is None:
        warnings.append("TICID header keyword missing from primary and table headers.")
    if raw_lc.sector is None:
        warnings.append("SECTOR header keyword missing from headers.")
    if raw_lc.pipeline_origin == "TESS_QLP":
        warnings.append(
            "QLP light curves utilize difference imaging or circular apertures with simplified quality masks; "
            "some SPOC quality bit checks may not apply."
        )

    segments_info = [
        {
            "segment_id": seg.segment_id,
            "start_btjd": round(seg.start_time, 4),
            "end_btjd": round(seg.end_time, 4),
            "duration_days": round(seg.end_time - seg.start_time, 3),
            "cadence_count": seg.cadence_count,
            "median_flux_scale": round(seg.median_flux_scale, 2),
        }
        for seg in norm_lc.segments
    ]

    return {
        "file_name": p.name,
        "file_size_bytes": file_size,
        "sha256": sha256_val,
        "pipeline_origin": raw_lc.pipeline_origin,
        "metadata": {
            "tic_id": raw_lc.tic_id,
            "sector": raw_lc.sector,
            "camera": raw_lc.camera,
            "ccd": raw_lc.ccd,
            "cadence_seconds": raw_lc.cadence_seconds,
            "time_convention": raw_lc.time_convention,
            "time_unit": raw_lc.time_unit,
        },
        "cadence_audit": {
            "raw_total_cadences": total_raw,
            "finite_positive_cadences": total_raw - q_rep["non_finite_cadences"],
            "non_finite_rejected": q_rep["non_finite_cadences"],
            "quality_bitmask_retained": q_rep["retained_cadences"],
            "quality_bitmask_rejected": q_rep["rejected_cadences"],
            "usable_cadences": usable_count,
            "retention_percentage": round(q_rep["retained_fraction"] * 100.0, 2),
            "flag_breakdown": q_rep["flag_breakdown"],
        },
        "processing_policy": {
            "quality_bitmask": quality_bitmask,
            "normalization_strategy": "segment_median_scaling",
            "gap_threshold_days": 0.5,
            "detrending_strategy": "transit_protected_savgol",
        },
        "segments": segments_info,
        "warnings": warnings,
        "is_valid": True,
    }
