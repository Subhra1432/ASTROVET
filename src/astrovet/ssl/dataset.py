"""
Dataset loader and view extractor for self-supervised representation learning from TESS FITS light curves.

Scientific safeguards:
- Reads real TESS SPOC light curves directly from FITS files.
- Preserves cadence, units, and astronomical time base.
- Filters quality flags according to standard SPOC bitmask policies.
- Segments and normalizes flux across gaps prior to windowing or folding.
- Extracts sliding windows and trial folded views binned to length N=200.
- Preserves complete provenance (file paths, SHA-256 hashes, TIC IDs, sectors, cadence).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from astrovet.lightcurves.reader import read_tess_fits_lightcurve
from astrovet.lightcurves.quality import filter_quality_cadences, DEFAULT_SPOC_QUALITY_BITMASK
from astrovet.lightcurves.normalization import segment_and_normalize_lightcurve
from astrovet.views.folding import fold_lightcurve, bin_folded_view
from astrovet.provenance import compute_sha256


def extract_views_from_fits(
    fits_path: str | Path,
    view_length: int = 200,
    stride: int = 100,
    trial_periods: Optional[List[float]] = None,
    quality_bitmask: int = DEFAULT_SPOC_QUALITY_BITMASK,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract real normalized training views of length `view_length` from an unlabeled TESS FITS file.
    Combines sliding cadence windows and trial-period folded views.
    Returns:
        (views_array of shape [M, view_length], provenance_metadata_dict)
    """
    p = Path(fits_path)
    if not p.is_file():
        raise FileNotFoundError(f"Input FITS light curve not found: {p}")

    sha256 = compute_sha256(p)
    raw = read_tess_fits_lightcurve(p)
    t_clean, f_clean, fe_clean, q_report = filter_quality_cadences(
        raw.time, raw.flux, raw.flux_err, raw.quality, bitmask=quality_bitmask
    )

    if len(t_clean) < view_length:
        raise ValueError(
            f"FITS file {p.name} contains only {len(t_clean)} valid cadences, fewer than required view length {view_length}."
        )

    processed = segment_and_normalize_lightcurve(t_clean, f_clean, fe_clean)

    views: List[np.ndarray] = []

    # 1. Sliding cadence windows over detrended continuous flux
    detrended = processed.flux_detrended
    for i in range(0, len(detrended) - view_length + 1, stride):
        window = detrended[i:i + view_length].copy()
        if np.all(np.isfinite(window)):
            med = float(np.median(window))
            if med > 0:
                window /= med
            views.append(window)

    # 2. Trial folded views across physically representative period grid
    periods = trial_periods or [1.5, 2.5, 4.0, 8.0, 12.0]
    epoch_ref = float(t_clean[0])

    for period in periods:
        if period <= 0:
            continue
        phase, _, sf, sfe, _ = fold_lightcurve(
            t_clean, processed.flux_detrended, processed.flux_err_norm,
            period_days=period, epoch_btjd=epoch_ref
        )
        fv = bin_folded_view(phase, sf, sfe, num_bins=view_length, period_days=period)
        arr = fv.flux.copy()
        nans = np.isnan(arr)
        if not np.all(nans):
            if np.any(nans):
                idx = np.arange(len(arr))
                arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
            med = float(np.median(arr))
            if med > 0:
                arr /= med
            views.append(arr)

    if len(views) == 0:
        raise ValueError(f"No valid views could be extracted from {p.name}")

    views_arr = np.array(views, dtype=np.float64)

    metadata = {
        "file_path": str(p.resolve()),
        "file_name": p.name,
        "sha256": sha256,
        "tic_id": raw.tic_id,
        "sector": raw.sector,
        "cadence_seconds": raw.cadence_seconds,
        "total_cadences": len(raw.time),
        "valid_cadences": len(t_clean),
        "retained_fraction": q_report["retained_fraction"],
        "num_segments": len(processed.segments),
        "views_extracted": len(views),
        "view_length": view_length,
    }

    return views_arr, metadata


def extract_views_from_directory(
    dir_path: str | Path,
    view_length: int = 200,
    stride: int = 100,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Scan a directory for TESS FITS light curves, extracting views across all discovered files.
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {root}")

    # Discover .fits and .fit files recursively
    fits_files = sorted([f for f in root.glob("**/*") if f.is_file() and f.suffix.lower() in [".fits", ".fit"]])
    if not fits_files:
        raise FileNotFoundError(f"No FITS light curves found in directory: {root}")

    if max_files:
        fits_files = fits_files[:max_files]

    all_views: List[np.ndarray] = []
    metadata_list: List[Dict[str, Any]] = []

    for fpath in fits_files:
        try:
            views_arr, meta = extract_views_from_fits(fpath, view_length=view_length, stride=stride)
            all_views.append(views_arr)
            metadata_list.append(meta)
        except Exception:
            # Skip non-lightcurve or corrupt FITS files
            continue

    if not all_views:
        raise ValueError(f"Failed to extract valid views from any FITS files in {root}")

    combined = np.vstack(all_views)
    return combined, metadata_list


def load_prepared_views(views_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a pre-extracted views archive (.npz) containing a 'views' array of shape (N, view_length).
    """
    p = Path(views_path)
    if not p.is_file():
        raise FileNotFoundError(f"Prepared views file not found: {p}")

    data = np.load(p, allow_pickle=True)
    if "views" not in data:
        raise KeyError(f"Expected array 'views' in {p.name}, found keys: {list(data.keys())}")

    views = np.array(data["views"], dtype=np.float64)
    if views.ndim != 2:
        raise ValueError(f"Expected 2D array of views, got shape {views.shape}")

    sha256 = compute_sha256(p)
    metadata = {
        "file_path": str(p.resolve()),
        "file_name": p.name,
        "sha256": sha256,
        "views_loaded": len(views),
        "view_length": views.shape[1],
    }
    return views, metadata
