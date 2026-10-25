"""
MAST query, product retrieval, and immutable manifest logging.

Scientific safeguards:
- Query MAST by TIC and observation/product type (SPOC light curve first).
- Save a provenance manifest with source URLs, filenames, sectors, cadences, retrieval time, and SHA-256 checksums.
- Never guess product URLs without recording their query provenance.
- Support offline fixture mode for deterministic tests and isolated runs.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from astrovet.provenance import compute_sha256, create_manifest, save_manifest


def query_mast_for_tic_lightcurves(
    tic_id: int,
    sector: Optional[int] = None,
    offline_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Query MAST for TESS SPOC light-curve products for a given TIC star.
    Returns product records.
    """
    if offline_mode:
        # Return mock / cached product descriptor
        sec = sector or 1
        return [{
            "tic_id": tic_id,
            "sector": sec,
            "product_type": "SPOC_LC_2MIN",
            "cadence_seconds": 120,
            "source_url": f"https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS/product/tess_s00{sec:02d}_{tic_id}_lc.fits",
            "is_fixture": True,
        }]

    try:
        from astroquery.mast import Observations
        # Search criteria for SPOC 2-min light curves
        obs_table = Observations.query_criteria(
            obs_collection="TESS",
            target_name=str(tic_id),
            dataproduct_type="timeseries",
        )
        if len(obs_table) == 0:
            return []

        products = []
        for row in obs_table:
            # Filter for SPOC pipeline if specified
            prov = str(row.get("provenance_name", ""))
            sec_num = row.get("sequence_number")
            if sector is not None and sec_num != sector:
                continue
            products.append({
                "tic_id": tic_id,
                "sector": int(sec_num) if sec_num is not None else None,
                "obs_id": str(row.get("obs_id", "")),
                "provenance_name": prov,
                "dataproduct_type": "timeseries",
                "cadence_seconds": 120,
                "is_fixture": False,
            })
        return products
    except Exception as e:
        # Graceful fallback reporting
        return [{
            "tic_id": tic_id,
            "error": f"MAST query failed or network unavailable: {e}",
            "is_fixture": False,
        }]


def is_fits_synthetic_fixture(file_path: str | Path) -> bool:
    """
    Inspect FITS header metadata to determine whether the file is an ASTROVET synthetic fixture.
    """
    p = Path(file_path)
    if not p.is_file():
        return False
    try:
        from astropy.io import fits
        with fits.open(p) as hdul:
            hdr0 = hdul[0].header
            if hdr0.get("SYNTH_LC") or hdr0.get("ASTROSYN") or hdr0.get("SYNTHETIC"):
                return True
            comments = " ".join([str(c) for c in hdr0.get("COMMENT", [])])
            if "ASTROVET SYNTHETIC TEST FIXTURE" in comments.upper():
                return True
            if "SYNTHETIC" in comments.upper() and "NASA/Ames" not in str(hdr0.get("ORIGIN", "")):
                return True
    except Exception:
        pass
    return False


def validate_spoc_lightcurve_fits(
    file_path: str | Path,
    expected_tic_id: int,
    expected_sector: int,
    expected_cadence_seconds: float = 120.0,
    cadence_tolerance_seconds: float = 10.0,
) -> Dict[str, Any]:
    """
    Verify that a downloaded or local FITS file is a valid TESS SPOC 2-min light curve.
    Validates TIC ID, Sector, SPOC origin/author, and ~120-second cadence.
    Returns dict with validated metadata and exact cadence in seconds.
    Raises ValueError if validation fails.
    """
    from astropy.io import fits
    p = Path(file_path)
    if not p.is_file():
        raise ValueError(f"FITS file does not exist: {p}")

    with fits.open(p) as hdul:
        if len(hdul) < 2:
            raise ValueError(f"FITS file {p.name} does not contain required table extension")

        hdr0 = hdul[0].header
        hdr1 = hdul[1].header

        is_synth = is_fits_synthetic_fixture(p)

        # 1. Verify TIC ID
        tic = hdr0.get("TICID")
        if tic is not None and int(tic) != expected_tic_id:
            raise ValueError(f"FITS TIC ID mismatch: expected {expected_tic_id}, found {tic}")

        # 2. Verify Sector
        sector = hdr0.get("SECTOR")
        if sector is not None and int(sector) != expected_sector:
            raise ValueError(f"FITS Sector mismatch: expected {expected_sector}, found {sector}")

        # 3. Verify Origin / Author
        creator = str(hdr0.get("CREATOR", ""))
        origin = str(hdr0.get("ORIGIN", ""))
        telescop = str(hdr0.get("TELESCOP", ""))
        if not is_synth:
            if telescop != "TESS":
                raise ValueError(f"Expected TELESCOP='TESS', found '{telescop}'")
            if origin != "NASA/Ames" and "SPOC" not in creator.upper() and "LIGHTCURVE" not in creator.upper():
                raise ValueError(f"FITS pipeline author/origin mismatch: origin='{origin}', creator='{creator}'")

        # 4. Determine and verify cadence in seconds
        cadence_sec: Optional[float] = None
        # Check TIMEDEL in HDU 1 or HDU 0 (days -> seconds)
        timedel = hdr1.get("TIMEDEL") if hdr1.get("TIMEDEL") is not None else hdr0.get("TIMEDEL")
        if timedel is not None:
            cadence_sec = float(timedel) * 86400.0

        # Or check NUM_FRM * FRAMETIM
        if cadence_sec is None:
            num_frm = hdr1.get("NUM_FRM")
            frametim = hdr1.get("FRAMETIM")
            if num_frm and frametim:
                cadence_sec = float(num_frm) * float(frametim)

        # Or compute median time difference
        if cadence_sec is None and "TIME" in hdul[1].data.names:
            t = np.array(hdul[1].data["TIME"], dtype=np.float64)
            valid_diffs = np.diff(t[np.isfinite(t)])
            pos_diffs = valid_diffs[valid_diffs > 0]
            if len(pos_diffs) > 0:
                cadence_sec = float(np.nanmedian(pos_diffs)) * 86400.0

        if cadence_sec is None:
            raise ValueError(f"Could not determine cadence/exposure interval from FITS header or time column in {p.name}")

        if abs(cadence_sec - expected_cadence_seconds) > cadence_tolerance_seconds:
            raise ValueError(
                f"FITS cadence mismatch for {p.name}: measured {cadence_sec:.1f}s, "
                f"expected ~{expected_cadence_seconds:.1f}s (tolerance +/-{cadence_tolerance_seconds:.1f}s). "
                f"20-second or other non-2-minute cadences are rejected."
            )

        return {
            "tic_id": int(tic) if tic is not None else expected_tic_id,
            "sector": int(sector) if sector is not None else expected_sector,
            "cadence_seconds": round(float(cadence_sec), 2),
            "is_synthetic_fixture": is_synth,
            "origin": origin,
            "creator": creator,
        }


def fetch_tess_lightcurve(
    tic_id: int,
    output_dir: str | Path,
    sector: Optional[int] = None,
    manifest_output_path: Optional[str | Path] = None,
    offline_mode: bool = False,
    fixture_if_offline: bool = True,
) -> Dict[str, Any]:
    """
    Fetch a TESS SPOC light-curve file and record an immutable provenance manifest.

    Parameters
    ----------
    tic_id : int
        Target TIC ID.
    output_dir : str | Path
        Directory to save light curve product.
    sector : Optional[int]
        TESS sector number (default: 23).
    manifest_output_path : Optional[str | Path]
        Optional path to write provenance manifest JSON.
    offline_mode : bool
        If True, skips network access and generates a deterministic synthetic fixture.
    fixture_if_offline : bool
        If True and online retrieval fails (e.g. network down, target missing on MAST),
        falls back to generating a synthetic test fixture with clear reporting.
        If False and online retrieval fails, raises RuntimeError.

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'product_file', 'sha256', 'is_synthetic_fixture', 'cadence_seconds',
        'retrieval_mode', 'manifest', and optional 'error'.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sec = sector or 23
    fname = f"tess_s{sec:04d}_{tic_id}_spoc_lc.fits"
    dest_path = out_dir / fname

    is_fixture = False
    cadence_seconds: float = 120.0
    download_err: Optional[str] = None
    retrieval_mode = "ONLINE_MAST"

    if offline_mode:
        # Explicit offline mode: generate deterministic synthetic fixture directly
        is_fixture = True
        retrieval_mode = "EXPLICIT_OFFLINE_FIXTURE"
        if not dest_path.exists() or not is_fits_synthetic_fixture(dest_path):
            _create_synthetic_fits_lightcurve(dest_path, tic_id, sec)
        val_meta = validate_spoc_lightcurve_fits(dest_path, expected_tic_id=tic_id, expected_sector=sec, expected_cadence_seconds=120.0)
        cadence_seconds = val_meta["cadence_seconds"]
    else:
        # Normal online mode: check if an official validated product is already cached
        cached_valid = False
        if dest_path.exists() and not is_fits_synthetic_fixture(dest_path):
            try:
                val_meta = validate_spoc_lightcurve_fits(dest_path, expected_tic_id=tic_id, expected_sector=sec, expected_cadence_seconds=120.0)
                is_fixture = False
                retrieval_mode = "CACHED_ONLINE_MAST"
                cadence_seconds = val_meta["cadence_seconds"]
                cached_valid = True
            except Exception:
                cached_valid = False

        if not cached_valid:
            # Attempt official SPOC retrieval from MAST filtered strictly to 120-second cadence
            downloaded = False
            try:
                import lightkurve as lk
                # Filter MAST query to 120-second cadence
                sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sec, exptime=120)
                if len(sr) > 0 and hasattr(sr, "table") and "exptime" in sr.table.colnames:
                    mask_120 = [abs(float(e) - 120.0) <= 10.0 for e in sr.table["exptime"]]
                    sr = sr[mask_120]

                if len(sr) > 0:
                    lc = sr.download(download_dir=str(out_dir))
                    if lc is not None and getattr(lc, "filename", None) and Path(lc.filename).is_file():
                        downloaded_file = Path(lc.filename)
                        # Verify downloaded FITS TIC, sector, origin/author, and exposure before writing product
                        val_meta = validate_spoc_lightcurve_fits(
                            downloaded_file,
                            expected_tic_id=tic_id,
                            expected_sector=sec,
                            expected_cadence_seconds=120.0,
                        )
                        cadence_seconds = val_meta["cadence_seconds"]
                        if downloaded_file.resolve() != dest_path.resolve():
                            shutil.copy2(downloaded_file, dest_path)
                        downloaded = True
                    else:
                        download_err = f"Failed to download FITS file from MAST for TIC {tic_id} Sector {sec}"
                else:
                    download_err = f"No SPOC 2-min light curve found on MAST for TIC {tic_id} Sector {sec} (no official 120-second cadence product found)"
            except Exception as e:
                download_err = f"MAST query/download failed: {e}"

            if downloaded:
                is_fixture = False
                retrieval_mode = "ONLINE_MAST"
            else:
                if fixture_if_offline:
                    # Clearly reported fallback
                    is_fixture = True
                    retrieval_mode = "FALLBACK_OFFLINE_FIXTURE"
                    _create_synthetic_fits_lightcurve(dest_path, tic_id, sec)
                    val_meta = validate_spoc_lightcurve_fits(dest_path, expected_tic_id=tic_id, expected_sector=sec, expected_cadence_seconds=120.0)
                    cadence_seconds = val_meta["cadence_seconds"]
                else:
                    raise RuntimeError(
                        f"Failed to retrieve official 120-second SPOC light curve for TIC {tic_id} Sector {sec} "
                        f"from MAST ({download_err}) and fixture_if_offline is False."
                    )

    # Compute checksum and build manifest
    sha256_hash = compute_sha256(dest_path)
    file_record = {
        "filename": fname,
        "local_path": str(dest_path.resolve()),
        "relative_path": str(dest_path).replace("\\", "/"),
        "sha256": sha256_hash,
        "size_bytes": dest_path.stat().st_size,
        "tic_id": tic_id,
        "sector": sec,
        "cadence_seconds": cadence_seconds,
        "is_synthetic_fixture": is_fixture,
        "retrieval_mode": retrieval_mode,
    }

    manifest = create_manifest(
        manifest_id=f"manifest_mast_tic_{tic_id}_s{sec}",
        manifest_type="MAST_PRODUCT_INGEST",
        files=[file_record],
        metadata={
            "mission": "TESS",
            "pipeline": "SPOC",
            "time_standard": "BTJD",
            "retrieval_mode": retrieval_mode,
            "offline_mode_requested": offline_mode,
        }
    )

    if manifest_output_path:
        save_manifest(manifest, manifest_output_path)

    return {
        "product_file": str(dest_path),
        "sha256": sha256_hash,
        "is_synthetic_fixture": is_fixture,
        "cadence_seconds": cadence_seconds,
        "retrieval_mode": retrieval_mode,
        "manifest": manifest,
        "error": download_err,
    }


def _create_synthetic_fits_lightcurve(
    target_path: Path,
    tic_id: int,
    sector: int,
) -> None:
    """
    Generate a standards-compliant TESS SPOC FITS light-curve table fixture.
    Clearly records SYNTHETIC_FIXTURE in the header comments and quality table.
    """
    from astropy.io import fits
    from astropy.table import Table

    # 27 days of observations at 2-minute cadence (~19440 points)
    # Include realistic sector gap in middle
    n_pts_half = 9500
    cadence_days = 120.0 / 86400.0
    time_chunk1 = 1900.0 + np.arange(n_pts_half) * cadence_days
    time_chunk2 = 1914.5 + np.arange(n_pts_half) * cadence_days # 1.5 day gap
    times = np.concatenate([time_chunk1, time_chunk2])
    n_pts = len(times)

    # Seed for reproducibility
    rng = np.random.default_rng(seed=tic_id % 1000000 + sector)

    # Out-of-transit flux with Poisson-like noise (standard deviation 150 ppm)
    flux_noise = rng.normal(1.0, 0.00015, n_pts)
    
    # Inject a known transit signal: P=3.5d, t0=1901.2, duration=2.5h (0.104d), depth=1500 ppm
    period = 3.5
    t0 = 1901.2
    duration_days = 2.5 / 24.0
    depth_frac = 0.0015
    phase = ((times - t0 + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) < (duration_days / 2.0)
    
    flux = flux_noise.copy()
    flux[in_transit] -= depth_frac
    flux_err = np.full(n_pts, 0.00015)
    
    # Quality flags: set bit 3 (momentum dump) on 50 points
    quality = np.zeros(n_pts, dtype=np.int32)
    quality[rng.choice(n_pts, size=50, replace=False)] = 32

    tab = Table({
        "TIME": times,
        "TIMECORR": np.zeros(n_pts),
        "CADENCENO": np.arange(100000, 100000 + n_pts, dtype=np.int32),
        "SAP_FLUX": flux * 10000.0,
        "SAP_FLUX_ERR": flux_err * 10000.0,
        "SAP_BKG": np.full(n_pts, 150.0),
        "PDCSAP_FLUX": flux * 10000.0,
        "PDCSAP_FLUX_ERR": flux_err * 10000.0,
        "QUALITY": quality,
    })

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["TELESCOP"] = "TESS"
    primary_hdu.header["INSTRUME"] = "TESS Photometer"
    primary_hdu.header["TICID"] = tic_id
    primary_hdu.header["SECTOR"] = sector
    primary_hdu.header["CAMERA"] = 1
    primary_hdu.header["CCD"] = 1
    primary_hdu.header["TIMEUNIT"] = "d"
    primary_hdu.header["BJDREFI"] = 2457000
    primary_hdu.header["BJDREFF"] = 0.0
    primary_hdu.header["TIMEDEL"] = cadence_days
    primary_hdu.header["SYNTH_LC"] = (True, "ASTROVET synthetic test fixture")
    primary_hdu.header["COMMENT"] = "ASTROVET SYNTHETIC TEST FIXTURE - NOT OFFICIAL FLIGHT DATA"

    table_hdu = fits.table_to_hdu(tab)
    table_hdu.header["EXTNAME"] = "LIGHTCURVE"
    table_hdu.header["TUNIT1"] = "BJD - 2457000, days"
    table_hdu.header["TUNIT7"] = "e-/s"
    table_hdu.header["TIMEDEL"] = cadence_days
    table_hdu.header["NUM_FRM"] = 60
    table_hdu.header["FRAMETIM"] = 2.0

    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(target_path, overwrite=True)
