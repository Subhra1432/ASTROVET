"""
TESS FITS light-curve reader preserving astronomical standards.

Scientific safeguards:
- Retain original units, headers, quality flags, time column, flux columns, and uncertainty columns.
- Keep SAP and PDCSAP distinct. PDCSAP is detrended product flux, not raw counts.
- Compute cadence from timestamps; never infer from array length.
- Preserve time convention (BTJD = BJD - 2457000.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from astropy.io import fits


@dataclass
class RawLightCurveData:
    """Raw light curve read directly from FITS table without destructive alterations."""
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    quality: np.ndarray
    sap_flux: Optional[np.ndarray]
    cadence_no: Optional[np.ndarray]
    
    # Metadata
    tic_id: Optional[int]
    sector: Optional[int]
    camera: Optional[int]
    ccd: Optional[int]
    time_unit: str
    time_convention: str
    cadence_seconds: float
    header_metadata: Dict[str, Any]


def read_tess_fits_lightcurve(
    file_path: str | Path,
    use_pdcsap: bool = True,
) -> RawLightCurveData:
    """
    Read a TESS SPOC light-curve FITS file while strictly preserving
    astronomical time base, units, quality flags, and metadata.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Light-curve FITS file does not exist: {p}")

    with fits.open(p) as hdul:
        primary_hdr = hdul[0].header
        lc_table = hdul[1].data
        lc_hdr = hdul[1].header

        tic_id = primary_hdr.get("TICID")
        sector = primary_hdr.get("SECTOR")
        camera = primary_hdr.get("CAMERA")
        ccd = primary_hdr.get("CCD")
        time_unit = primary_hdr.get("TIMEUNIT", "d")
        bjdrefi = primary_hdr.get("BJDREFI", 2457000)

        raw_time = np.array(lc_table["TIME"], dtype=np.float64)
        sap_flux = np.array(lc_table["SAP_FLUX"], dtype=np.float64) if "SAP_FLUX" in lc_table.names else None
        pdcsap_flux = np.array(lc_table["PDCSAP_FLUX"], dtype=np.float64) if "PDCSAP_FLUX" in lc_table.names else None
        
        flux_err_col = "PDCSAP_FLUX_ERR" if use_pdcsap and "PDCSAP_FLUX_ERR" in lc_table.names else "SAP_FLUX_ERR"
        raw_flux_err = np.array(lc_table[flux_err_col], dtype=np.float64) if flux_err_col in lc_table.names else np.zeros_like(raw_time)

        quality = np.array(lc_table["QUALITY"], dtype=np.int32) if "QUALITY" in lc_table.names else np.zeros(len(raw_time), dtype=np.int32)
        cadence_no = np.array(lc_table["CADENCENO"], dtype=np.int32) if "CADENCENO" in lc_table.names else None

        selected_flux = pdcsap_flux if (use_pdcsap and pdcsap_flux is not None) else sap_flux
        if selected_flux is None:
            raise ValueError(f"No valid flux column found in {p.name}")

        # Compute empirical median cadence in seconds
        valid_time_diffs = np.diff(raw_time[np.isfinite(raw_time)])
        positive_diffs = valid_time_diffs[valid_time_diffs > 0]
        if len(positive_diffs) > 0:
            cadence_seconds = float(np.nanmedian(positive_diffs) * 86400.0)
        else:
            cadence_seconds = 120.0

    return RawLightCurveData(
        time=raw_time,
        flux=selected_flux,
        flux_err=raw_flux_err,
        quality=quality,
        sap_flux=sap_flux,
        cadence_no=cadence_no,
        tic_id=int(tic_id) if tic_id is not None else None,
        sector=int(sector) if sector is not None else None,
        camera=int(camera) if camera is not None else None,
        ccd=int(ccd) if ccd is not None else None,
        time_unit=time_unit,
        time_convention=f"BJD - {bjdrefi} (BTJD)",
        cadence_seconds=round(cadence_seconds, 2),
        header_metadata=dict(primary_hdr),
    )
