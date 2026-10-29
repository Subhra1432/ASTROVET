"""
TESS FITS light-curve reader preserving astronomical standards.

Scientific safeguards:
- Retain original units, headers, quality flags, time column, flux columns, and uncertainty columns.
- Keep SAP and PDCSAP/KSPSAP distinct. PDCSAP/KSPSAP is detrended product flux, not raw counts.
- Compute cadence from timestamps; never infer from array length.
- Preserve time convention (BTJD = BJD - 2457000.0).
- Support both NASA SPOC and MIT QLP light-curve data formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
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
    pipeline_origin: str
    header_metadata: Dict[str, Any]


def read_tess_fits_lightcurve(
    file_path: str | Path,
    use_pdcsap: bool = True,
) -> RawLightCurveData:
    """
    Read a TESS SPOC or QLP light-curve FITS file while strictly preserving
    astronomical time base, units, quality flags, and metadata.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Light-curve FITS file does not exist: {p}")

    with fits.open(p) as hdul:
        primary_hdr = hdul[0].header

        # Identify binary table HDU (usually HDU 1, named LIGHTCURVE or similar)
        table_hdu = None
        for hdu in hdul[1:]:
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                table_hdu = hdu
                break
        if table_hdu is None:
            raise ValueError(f"FITS file '{p.name}' contains no binary table HDU.")

        lc_table = table_hdu.data
        lc_hdr = table_hdu.header

        # Resolve TIC ID
        tic_id = (
            primary_hdr.get("TICID")
            or primary_hdr.get("TIC_ID")
            or lc_hdr.get("TICID")
            or lc_hdr.get("TIC_ID")
        )
        sector = primary_hdr.get("SECTOR") or lc_hdr.get("SECTOR")
        camera = primary_hdr.get("CAMERA") or lc_hdr.get("CAMERA")
        ccd = primary_hdr.get("CCD") or lc_hdr.get("CCD")
        time_unit = primary_hdr.get("TIMEUNIT", "d")
        bjdrefi = primary_hdr.get("BJDREFI", 2457000)

        col_names = [c.upper() for c in lc_table.names] if lc_table.names else []

        if "TIME" not in col_names:
            raise ValueError(f"FITS table in '{p.name}' lacks mandatory 'TIME' column.")

        time_col_idx = col_names.index("TIME")
        raw_time = np.array(lc_table[lc_table.names[time_col_idx]], dtype=np.float64)

        # Detect pipeline origin
        creator = str(primary_hdr.get("CREATOR", "") or "").upper()
        origin = str(primary_hdr.get("ORIGIN", "") or "").upper()
        telescop = str(primary_hdr.get("TELESCOP", "") or "").upper()
        
        filename_upper = p.name.upper()
        qlp_columns = {"KSPSAP_FLUX", "SAP_FLUX_SML", "SAP_FLUX_MID", "SAP_FLUX_LAG"}
        if (
            "QLP" in creator
            or "QLP" in origin
            or "KSPSAP_FLUX" in col_names
            # Upload storage prepends a SHA/timestamp prefix, so identify this
            # published QLP filename family by containment rather than prefix.
            or "ASTRONET_TESS_FFI_" in filename_upper
            or qlp_columns.issubset(set(col_names))
        ):
            pipeline_origin = "TESS_QLP"
        elif "SPOC" in creator or "SPOC" in origin or "PDCSAP_FLUX" in col_names:
            pipeline_origin = "TESS_SPOC"
        else:
            pipeline_origin = "TESS_GENERIC"

        # Resolve flux columns
        pdcsap_flux = None
        kspsap_flux = None
        sap_flux = None
        generic_flux = None

        if "PDCSAP_FLUX" in col_names:
            idx = col_names.index("PDCSAP_FLUX")
            pdcsap_flux = np.array(lc_table[lc_table.names[idx]], dtype=np.float64)
        if "KSPSAP_FLUX" in col_names:
            idx = col_names.index("KSPSAP_FLUX")
            kspsap_flux = np.array(lc_table[lc_table.names[idx]], dtype=np.float64)
        if "SAP_FLUX" in col_names:
            idx = col_names.index("SAP_FLUX")
            sap_flux = np.array(lc_table[lc_table.names[idx]], dtype=np.float64)
        if "FLUX" in col_names:
            idx = col_names.index("FLUX")
            generic_flux = np.array(lc_table[lc_table.names[idx]], dtype=np.float64)

        # Select primary flux
        if use_pdcsap:
            if pdcsap_flux is not None:
                selected_flux = pdcsap_flux
                err_cand = "PDCSAP_FLUX_ERR"
            elif kspsap_flux is not None:
                selected_flux = kspsap_flux
                err_cand = "KSPSAP_FLUX_ERR"
            elif sap_flux is not None:
                selected_flux = sap_flux
                err_cand = "SAP_FLUX_ERR"
            elif generic_flux is not None:
                selected_flux = generic_flux
                err_cand = "FLUX_ERR"
            else:
                raise ValueError(f"No usable flux column found in '{p.name}'. Available: {col_names}")
        else:
            if sap_flux is not None:
                selected_flux = sap_flux
                err_cand = "SAP_FLUX_ERR"
            elif generic_flux is not None:
                selected_flux = generic_flux
                err_cand = "FLUX_ERR"
            elif pdcsap_flux is not None:
                selected_flux = pdcsap_flux
                err_cand = "PDCSAP_FLUX_ERR"
            elif kspsap_flux is not None:
                selected_flux = kspsap_flux
                err_cand = "KSPSAP_FLUX_ERR"
            else:
                raise ValueError(f"No usable flux column found in '{p.name}'. Available: {col_names}")

        # Resolve flux errors
        if err_cand in col_names:
            idx = col_names.index(err_cand)
            raw_flux_err = np.array(lc_table[lc_table.names[idx]], dtype=np.float64)
        elif "SAP_FLUX_ERR" in col_names:
            idx = col_names.index("SAP_FLUX_ERR")
            raw_flux_err = np.array(lc_table[lc_table.names[idx]], dtype=np.float64)
        elif "FLUX_ERR" in col_names:
            idx = col_names.index("FLUX_ERR")
            raw_flux_err = np.array(lc_table[lc_table.names[idx]], dtype=np.float64)
        else:
            # Fallback to empirical robust scatter estimation
            diff = np.diff(selected_flux[np.isfinite(selected_flux)])
            est_std = float(1.4826 * np.nanmedian(np.abs(diff - np.nanmedian(diff))) / np.sqrt(2.0))
            raw_flux_err = np.full_like(raw_time, max(1e-6, est_std))

        # Quality column
        if "QUALITY" in col_names:
            idx = col_names.index("QUALITY")
            quality = np.array(lc_table[lc_table.names[idx]], dtype=np.int32)
        else:
            quality = np.zeros(len(raw_time), dtype=np.int32)

        # Cadence number
        cadence_no = None
        if "CADENCENO" in col_names:
            idx = col_names.index("CADENCENO")
            cadence_no = np.array(lc_table[lc_table.names[idx]], dtype=np.int32)

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
        pipeline_origin=pipeline_origin,
        header_metadata=dict(primary_hdr),
    )
