"""Download and render small TESS pixel cutouts for dashboard context."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def get_tess_cutout(tic_id: int, sector: int, cache_dir: str | Path) -> Path:
    """Return a cached 11x11 TESSCut FITS cutout, downloading it from MAST if needed."""
    tic_id = int(tic_id)
    sector = int(sector)
    if tic_id <= 0 or sector <= 0:
        raise ValueError("A valid TIC identifier and TESS sector are required.")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_path = cache_root / f"tic{tic_id}_sector{sector:04d}_11x11_tesscut.fits"
    if cached_path.is_file():
        _validate_cutout(cached_path, sector)
        return cached_path

    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs, Tesscut

    catalog = Catalogs.query_criteria(catalog="TIC", ID=tic_id)
    if len(catalog) == 0:
        raise FileNotFoundError(f"TIC {tic_id} was not found in the MAST catalog.")
    ra = float(catalog[0]["ra"])
    dec = float(catalog[0]["dec"])
    manifest = Tesscut.download_cutouts(
        coordinates=SkyCoord(ra, dec, unit="deg"),
        size=11,
        sector=sector,
        path=str(cache_root),
    )
    if len(manifest) == 0:
        raise FileNotFoundError(f"MAST returned no TESS pixel cutout for TIC {tic_id}, Sector {sector}.")
    source_path = Path(str(manifest["Local Path"][0]))
    if not source_path.is_file():
        raise FileNotFoundError("The TESSCut download did not produce a FITS file.")
    _validate_cutout(source_path, sector)
    if source_path.resolve() != cached_path.resolve():
        shutil.copy2(source_path, cached_path)
    return cached_path


def _validate_cutout(path: Path, expected_sector: int) -> None:
    with fits.open(path, memmap=False) as hdul:
        if "PIXELS" not in hdul or "FLUX" not in hdul["PIXELS"].columns.names:
            raise ValueError("The FITS file does not contain TESS target pixels.")
        actual_sector = hdul[0].header.get("SECTOR", hdul["PIXELS"].header.get("SECTOR"))
        if actual_sector is not None and int(actual_sector) != int(expected_sector):
            raise ValueError("The downloaded TESS cutout sector does not match the requested sector.")


def render_tess_pixel_preview(
    cutout_path: str | Path,
    output_path: str | Path,
    tic_id: int,
    sector: int,
) -> Dict[str, Any]:
    """Render a median image of quality-clear TESS pixel cadences as a PNG."""
    cutout_path = Path(cutout_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with fits.open(cutout_path, memmap=False) as hdul:
        pixels = hdul["PIXELS"].data
        flux = np.asarray(pixels["FLUX"], dtype=float)
        quality = np.asarray(pixels["QUALITY"], dtype=np.int64)
        if flux.ndim != 3:
            raise ValueError("The TESS pixel flux data are not a 3D cadence-by-pixel array.")
        good_cadences = quality == 0
        used_quality_clear = bool(np.any(good_cadences))
        if not np.any(good_cadences):
            good_cadences = np.ones(len(flux), dtype=bool)
        image = np.nanmedian(flux[good_cadences], axis=0)
        if not np.isfinite(image).any():
            raise ValueError("The pixel cutout contains no finite flux values to display.")
        aperture = np.asarray(hdul["APERTURE"].data) if "APERTURE" in hdul else None

    finite_values = image[np.isfinite(image)]
    low, high = np.nanpercentile(finite_values, [2, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
        if high <= low:
            high = low + 1.0

    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    heatmap = ax.imshow(image, origin="lower", cmap="inferno", interpolation="nearest", vmin=low, vmax=high)
    if aperture is not None and aperture.shape == image.shape and np.unique(aperture).size > 1:
        ax.contour(aperture > 0, levels=[0.5], colors=["cyan"], linewidths=0.8)
    frame_label = "quality-clear frames" if used_quality_clear else "available frames (no quality-clear frames)"
    ax.set_title(
        f"TESS target pixel cutout — TIC {int(tic_id)}, Sector {int(sector)}\n"
        f"Median of {int(np.count_nonzero(good_cadences)):,} {frame_label} ({image.shape[1]} × {image.shape[0]} pixels)"
    )
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    fig.colorbar(heatmap, ax=ax, label="Median pixel flux")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(output_path),
        "cutout_path": str(cutout_path),
        "cadences_used": int(np.count_nonzero(good_cadences)),
        "used_quality_clear": used_quality_clear,
        "pixel_shape": [int(image.shape[0]), int(image.shape[1])],
        "sector": int(sector),
        "tic_id": int(tic_id),
    }
