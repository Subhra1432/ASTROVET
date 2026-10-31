# ASTROVET Light Curve Upload & Vetting Workflow

## Overview

ASTROVET provides a secure, single-action upload and vetting workflow for TESS light curves. Users can supply raw FITS files from the **TESS SPOC** (Science Processing Operations Center) or **TESS QLP** (Quick-Look Pipeline) pipelines via the browser dashboard, REST API, or command-line interface.

The workflow executes end-to-end scientific validation and transit vetting while strictly preserving uploaded data unchanged and preventing remote code execution or filesystem traversal.

After vetting, the dashboard also tries to fetch a matching 11 × 11 TESS pixel cutout from MAST TESSCut using the candidate TIC and sector. It caches the FITS cutout under `data/pixel_cutouts/` and saves a median, quality-filtered pixel preview as `artifacts/reports/<candidate_id>_pixel_view.png`. This spatial view is supplemental; the uploaded light-curve FITS remains the input for transit searching. If MAST has no cutout or cannot be reached, vetting still completes and the dashboard reports that the pixel view is unavailable. See the [MAST TESS data products](https://archive.stsci.edu/missions-and-data/tess/data-products) and [TESSCut guide](https://astroquery.readthedocs.io/en/stable/mast/mast_cut.html).

---

## 1. Security Architecture & Ingestion Guardrails

1. **Unchanged File Preservation:**
   - Raw uploaded bytes are written directly to disk without alteration.
   - Storage directory: `data/uploads/` (explicitly excluded via `.gitignore`).
2. **Sanitized Unique Naming:**
   - Filenames are scrubbed against directory traversal attacks (`../`, `..\`) and invalid characters.
   - Files are stored with content-addressed prefixes and UTC timestamps:
     `{sha256[:12]}_{timestamp}_{sanitized_filename}`.
3. **Strict Size & Extension Boundaries:**
   - Maximum upload payload: 50 MB (`MAX_UPLOAD_SIZE_BYTES = 52,428,800`).
   - Accepted file extensions: `.fits`, `.fits.gz`.
4. **Magic Byte Verification:**
   - Uncompressed `.fits` files must start with the standard FITS header `SIMPLE  =`.
   - Compressed `.fits.gz` files must match the gzip magic signature (`0x1F`, `0x8B`).
5. **No Arbitrary Server Paths:**
   - Browser API requests cannot supply arbitrary local server paths. Only pre-configured allowed fixtures or files in `data/uploads/` can be accessed by the server.

---

## 2. Scientific Preprocessing & Audit

When a file is uploaded, ASTROVET immediately runs `audit_lightcurve_fits()`:

1. **Pipeline Origin Detection:**
   - Automatically differentiates between **TESS SPOC** (using `PDCSAP_FLUX` and 32-bit quality masks) and **TESS QLP** (using `KSPSAP_FLUX` or circular aperture flux).
2. **Cadence Attrition Accounting:**
   - Tracks exact cadence counts before and after every step:
     - `raw_total_cadences`: Raw cadences in table HDU.
     - `non_finite_rejected`: NaN/Inf/non-positive flux cadences removed.
     - `quality_bitmask_rejected`: Cadences flagged by quality bits (e.g. cosmic rays, momentum dumps, stray light).
     - `usable_cadences`: Finite, high-quality cadences retained for analysis.
   - Enforces a minimum threshold of **50 usable cadences**; files failing this threshold are rejected before downstream processing.
3. **Segment & Normalization Analysis:**
   - Detects orbit segments separated by data gaps ($>0.5$ days).
   - Computes per-segment median flux scaling factors.
   - Protects transits using Savitzky-Golay detrending with transit masking, ensuring transit depths are never flattened or distorted.
4. **Cryptographic SHA-256 Provenance:**
   - Computes SHA-256 digest of the raw file and associates it with all emitted audit reports.

---

## 3. Interfaces

### A. Dashboard UI (Single-Action Upload & Vetting)

1. Navigate to the ASTROVET dashboard (`http://127.0.0.1:8765/`).
2. In the **Ingest & Vet Light Curve** panel, drag and drop a `.fits` / `.fits.gz` file or click **Choose File**.
3. (Optional) Provide a candidate identifier or TIC ID; if omitted, the system extracts the TIC ID and sector directly from the FITS primary header.
4. Click **Upload & Run Complete Vetting Pipeline**.
5. The dashboard presents:
   - **Upload & Cadence Audit**: File SHA-256, pipeline origin, cadence attrition table, segment breakdowns, and scientific warnings.
   - **Vetting Verdict & Priority Score**: Physics plausibility checks, TLS periodogram results, and diagnostic metrics.
   - **Interactive Phase-Fold & Transit Plots**: Transit-protected folded light curves.

### B. REST API Endpoints

- **`POST /api/upload_and_vet`**:
  - Multipart form data: `file` (binary FITS), optional `candidate_id`, `tic_id`.
  - Executes upload validation, storage, FITS audit, and full vetting in a single synchronous call.
  - Returns JSON containing `audit`, `vetting_result`, and execution status.

- **`POST /api/upload`**:
  - Uploads and validates a FITS file without immediately running the full vetting pipeline.
  - Returns `stored_path`, `sha256`, and initial file metadata.

- **`GET /api/vet`**:
  - Securely runs vetting on a previously uploaded file or allowable fixture via query parameter `fits_path`.
  - Enforces strict path allowlisting to prevent arbitrary filesystem reads.

### C. Command-Line Interface (CLI)

Validate and audit any FITS light curve locally:

```bash
astrovet data validate-fits --fits-file data/raw/mast/tess_s0023_390651552_spoc_lc.fits
```

Output:
```
============================================================
ASTROVET FITS Light Curve Audit Report
============================================================
File: tess_s0023_390651552_spoc_lc.fits (1,958,400 bytes)
SHA-256: d53bccebc171d374fbafba442b0a969e63f9ebba512fb0ebbbbaeaeb2e6283db
Pipeline Origin: TESS_SPOC
Target TIC: 390651552 (Sector 23)

--- Cadence Attrition ---
  Raw Total Cadences:       19,000
  Non-finite / NaN:         50
  Quality Mask Filtered:    0
  Usable Clean Cadences:    18,950 (99.74% retained)

--- Orbit Segments ---
  Segments: 2
  Segment 1: BTJD 1928.50 -> 1941.00 (12.5 d, 8,900 cadences)
  Segment 2: BTJD 1942.50 -> 1955.00 (12.5 d, 10,050 cadences)

Status: VALID - Suitable for transit vetting.
============================================================
```

---

## 4. Scientific Guardrails

- **E Label Scope:** The label `E` from the Astronet-Triage dataset represents periodic eclipsing signals, which include transiting exoplanets as well as eclipsing binaries. It is not a confirmed exoplanet label.
- **Model Decoupling:** The catalog tabular classifier (`tce_e_vs_bj_classifier.joblib`) is strictly decoupled from the hybrid fusion inference path. Attempting to use the catalog classifier as a candidate vetting model triggers an explicit `ValueError`.
- **Preservation of Raw Physics:** Transit depths, durations, and shapes are never clipped or artificially flattened by detrending algorithms.
