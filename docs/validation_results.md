# Upload and Light-Curve Validation Results

**Run date:** 2026-10-30  
**Scope:** Real-file smoke validation of FITS audit, preprocessing, browser upload, CLI/report parity, and upload rejection behavior. This is a focused integration check, not broad validation over the TESS archive.

## Real FITS files exercised

| Pipeline | Source file | Usable cadences | Result |
|---|---|---:|---|
| SPOC | TIC 1003831, Sector 8, `tic1003831_s0008_tess2019032160000-s0008-0000000001003831-0136-s_lc.fits` | 12,907 | Pipeline identified as `TESS_SPOC`; upload audit passed. |
| QLP | TIC 592638, Sector 39, `astronet_tess_ffi_s0039-0000000000592638_tess_v01_llc.fits` | 4,343 | Pipeline identified as `TESS_QLP`; upload audit passed. |

Both files were run through the real vetting workflow, without synthetic fallback. CLI and browser upload used the same fusion model, its matching self-supervised checkpoint (`ssl_encoder_trainhosts_v1.npz`), data, search settings, and candidate measurements where available. The model/checkpoint SHA-256 compatibility guard rejects mismatched artifacts.

## Browser, CLI, and saved report parity

The multipart browser upload endpoint was exercised for each real FITS file. The generated report was downloaded through the browser report URL and compared to the CLI-generated report on score, triage, and candidate/BLS/TLS periods.

| Candidate | Raw score | Triage | Candidate, BLS, TLS period (days) | Parity |
|---|---:|---|---|---|
| `1003831-1-S8` (SPOC; supplied P=1.65114 d, epoch=1518.2 BTJD, duration=0.872646 h, depth=3056.04 ppm) | 0.6942 | `PLANET_LIKE` | 1.65114, 1.64938, 1.65021 | Exact score/triage and period agreement within 1e-5 days. |
| `592638-1-S39` (QLP; blind recovery) | 0.0690 | `REVIEW_REQUIRED` | 25.74145, 18.65158, 25.74145 | Exact score/triage and period agreement within 1e-5 days. |

The SPOC source candidate is marked `UNREVIEWED` in the available catalog record, so its `PLANET_LIKE` output is not a confirmed-planet correctness measurement. The QLP BLS/TLS disagreement appropriately results in `REVIEW_REQUIRED`; no independent disposition label was available for this QLP example.

Browser inputs now include optional transit depth in ppm. It is passed through both browser vetting routes to the same report workflow as CLI `--depth`, enabling matched candidate inputs.

## Cleaning and transit preservation

A synthetic 1000 ppm box-shaped transit was injected into each of the two real light curves and measured before and after the current preprocessing/detrending path. Both files recovered 1000.0 ppm from `flux_raw_norm` and `flux_detrended` (100% depth recovery in this controlled injection). The audits reported usable-cadence retention of 72.70% for SPOC and 90.55% for QLP.

This checks depth preservation for one injection per file. It does not establish completeness across transit periods, durations, depths, stellar variability, instrumental artifacts, or all quality flags. A parameter-grid injection-and-recovery campaign remains needed for that claim.

## Rejected upload cases

| Input | Observed response |
|---|---|
| Malformed FITS payload with FITS signature but invalid HDU content | HTTP 400 with a FITS validation error |
| CSV submitted as an upload | HTTP 400 with an invalid file-extension message |
| Request larger than the configured 50 MB upload limit | HTTP 413 |

The request-size check runs before reading the declared request body. These checks cover representative failures; they do not replace fuzzing or a full security review.

## Docker baseline and hardware check

**Checked:** 2026-10-30 on the workstation running this dashboard.

- Docker Desktop is installed and its daemon is running. The pinned NASA ExoMiner image is present locally (about 16.3 GB).
- The dashboard uses the lightweight published-catalog lookup by default. The advanced option **Also run NASA ExoMiner in Docker for a comparison** explicitly requests the slower container run.
- A real SPOC candidate was sent through that container path. ASTROVET's integration input was corrected to give the official pipeline an integer TIC CSV in its writable output mount. ExoMiner began inference, then returned `All model predictions failed ... Probably due to out-of-memory`. Docker reports about 8.15 GB of memory available on this machine. The run produced no baseline score, so the app leaves it unavailable and explains why.
- The pinned ExoMiner container's `run_pipeline.py` calls `tf.config.set_visible_devices([], 'GPU')` and prints that GPUs are disabled. Its log also reported no CUDA driver inside the container. ASTROVET's current self-supervised encoder is vectorized NumPy designed for CPU use. Docker provides an isolated runtime; it does not automatically accelerate CPU implementations with the desktop GPU.
- The dashboard is now started with the trained SSL encoder and fusion artifact using `scripts/start_dashboard_with_trained_model.ps1`. A real SPOC upload returns ASTROVET score `0.6942` and selected-sample calibrated score `0.7178` (displayed as `71.8%`). That score is limited to a selected TOI follow-up cohort; it is not a planet probability for the general TESS population.

The ExoMiner score must stay unavailable until either the candidate is found in the reference catalog or the container can finish inference within its available memory. Do not replace a missing score with zero or a fabricated value. Enabling GPU use would require a separately validated GPU-capable ASTROVET implementation and a changed ExoMiner image; the pinned NASA baseline is left unchanged.

## Automated checks

- Repository test suite: **76 passed, 1 warning** after the browser-depth field addition.
- Added model/checkpoint compatibility tests ensure the inference encoder matches the checkpoint used to train the fusion model.
- Manual real-file integration harness verified upload audit, report retrieval, CLI/browser parity, and the three rejection cases listed above.
- The later UI/ExoMiner container changes have not had a full repository test-suite rerun yet; their selected SPOC API and container paths were exercised directly.

## Validation boundary and next evidence

The current evidence demonstrates that these two real examples pass end-to-end and that the supplied transit depth survives this specific preprocessing check. It does **not** demonstrate general performance across TESS sectors or establish scientific planet-confirmation capability. Before presenting this as broadly validated, run a host-star-disjoint, disposition-labeled cohort spanning SPOC and QLP; measure candidate-level precision/recall and calibration; and perform injection/recovery grids stratified by cadence, sector, depth, period, duration, gaps, and stellar variability. Preserve per-file manifests, model/checkpoint hashes, and report outputs for those runs.
