# ASTROVET Implementation Status

Last audited: 2026-10-29

## Operational in this checkout

- **Current official disposition data:** Downloaded the NASA Exoplanet Archive TOI disposition table (8,151 rows) and all 107 latest single-sector MAST TCE-statistics catalogs (156,431 TCE rows; 248.09 MB). SHA-256 manifests are stored beside the downloaded data under `D:\Final Year Projet\ASTROVET_data`. A reproducible matcher created 18,121 period/epoch matches and a one-record-per-TIC/TOI hard-label research cohort: 1,782 events on 1,734 hosts (CP 565, KP 489, FP 706, FA 22). PC/APC/unset and ambiguous labels are excluded from the hard-label fit.
- **Research-only planet-disposition prototype:** Trained a leakage-safe logistic classifier on MAST DV TCE statistics, using host-disjoint 60/15/10/15 train/validation/calibration/test partitions. The untouched test set contains 266 events on 261 hosts. Test ROC-AUC `0.8577`, PR-AUC `0.8645` (test positive prevalence `0.5865`), Brier `0.1515`, ECE `0.1090`; at threshold 0.5, precision `0.8239`, recall `0.8397`, F1 `0.8317`. This is a research-only catalog-statistics baseline on the selected TOI-follow-up cohort, not the SSL hybrid or an upload-ready planet probability. Model bundle and metrics are under `D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype`.
- **Real MAST upload-path smoke run:** Downloaded an observed SPOC 120-second FITS for TOI 564.01 / TIC 1003831 / Sector 8 and ran the existing upload-equivalent CLI with the real SSL checkpoint and BLS/TLS. BLS recovered 1.64938 d (0.11% from the catalog ephemeris), TLS recovered 1.65021 d (0.06%), and the report correctly remained `REVIEW_REQUIRED` with uncalibrated score and unavailable ExoMiner reference. Report and diagnostic plot are under `D:\Final Year Projet\ASTROVET_data\artifacts\reports\real_toi_564_01_sector8`.

- **SSL-plus-physics fusion head:** Added `scripts/train_ssl_physics_fusion.py`. It extracts the same 16-dimensional real SSL embedding and 11 physics/missingness features used by upload inference, excluding ExoMiner catalog scores. It processed 1,771/1,782 labeled events from real matched SPOC FITS; 11 lacking a matching downloaded sector product are recorded in the feature audit and excluded. Training used the pre-existing 1,040-host-only SSL checkpoint and host-disjoint train/validation/calibration/test partitions (1,033/260/172/258 hosts represented after extraction). Validation selected logistic-regression C=10.0. Untouched test: 263 events / 258 hosts, ROC-AUC `0.7768`, PR-AUC `0.8251` (test prevalence `0.5817`), raw Brier `0.1909`, raw ECE `0.0882`; calibration-split Platt scaling yields test Brier `0.1903`, ECE `0.0510`. At a 0.5 calibrated-score threshold, precision `0.7074`, recall `0.8693`, F1 `0.7801`. Feature matrix, excluded-row audit, split IDs, metrics, and bundle are under `D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype\ssl_physics_fusion`.
- **Fusion artifact integration:** Fitted fusion bundles can now carry a train-only feature scaler and map/reorder named inference features safely. The dashboard reads `ASTROVET_FUSION_MODEL`; report/UI text exposes the calibration scope and identifies this selected-TOI-cohort score as not population-wide. The fitted model is opt-in and does not silently replace the default model.

- **Real Dataset Ingestion & Audit:** Full audit and validation of `D:\Final Year Projet\tces_with_labels_v3.csv` (24,926 candidates, SHA-256 `8c4412b073ac...`).
- **Reproducible Training Pipeline:** End-to-end training CLI (`astrovet train` and `astrovet model train`) implementing host-disjoint splits, train-only preprocessing, validation-only tuning, calibration-only Platt scaling, and single untouched test evaluation.
- **Trained Catalog Baseline:** Saved as `tce_e_vs_bj_classifier.joblib` with a host-disjoint split manifest and evaluation report. This model is intentionally not auto-loaded by candidate-vetting inference.
- **Local UI & Safe Upload Workflow:** Candidate inspection and browser dashboard server with drag-and-drop TESS SPOC/QLP FITS upload control. Enforces strict size/extension limits, FITS/gzip magic byte checks, sanitized paths, SHA-256 integrity, cadence attrition accounting, segment-aware normalization, and single-action end-to-end vetting with arbitrary server path rejection.
- **Real TESS SPOC/QLP FITS Ingestion:** Automated pipeline origin detection (SPOC `PDCSAP_FLUX` vs QLP `KSPSAP_FLUX`), quality filtering, segmentation, and normalization.
- **Periodicity Screening:** BLS and TLS periodicity search with harmonic and ephemeris verification.
- **Physics Diagnostics:** Transit model fit, odd-even consistency test, secondary eclipse search, and diagnostic report generation.
- **Per-candidate explanations:** Reports contain exact additive logit contributions for compatible fitted logistic fusion models, or an explicitly labeled arithmetic breakdown for the fallback heuristic. Representation-only SSL embeddings are not presented as planet-probability explanations.
- **Self-supervised input stability:** Folded light-curve views are robustly centered/scaled regardless of FITS flux units, cadence masks use the normalized baseline, optimizer gradients are clipped, and non-finite model states fail clearly. Directory pretraining records skipped-file reasons in checkpoint provenance.
- **Real QLP archive:** The downloaded archive is at `D:\Final Year Projet\ASTROVET_data\LC.tar.gz` (MD5 verified) and 11,511 host-filtered FITS products are available in `LC_train_hosts`. The host split manifest contains 11,487 training hosts.
- **Real-data SSL checkpoint:** `D:\Final Year Projet\ASTROVET_data\artifacts\models\ssl_encoder_real_qlp_trainhosts_v2.npz` was trained on 97,857 views extracted from 1,000 deterministic QLP FITS files restricted to training hosts. No files were skipped. Joint loss decreased from `1.296555` to `0.707230` over five epochs; reconstruction loss decreased from `1.232723` to `0.698956` and view-consistency loss from `0.127663` to `0.016548`. The artifact is a representation encoder, not a classifier.
- **Dashboard encoder integration:** The upload-and-vet endpoint automatically discovers the project-local real-data checkpoint or accepts `ASTROVET_SSL_CHECKPOINT`; the dashboard reports whether the SSL representation was actually used and displays a per-score explanation.
- **Real-data end-to-end inference:** `D:\Final Year Projet\ASTROVET_data\artifacts\reports\real_qlp_case_183978641_with_periodograms\183978641-1-S39_report.md` and its plot were generated from an observed QLP FITS file. TLS recovered the supplied 10.029754-day ephemeris (0.01% difference), while BLS selected a different period (29.02% difference); ASTROVET correctly exposes this as `PERIODICITY_MISMATCH` and `BLS_TLS_PERIOD_DISAGREEMENT` and routes the candidate to `REVIEW_REQUIRED`.

## Evaluated Test Performance (Real Labeled Cohort)

- **Test Cohort:** 2,878 candidates on 2,873 independent host stars (zero host leakage from train/val/cal).
- **ROC-AUC:** `0.8901`
- **PR-AUC:** `0.5368`
- **Expected Calibration Error (ECE):** Reduced from `0.2027` (raw) to `0.0222` (calibrated).
- **Brier Score:** Reduced from `0.1417` (raw) to `0.0825` (calibrated).
- **Precision / Recall / F1:** Precision `0.6069`, Recall `0.4077`, F1 `0.4877` (at default decision threshold 0.50).

## Remaining Boundaries & Limitations

- **Target meaning:** E is a periodic eclipsing signal label and includes planet transits as well as non-contact eclipsing binaries. The trained E-versus-B/J score does not identify planets and must not be displayed as a planet probability.
- **Catalog-only scope:** The trained classifier uses tabular transit/stellar parameters (`Period`, `Duration`, `Depth`, `SMass`, `SRad`, `Tmag`, etc.) only. It is a baseline, not the hybrid model; its artifact is blocked from fusion inference to prevent misuse.
- **SSL scope:** Self-supervised pretraining learns representations from unlabeled TESS curves. It does not, by itself, learn a planet/non-planet decision boundary. The real-data encoder checkpoint is a validated training artifact for software integration, but its scientific utility requires downstream labeled evaluation.
- **Hybrid inference boundary:** An SSL-plus-physics decision head is now trained from the matched raw-FITS labeled cohort and can be explicitly enabled for upload inference. It is trained on a small, selected TOI follow-up sample and must be treated as a research candidate-prioritization score. Broader independent host/sector/pipeline validation and decision-threshold/abstention review are still needed before population-level or autonomous disposition claims. The separate E-vs-B/J classifier remains a different task and stays blocked from planet-vetting inference.
- **Upstream Containerized Baseline:** The untouched NASA ExoMiner source checkout and its prediction catalog are present beside ASTROVET at `D:\Final Year Projet\ExoMiner-main\ExoMiner-main`; default adapter/report discovery uses that catalog. Live NASA pipeline execution follows the official container interface but requires Docker/Podman daemon. Docker CLI is installed on the audited machine, but its daemon was unavailable during this audit, so live container execution was not verified.
- **Long-Period & Shallow Candidates:** Subgroup evaluation shows reduced PR-AUC for long periods ($P \ge 15$ d: PR-AUC 0.1296) and shallow depths ($<500$ ppm: PR-AUC 0.1782) due to lower signal-to-noise ratio and extreme class imbalance.

## Authoritative Data and Baseline References

- [NASA ExoMiner source and model documentation](https://github.com/nasa/ExoMiner)
- [MAST TESS mission data and products](https://archive.stsci.edu/missions-and-data/tess)
- [MAST TESS TCE bulk-download catalog](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html)
- [NASA Exoplanet Archive TOI disposition definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
- [NASA Exoplanet Archive TAP/API guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [MAST QLP High-Level Science Product](https://archive.stsci.edu/hlsp/qlp)
- **Hybrid head real-data end-to-end smoke:** Ran held-out real FITS candidate `2758565-1-S29` through the fused encoder/physics model and full default BLS/TLS workflow. Raw score `0.2917`, selected-cohort calibrated score `0.4005`; BLS and TLS recovered discrepant low-significance periods (4.62% and 143.49% from the supplied ephemeris, respectively), so ASTROVET correctly returned `REVIEW_REQUIRED` with periodicity mismatch/agreement reasons. Report and plot are under `D:\Final Year Projet\ASTROVET_data\artifacts\reports\fusion_model_full_smoke_test`.
