# ASTROVET Project Progress Report

**Prepared:** 29 October 2026
**Project:** ASTROVET — hybrid physics-informed, self-supervised and explainable TESS exoplanet-candidate triage
**Primary checkout:** `D:\Final Year Projet\ExoMiner-main\astrovet`
**Data and model artifacts:** `D:\Final Year Projet\ASTROVET_data`
**Status basis:** Source code, project documentation, trained model artifacts, audit manifests and real-data inference reports available in the local workspace at the time of preparation.

> **Scientific-use statement:** ASTROVET is a candidate-prioritization and human-review aid. The current fusion model was trained and evaluated on a selected TOI follow-up cohort. Its calibrated score must not be presented as a population-wide planet probability, statistical validation, or planet confirmation.

## 1. Executive summary

ASTROVET has progressed from a vetting pipeline prototype to a working real-data workflow with a trained downstream model. It accepts TESS light-curve FITS data, filters and normalizes observations, constructs folded transit views, calculates signal and physical diagnostics, computes a self-supervised representation, and can use a fitted physics-plus-SSL fusion model to rank a candidate. It can produce a Markdown evidence report, a diagnostic plot, and an explanation of the logistic model's additive feature contributions.

The work includes two distinct kinds of model that should not be confused:

1. A catalog-statistics baseline trained on the Astronet-Triage-style E-versus-B/J target. This task is **not planet identification**, because the `E` label includes both transiting planets and eclipsing binaries.
2. A newer SSL-plus-physics head trained on NASA Exoplanet Archive dispositions matched to real MAST light curves. It addresses CP/KP versus FP/FA labels, but only within a selected TOI follow-up cohort.

The newer fusion head processed 1,771 of 1,782 hard-labeled cohort events. On its host-disjoint test partition (263 events / 258 hosts), it achieved ROC-AUC 0.7768 and PR-AUC 0.8251 at test prevalence 0.5817. Platt scaling reduced test ECE from 0.0882 to 0.0510, while Brier score changed modestly from 0.1909 to 0.1903. At a 0.5 threshold, precision was 0.7074, recall 0.8693, and F1 0.7801. These metrics describe the selected cohort; broader independent validation is still required.

A held-out real FITS candidate was also processed through the complete CLI path including BLS and TLS. The model score was 0.2917 raw and 0.4005 after selected-cohort calibration. The period searches disagreed, so the system returned `REVIEW_REQUIRED`. This confirms that the implemented pipeline can surface conflicting evidence rather than silently calling a candidate a planet.

## 2. Project goal and scope

ASTROVET is intended to help vet TESS Threshold-Crossing Events and candidate light curves. The design combines:

- Real observed TESS SPOC or QLP light curves.
- Automated quality filtering, normalization and transit-view construction.
- Independent periodicity screening with BLS and TLS.
- Physics-informed diagnostics such as transit fitting, odd/even comparison and secondary-eclipse screening.
- Self-supervised representation learning from unlabeled light curves.
- A supervised downstream fusion head trained using vetted dispositions.
- Reviewable explanations and explicit uncertainty / missing-modality signals.
- NASA ExoMiner/ExoMiner++ as a reference baseline when available, rather than a replacement implementation.

The code keeps the upstream ExoMiner checkout separate from the ASTROVET package. The default adapter can use the available catalog. Live upstream container execution depends on a working Docker or Podman daemon; it was not verified in the local audit because the daemon was unavailable.

## 3. Implemented architecture

### 3.1 Ingestion and preprocessing

The FITS reader recognizes the TESS light-curve products and keeps pipeline-specific flux columns distinct: SPOC `PDCSAP_FLUX` and QLP `KSPSAP_FLUX` are supported, with metadata retained. Quality filtering records the policy and cadence attrition rather than silently discarding observations. Normalization is segment-aware and returns both normalized and detrended flux, with transit windows protected during detrending when ephemerides are supplied.

### 3.2 Transit views and signal search

The pipeline folds a light curve using period and epoch and creates a primary transit view, full-orbit view, odd/even views and individual event stamps. BLS and TLS can independently search for periodic transit-like structure and compare recovered periods and epochs to the supplied ephemeris. Harmonics and disagreements are surfaced as evidence for review.

### 3.3 Physics diagnostics

The physics branch calculates approximate transit-model fit values, odd/even depth consistency, secondary-eclipse signals, transit-event consistency, period/duration/depth plausibility and BLS/TLS availability or agreement. These are screening diagnostics, not proof that a signal is planetary. Missing diagnostics are represented with availability indicators and should not be interpreted as negative evidence.

### 3.4 Self-supervised encoder

`SelfSupervisedTransitEncoder` learns a 16-dimensional representation from folded light-curve views using masked reconstruction and view-consistency objectives. It does not use the disposition target to learn the representation. A real-data checkpoint was trained using 1,050 FITS files restricted to training hosts, producing 173,718 views over five epochs. The stored training losses decreased from 1.149569 to 1.127902 overall, 1.139346 to 1.120470 for reconstruction, and 0.020447 to 0.014864 for view consistency.

A separate QLP pretraining artifact is also documented: 97,857 views from 1,000 deterministic QLP files, with no skipped files. Both artifacts are encoders, not classifiers; the fusion head is what learns the disposition decision boundary.

### 3.5 Fusion head and explainability

The current research fusion head combines 16 SSL embedding values with 11 physics and diagnostic-availability features. The upstream ExoMiner catalog score is deliberately excluded from this fusion model so an uploaded light curve does not depend on locating a catalog score. Training-only scaling and feature-name mapping are stored in the model artifact; inference reorders named features and rejects missing trained inputs rather than feeding a misaligned vector.

The downstream classifier is logistic regression. Reports can show the largest additive logit contributions and reconstructed logit. Such contributions explain model associations for this fitted head; they are not causal explanations or direct physical evidence.

## 4. Data assembled and label policy

### 4.1 Source datasets

- NASA Exoplanet Archive TOI dispositions: 8,151 rows downloaded through the archive TAP interface.
- MAST single-sector TCE-statistics catalogs: 107 current files, 156,431 TCE rows, approximately 248.09 MB.
- Real matched MAST SPOC FITS products for the selected labeled cohort: the product download manifest records 1,752 downloaded files, sizes and checksums.
- A separate locally available QLP light-curve archive is documented in the implementation-status notes.

### 4.2 Cohort construction

TCEs were matched to TOI records using TIC identity and tight period/epoch agreement. Matching disallowed harmonic-only matches. The hard-label cohort contains 1,782 events on 1,734 hosts: 565 CP, 489 KP, 706 FP and 22 FA. CP/KP are treated as positive; FP/FA as negative. PC, APC, unset and unresolved labels are excluded from hard-label training. This is a reproducible research cohort, not a random survey of all TESS targets.

Eleven cohort rows had no matching downloaded FITS sector product and were excluded from the FITS feature cohort. Their exclusions are recorded in `planet_fusion_feature_audit.json`; the resulting cohort used for fusion contains 1,771 events.

### 4.3 Leakage safeguards

The fusion workflow splits by host TIC into train, validation, calibration and test groups. After FITS availability filtering, the represented groups were:

| Partition | Events | Hosts | Positive / negative |
|---|---:|---:|---:|
| Train | 1,061 | 1,033 | 615 / 446 |
| Validation | 270 | 260 | 176 / 94 |
| Calibration | 177 | 172 | 103 / 74 |
| Test | 263 | 258 | 153 / 110 |

The SSL checkpoint used for this fusion run was pretrained using training-host light curves only. Feature scaling was fitted on train only, regularization was selected on validation, Platt calibration was fitted on the separate calibration partition, and the test partition was reserved for evaluation.

## 5. Models and measured results

### 5.1 Catalog-statistics classifier (separate task)

A research classifier was trained on MAST DV catalog features. Its target was CP/KP-versus-FP/FA for the TOI cohort in the newer disposition dataset. The separate older TCE-vetting model trained on `tces_with_labels_v3.csv` uses E-versus-B/J labels, which do **not** mean planet versus non-planet. It is intentionally prevented from loading as the planet-vetting fusion model.

The TOI catalog-statistics research prototype reported an untouched test ROC-AUC of 0.8577 and PR-AUC of 0.8645 on 266 events / 261 hosts. This is a different feature branch and three more test events were available than for the FITS fusion run, so its scores should be treated as a reference, not a strict apples-to-apples model comparison.

### 5.2 SSL-plus-physics fusion model

The fusion model evaluation artifact reports:

| Metric | Held-out test result |
|---|---:|
| Events / hosts | 263 / 258 |
| Positive prevalence | 0.5817 |
| ROC-AUC (raw) | 0.7768 |
| PR-AUC (raw) | 0.8251 |
| Brier (raw → calibrated) | 0.1909 → 0.1903 |
| ECE (raw → calibrated) | 0.0882 → 0.0510 |
| Precision / recall / F1 at calibrated score 0.5 | 0.7074 / 0.8693 / 0.7801 |

At that 0.5 cutoff the confusion matrix is `[[55, 55], [20, 133]]` in negative/positive class order. Thus 55 of 110 negative events are false positives at that threshold. The score cutoff is a reported diagnostic, not a final operating threshold. The deployed triage logic also checks physics, periodicity agreement, missing modalities and uncertainty, so the exact application decisions still need separate evaluation against the intended review policy.

### 5.3 Real-data end-to-end example

Candidate `2758565-1-S29` was run from a real TESS FITS file through the trained fusion artifact, physics branch and full BLS/TLS search:

- Raw fusion score: 0.2917.
- Selected-TOI-cohort calibrated score: 0.4005.
- Transit fit: SNR 27.4 and reduced chi-square 3.313; odd/even comparison passed.
- BLS: 3.6073 days, SNR 4.35, 4.615% from supplied ephemeris.
- TLS: 9.20824 days, SDE 5.42, 143.488% from supplied ephemeris.
- BLS/TLS period disagreement: 60.825%.
- Triage: `REVIEW_REQUIRED`, with periodicity mismatch and BLS/TLS disagreement reasons.

The report labels calibration scope, shows diagnostic evidence and includes additive feature contributions. The report's candidate metadata itself says `UNREVIEWED`; the disposition label remains in the training cohort, not presented as a prediction-time input.

## 6. Interface and how to run it

A local browser dashboard supports FITS upload and single-action vetting. The trained fusion model is opt-in through `ASTROVET_FUSION_MODEL`; the encoder path is configured through `ASTROVET_SSL_CHECKPOINT`. A PowerShell launcher sets both paths and starts the configured dashboard:

```powershell
cd "D:\Final Year Projet\ExoMiner-main\astrovet"
.\scripts\start_dashboard_with_trained_model.ps1
```

The dashboard served successfully at `http://127.0.0.1:8081` during the current session. A real FITS file was processed successfully through the CLI with the same trained model and full BLS/TLS path. The browser page itself was checked for HTTP 200, but a multipart FITS upload through the browser has **not** yet been verified end to end.

To retrain the fusion head from the saved cohort, downloaded-product manifest, host-split manifest and real encoder checkpoint:

```powershell
cd "D:\Final Year Projet\ExoMiner-main\astrovet"
py scripts\train_ssl_physics_fusion.py `
  --cohort "D:\Final Year Projet\ASTROVET_data\labels\planet_cohort\planet_disposition_cohort.csv" `
  --download-manifest "D:\Final Year Projet\ASTROVET_data\raw\planet_disposition_lc_batch\download_manifest.json" `
  --split-manifest "D:\Final Year Projet\ASTROVET_data\labels\planet_cohort\planet_ssl_host_split_manifest.json" `
  --ssl-checkpoint "D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype\ssl_encoder_trainhosts_v1.npz" `
  --output-dir "D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype\ssl_physics_fusion"
```

## 7. Reproducibility and repository artifacts

Important source and documentation files:

- `scripts/prepare_planet_disposition_cohort.py` — join TCE statistics to TOI dispositions and build the hard-label cohort.
- `scripts/download_disposition_tce_lightcurves.py` — obtain matched MAST products and record product provenance.
- `scripts/train_planet_disposition_prototype.py` — catalog-statistics research baseline.
- `scripts/train_ssl_physics_fusion.py` — common SSL/physics feature extraction, host-grouped tuning, calibration, evaluation and artifact saving.
- `scripts/start_dashboard_with_trained_model.ps1` — configure the trained model and launch the upload UI.
- `src/astrovet/models/fusion.py` — fusion-model artifact loading, feature alignment, calibration and inference explanations.
- `src/astrovet/explain/report.py` — real-data vetting pipeline and report generation.
- `src/astrovet/ui/server.py` and `src/astrovet/ui/static/index.html` — upload server and dashboard.
- `docs/implementation_status.md` and `docs/model_card.md` — current metrics, model scope, source links and limitations.

Research data and model artifacts are stored outside the source package under `D:\Final Year Projet\ASTROVET_data`, with source files, manifests, feature matrices, split IDs, evaluation JSON and reports. These paths are specific to this Windows machine and need to be configured on another system.

## 8. Checks performed and not yet performed

### Performed for the current fusion integration

- Python compilation checks passed for the fusion trainer, fusion model, report generator and UI server.
- The fusion artifact loaded with 27 named features, a fitted scaler and a fitted calibrator.
- Real held-out FITS CLI inference completed with the fitted encoder and fusion model.
- Full BLS/TLS search and report generation completed for the real held-out example.
- The local dashboard home page returned HTTP 200 on port 8081.

### Not yet completed after the latest fusion integration

- A browser multipart upload of a real FITS file and inspection of the returned dashboard JSON/result.
- A complete rerun of the project unit/integration suite after the latest fusion, report and UI changes. An earlier project audit reported 74 passing tests, but that result predates this integration.
- A blinded independent validation outside the selected TOI-follow-up cohort.
- Evaluation of the exact deployed triage/review policy and subgroup metrics on a new, untouched cohort.
- Live ExoMiner++ container execution; the local Docker/Podman daemon was unavailable in the prior audit.

## 9. Remaining work and completion gates

### Engineering / release checks

- [ ] Run the full test suite after the fusion integration.
- [ ] Submit a real FITS through the dashboard browser form at `http://127.0.0.1:8081` and confirm score, SSL availability, diagnostics, XAI explanation, report and plot all return correctly.
- [ ] Test bad files, incomplete ephemerides, MAST retrieval failure and missing modalities.
- [ ] Evaluate exact production triage outcomes, including `PLANET_LIKE`, `FALSE_POSITIVE_LIKE` and `REVIEW_REQUIRED`; do not rely only on metrics at a generic 0.5 threshold.
- [ ] Confirm artifact installation and model-path configuration on a clean machine; document CPU/runtime and storage requirements.
- [ ] Verify Docker/Podman live baseline only if it is a required project acceptance criterion.

### Scientific validation (the key open item)

- [ ] Build an independent labeled cohort of hosts not used by training, validation or calibration.
- [ ] Evaluate across later sectors and, where feasible, another TESS pipeline such as QLP. Keep TIC hosts disjoint as well as reserving the desired sector/pipeline domain.
- [ ] Freeze labels, release date, source manifests, model artifact and thresholds before opening the final test set.
- [ ] Select thresholds on validation data based on reviewer requirements for recall, precision and false-positive rate; leave final test untouched.
- [ ] Report ROC-AUC, PR-AUC (with test prevalence), confusion matrix, precision, recall, false-positive rate, Brier, ECE, review/abstention coverage and confidence intervals.
- [ ] Evaluate subgroups by pipeline, sector, cadence, period, depth, magnitude and SNR; investigate poor-performance groups before deployment.
- [ ] Calibrate on a separate calibration cohort from the intended use population. Do not transfer the current selected-TOI calibration as a general population calibration.

## 10. Risks and limitations

1. **Selection bias:** TOI targets are selected for follow-up and do not represent the full TESS target population. Test prevalence is about 58%, so score calibration will not transfer automatically to lower-prevalence populations.
2. **Independent data scale:** The disposition cohort is modest, with a single selected TCE event per TOI identity and one downloaded sector product for each matched event where available. It does not fully sample repeated observations, all sectors or all pipelines.
3. **Operating point:** At threshold 0.5, the held-out fusion model false-positive rate was 50% (55/110 negatives). High recall comes with substantial false positives at that operating point. The current application uses extra review rules; evaluate those exact rules before assigning operational meaning.
4. **Weak or conflicting periodogram peaks:** BLS/TLS can identify periodic signals but cannot by themselves establish a planet. Disagreements correctly drive human review.
5. **Approximate diagnostics:** The transit-template fit and derived physical checks are screening measures with model and data assumptions.
6. **XAI scope:** Logistic contributions show how features affect the fitted score; SSL coordinates do not map directly to physical causes.
7. **Reference baseline:** The ExoMiner catalog branch can be unavailable for a candidate; live execution requires its container runtime.
8. **No autonomous confirmation:** Candidate dispositions require human expertise and, where appropriate, follow-up observations or statistical validation.

## 11. Authoritative references

- [NASA ExoMiner source and model documentation](https://github.com/nasa/ExoMiner)
- [MAST TESS mission data and products](https://archive.stsci.edu/missions-and-data/tess)
- [MAST TESS TCE bulk-download catalogs](https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html)
- [NASA Exoplanet Archive TOI disposition definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
- [NASA Exoplanet Archive TAP/API guide](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
- [MAST QLP High-Level Science Product](https://archive.stsci.edu/hlsp/qlp)
- [Astronet-Triage-v2 dataset, Zenodo](https://zenodo.org/records/7411579)
- [Tey et al. (2023), Astronomical Journal](https://doi.org/10.3847/1538-3881/acad85)

---

**Summary:** The real-data ingestion, self-supervised representation, physics diagnostics, trained fusion head, explainable reporting, and an end-to-end CLI example are in place. The software is ready for further controlled evaluation. The largest remaining task before broad scientific use is independent out-of-cohort validation, followed by threshold selection and browser upload verification.
