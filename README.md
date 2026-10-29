# ASTROVET: Explainable TESS Transit-Candidate Vetting Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Baseline Pin](https://img.shields.io/badge/NASA%20ExoMiner%2B%2B-v1.0.0%20(d877a23)-blue.svg)](https://github.com/nasa/ExoMiner/releases)

**ASTROVET** is a hybrid, explainable machine-learning framework designed to vet and prioritize TESS (Transiting Exoplanet Survey Satellite) Threshold Crossing Events (TCEs) for human review and telescope follow-up.

> [!IMPORTANT]
> **Scientific Status:** ASTROVET is a **candidate-prioritization and triage aid**, NOT a planet confirmation engine. A high model score is never a planet confirmation or statistical validation. A low score is never proof that a target has no planet.

---

## 1. Architecture Overview

ASTROVET combines four complementary evidence branches into a late-fusion layer:

1. **NASA ExoMiner++ reference adapter:** The code records release metadata (`v1.0.0`, commit `d877a23`) and supports published-catalog lookup or container execution. In the requested D: workspace, the untouched NASA upstream source checkout and its prediction catalog are present beside ASTROVET and the catalog is auto-discovered for local lookups. Missing baseline predictions remain unavailable; ASTROVET does not invent scores.
2. **Transit Periodicity Verification (BLS & TLS):** Box Least Squares (Astropy BLS) and Transit Least Squares (TLS, `transitleastsquares>=2.0.0`) recover or compare transit-like periodicity. TLS uses limb-darkened transit templates. Period, epoch (BTJD), duration, and common period harmonics are compared; a detected periodic signal is not a planet validation.
3. **Physics-Informed Diagnostics:** Approximate limb-darkened transit template fitting (for residual baseline estimation), cadence-level and per-event odd vs even depth tests (eclipsing binary screen with propagated uncertainty), secondary eclipse search (phase ~0.5), and cross-event depth/timing scatter.
4. **Self-Supervised Light-Curve Representations:** Neural representation learner (`SelfSupervisedTransitEncoder`) optimizing a joint Masked Cadence Autoencoding (MAE) and View-Consistency contrastive objective over physically augmented folded transit views (cadence dropouts and bounded jitter), alongside an unsupervised PCA prototype encoder (`LightCurvePCAPrototypeEncoder`). Supports pretraining directly from real unlabeled TESS FITS light curves (`--input-fits` or `--input-fits-dir`) or pre-extracted views, recording complete checkpoint provenance (`REAL_TESS_FITS` vs `--synthetic-demo`). Pretraining over the full 60+ sector TESS archive requires GPU cluster compute.
5. **Fusion & Evidence Report:** The code contains a logistic fusion model and Platt calibration helper, but no production training/evaluation command or fitted model is bundled. Current default scores are uncalibrated heuristic triage scores. Missing modalities are tracked; reports include diagnostic plots and reason codes.

```text
[ TESS SPOC Light Curves & Catalogs ]
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
[  ExoMiner++  ]   [ Physics  ]   [  SSL Neural/ ]
[   Baseline   ]   [  Branch  ]   [ Encoder/PCA  ]
     │            │            │
     └────────────┬────────────┘
                  ▼
      [ Fusion & Calibration ]
                  ▼
     [ Triage & Evidence Report ]
```

---

## 2. Workspace & Boundary Structure

This ASTROVET checkout is kept separate from the NASA upstream source. In the requested D: workspace, the untouched upstream source is at `D:\Final Year Projet\ExoMiner-main\ExoMiner-main` and its published prediction catalog is auto-discovered for local baseline lookups. The upstream directory has no Git metadata in this workspace; its files are not modified by ASTROVET.

- `D:\Final Year Projet\ExoMiner-main\astrovet`: ASTROVET source, configuration, tests, and documentation.
- `data/raw/mast/`: locally downloaded observations, ignored by Git.
- `data/fixtures/`: synthetic FITS files used by automated tests only.
- `artifacts/` and `data/checkpoints/`: generated reports/checkpoints, not validated model releases.

```text
astrovet/
├── pyproject.toml               # Package specifications and dependencies
├── README.md                    # Project documentation, CLI guide, quickstart
├── configs/                     # YAML configuration files
│   ├── data/tess_spoc_2min.yaml
│   ├── baseline/exominerpp.yaml
│   ├── preprocess/default.yaml
│   ├── ssl/masked_views.yaml
│   ├── model/fusion.yaml
│   └── evaluation/grouped_temporal.yaml
├── docs/                        # Scientific documentation
│   ├── data_dictionary.md       # Full parameter dictionary
│   ├── label_policy.md          # Disposition policy and host-grouping rules
│   ├── model_card.md            # Model card and limitations
│   └── decisions/               # Architecture Decision Records (ADRs)
├── src/astrovet/                # Core implementation package
│   ├── cli.py                   # Click-based CLI entry point
│   ├── ids.py                   # Compound candidate identity handling
│   ├── provenance.py            # SHA-256 manifests and environment metadata
│   ├── catalogs/                # CSV audit & candidate resolution
│   ├── ingest/                  # MAST product fetching & FITS inventory
│   ├── lightcurves/             # Quality bitmasking & segmented normalization
│   ├── views/                   # Phase folding, binning & periodograms
│   ├── physics/                 # Transit model, odd-even, secondary diagnostics
?   ??? ssl/                     # Self-supervised neural encoder, dataset view extractor, PCA prototype & augmentations
?   ??? ui/                      # Interactive local review dashboard (astrovet ui) & HTML report viewer
│   ├── baseline/                # ExoMiner++ version-pinned adapter
│   ├── models/                  # Late fusion & Platt calibration
│   ├── evaluation/              # Grouped splits, PR-AUC, Brier score, ECE
│   └── explain/                 # Reason codes, diagnostic plots & reports
├── tests/                       # Unit, contract & integration tests
├── data/fixtures/               # Clearly marked synthetic test fixtures
└── artifacts/                   # Output diagnostic plots and vetting cards
```

---

## 3. Dataset Audit & Label Policy

### Input Dataset Audit (`tces_with_labels_v3.csv`)
The requested `tces_with_labels_v3.csv` and NASA prediction catalog are not present in this checkout. No row counts or label statistics are claimed. Audit an actual supplied file with `astrovet data audit --input <path> --output artifacts/audit_report.json` before using it. The current command audits structure and identifiers; it does not build model-ready labels or features from arbitrary catalogs.

### Label Policy
- Confirmed Planets (`CONFIRMED_PLANET`), Planet Candidates (`PLANET_CANDIDATE`), False Positives (`FALSE_POSITIVE`), and False Alarms (`FALSE_ALARM`) are retained as separate states.
- Ambiguous/unresolved records (`AMBIGUOUS_UNRESOLVED`) are **never converted into negative examples**.
- All dataset partitions (Train, Validation, Calibration, Test) are grouped strictly by host star (`tic_id`) to prevent cross-sector and multi-candidate leakage.

---

## 4. Quickstart & Installation

Because `astrovet` uses a `src/` package layout, install the package in editable mode before executing CLI commands:

```powershell
# Navigate into the ASTROVET project directory
cd "D:\Final Year Projet\ExoMiner-main\astrovet"

# Install package in editable mode
py -m pip install -e .

# Verify system environment and dependencies
py -m astrovet.cli doctor

# Run full test suite
py -m unittest discover tests/ -v
```

---

## 5. CLI Usage & Key Workflows

### 1. System Health & Baseline Verification
```powershell
py -m astrovet.cli doctor
```

### 2. Audit a Candidate Catalog CSV
```powershell
py -m astrovet.cli data audit --input "C:\\path\\to\\your\\tces_with_labels_v3.csv" --output "artifacts/audit_report.json"
```

### 3. Fetch TESS SPOC Light Curve with Provenance Manifest
ASTROVET defaults to **online retrieval** (`--online`), querying the STScI MAST archive for official TESS SPOC 2-minute cadence light curves. If network access is unavailable or offline testing is required, explicit offline and fallback controls are provided:

```powershell
# Online official MAST retrieval (default):
py -m astrovet.cli data fetch --tic 390651552 --sector 23 --output-dir "data/raw/mast" --manifest "data/manifests/mast_ingest.json"

# Strict online mode with fallback disabled (fails clearly if MAST is unreachable or target not found):
py -m astrovet.cli data fetch --tic 390651552 --sector 23 --online --no-fallback-fixture --output-dir "data/raw/mast"
```

### 4. Run ExoMiner++ Baseline Catalog Adapter
```powershell
py -m astrovet.cli baseline run --candidate-id "390651552-1-S23"
```

### 5. Pretrain Self-Supervised Neural Encoder & Checkpoint
Train the self-supervised representation learner (`SelfSupervisedTransitEncoder`) with joint Masked Cadence Autoencoding (MAE) and view-consistency contrastive objectives directly from real TESS light curves:
```powershell
# Pretrain on real TESS FITS light curve (e.g. downloaded SPOC 2-min file):
py -m astrovet.cli ssl pretrain --input-fits "data/raw/mast/tess_s0023_390651552_spoc_lc.fits" --epochs 5 --output-checkpoint "data/checkpoints/ssl_transit_encoder.npz"

# Pretrain across an entire directory of TESS FITS files:
py -m astrovet.cli ssl pretrain --input-fits-dir "data/raw/mast" --epochs 5 --output-checkpoint "data/checkpoints/ssl_transit_encoder.npz"

```
*(A vetting run without `--ssl-checkpoint` omits SSL embeddings and records that modality as unavailable. Self-supervised pretraining alone does not train a planet/non-planet decision head.)*

### 5a. Rebuild the Real-Label Research Cohort and Prototype

The project workspace includes scripts to match official MAST sector TCE statistics to the NASA Exoplanet Archive TOI disposition snapshot. `CP`/`KP` are positive; `FP`/`FA` are negative; `PC`/`APC`/unset remain unlabeled and are excluded. The generated research-only classifier uses host-disjoint train/validation/calibration/test splits. It is **not** connected to FITS upload inference and its probabilities are not population-calibrated because TOIs are a selected follow-up sample.

```powershell
py scripts\prepare_planet_disposition_cohort.py `
  --tce-dir "D:\Final Year Projet\ASTROVET_data\raw\mast_tce_single_sector" `
  --toi-csv "D:\Final Year Projet\ASTROVET_data\labels\nasa_exoplanet_archive_toi_dispositions.csv" `
  --output-dir "D:\Final Year Projet\ASTROVET_data\labels\planet_cohort"

py scripts\train_planet_disposition_prototype.py `
  --cohort "D:\Final Year Projet\ASTROVET_data\labels\planet_cohort\planet_disposition_cohort.csv" `
  --output-dir "D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype"
```

The scripts save source hashes, disposition counts, host split IDs, a test evaluation report, and a catalog-statistics research artifact. The catalog-statistics artifact is not a FITS-vetting model. The next workflow builds and evaluates an SSL-plus-physics head using a single inference-aligned feature vector:

```powershell
py scripts\train_ssl_physics_fusion.py `
  --cohort "D:\Final Year Projet\ASTROVET_data\labels\planet_cohort\planet_disposition_cohort.csv" `
  --download-manifest "D:\Final Year Projet\ASTROVET_data\raw\planet_disposition_lc_batch\download_manifest.json" `
  --split-manifest "D:\Final Year Projet\ASTROVET_data\labels\planet_cohort\planet_ssl_host_split_manifest.json" `
  --ssl-checkpoint "D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype\ssl_encoder_trainhosts_v1.npz" `
  --output-dir "D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype\ssl_physics_fusion"
```

The output includes `astrovet_ssl_physics_fusion_research.joblib`, host split IDs, an event-level feature audit, and untouched test metrics. The trained bundle can be enabled for the UI explicitly; in PowerShell, set the two environment variables before starting the dashboard:

```powershell
$env:ASTROVET_SSL_CHECKPOINT = "D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype\ssl_encoder_trainhosts_v1.npz"
$env:ASTROVET_FUSION_MODEL = "D:\Final Year Projet\ASTROVET_data\artifacts\models\planet_disposition_prototype\ssl_physics_fusion\astrovet_ssl_physics_fusion_research.joblib"
py -m astrovet.cli ui --port 8080
```

Or start the dashboard with these trained artifacts in one command:

```powershell
.\scripts\start_dashboard_with_trained_model.ps1
```

The dashboard and reports label the output as a selected-TOI-cohort calibrated score, not a population-wide planet probability. It still requires human review and independent validation on a broader sample before any scientific population claims.

### 6. Run Transit Periodicity Search (BLS & TLS)
Run Box Least Squares (Astropy BLS) and Transit Least Squares (TLS) on real light curve FITS data to recover or verify periodicity:
```powershell
# Run period search with known candidate ephemeris for validation:
py -m astrovet.cli periodogram search --fits-file "data/raw/mast/tess_s0023_390651552_spoc_lc.fits" --period 1.46580 --epoch 1931.1684 --duration 1.000

# Blind periodicity search without candidate ephemeris:
py -m astrovet.cli periodogram search --fits-file "data/raw/mast/tess_s0023_390651552_spoc_lc.fits" --min-period 0.5 --max-period 15.0
```

### 7. Run Candidate Vetting Slice & Generate Evidence Report
Execute candidate vetting with BLS/TLS verification, physics diagnostics, and configurable baseline/encoder branches:
```powershell
# Standard vetting on real observed FITS data with periodogram verification:
py -m astrovet.cli explain --candidate-id "390651552-1-S23" --fits-file "data/raw/mast/tess_s0023_390651552_spoc_lc.fits" --period 1.46580 --epoch 1931.1684 --duration 1.000

# Blind vetting with automated ephemeris recovery when catalog values are absent:
py -m astrovet.cli explain --candidate-id "390651552-1-S23" --fits-file "data/raw/mast/tess_s0023_390651552_spoc_lc.fits"

# Using a trained self-supervised neural encoder checkpoint:
py -m astrovet.cli explain --candidate-id "390651552-1-S23" --fits-file "data/raw/mast/tess_s0023_390651552_spoc_lc.fits" --encoder-mode ssl --ssl-checkpoint "data/checkpoints/ssl_transit_encoder.npz"

# Opting into containerized baseline execution (Podman/Docker):
# Note: If container daemon is absent, reports CONTAINER_RUNTIME_UNAVAILABLE honestly rather than claiming live execution.
py -m astrovet.cli explain --candidate-id "390651552-1-S23" --fits-file "data/raw/mast/tess_s0023_390651552_spoc_lc.fits" --baseline-mode container --container-fallback
```

The browser dashboard automatically loads `artifacts/models/ssl_encoder_real_qlp_trainhosts_v2.npz` when present, then falls back to `data/checkpoints/ssl_transit_encoder.npz`. Set `ASTROVET_SSL_CHECKPOINT` to override this path. Without a fitted checkpoint, the UI reports the SSL representation as unavailable; the score remains an uncalibrated heuristic unless a compatible, validated fusion artifact is provided.
*(Real-Data Requirement: Normal workflows require observed FITS data; missing data raises a clear error. Synthetic fixtures are test-only and are not used as scientific results.)*

---

## 6. Vetting Outputs & Evidence Card

Running `astrovet explain` generates:
1. `artifacts/reports/<candidate_id>_report.md` (Markdown evidence card with Section 4: Periodicity Search & Verification)
2. `artifacts/reports/<candidate_id>_diagnostics.png` (6-panel diagnostic plot)

### 6-Panel Diagnostic Plot Architecture:
- **Panel 1 (Top-Left):** Raw and detrended time series across full observational baseline with quality-filtered cadences.
- **Panel 2 (Top-Right):** Primary transit zoom phase-folded on best ephemeris with transit window and template fit.
- **Panel 3 (Mid-Left):** Box Least Squares (BLS) power spectrum across period search grid with peak period indicator.
- **Panel 4 (Mid-Right):** Transit Least Squares (TLS) power spectrum with Signal Detection Efficiency (SDE) peak indicator.
- **Panel 5 (Bottom-Left):** TLS folded light curve with realistic Mandel-Agol limb-darkened transit model profile.
- **Panel 6 (Bottom-Right):** Odd vs Even transit comparison and full-orbit phase curve (secondary eclipse screen).

> [!CAUTION]
> **Scientific Notice:** Periodicity detection (BLS/TLS) confirms signal recurrence and geometric profile consistency, **NOT** exoplanetary validation. Stellar activity, rotational modulation, and eclipsing binaries also produce strong periodogram peaks.

### 8. Launch Interactive Browser UI
Start the local dashboard for human review and candidate vetting:
```powershell
py -m astrovet.cli ui --port 8080
```
Open `http://127.0.0.1:8080/` in a web browser to review candidate ephemerides, trigger on-demand vetting with BLS and TLS metrics, examine physical diagnostics, and inspect diagnostic plots.
